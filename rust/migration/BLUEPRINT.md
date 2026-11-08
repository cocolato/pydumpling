# Blueprint: native value walker for pydumpling

Scope: the bounded capture walk in `pydumpling/snapshot/values.py` and the
`SnapshotContext` helpers it calls. Everything else stays in Python: the
writer, reduction, the debugger, the report, and the CLI. The pure-Python walk
remains the reference implementation and the fallback when the extension is
not built, disabled (`PYDUMPLING_NATIVE=0`), or asked for an unlimited
snapshot.

Why only this part: capture runs while the failing process waits, and the
profile showed that the value walk dominates bounded capture time. The
debugger and file handling are I/O bound or built on `pdb`, and a whole-project
port would replace a pure-Python, all-platform wheel with per-platform builds
for no measurable gain.

## 1. Codebase audit, legacy census and latent bugs

Legacy census (`migration_telemetry.json`, `scan-codebase --phase before` on
`values.py`, `identity.py`, `context.py`):

| Metric | Value |
| :-- | --: |
| Files | 3 |
| Code LOC | 567 |
| Comment density | 14.1% |
| Largest file | 429 LOC (`values.py`) |
| Max nesting depth | 7 |
| Average function length | 18 LOC |
| Memory or state hazards | 0 (pure Python) |

The contract the port has to keep is strict:

- Capture never runs application code. Types are compared by identity,
  through `type.__dict__` descriptors, and containers are read through their
  builtin base.
- The order of memo entries, node counts and truncation counters is part of
  the snapshot. The pickled snapshot must be byte-identical.
- Recursion overflow is handled by `_root_value`, and its result depends on
  the Python stack depth.

The native walk keeps that contract by construction: it **never runs
application code**. Wherever the Python walk would (a mapping's own
`items()`, a str-subclass `__qualname__`, a class dict with keys that are not
exact str, keys hashed by the private-attribute filter, argument shapes its
callers never pass), and wherever the result depends on the Python recursion
limit or the native stack runs short, the walker undoes the call and hands
the whole call to the Python walk before any such code runs (near CPython's
own C recursion limit, see section 3). Application hooks therefore run
exactly as often as without the extension.

Audit findings (logged with `log-opportunity`):

| Category | Finding | Where |
| :-- | :-- | :-- |
| performance | The value walk is the hot path of a bounded capture, paying about 1–3 µs of interpreted dispatch per node. | `values.py::ValuesMixin.value` |
| performance | Unlimited snapshots probe each value with `dill.dumps` before descending. A deep chain ending in one unserializable leaf re-serializes each subtree on the way down: O(depth²). | `ValuesMixin.value`, `SnapshotContext._serializable` |
| hygiene | `_attribute_mapping` registers the id of a temporary filtered dict in `_namespaces`. It is only freed unremembered when the node budget is exhausted, after which nothing else is captured, so the stale id is never read. Benign; noted, not changed. | `ValuesMixin._attribute_mapping` |
| safety | A native port must not recurse on the C stack unboundedly. Python 3.12+ frames do not use it, but a Rust walk does. | new code |

The adversarial review of the port also confirmed these legacy behaviours.
The native walk reproduces them byte for byte (or hands the call to Python),
so they are reported here and not changed:

| Finding | Where |
| :-- | :-- |
| A hashable dict subclass inside a set is copied to an unhashable `FakeDict`, so `captured.add` raises and the whole set is lost. | `ValuesMixin.value` |
| A non-dict namespace is read through its own `items()` up to three times; a one-shot view loses entries in later passes. | `ValuesMixin._mapping` |
| After a `RecursionError` in a frame variable, `_root_value` forgets the half-built copies but keeps their node count and counters. | `ValuesMixin._root_value` |
| Instances whose `__dict__` is a member descriptor (`SimpleNamespace`, modules) lose their attributes. | `ValuesMixin._attributes` |
| The first capture of an exception materializes its `__dict__`, so a second capture differs. | `ExceptionsMixin` |
| A str-subclass `__qualname__` runs its own `__radd__` and slicing. | `SnapshotContext._type_name` |
| On Python 3.15 an over-long int's `repr` holds its address, so exception messages differ between dumps. | `ExceptionsMixin._exception_message` |

## 2. Target safe Rust architecture and module graph

Crate `rust/` (`pydumpling-native`), PyO3 0.30.1, `abi3-py312` (one binary for
CPython 3.12+), `stacker` for the thread's real stack bounds,
`#![forbid(unsafe_code)]`, every file at most 400 LOC. Python 3.11 keeps the
Python walk: there, C calls count toward the recursion limit, so a call handed
back from native code would overflow earlier than the Python walk alone, at
any recursion limit. From 3.12 they count toward CPython's own C recursion
limit only, which recursion through C calls reaches, under a raised
recursion limit or, in C code, at the default one, as does, from 3.14, a
small thread stack; there the same holds (see section 3).

```
lib.rs          module `pydumpling._native`: Statics, Walker, Dispatch
├─ dispatch.rs  descriptor: ctx.value / ctx._mapping / Class.value(ctx) / Class.value(self=ctx) → walker or Python; refuses same-thread re-entry
├─ arguments.rs the argument shapes the walker reads without running application code
├─ walker.rs    Walker pyclass, State (read from the instance dict and validated per call), retire checks, GC traverse
├─ shadow.rs    per-call check, without hooks, that the context's classes hold the stock walk methods and no state attribute; finds the instance dict
├─ mirror.rs    syncs the memo with ctx.objects / ctx._sources: adopts Python's entries, rebuilds on replacement, notes originals held without a copy
├─ flush.rs     writes the unwritten copies in the Python walk's order, Python's later entries behind them, in-place entries only where still held
├─ transaction.rs  per-call transaction, recursion headroom probe, stack budget, hand-back (HandBack marker), retire
├─ walk.rs      value(): exact scalars, containers, FakeClass objects
├─ copies.rs    _scalar, _fill_copy, _filled_slot, _layout
├─ mapping.rs   _mapping, _attribute_mapping, _root_value
├─ keys.rs      key selection: preferred names, link order, limit + 1
├─ text.rs      budgets, counters, _text/_code_text/_bounded_bytes/_type_name/_omit fast paths
├─ types.rs     per-type, per-call facts from the MRO (bases, __dict__ descriptor, non-str class-dict keys)
├─ statics.rs   per-import table of builtin and fake types, unbound base methods
└─ memo.rs      insertion-ordered identity memo with journaled rollback (pure Rust)
```

| Legacy file and symbol (Python) | Target Rust module and type/fn | Transformation |
| :-- | :-- | :-- |
| `values.py::ValuesMixin.value` | `walk.rs::State::value`, `container`, `object` | Exact-type pointer dispatch; same step order |
| `values.py::ValuesMixin._mapping` | `mapping.rs::State::mapping`, `fill_mapping` | `try`/`except RuntimeError` becomes `Halt::Py` matching |
| `values.py::_key_order`, `_plain_key`, selection | `keys.rs::KeyOrder`, `Chosen`, `State::select` | Lazy generator chain becomes explicit passes; values kept from the same view |
| `values.py::_attribute_mapping`, `_root_value` | `mapping.rs` | Native never overflows; it hands back instead |
| `values.py::_fill_copy`, `_filled_slot`, `_scalar`, `_layout` | `copies.rs` | `_layout` called once per type and call |
| `values.py::_attributes`, `_container_base`, `_scalar_copy_type`; `identity.py::_type_mro`, `_derives` | `types.rs::TypeInfo::compute` | MRO read once per type and call, not per value |
| `context.py::_remember`, `objects`, `_sources` | `memo.rs::Memo<Entry>`, `text.rs::remember`, `mirror.rs`, `flush.rs` | `HashMap` index plus ordered `Vec`, with undo journal; written to `objects` lazily, in the same order |
| `context.py::_omit`, `_type_name`, `_text`, `_code_text`, `_bounded_bytes`, `_has_nodes` | `text.rs` | Fast path natively; slow path calls the Python helper (same side effects), or hands back if a tracing hook would see it |
| (new) | `dispatch.rs`, `arguments.rs`, `transaction.rs`, `release.rs` | No extra Python frame; transactional hand-back; copies written before the cyclic collector clears a walker |

## 3. API ergonomics and call sites

Call sites do not change. `frames.py`, `exceptions.py` and the tests keep
calling `self.value(...)` and `self._mapping(...)`; the class attribute is a
native descriptor.

Before (`values.py`):

```python
class ValuesMixin:
    def value(self, value, depth=0):
        cached = self.objects.get(id(value), _UNMEMOIZED)
        ...
```

After (`values.py`; the method body is unchanged):

```python
if _native is not None:

    class _Walker(_native.Walker):  # writes its copies before a cycle is cleared
        __slots__ = ()
        __del__ = _native.Walker.release  # native: pushes no Python frame


class ValuesMixin:
    _walker = None
    _stock_walk = None  # recorded by context.py for SnapshotContext

    def _start_walker(self):  # last in __init__: it reads the state set before
        if _native is None or os.environ.get("PYDUMPLING_NATIVE") == "0":
            return
        # The walker checks the class and the state before every call.
        stock = object.__getattribute__(self, "_stock_walk")
        if stock is not None:
            object.__setattr__(self, "_walker", _Walker(_STATICS, stock))

    def __getstate__(self):  # copies and pickles see every copy
        try:  # no class hook reads, removes or restores the walker
            walker = object.__getattribute__(self, "_walker")
        except AttributeError:
            return _detached(object.__getstate__(self))
        if walker is not None:
            walker.flush()  # from now on it writes each call's copies at once
        try:
            object.__delattr__(self, "_walker")
        except AttributeError:  # only the class default: no walker
            return _detached(object.__getstate__(self))
        try:
            return _detached(object.__getstate__(self))  # never the live dict
        finally:
            object.__setattr__(self, "_walker", walker)


if _native is not None:
    _STATICS = _native.Statics(globals(), _identity)
    ValuesMixin.value = _native.Dispatch(ValuesMixin.value, owner=ValuesMixin)
    ValuesMixin._mapping = _native.Dispatch(
        ValuesMixin._mapping, mapping=True, owner=ValuesMixin
    )
```

Design decisions that the first test run forced:

- Instance attributes (`self.value = walker.value`) failed for throwaway
  contexts (`SnapshotContext(o).value(x)`): the context died while its bound
  method ran. A strong walker→context reference would fix that but create a
  cycle, so originals would outlive the capture until a full GC.
- The descriptor returns a bound method holding the context. The walker
  never stores the context, so there is no cycle and no extra Python frame.

Decisions the adversarial review forced (test iteration 4):

- **Hand back before, not after.** Handing a call back after the native walk
  had already called an application `items()` ran that code twice. Every
  place the Python walk would run application code is now a hand-back point
  reached before any is run, and the reviewers' hook counts match exactly.
- **Stack budget from the real thread.** A fixed 256 KiB budget crashed
  (SIGBUS) on small thread stacks. The budget is now
  `min(256 KiB, stacker::remaining_stack() − 64 KiB)`.
- **Hidden frames.** CPython 3.13+ runs each `__init__` reached from a class
  call under a shim frame that counts toward the recursion limit but is not
  visible through `f_back`. (Superseded in iteration 5 by the headroom probe.)
- **No cached state across calls.** Budgets, counters, the node count and
  per-type facts are reread on every call, because Python code (such as an
  exception's `__reduce_ex__`) runs between calls and may change a class's
  `__bases__` or the context.
- **One table per import.** The table of fake classes is built from the
  `values` module's own globals, so a reloaded or vendored package copies into
  its own classes.
- **Methods stay methods.** A subclass that overrides a walk method gets no
  walker. `__getstate__` writes the walker's copies first, so copies and
  pickles capture in Python from the same state.

Decisions the second adversarial review forced (test iteration 5):

- **Recursion headroom is measured, not estimated.** Counting visible frames
  twice was wrong under an audit hook and slow on deep stacks. One failed
  `sys._getframe(min((limit − 97) / 2, 1024))` now proves a stack shallow
  enough for the first 32 walk levels; otherwise, and every time the walk
  needs more, the walker recurses into `values._headroom(n)` and lets
  `RecursionError` answer (exponential, then binary search). The interpreter
  counts the hidden shim frames itself, the common case raises no audit
  event, and the check costs about a microsecond per call.
- **The memo is mirrored incrementally.** Exporting the whole memo on every
  hand-back cost 30–125 ms with 300,000 entries. `ctx.objects` now holds the
  walker's oldest entries in order; the newer ones are written before any
  Python code reads them (a hand-back, `__getstate__`, a retire, the walker's
  release, or its move to another context's dicts), and the copies Python
  appends are adopted from the tail. A hand-back on a 300,000-entry memo
  takes 0.17 ms.
- **Retire for state only Python reproduces.** Options that are not exactly
  `DumpOptions`, budgets that are not `None` or a positive exact int, a code
  text limit below 9, `_unlimited` set, a `_truncation` that is not an exact
  `Counter`, `_namespaces` that is not an exact set, `_nodes` that is not an
  exact non-negative int, `objects` or `_sources` that are not exact dicts,
  or an instance attribute that shadows a walk method: the walker writes its
  copies and leaves the context to the Python walk for good.
- **Lookups that could run application code hand back.** A key colliding
  with a name the walk reads in a copy's or a placeholder's instance dict,
  and a foreign getset planted as a metaclass's `__dict__`.
- **Same-thread re-entry runs in Python.** A finalizer or signal handler
  that captures on the same context during a native walk takes the Python
  walk; another thread still gets `RuntimeError`. (Superseded by the third
  review: that Python walk read a memo missing the call's own copies.)
- **The class attribute keeps routing.** `SnapshotContext.value` is the
  descriptor and routes `SnapshotContext.value(ctx, ...)`: the Python function
  run directly on a context whose walker holds unwritten copies would read a
  stale `objects`. `ctx.value` is a real `types.MethodType`; `__doc__`,
  `__name__`, `__qualname__`, `inspect.signature`, `inspect.getsource`,
  pickling, copying and weak references follow the function.
- **Stack budget.** `min(256 KiB, (remaining − 64 KiB) / 4)`: collections and
  finalizers that run between two checks need the rest.

Decisions the third review forced (a correctness follow-up and its review):

- **Walk methods are checked before every call, without hooks.** The walker
  compares each walk method, and `__getattribute__`, `__setattr__` and
  `__delattr__`, with the stock class's (`_stock_walk`, which
  `values._walk_methods` reads the same way). Class dicts are read through
  the `type.__dict__` descriptors, never a metaclass, and a lookup takes the
  raw class attribute without its `__get__`; only the classes in front of a
  name's owner are searched, and the MRO tuple is compared by identity. The
  instance dict is scanned on every call: its length is no mutation token.
  A class patched after its contexts were made is therefore seen, and no
  check reads the context through a patched `__getattribute__` first.
- **The receiver passed by keyword routes too.**
  `SnapshotContext.value(self=ctx, ...)` reaches the walker like
  `SnapshotContext.value(ctx, ...)`, so it cannot read a stale `objects`.
- **Tracing hooks see only the Python walk.** The walker runs Python code in
  a few places: the stack probe and pydumpling's own helpers (`_text`,
  `_code_text`, `_bounded_bytes`, `_layout`, `FakeClass.__repr__`). Before
  any of them, if a profile or trace function or a `sys.monitoring` tool is
  set, the call goes to the Python walk instead, so the hook, which is
  application code, runs exactly where the Python walk runs it. The answer
  is read once per call through `sys` functions cached at import; a call
  that needs no Python code stays native under a profiler or coverage.
  (Refined by the fifth review: hooks set from C or during the call.)
- **Signal handlers and collection callbacks do not hand back.** Inside a
  native call a Python signal handler, and a collection with its callbacks
  and finalizers (an allocation only schedules one), runs only where the
  walker checks for signals, every 4096 nodes so that Ctrl-C still
  interrupts a long capture, or in its Python code. Either ends the call as
  it would end the Python walk's, unless
  it captures through the same context (refused, below). Handing back
  whenever a handler was installed turned the walker off in every
  `asyncio.run` program (asyncio installs a SIGINT handler) and every server
  with a SIGTERM handler, and was slower there than the Python walk alone.
- **Same-thread re-entry is refused.** A finalizer, collection callback,
  audit hook or signal handler that captures through the same context while
  the walker runs gets `RuntimeError`, as does a copy of the context made
  then: the call's own copies are not yet written where the Python walk
  could share them. Other contexts capture normally.
- **The memo stays mirrored lazily.** Writing every call's copies to
  `ctx.objects` and looking earlier ones up there halved the capture
  speed-up (7.1× to 3.5×) and raised peak RSS above the Python walk's, only
  to honor code outside the walk changing that internal dict, which nothing
  in pydumpling does. Replacing the dict is still honored.
- **Copies keep subclass slots; frame access is optional.** `__getstate__`
  returns Python's own `(dict, slots)` state without the walker, and an
  audit hook that refuses `sys._getframe` at import only turns off the
  frame-locals fast path.

Decisions the fourth review forced (the review of that follow-up):

- **State is read from the instance dict, not through the class.** The
  walker reads the options, budgets, counters, `objects`, `_sources`,
  `_nodes`, `_layouts` and the rest of `values._NATIVE_STATE` from the
  context's own dict, reached through the class's plain `__dict__`
  descriptor once the class check passed, and writes `_nodes` back there;
  `_walker` goes through `object.__getattribute__` and `object.__setattr__`.
  The class check covers the state names too: a class attribute of one of
  them, such as a property set when the class was made or patched on after a
  call, sends the context to the Python walk before the walker reads it, so
  that property and the class's `__getattribute__` and `__setattr__` run
  exactly as in the Python walk. So does a missing state attribute, a
  `_layouts` that is not an exact dict, or a tally that is not an exact int
  below 2**63 − 1. The Python pre-check at construction is gone; the
  per-call check covers it.
- **Hashing a copy is Python code.** Adding a copy that is not a plain key,
  such as a named tuple's, to a set or using it as a dict key may run
  `FakeContainer.__hash__`; under a tracing hook the call goes to Python
  first. The gate is conservative: a copy whose hash is C, such as an enum
  member's or an object's, hands back too, which costs speed only under a
  hook, never the snapshot.
- **After a copy, every call writes its copies at once.** A shallow copy
  shares `objects` with its original, so once `__getstate__` has written the
  walker's copies, the walker writes each call's copies when it commits, and
  the copy and the original share every copy, as in the Python walk.
- **Copies survive a replaced `_sources`.** A retire or a flush writes the
  copies into the dicts the walker mirrored, without reading the context
  again, so replacing `_sources` alone loses none. (Completed by the fifth
  review for a plain dict in its place.)
- **Hostile context classes.** A context class whose MRO has a class-dict key
  that is not an exact str captures in Python, and the check looks nothing
  up in it. Releasing a walker imports nothing.
- **Cost.** The check reads each walk, state and copy-hook name along the
  MRO, `__copy__`, `__replace__`, `__getattr__` and `__getattribute__`
  along a metaclass's other than `type`, and the instance dict. With the
  rest of the per-call state reading it brings a memo hit on
  `SnapshotContext` from 0.29 µs, in the first native build, to about
  1.08 µs (the seventh review's changes add 0.11 µs, the eighth's 0.07 µs,
  the ninth's and tenth's nothing measurable), more with a longer MRO or a
  metaclass, and up to about 0.1 µs more once the dicts hold entries the
  walker synced with, where each call checks the keys ending them
  (1.15–1.17 µs against the ninth's 1.07 µs in the same runs, once the
  walker wrote them); the Python walk's memo hit costs
  0.05 µs. A dump makes a few calls per frame: over the six benchmark
  workloads, in two interleaved runs, capture is 12–13% slower than in the
  first native build and save 4–5% (geomean throughout; the tenth review's
  changes cost nothing measurable of either), still 3.5 times as fast as
  the Python walk to capture and 2.1 times to save. A throwaway context
  capturing one small value is faster in the Python walk.

Decisions the fifth review forced (the review of that change):

- **Hooks are asked for before each Python call.** A profile or trace
  function set from C without an object, as yappi sets one, is invisible to
  `sys.getprofile` and `sys.gettrace`, and on 3.12 and 3.13 a tool that
  freed its id keeps its events; a signal handler, a collection or another
  thread can set a hook while a native call runs. `sys.monitoring._all_events()`
  shows every profile and trace function (tool ids 6 and 7) and every tool's
  global events, from any thread, so the walker asks it before each Python
  call, at about 10 ns; the six tool ids, for a tool with only per-code
  events, are asked once per call. Asking them every time too cost about
  15% on calls that truncate many strings. Without `_all_events` the walker
  asks `sys.getprofile` and `sys.gettrace`.
- **A copy's in-place replacement is adopted.** A shallow copy's Python walk
  may capture, as a root, a mapping its original omitted by depth, and so
  replace the placeholder in the `objects` they share under the same key:
  once the context has been copied, every call rechecks the walker's depth
  placeholders, as after a hand-back.
- **Python's entries go after the walker's.** Entries Python code adds to
  `objects` or `_sources` between calls, while the walker holds unwritten
  copies, are taken out (their keys are exact ints, which run no code;
  widened by the sixth review to every key whose hash is builtin),
  the walker writes its copies, and they go back after them, so both dicts
  have the Python walk's order; one for an original the walker copied
  replaces that copy in place, and the walker rebuilds its memo from
  `objects`. Replacing `_sources` alone, with any dict, first writes the
  unwritten copies into the old pair, as the Python walk wrote them there.
- **A released walker writes what either dict lacks:** a placeholder it
  replaced in place too, and the originals into a `_sources` the context
  still holds after `objects` was replaced.
- **The default state is built as without a walker.** `__getstate__` reads,
  removes and restores `_walker` with `object.__getattribute__`,
  `object.__delattr__` and `object.__setattr__` around
  `object.__getstate__`, so a class with a `_walker` slot and `__getattr__`
  runs `__getattr__` as often as one that captures in Python.
- **A `__dict__` descriptor that refuses the context**, such as another
  class's placed in the context class before 3.15, which accepts it, sends
  it to the Python walk.

Decisions the sixth review forced (the review of the fifth review's fixes):

- **Python's entries are noticed by count, both ways.** An original Python
  code keeps in `_sources` alone, as `exception` keeps a builtin exception,
  and an entry it removes from either dict make the walker write its
  unwritten copies first, as an entry added to `objects` did. Before, such
  an original went after copies later calls made, and a removed entry of
  Python's own dropped the walker's unwritten copies from its memo.
- **A replaced placeholder goes where it was replaced.** The walker keeps
  the memo length at which a call replaced a copy it had already written,
  and writes that original into a `_sources` that lacks it, such as a
  replaced one, after the copies made before, as the Python walk appended
  it. A hand-back forgets the replacements the undone call recorded.
- **A memo key that is not an id goes to the Python walk.** The memo is
  keyed by address, so a key in `objects` other than an exact int, such as
  a str, a float equal to an id, an int subclass or an object with
  `__index__`, sends the context to the Python walk for good, which looks
  it up as it would; the type check runs no code. Python's entries are
  moved behind the walker's when every key's hash and comparison are
  builtin (an exact str, int, float, complex, bool, bytes, `None` or tuple
  of them); otherwise they stay, and so does a copy Python wrote for an
  original the walker had yet to write.
- **A replaced `_sources` releases the originals at the next call.** The
  memo is rebuilt from `objects` with the originals the new `_sources`
  holds; before, it kept the old ones alive until `objects` was replaced.
- **A copy made during a hand-back makes the walker write at once.** While
  the Python walk takes a call, `_walker` holds a `HandBack` instead of
  `None`. It routes the context's calls to Python as `None` does, and its
  `flush`, which `__getstate__` calls, notes the copy.
- **Naming a placeholder needs no Python code.** An ASCII str no longer
  than the limit fits, a byte per character, so the walker keeps it as
  `_bounded_text` would (`str.isascii` is C). `tuple` and `frozenset`, the
  placeholders the walker names lazily, always fit the code text limits it
  accepts (9 or more), so no hook can make a pending one go missing, and
  strings that fit need no helper under a hook either.
- **A tally is read only as an exact int.** Its type is checked before its
  value, so its `__index__` never runs.
- **The default state is detached in both walks.** `__getstate__` returns a
  copy of the instance dict with or without a walker, so a subclass that
  edits it, or a `__setstate__` that adopts it, behaves alike in both.
- **A walker pickles and copies as `None`**, once it has written its
  copies, as for `__getstate__`. This review also made it the instance
  dict's first key, so that a reducer handing over `self.__dict__` met it
  before `objects`; the seventh review undid that (below).

Decisions the seventh review forced (the review of the sixth review's fixes):

- **The walker is the instance dict's last key again.** An instance dict
  releases its values in its class's shared-key order on 3.12 and 3.13,
  which the class's first instance fixes, so a walker
  released first found `objects` and `_sources` still held and wrote every
  unwritten copy into dicts about to be freed: about 70 ms and a memory
  peak above the Python walk's for 50,000 ints. It starts at the end of
  `__init__` again, and releasing a context writes nothing.
- **A context class with a copy hook of its own captures in Python.**
  `__reduce_ex__`, `__reduce__`, `__getstate__`, `__copy__` and
  `__deepcopy__` join the names the per-call check reads
  (`values._COPY_PROTOCOL`): such a hook may build a state from the instance
  dict, where the walker comes after `objects`, or give a shallow copy the
  walker itself. The Python walk's memo is always complete when copied.
- **An entry the Python walk wrote in place stays removed.** The Python
  walk writes a replaced placeholder, and an original `_sources` already
  held (as `exception` keeps a builtin exception), over the entry at once,
  so Python code that removes it later keeps it removed. The walker notes
  whether `_sources` held a key it replaced, and the ids `_sources` holds
  without a copy, and writes those entries only where the dict still holds
  one. Ids are noted as Python adds them, counted, and anew when the memo is
  rebuilt; one removed before the walker copied it is forgotten, including
  by a flush outside a call, such as `__getstate__`, which checks for
  removals before writing.
- **One dict for `objects` and `_sources` retires the walker.** The Python
  walk writes every original over its own copy there.
- **A helper's tally is checked before the helper runs.** `_text`,
  `_code_text` and `_bounded_bytes` add to a tally themselves; the walker
  reads it as an exact int first, so one of another type hands the call
  back before the helper adds to it, and the Python walk adds to it once.

Decisions the eighth review forced (the review of the seventh review's
fixes):

- **A removal inside a hand-back is noticed past the additions.**
  Application code that a hand-back's Python walk runs, such as a mapping's
  `items()`, can remove an entry from `objects` or `_sources` while that
  walk adds its own, so the count grows. At the hand-back's resync, and
  before every call once the context was copied, the walker checks that the
  key ending the part of each dict it mirrored is still the one it saw
  there, and rebuilds its memo from the dicts otherwise. Before, it kept a
  removed copy, or an original held without a copy, and wrote it back.
- **An original Python code removes is let go.** A `_sources` shorter than
  when last mirrored rebuilds the memo, at the next call or at a flush such
  as `__getstate__`, so the original is held nowhere once its caller lets it
  go and its id can be reused, as in the Python walk. Before, the memo kept
  it alive until the context was released.
- **A `_walker` slot captures in Python.** An instance releases its slots
  before its dict, so a walker in a slot was released before `objects` and
  `_sources` and wrote its unwritten copies into them. The slot is no longer
  taken for the stock attribute.
- **Copy hooks are looked for where `copy` and `pickle` look.**
  `__getnewargs_ex__` and `__getnewargs__`, which `object.__reduce_ex__`
  calls before it reads the state, and `__getattr__`, which can answer
  `__deepcopy__`, join `values._COPY_PROTOCOL`. An instance attribute of a
  copy-hook name, and a metaclass `__copy__`, `__getattr__` or
  `__getattribute__` other than `type`'s, any of which can answer a lookup
  `copy` or `pickle` makes, send the context to the Python walk too, checked
  before every call; so does a metaclass dict key that is not an exact str,
  whose `__eq__` looking a name up may run.
- **A tally's keys must be plain.** Reading a helper's tally before the
  helper runs looks a reason up in `_truncation`; a key there that is not
  of a plain kind, whose `__eq__` that lookup may run where the Python walk
  runs none, sends the context to the Python walk.
- **Releasing is tested on a fresh class.** Which value an instance dict
  releases first depends on its class's first instance, so the test makes a
  new class and first instantiates it natively, with and without slots: an
  instance the Python walk makes first, which never sets `_walker`, puts
  the walker last whatever `__init__` does. On `SnapshotContext` the test
  passed only because earlier tests had made one natively.
- **Outcomes compare what a pickle keeps.** An outcome that does not
  pickle compares by an exception's `args`, slots and other members, and a
  dict, list or tuple subclass's items, besides the instance dict.

Decisions the ninth review forced (the review of the eighth review's
fixes):

- **A flush checks where Python left the dicts.** A copy, pickle,
  `__getstate__`, hand-back, retire or release flushes without the sync a
  call makes first, and checked counts only: after a copy, Python code that
  removed an entry and added two left the count larger, and the flush wrote
  the memo back with the removed entry in it. Every flush but a commit's,
  whose sync has just checked, now checks the key ending each mirrored part
  and rebuilds the memo otherwise.
- **The sync checks that key whenever a count changed.** Before, only a
  hand-back's resync and, once the context was copied, every call did;
  before any copy, a removal with more additions between two calls went
  unnoticed.
- **Unwritten copies go before what Python added, after a removal.**
  Writing its copies, the walker took Python's additions to be the entries
  past the counts it last wrote; after a removal some of those were the
  walker's own, and additions stayed ahead of its copies. It now takes them
  to begin after the entry it wrote last or, if Python removed that one, at
  the first key it does not know. Found by an out-of-tree differential
  fuzzer that mixes removals and additions between calls, padding
  `_sources` so that its count changes.
- **A `_sources` key of another kind goes to the Python walk.** Rebuilding
  the memo, noting the keys Python added to `_sources`, and finding one
  among the entries a flush puts after the walker's, a key not of a plain
  kind now sends the context to the Python walk before a lookup of an
  original could run its `__eq__`; letting go of the originals Python
  removed reads `_sources`'s keys in order instead of looking each id up.
  Before, a rebuild after a removal ran such a key's `__eq__`, where the
  Python walk's next call, a memo hit, ran none.
- **`__replace__` is a copy hook.** `copy.replace` (3.13+) looks
  `__replace__` up on the class, through the metaclass, so it joins
  `values._COPY_PROTOCOL` and the metaclass names. `@dataclass` adds one
  from 3.13, so a dataclass context captures in Python there.
- **The hooks test reads the metaclass.** Its metaclass logs reads of
  `__mro__`, `__bases__`, `__dict__`, `__name__` and `__flags__`, and a
  native call reads none.

Decisions the tenth review forced (the review of the ninth review's
fixes):

- **Every sync checks the key ending each dict.** Checked only when a
  count changed, a removal with as many additions between two calls went
  unnoticed before any copy, as was accepted; but once a flush checked
  where Python left the dicts, a later flush noticed it and put Python's
  addition, made before the walker's next copies, after them. The order
  was right only if Python made its change after those copies, and the
  flush cannot tell which. Each call's sync now checks, before the call
  copies, and the accepted difference is gone. It costs nothing while
  neither dict holds an entry the walker synced with, and up to about
  0.1 µs per call once either does, whether the walker wrote it, through a
  hand-back, a copy or a flush, or Python added it, as `exception` does
  for a builtin exception late in a dump; capture and save are unchanged
  in the benchmark.
- **Python's additions begin at the first key the walker does not know.**
  After a removal the walker took them to begin after the entry it wrote
  last, wherever that was; Python code that removed that entry, added
  another and put it back, in `objects` or under the same `_sources` key,
  made the walker put its copy after both. One forward scan now stops at
  that entry or at the first key the walker does not know, whichever comes
  first, and after as many entries as the walker last saw in that dict,
  since Python only removes from those and appends its own: before, an
  original the walker wrote that Python put into a `_sources` it had
  replaced counted as the walker's. Two reviewers found the first independently, one through the
  test's docstring, which claimed the case but put back only a new
  `_sources` key; the out-of-tree fuzzer found the second once it stopped
  padding `_sources`, which the check at every sync made unnecessary.
- **`__class__` is a copy hook.** `copy.replace` calls the `__replace__`
  of the class `obj.__class__` names, which a class attribute of that
  name, a property or a plain class as proxy classes set, answers in place
  of `type(obj)`; it joins `values._COPY_PROTOCOL`.

Decisions the eleventh review forced (the review of the tenth review's
fixes):

- **A memo entry's own class is not asked.** A memo hit in `_mapping`
  called `isinstance(entry, TruncatedValue)`, which reads the `__class__`
  of an entry not derived from it, and handed the call back when that
  named a placeholder the entry was not exactly. An entry Python code
  added, such as a mock or a lazy proxy that answers `__class__` itself,
  ran that code once in the walker and again in the Python walk, which a
  stateful one answered differently. The walker now decides by the exact type: it reads an exact
  placeholder, knows a builtin scalar or container is none, and hands
  anything else back before running anything. This was in the first
  native build.
- **The tests pin the tenth review's fixes.** With the check at every
  sync reverted, the parameters that held a copy the walker had yet to
  write still passed, because the call after Python's change only found
  that copy again: it now makes a new copy. And the entry the walker wrote
  last, put back after an addition, always landed past the entries the
  walker last saw, which the cap alone handled: a new test removes two
  earlier entries too, so only the stop at the first unknown key puts the
  walker's copy ahead of Python's addition.

Decisions the twelfth review forced (the review of the eleventh review's
fixes):

- **A walker writes before the cyclic collector clears it.** A captured
  object that refers back to its context puts the context, its walker
  and the originals the memo keeps alive in one reference cycle, which
  only the cyclic collector frees, clearing its objects in an order of its
  own. When it cleared the walker first, the memo was gone before the
  walker was freed, so `objects` kept elsewhere lacked the copies the
  walker had yet to write, even with the returned copy still held. Writing
  them while being cleared would be too late: the collector has already
  found those copies unreachable and clears them too. The walkers `values`
  makes now have a finalizer, `__del__` on a subclass, as PyO3 sets no
  `tp_finalize`. CPython runs it before it clears any object of a cycle,
  then keeps alive whatever it made reachable again: copies the walker
  writes into a dict that outlives the cycle survive it intact, and
  originals it writes into a `_sources` kept elsewhere keep the cycle
  alive, as the Python walk's dicts did. The finalizer runs once, so a
  walker kept alive that way writes each call's copies when it ends from
  then on. This was in the first native build.

Decisions the thirteenth review forced (the review of the twelfth review's
fix):

- **The finalizer is native.** A `__del__` written in Python pushes a
  frame, and failed before it wrote anything when the collector ran where
  no frame fits, at the recursion limit, or with a signal pending, which
  it swallowed, or under a tracer that raised: the cycle lost the copies
  again. `__del__` is now the method `_native.Walker.release` itself,
  which CPython calls with no Python frame, so none of those can stop it
  (a spent C stack still can: see the fifteenth review). It
  raises nothing, and leaves a walker that is in use alone.
- **A walker tells its dying context's dicts from kept ones.** While
  finalizers run, the dying context still holds both dicts, so the
  reference count took each for kept, and the finalizer wrote into both.
  On 3.12 and 3.13 the collector does not track a dict that holds only
  objects it does not track, as the walker's still-empty dicts, so they
  are outside the objects it frees, and what the finalizer wrote into them
  made the whole cycle reachable again: it took a second collection to
  free, and then the cycle stayed alive for as long as something kept
  `objects`. The walker now holds a weak reference to its context, with a
  callback that notes the context was freed first, and took a reference
  found dead with its callback not run to mean that the context dies with
  the walker, so that one holder of each dict is its own; the fourteenth
  review showed where that does not hold.
- **Originals go into `_sources` only when it is kept.** When only
  `objects` is kept, the copies are written into it, and the originals,
  which keep the cycle alive, into a copy of `_sources` that is dropped:
  a flush writes both dicts together. The test counts what one collection
  leaves alive, which, on 3.13, fails with this or the reference count's
  adjustment reverted, and collects a cycle in the deepest frame the
  recursion limit allows.

Decisions the fourteenth review forced (the review of the thirteenth
review's fixes):

- **A walker freed by reference counting has no dying context.** A
  context that a finalizer of the collected cycle frees by reference
  counting, as a generator owning it, or an object whose `__del__` lets
  go of it, does when the collector finalizes that object first, released
  both dicts before its walker; the walker's weak reference had been
  cleared with no callback all the same, so the walker took the context
  for one dying with it, and a dict kept once elsewhere lacked the copies
  the first native build wrote. The finalizer now reads the walker's own
  reference count: CPython calls a finalizer from deallocation with it at
  one, while the collector holds the walker as it finalizes it.
- **The memo's references are the walker's own.** A walk that reaches the
  context, as at the default depth, memoizes both dicts as originals,
  which counted as another holder: the finalizer wrote into the dying
  context's dicts and, before 3.14, kept the cycle alive another
  collection. The count leaves out the memo's reference too.
- **A dying context counts only where dicts go untracked.** From 3.14 the
  collector tracks every dict, and what the finalizer writes into one that
  only the cycle holds is freed with it, so the walker there takes no
  reference for its context's own and keeps no weak reference: a context
  brought back by another finalizer or by `gc.DEBUG_SAVEALL`, a walker
  taken off a living context and a dict taken out of the context all get
  the copies, as in the Python walk. Before 3.14, which the walker tells
  from `gc.is_tracked({})`, it still does, and there the collector also
  clears a weak reference that only dying objects hold, whether or not its
  referent lives, so the walker goes by the callback alone: a context it
  was not told was freed is taken to die with it. The weak reference now
  follows the context of the walker's latest call, so a walker moved to
  another context, or one whose context came back, takes that one for its
  own.
- **Counts are read without calling Python.** `sys.getrefcount` is a call,
  which failed where the C recursion budget was spent, and the finalizer
  then wrote nothing; the walker now reads each count itself. CPython
  itself refuses there to call the native `__del__` at all, on every
  version (see the fifteenth review), which the walker cannot change while
  PyO3 sets no `tp_finalize`.
- **Drop writes only for a walker without a finalizer.** Once the
  finalizer ran, the walker may be freed while the collector clears the
  cycle, after the copies it holds were cleared. The fifteenth review
  removed that write.
- **The recursion-limit test checks itself.** Under a Python tracer the
  tracer's own call overflowed a frame deeper, so the collection ran where
  a frame fits, and the test passed with a Python `__del__`; it now skips
  unless no Python frame fits where it collects.

Decisions the fifteenth review forced (the review of the fourteenth
review's fixes):

- **Only the finalizer writes.** Where the C stack is spent, CPython
  refuses to call the walker's finalizer on every version, not only on
  3.13, and Drop, which went by whether the finalizer ran, took such a
  walker for one without a finalizer. When the collector freed it, Drop
  wrote copies the collector had cleared, or was about to clear, into a
  dict kept elsewhere: emptied instances, lists and dicts. And it ran
  while the `RecursionError` that had spent the stack was on its way up,
  so the dict method its write called failed, and PyO3 took the pending
  error with that failure: the program got `SystemError: error return
  without exception set` in place of its `RecursionError`, and on 3.15 the
  interpreter could crash. The walker cannot tell a refused finalizer from
  no finalizer, nor a collection from reference counting, without calling
  Python where that fails too, so Drop no longer writes at all: a walker
  whose finalizer CPython refuses loses the copies it had yet to write,
  whatever frees it. The `released` flag went with it, which a direct call
  of `release` also set, so that a walker with no finalizer then wrote
  nothing as reference counting freed it. pydumpling never makes one; a
  walker of the native class itself now never writes as it is freed.
- **The C-stack loss is on every version, and has two causes.** Besides
  CPython's refusal, the walker's write calls `dict.__reversed__` once a
  previous write has run, which the same check refuses a level earlier;
  the write then fails, and the walker drops its copies. Reading the
  newest keys without that call would cost a scan of the dict per write
  and would still leave the refusal, so the difference below now says
  both (the sixteenth review corrected when that call comes, and how far
  up).
- **A finalizer that makes the walker write delays the cycle.** Before
  3.14, a finalizer of the cycle that copies or pickles the context, or
  calls it in a way that hands the call back, makes the walker write what
  it holds into its untracked dicts, whichever finalizer runs first; the
  cycle then takes a second collection, as the difference below now says.
  Writing only into dicts something outside the cycle holds would leave
  that copy or pickle without the copies.
- **The tests no longer depend on order.** The `DEBUG_SAVEALL` case took
  the first context it found in `gc.garbage`, which a test before it could
  have left there; it now collects first and finds its context by id. The
  test of a context freed inside another finalizer relies on the collector
  finalizing that object before the walker, which holds within one
  generation only, so it collects first and turns automatic collection
  off until it is done.

Decisions the sixteenth review forced (the review of the fifteenth
review's fixes):

- **A walker that wrote only `objects` keeps its originals.** Before 3.14,
  when only `objects` is kept, the finalizer writes the originals into a
  copy of `_sources` that it drops, and the walker noted that copy as the
  dict it had last written. The next sync then found the real `_sources`
  shorter, took the originals it lacked for ones Python code had removed,
  and let go of them: a context that captured again, through a later
  finalizer of the cycle or once brought back, got for a new object at the
  id of a gone one that one's copy. The walker now notes the dict the
  context holds, so its memo keeps those originals alive while `_sources`
  lacks them. The copy of `_sources` is also made before the mirror is
  taken apart, so a failure there leaves the mirror whole.
- **What follows a refused finalizer is tested on every version.** The
  C-stack test reaches CPython's refusal only where the version and the
  stack's layout put the collection there: here 3.13 for a collection in
  the deepest frame, and 3.15 for some sizes of the environment. A new
  test puts a builtin that writes nothing in place of `_Walker.__del__`,
  which CPython calls instead, so that nothing writing as the walker is
  freed, in a collection or as an exception unwinds, is checked
  everywhere. The C-stack test skips where the main thread's stack
  overflows before CPython's own limit, as on 3.12 and 3.13 with less than
  about 7.5 MB of it.
- **Child interpreters import the checkout from any directory.** The
  tests' environment for them put the package directory on `PYTHONPATH`,
  not the repository root, which the working directory hid.
- **The C-stack difference says when and where.** The `dict.__reversed__`
  that a refusal fails comes before the walker writes whenever either
  dict held entries at its last sync or Python code added some since, as
  `exception()` does to `_sources`, not only after an earlier write; it is
  refused one C level before the finalizer on 3.12 and 3.13 and across
  several on 3.14 and later; and a refusal of the call after the writes
  loses nothing. Reference counting frees a walker at a spent stack only
  on 3.12: from 3.13 CPython defers such a deallocation until the stack is
  shallower. And CPython never runs a refused finalizer again.

Decisions the seventeenth review forced (the review of the sixteenth
review's fixes):

- **A walker owes `_sources` the originals it kept.** The sixteenth
  review's fix kept them in the memo alone, which other paths let go of
  while `objects` still held their copies: a retirement to the Python
  walk, which empties the memo; a removed entry of `objects`, or `objects`
  replaced, which rebuilds it from the dicts; the walker going while a copy of the
  context, shallow or made by a later finalizer of the cycle, shares the
  dicts; and a refused write after the copies were in, which emptied the
  memo. Each then gave a new object at the id of a gone one that one's
  copy. The walker now keeps those originals apart, in the Python walk's
  order, where the collector sees them, and goes on from `_sources` as it
  last saw it before the release; its next write, at the next call's sync
  or end, a copy, a pickle, a hand-back or a retirement, puts them there
  first, in front of what Python code added since, as it does its
  unwritten copies. Only a walker that Python code takes off a living
  context still loses them, and a `_sources` that Python code puts in
  place of the context's before the walker writes lacks them, as it does
  the walker's unwritten copies.
- **A failed write leaves nothing half kept.** A failed write in the
  finalizer emptied the memo but left standing the count of its copies
  that `objects` holds, so the walker's next write took every entry of the
  dicts for Python's, and would have put what it owes `_sources` in front
  of them all. It now forgets that count with the memo, and takes those
  copies from `objects` at its next sync, as copies Python code added;
  what it owes `_sources` it keeps.

Decisions the eighteenth review forced (the review of the seventeenth
review's fixes), each before 3.14 only, where the walker owes anything:

- **The walker owes every original it left out.** It owed only those the
  real `_sources` lacked, so an original that Python code had since put
  there itself, as `_builtin_exception` does an exception, kept the place
  Python gave it, after the walker's later copies. The walker now owes
  every original its write put into the spare, and its next write puts
  each where the copy was made, as the Python walk did, Python's later
  value then going in at that place.
- **An id the walker owes is not one it wrote.** After a noticed removal,
  the walker took an entry Python code wrote at an owed id, within the
  part of `_sources` it last saw, for its own, and wrote the original over
  Python's value. Owed ids now end that part, so Python's entry goes back
  after the original, with Python's value, as in the Python walk.
- **The release forgets what it noted in the spare.** The walker went on
  from `_sources` as it last saw it, but kept the notes its write into the
  spare took of the ids Python had added since, and so put an exception
  `_builtin_exception` kept before its own originals. Those notes are now
  restored with the rest.
- **A failed release keeps what it already owed.** A second release whose
  write failed, as on a `_sources` key whose comparison raises, dropped
  the originals an earlier one owed. What the walker owed and did not
  write stays owed, in front; after a failure, only what `_sources` still
  lacks counts as written.

Decisions the nineteenth review forced (the review of the eighteenth
review's fixes):

- **The walker owes the writes it made, not a difference of dicts.** The
  eighteenth review's fixes worked out what the walker owed from the copy
  of `_sources` its release wrote into, once the write was done, and each
  case the reading missed was a defect: an original written over an entry
  Python code had put there before the walker copied it (or at a float or
  complex key equal to its id), or over a depth placeholder's, was never
  owed, so `_sources` kept Python's value and a rebuild of the memo let the
  original go, giving a new object at its id the old copy; Python's value,
  put back over the original in the copy, was owed in its place, and
  written back once Python code had removed it; and a second release that
  failed partway through what was owed put the part it wrote behind the
  rest. The write now notes each write into `_sources` as it makes it
  there, with the original and how it goes in (as a new entry, in place,
  or as a replacement), and that note is what the walker owes: its next
  write makes each write again, in order, as it would have made it then. A
  write that fails among what the walker owed made none of the rest, which
  it still owes, in order.
- **An in-place write the walker owes is one it wrote.** The eighteenth
  review's rule that an id the walker owes ends the part of `_sources` it
  last saw holds for a new entry only. One that goes over an entry `_sources`
  holds, the walker's own original under a placeholder it replaced, or
  Python's, stays where it is, as without a release.
- **A rebuild in the release notes what it let go of.** Restoring what the
  walker noted of `_sources` before its release undid the notes its memo's
  rebuild took, when Python code had removed a copy from `objects`, of the
  originals `_sources` then held without a copy; after a later noticed
  removal, the walker took such an original, and every entry after it,
  for Python's, and put them after what it owed. The release now notes
  them again.
- **A failed write keeps no copy whose original is held nowhere.** On every
  version, the walker wrote each copy before its original, so a write into
  `_sources` that failed, as on a key whose comparison raises, left that
  copy in `objects` while emptying the memo let go of the original, and a
  new object at its id got the old copy. Each original now goes in before
  its copy.
- **A failed write keeps Python's entries.** On every version, the walker
  takes the entries Python code added after its own out of both dicts
  before it writes, and a write that failed left them out for good. They
  now go back after the walker's whatever the write did.

Decisions the twentieth review forced (the review of the nineteenth
review's fixes):

- **A write keeps what it has to make until it has made all of it.** A
  write that failed after it made what the walker owed `_sources` no longer
  owed it, and the next write, before 3.14, took those originals for
  Python's and put them behind the copies made since; one that failed
  before it reached a copy the walk replaced in place let the replacement
  go, and `objects` kept the depth placeholder for good, on every version.
  Both now go only once the whole write succeeds, as the memo entries past
  `mirrored` do, and writing one again puts it where it was. A release
  whose write fails owes the writes it made, which begin with what it
  owed, unless it failed among those, when it owes all it owed still.
- **Entries move only in a dict whose every key compares as a builtin.**
  The walker took Python's later entries out, and put them back, when
  their own keys were of plain kinds; but looking a key up compares it
  with each key of the same hash, and one of another kind whose comparison
  raised partway left entries taken out and never put back, on every
  version. The walker now moves entries only in a dict all of whose keys
  are of plain kinds, which costs a pass over the dict's keys at each
  write that finds entries Python added.
- **What the C stack bounds is documented, not fixed.** A call itself
  makes more nested C calls than the Python walk before it can hand back,
  so near CPython's C recursion limit it can raise where the Python walk
  returns; nothing short of a Python descriptor avoids that, so it is an
  accepted difference, below, as is a write the walker defers in place
  over a value Python code wrote there later.
- **Tests for fixes that had none.** The nineteenth review's replacement
  writes a release owes, in place or appended, and the seventeenth's
  failed release forgetting what `objects` holds, now each have a test
  that fails with the fix reverted.

Decisions the twenty-first review forced (the review of the twentieth
review's fixes):

- **A pass over every key only where the walker has not read them all.**
  The twentieth review's pass over every key of a dict, at each write
  that found entries Python added, cost a dump whose exception chain
  alternates builtin exceptions with others a pass over `_sources` per
  pair: quadratic in the chain, and with the default options twice the
  time of the round before, slower than the Python walk. The walker reads
  each key of either dict as Python adds it, or as it rebuilds, and keeps
  in `foreign` whether one was not of a plain kind; it now passes over
  every key once it has met one, or after a removal it notices, which may
  let keys it never read into the part of the dict it last saw, and
  otherwise checks the keys Python added since. A key a removal it does
  not notice lets in, it does not check (see the accepted differences).
- **A write that fails keeps its notes.** On every version, where a key
  not of a plain kind kept Python's entries in place, the walker noted the
  ids `_sources` held anew after its write, and so, after a write that
  failed, forgot that an original the walk copied in place of an exception
  `exception()` kept goes in only over an entry `_sources` holds; once the
  key was gone, its next write put back the original Python code had
  removed. It now adds to its notes there, as where it moves the entries.
- **An appended replacement goes in as a new entry.** On every version,
  the original of a copy the walk put over a depth placeholder, where
  `_sources` no longer held the original, went in over any entry, so where
  a key not of a plain kind kept Python's entries in place, it replaced
  the value Python code wrote at that id later; a release owed it so
  before 3.14. It now goes in as a new entry does, and is owed so.
- **A failed release keeps what it wrote ahead.** Before 3.14, a release
  whose write failed let go of the memo, so after a removal the walker's
  next write knew none of the originals in `_sources` for its own, and put
  every one, with Python's, after those it owed. It now takes the copies
  `objects` then holds for its own, with their originals.
- **What the C stack bounds needs no raised limit.** Sections 2 and 3 said
  only a raised recursion limit reaches CPython's C limit; recursion that
  runs in C reaches it at the default limit too, and, from 3.14, so does a
  thread with a small stack, where the walker raises some levels earlier
  than the Python walk. Both sections now say so, with the levels measured.

Decisions the twenty-second review forced (the review of the twenty-first
review's fixes):

- **A commit that writes checks the dicts.** A walker that writes each
  call's copies as the call ends, as after a copy of the context, took the
  dicts as the call's sync left them, though Python code the walk ran, such
  as a collection's callbacks, may have removed entries since; with the
  twenty-first review's checks of only the keys Python added, a key not of
  a plain kind that such a removal let into the part of `_sources` the
  walker saw raised as it moved Python's entries, losing one, and failing
  the call, on every version. The commit now checks where Python code left
  each dict, as every other write does, at about 0.1 µs a call (8% on a
  loop of small calls after a copy).
- **An appended replacement's id ends the part the walker saw.** On every
  version, after a removal the walker noticed, an entry Python code wrote
  at the id of a copy the walk replaced in place, where the walker had yet
  to append its original, counted as the walker's own, and the original
  went over Python's later value. Such an id now ends that part, as an id
  the walker owes a new entry does.
- **Docs say what the code does.** What a removal the walker does not
  notice leaves (a key not of a plain kind it lets in is never read, keeps
  the context on the walker, and can fail a call or a copy as the walker
  moves entries or looks an original up), the entries a failed release's
  next write takes for its own (Python's copies in `objects` among them),
  the recursion the C limit's levels were measured in (plain Python
  recursion never reaches that limit), where an owed original goes, and
  which of the twenty-first review's tests guard the pass over every key
  rather than fail with a fix reverted.

Accepted, documented differences from the Python walk:

- The walker writes its copies to `ctx.objects` and `ctx._sources` lazily
  until the context is first copied. Code other than the Python walk,
  `copy` and `pickle` that reads them between calls sees fewer entries, or
  a depth placeholder the walker has since replaced in place, and so does
  a reducer that leaves the walker out of the state it builds from
  the instance dict. An in-place change to what the walker already wrote
  there, such as an overwritten key, goes unnoticed, as does one to an
  entry of `_sources` the walker has yet to write its original over in
  place, such as an exception `exception()` kept that the walk then
  copied: the walker's original goes over the value Python wrote there
  later, which the Python walk, having written the original first, keeps.
  `clear()` keeps the copies the walker has yet to write. Python's entries are noticed, at
  every call and flush, by count and by the entry ending the part of each
  dict the walker last wrote: additions, removals and a replaced dict are
  noticed, and the walker's unwritten copies go before what Python added
  since. A removal that leaves that entry where it was, by its id in
  `objects` and by its key in `_sources`, goes unnoticed: as when Python
  removes that entry and puts its own back, or removes an earlier entry,
  adds one, and then removes the entry the walker wrote last and puts it
  back after the one it added. The walker may then
  return a removed copy, write its own copy over an entry Python added for
  the same original, count nodes differently, keep a removed original
  alive until a rebuild or its release, write its unwritten copies after
  Python's additions, leave out an original removed before the walker
  copied it, or, where Python added among them a key not of a plain kind,
  which it never reads, keep the context on the walker and run that key's
  comparison as it moves Python's later entries or looks an original up,
  as it does to adopt a copy Python added: an exception it raises there
  goes to the call, copy or pickle that made the walker write or look,
  and to each later one until the key is gone, and loses the entries the
  walker had taken out. After a noticed removal, Python's additions are taken to
  begin after the entry the walker wrote last, or at the first key it does
  not know if that comes first, and no later than after as many entries as
  the walker last saw: an entry Python adds before both, in place of entries
  of the walker's it removed, for an original the walker wrote or noted,
  or a `_sources` key that is not an id, stays ahead of the walker's
  unwritten copies, as does that last entry when Python removes it and
  puts it back first; after a release whose write failed, which let go of
  the memo, so does one for any copy `objects` then holds, Python's own
  included, ahead of what the walker owes. A walker removed from its context while something
  else still holds it writes its copies only when it is released.
- An object that only the walker's unwritten copies keep alive, in a
  reference cycle the collector frees while something else keeps the dict
  the copy goes into, is written there by the walker's finalizer and
  survives with all it refers to, intact, as in the Python walk, where the
  dict held it all along: an original in `_sources`, or a copy in
  `objects` that is the captured object itself or refers to it. The
  collector has by then run their own finalizers, cleared weak references
  to them (every one before 3.15, from 3.15 those with a callback), and
  run the callbacks of those that live on, such as `weakref.finalize`
  cleanups and the pruning of weak containers, as for every object it
  finds unreachable.
- Before 3.14 the collector leaves a dict that holds nothing it tracks
  untracked, as the walker's dicts are until it writes them, and the
  walker's finalizer takes a context it was not told was freed for one
  that dies with it, and that context's reference to each dict for one
  that goes with the cycle. So, when the collector frees the walker:
  - a dict Python code took out of the context, or replaced, after its
    last call, and the dicts of a living context whose walker Python code
    took off it, lack the copies the walker had yet to write;
  - a context that another finalizer, or `gc.DEBUG_SAVEALL`, brings back
    lacks them, unless it captures again before it goes for good;
  - when only `objects` is kept, the originals go into neither dict: the
    walker keeps the writes into `_sources` it left out, originals and
    all, and makes them at its next write, once a context that lives on
    is used again; a walker that Python code takes off the context first
    lets go of them, and a new object may then take one's id and get its
    copy;
  - a cycle takes a second collection to free when another object of it
    holds `objects` or `_sources`, or keeps the walker's weak reference to
    the context (from `weakref.getweakrefs`) reachable, or when a finalizer
    of the cycle makes the walker write: one that captures through the
    context after the walker's finalizer ran, or one that, before or after
    it, copies or pickles the context or calls it in a way that hands the
    call back. The walker then writes what it holds into a dict the
    collector did not track, which keeps it alive until the next
    collection.
  pydumpling itself never takes the dicts or the walker out of a context.
- Where the C stack is spent, CPython refuses to call the walker's
  finalizer, on every version, and, from a little higher up (one C level
  on 3.12 and 3.13, several on 3.14 and later), the `dict.__reversed__`
  that the walker's write calls before it writes, whenever either dict
  held entries at the walker's last sync or Python code added some since,
  as `exception()` does to `_sources`. A walker the collector frees there,
  or, on 3.12, one reference counting frees there (from 3.13 CPython frees
  it once the stack is shallower), loses the copies it had yet to write,
  though the dicts are kept elsewhere. CPython never runs a refused
  finalizer again: a walker that lives on, as one `gc.DEBUG_SAVEALL`
  keeps, also loses the copies of its later calls it has yet to write
  when it goes.
- Near CPython's C recursion limit a call through the walker can raise
  `RecursionError` where the Python walk returns. Python code reaches that
  limit under a recursion limit raised past it when its recursion makes a
  C call at each level, as `list(map(f, ...))`, a property or a C-wrapped
  function does (plain Python-to-Python recursion stops at the Python
  limit, in both walks alike); at the default limit from a callback deep in
  recursion that runs in C, as a `__repr__` or `__eq__` of an item of
  deeply nested containers is; and, from 3.14, where CPython bounds the C
  stack by its size, in a thread with a small stack. Before
  it can hand a call back the walker makes nested C calls the Python walk
  does not (the descriptor's own call, reading the walker off the context,
  the sync's reads of the dicts, and, on a hand-back, its write), and the
  hand-back runs the Python walk beneath them, which may capture part of
  the value before it raises, as the Python walk does at its own limit.
  How far the limit moves depends on the recursion, and varies with its
  frames: in the `list(map(...))` recursion measured under a raised limit,
  by about a level on 3.12 and 3.13 and two to five from 3.14, and for
  `save_dumping` by a level on 3.14 and none elsewhere; in nested `repr` at
  the default limit, by two levels on 3.12 and 3.13 and sixteen to
  twenty-five from 3.14, and for `save_dumping` by none on 3.12 and 3.13
  and three to six from 3.14; in a thread with a 512 KiB stack, by up to
  four levels from 3.14.
- A key of either dict that is not of a plain kind (an exact str, int,
  float, complex, bool, bytes, `None` or tuple of them), such as an int
  subclass or a frozenset, stays where Python put it, ahead of the walker's
  unwritten copies, and so, while the dict holds it, do the entries Python
  adds to that dict, as the walker moves entries only where looking them up
  runs no key's own methods (but for a key a removal the walker does not
  notice lets in, above). Its `__eq__` may run when the walker writes an
  id it collides with. An exception that raises there goes to the call,
  copy or pickle that made the walker write, and to each later one until
  the key is gone; from a dict Python code replaced since, which only the
  walker then holds, taking the walker off the context (`ctx._walker =
  None`) ends that. In the walker's finalizer, which
  cannot raise, it ends the write, and the walker lets go of the copies it
  had yet to write and of their originals, which the Python walk, having
  written each at once, still holds. Every copy the walker wrote keeps its
  original, and Python's own entries stay. One in `objects` that is not an
  id, and one in `_sources` not of a plain kind, send the context to the
  Python walk once the walker, having read it, has written what it holds; one in `_sources`
  of a plain kind that is not an id the walker leaves alone, and never
  takes for an original held without a copy.
- A copy or pickle made outside the context class's own protocol, through
  `copyreg`, a pickler's `reducer_override` or `dispatch_table`, the `copy`
  module's dispatch tables or a by-hand copy of the instance dict, reads the
  instance dict as it stands. A deep copy or pickle meets `objects` before
  the walker, which writes its copies too late for it, so the copy holds
  only those written so far; a shallow one shares the walker, or, made
  during a hand-back, the `HandBack` standing in for it, and with it the
  memo: capturing through one context while the other's call runs raises
  `RuntimeError`, and once either retires the two may diverge. A copy hook
  given to the context class after its last call is seen at its next call,
  so a copy in between is one of these. Copy contexts with `copy`, `pickle`
  or `__getstate__`, or give the class its hook before using it.
- Replacing `_sources` releases the originals only it held at the next
  call, not at once: a finalizer of one runs inside that call, where
  capturing through the context raises `RuntimeError`, and its id is not
  free before.
- A context captured as a value, such as one held by a frame that a dump
  captures, shows its `_walker` attribute, only the copies written so far,
  and its node count as the call that captures it began.
- A call that passes the value or source by keyword runs in the Python walk:
  correct, but without the speed-up. pydumpling passes them positionally.
- `_walker` is the walker's own attribute, set with `object.__setattr__`: a
  class attribute of that name, such as a property, is unsupported; a slot
  of that name makes the context capture in Python.
- Patching the `values` module's helpers (`FakeClass`, `_plain_key`) after a
  context is created is not seen by its walker; patching the context's walk
  methods is. Patch before creating contexts, or set `PYDUMPLING_NATIVE=0`.
  Methods patched onto `collections.Counter`, such as `__missing__`, are
  not run by the walker's own counts, which write the tally dict directly.
- An entry Python code writes for an original the walker copied but has
  yet to write, as `exception()` keeps an original in `_sources` in place
  of the walker's entry, and removes again before the next call, the walker
  writes back, whatever else Python added or removed: the dicts look as if
  Python had written nothing. Noticing it would mean writing every call's
  copies at once, about 65% more per call.
- Code that runs inside a native call (a finalizer, a collection callback,
  an audit hook, or a signal handler at one of the walker's signal checks)
  and captures through the same context, or copies it, gets `RuntimeError`;
  the Python walk would capture there. Set `PYDUMPLING_NATIVE=0` for such
  code.
- Under a profiler, tracer or `sys.monitoring` tool, a call that needs
  Python code runs in the Python walk; one that does not stays native and
  the hook sees none of it, as with any C accelerator. `_all_events` is
  interpreter-wide, so a hook on another thread sends such calls to the
  Python walk too. A hook set inside one of pydumpling's helpers sees the
  rest of that helper; the Python walk reruns the call if it needs Python
  code again, and otherwise the call commits natively. A tool that takes
  an id during a call with only per-code events is seen from the next
  call; on 3.12 and 3.13, one that freed its id and kept only per-code
  events is seen by no check, and sees the stack probe and helpers the
  walker calls.
- Introspection of the class attribute: it is a `Dispatch`, so
  `inspect.isfunction` is false, `__module__` is `pydumpling._native`,
  `inspect.getsourcefile` raises `TypeError` (`getsource` and `signature` work
  through `__wrapped__`), and `mock.create_autospec` sees `(*args, **kwargs)`.
  `ctx.value.__wrapped__` exists; calling it directly on a native context
  bypasses the walker and may read a stale `objects`.
- From a stack deeper than `min((limit − 97) / 2, 1024)` frames (451 at the
  default limit of 1000), the start check raises one `sys._getframe` audit
  event per call; the Python walk raises none.
- With a colliding instance-dict key, the Python walk's own hook count can
  change after its first runs, as `LOAD_ATTR` specializes; it settles after a
  few warm-up runs, and then the counts match.
- Using one context from two threads at once raises `RuntimeError` instead of
  interleaving the two captures.
- A str-subclass keyword *name* in `**kwargs` is hashed once by PyO3 while the
  arguments are collected.

## 4. Test harness, property fuzzing and parity strategy

- **Existing suite, both modes.** The whole suite runs with the native
  walker and with `PYDUMPLING_NATIVE=0`, on CPython 3.12 to 3.15 (3.11 keeps
  the Python walk).
- **A/B parity oracle.** For N seeded random value graphs (nested containers,
  subclasses, slots, named tuples, enums, cycles, shared references, long
  strings, big ints, OrderedDicts with non-plain keys, frame-like namespaces
  with priorities) and several budgets, `pickle.dumps` of the native snapshot
  and of the Python snapshot must be **byte-identical**. Epsilon is exactly 0:
  there is no floating-point arithmetic; values are copied, not computed.
  Memo order, counter order and object sharing all show up in the pickle
  memo, so byte identity is the strictest available oracle.
- **Recursion and hand-back.** Nesting up to and past the recursion limit for
  lists, dicts and objects must be byte-identical, including the
  `recursion` truncation. `Walker.handbacks` shows which calls went back to
  Python.
- **Targeted regressions.** One test per review finding: non-dict mappings,
  errors part way through a walk, a collection during a walk, odd type names,
  class-level and bound calls, subclass overrides, small thread stacks (in a
  child process), hidden `__init__` frames near the limit, non-singleton empty
  bytes, class dicts with non-str keys, arguments of any shape, copies, frame
  namespaces staying native, changed `__bases__`, a re-imported package; and,
  from the second review, context state changed between calls (25 cases),
  walkers removed or moved, options subclasses, method objects under pickle,
  copy and weak references, non-UTF-8 keyword names, application code behind
  builtin lookups, deep caller stacks under a raised limit, and an audit hook
  refusing frame access (in a child process); and, from the third review,
  walk methods patched on the instance or class after construction, the
  receiver passed by keyword, copies of slotted subclasses, frame access
  denied at import, re-entry refused from collection callbacks, audit hooks
  and copies, tracing hooks around hand-backs and the stack probe, signal
  handlers and `asyncio.run` keeping the walker, and no `__import__`,
  metaclass or descriptor hook run by a native call; and, from the fourth
  review, state attributes behind properties and the context class's own
  attribute hooks (defined on it or patched on after a call, with access
  counts equal to the Python walk's), tallies of other types, a replaced
  layout cache or `_sources`, capturing on a context and its shallow copy,
  container hashes under tracing hooks, a released walker importing
  nothing, a non-str key in a context class dict, and a signal handler that
  captures inside a native walk; and, from the fifth review, a copy
  replacing a placeholder its original omitted, copies of a class with a
  `_walker` slot and `__getattr__`, another class's `__dict__` descriptor,
  released walkers keeping a replaced placeholder and the originals in
  `_sources`, Python's memo entries kept in order (new or replacing a copy),
  `_sources` replaced by a plain dict, a profile function set from C and a
  freed tool id, and a profiler set by a collection inside a native walk;
  and, from the sixth review, originals Python keeps in `_sources` alone,
  entries it removes, a root capture after `_sources` was replaced, the
  originals a replaced `_sources` held released, memo keys that are not
  ids, a copy made during a hand-back, an interrupted tuple's placeholder
  under a profiler, strings that fit staying native under a profiler, a
  tally with `__index__`, the detached default state, and a reducer that
  hands over the instance dict; and, from the seventh review, a replaced
  placeholder or kept original that Python code removes before or after the
  copy (across a state read, a hand-back and a copy), releasing a context
  writing nothing, one dict for `objects` and `_sources`, a context class
  with a copy hook of its own, a replacement a hand-back undid, and a tally
  a helper adds to once; and, from the eighth review, a removal inside a
  hand-back masked by its additions (from either dict), a removal and an
  addition between calls after a copy, an original Python removes being
  released, a `_walker` slot, copy hooks on the instance, as
  `__getattr__`, `__getnewargs__` or `__getnewargs_ex__`, or on the
  metaclass (from the start or given later), a metaclass dict key that is
  not a str, a `_truncation` key that is not plain, and releasing a fresh
  class first made natively; and, from the ninth review, a call, copy,
  pickle or state read after Python removed an entry and added more, after
  a copy or a hand-back (compared, and by a finalizer), the walker's
  unwritten copies going before what Python added after a removal (of the
  entry the walker wrote last too, put back, or with a key that is not an
  id), a `_sources` key of another kind found by a call, a copy or a
  rebuild, `__replace__` on the class or the metaclass and `copy.replace`,
  a dataclass context, and no read of the metaclass by a native call; and,
  from the tenth review, a removal with as many additions between calls
  before any copy, with or without a copy the walker has yet to write,
  then a call, copy, pickle, state read or hand-back, the entry the walker
  wrote last put back after an addition, in `objects` or under the same
  `_sources` key, and among the first entries once two earlier ones were
  removed too, an original the walker wrote put into a replaced
  `_sources`, and `copy.replace` through a `__class__` property or plain
  class attribute; and, from the eleventh review, a memo entry Python adds
  for a mapping whose class answers `__class__` itself, through a property
  or `__getattribute__`, with plain and builtin entries as controls; and,
  from the twelfth review, a context that a captured object refers back to,
  freed by the cyclic collector while `objects`, `_sources`, both or
  neither are kept elsewhere, with or without the copy, and the context a
  kept `_sources` keeps alive capturing again before a second collection;
  and, from the thirteenth review, what one collection of that cycle
  leaves alive, and the cycle collected where no Python frame fits; and,
  from the fourteenth review, a context freed by a finalizer of the
  collected cycle (a generator's, an object's), a captured context's
  dicts at the default depth, a walker moved to a second context collected
  with it, and, from 3.14, a context brought back (collected again, freed
  by reference counting, saved by `gc.DEBUG_SAVEALL`), a walker taken off
  a living context, and a replaced `objects`; and, from the fifteenth
  review, a context freed by reference counting before the collector
  frees its walker, a replaced `objects` or `_sources` at depths that free
  the walker while the collector clears the cycle, intact or, before 3.14,
  empty, and, in a child interpreter, a cycle freed where the C stack is
  spent, which must keep its `RecursionError` and lose the copies rather
  than get them cleared; and, from the sixteenth review, a walker whose
  finalizer writes nothing, freed in a collection or as an exception
  unwinds, which must lose its copies and leave the exception alone, and
  a context that captures again after its walker wrote only `objects`;
  and, from the seventeenth review, such a context brought back and used
  as it was, after Python code added an original, retired by its options
  or a key, with an entry of `objects` removed or the dict replaced,
  copied, or copied by a later finalizer, which must give each new object
  a copy of its own and both dicts the Python walk's entries in its order;
  and, from the eighteenth review, such a context after Python code kept
  an original itself, after a removal and an addition or Python's own
  entry at an owed id, and released again where the write fails; and,
  from the nineteenth review, such a context after an entry of `objects`
  was removed, or after Python's own entries at the ids of originals it
  copied, and released again where the write fails partway through what
  it owed, a copy replaced in place, which must keep its place in
  `_sources`, a release whose write fails, which must leave no copy
  without its original, a write that fails, which must keep Python's own
  entries in either dict, and, before 3.14, a walker whose `_sources`
  Python code changed, whose release must owe what its lazy write would
  have written; and, from the twentieth review, a write that fails after
  it made what the walker owed, or before a copy the walk replaced in
  place, which the next write must make in the Python walk's order, and
  a release after the first whose write fails too, which must keep each
  copy it wrote with its original,
  Python's entries in a `_sources` holding a key that refuses to compare
  partway, which must all stay, the originals of copies put over depth
  placeholders that a release owes, in place or appended, and a second
  release that fails past copies the walker wrote and an original Python
  added; and, from the twenty-first review, an original Python removed
  after the walk copied it in place, where such a key makes a write fail,
  which must stay removed, a copy put over a depth placeholder whose
  original Python removed, with or without a release, which must keep the
  value Python wrote at that id later, and a removal after a release whose
  write fails, which must keep the walker's earlier originals ahead of
  those it owes; and, from the twenty-second review, such a copy after a
  removal the walker notices, which must also keep Python's later value,
  and Python's changes to `_sources` inside a call of a walker that writes
  as each call ends, which it must notice.
  The C-stack test reaches CPython's refusal only on some versions and
  stack layouts (see the sixteenth review), and otherwise passes as the
  Python walk does.
  Each of these fails with its fix reverted, where the fix applies (the
  seventeenth to twenty-first reviews' before 3.14 only, but for the
  nineteenth's to twenty-first's failed writes, the twenty-first's entries
  kept in place and the twenty-second's fixes, on every version), but for
  a few parameters
  kept as controls, such as a call after a copy, which an earlier review
  had made pass; a context let go, which guards the walker's traversal
  of what it owes `_sources`: it fails, the collector then leaking the
  original that refers back to the context, only if the walker hides those
  originals from the collector; and a key that refuses to compare let in
  by a removal, which guards the twenty-first review's pass over every key:
  it fails only if the walker skips that pass after a removal it notices. An outcome that does not pickle, such as
  one holding a local class, compares by type and parts; before, both
  walks' pickling errors compared equal.
- **Rust tests.** Unit tests, doc-tests and `proptest` model-based tests of
  `Memo` against a reference model of Python dict semantics with
  transactions.
- **Coverage.** The extension is built with `-C instrument-coverage`, the
  pytest suite and parity harness are run against it, and `llvm-cov` reports
  line and branch coverage. Target: ≥95%. Unreachable defensive branches are
  documented if the target is missed.

## 5. 4-stage benchmarking and PGO pipeline

Harness `rust/bench/bench_walk.py`: realistic and stress workloads, full
`save_dumping` and capture-only timings, median of at least 15 runs over 3
rounds, peak RSS from `ru_maxrss`, `PYTHONHASHSEED=0`.

| Stage | Build |
| :-- | :-- |
| `legacy_baseline` | Frozen copy of the pre-change package (pure Python) |
| `safe_rust` | First native build: release, fat LTO, `codegen-units=1` |
| `soa_simd` | Data-layout and hot-path pass. SIMD has no target here (pointer chasing through Python objects), so this stage measures layout and allocation work and says so. |
| `pgo_lto` | `-Cprofile-generate`, then run the bench and test workloads, `llvm-profdata merge`, then `-Cprofile-use` with fat LTO |

## 6. Phased execution roadmap and open questions

1. Scaffold the crate, statics and memo. **Done.**
2. Port `value`, `_mapping`, copies and key selection. **Done.**
3. Dispatch and the transactional hand-back. **Done.**
4. Suite in both modes, A/B parity harness, recursion checks. **Done.**
5. Adversarial review of semantic divergences; self-heal (≤5 iterations).
   **Done** (iterations 4 and 5; see the decisions above).
6. Coverage, benchmarks for all stages, PGO, infographic.

Open questions for the maintainer:

- **Packaging.** Today `rust/build_native.py` builds the extension in place,
  and the published wheel stays pure Python. Shipping it needs
  maturin/cibuildwheel and per-platform abi3 wheels, which is a release
  decision.
- **Free-threaded CPython.** abi3 does not cover 3.13t/3.14t; those
  interpreters use the Python walk.
- **Stack budget.** At most 256 KiB of native stack, and at most a quarter
  of what remains above a 64 KiB reserve, before a hand-back. Deep
  `max_depth=None` captures beyond roughly 90–220 levels run in Python, and
  so does any walk level fewer than 64 frames from the recursion limit.
