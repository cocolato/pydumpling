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
the whole call to the Python walk before any such code runs. Application
hooks therefore run exactly as often as without the extension.

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
back from native code would overflow earlier than the Python walk alone.

```
lib.rs          module `pydumpling._native`: Statics, Walker, Dispatch
├─ dispatch.rs  descriptor: ctx.value / ctx._mapping / Class.value(ctx) → walker or Python; same-thread re-entry
├─ arguments.rs the argument shapes the walker reads without running application code
├─ walker.rs    Walker pyclass, State (validated against the context per call), retire checks, GC traverse
├─ mirror.rs    incremental mirroring of the memo into ctx.objects / ctx._sources
├─ transaction.rs  per-call transaction, recursion headroom probe, stack budget, hand-back, retire
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
| `context.py::_remember`, `objects`, `_sources` | `memo.rs::Memo<Entry>`, `text.rs::remember`, `mirror.rs` | `HashMap` index plus ordered `Vec`, with undo journal; written to `objects` lazily, in the same order |
| `context.py::_omit`, `_type_name`, `_text`, `_code_text`, `_bounded_bytes`, `_has_nodes` | `text.rs` | Fast path natively; slow path calls the Python helper (same side effects) |
| (new) | `dispatch.rs`, `arguments.rs`, `transaction.rs` | No extra Python frame; transactional hand-back |

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
class ValuesMixin:
    _walker = None
    _stock_walk = None  # recorded by context.py for SnapshotContext

    def _start_walker(self):  # called last in SnapshotContext.__init__
        if (
            _native is None
            or self._unlimited
            or type(self.options) is not DumpOptions
            or os.environ.get("PYDUMPLING_NATIVE") == "0"
        ):
            return
        stock = self._stock_walk
        if stock is None or any(
            method is not original
            for method, original in zip(_walk_methods(type(self)), stock)
        ):
            return  # a subclass overrides a walk method: Python calls it
        self._walker = _native.Walker(_STATICS)

    def __getstate__(self):  # copies and pickles see every copy
        walker = self.__dict__.get("_walker")
        if walker is not None:
            walker.flush(self)
        state = self.__dict__.copy()
        state.pop("_walker", None)
        return state


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
  walk; another thread still gets `RuntimeError`.
- **The class attribute keeps routing.** `SnapshotContext.value` is the
  descriptor and routes `SnapshotContext.value(ctx, ...)`: the Python function
  run directly on a context whose walker holds unwritten copies would read a
  stale `objects`. `ctx.value` is a real `types.MethodType`; `__doc__`,
  `__name__`, `__qualname__`, `inspect.signature`, `inspect.getsource`,
  pickling, copying and weak references follow the function.
- **Stack budget.** `min(256 KiB, (remaining − 64 KiB) / 4)`: collections and
  finalizers that run between two checks need the rest.

Accepted, documented differences from the Python walk:

- The walker writes its copies to `ctx.objects` lazily. Code other than the
  Python walk, `copy` and `pickle` that reads `ctx.objects` between calls sees
  fewer entries, and an in-place change that keeps its length (such as
  `clear()` before anything was written) goes unnoticed; replacing the dict is
  noticed. A walker removed from its context while something else still holds
  it writes its copies only when it is released.
- Patching `SnapshotContext` or the `values` module's helpers (`FakeClass`,
  `_plain_key`) after a context is created is not seen by its walker. Patch
  before creating contexts, or set `PYDUMPLING_NATIVE=0`.
- Introspection of the class attribute: it is a `Dispatch`, so
  `inspect.isfunction` is false, `__module__` is `pydumpling._native`,
  `inspect.getsourcefile` raises `TypeError` (`getsource` and `signature` work
  through `__wrapped__`), and `mock.create_autospec` sees `(*args, **kwargs)`.
  `ctx.value.__wrapped__` exists; calling it directly on a native context
  bypasses the walker and may read a stale `objects`.
- From a stack deeper than 1024 frames, the start check raises one
  `sys._getframe` audit event per call.
- With a colliding instance-dict key, the Python walk's own hook count can
  change after its first runs, as `LOAD_ATTR` specializes; it settles after a
  few warm-up runs, and then the counts match.
- Using one context from two threads at once raises `RuntimeError` instead of
  interleaving the two captures.
- A str-subclass keyword *name* in `**kwargs` is hashed once by PyO3 while the
  arguments are collected.

## 4. Test harness, property fuzzing and parity strategy

- **Existing suite, both modes.** All 276 tests run with the native walker
  and with `PYDUMPLING_NATIVE=0`, on CPython 3.13, 3.14 and 3.15.
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
  from the second review, context state changed between calls (17 cases),
  walkers removed or moved, options subclasses, method objects under pickle,
  copy and weak references, non-UTF-8 keyword names, application code behind
  builtin lookups, deep caller stacks under a raised limit, and an audit hook
  refusing frame access (in a child process).
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
