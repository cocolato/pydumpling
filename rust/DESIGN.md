# Native value walker: design

This crate builds `pydumpling._native`, an optional native implementation of
the bounded value walk in `pydumpling/snapshot/values.py`. The pure-Python
walk there is the reference and the fallback: the native walk must produce a
snapshot that pickles to the same bytes, run application code exactly as
often, and raise where the Python walk raises. This document describes the
design as it stands, for whoever changes it.

## Scope

Only the bounded capture walk is native: `ValuesMixin.value`, `_mapping`
and the `SnapshotContext` helpers they call (`_attribute_mapping`,
`_root_value`, `_fill_copy`, `_filled_slot`, `_scalar`, `_attributes`,
`_remember`, `_has_nodes`, `_omit`, `_type_name`, and the fast paths of
`_text`, `_code_text`, `_bounded_text`, `_bounded_bytes`). The walker calls
the Python `_layout` once per type and call and caches its answer.
The writer, reduction, the debugger, the report and the CLI stay in Python.
Capture runs while the failing process waits, and profiling showed the
value walk dominating bounded capture time, at about 1-3 us of interpreted
dispatch per node. The rest is I/O bound or built on `pdb`; porting it
would trade a pure-Python, all-platform wheel for per-platform builds with
no measurable gain.

## Build and supported range

    python rust/build_native.py [--debug] [cargo build arguments...]

The script runs `cargo build --lib` against the running interpreter
(`PYO3_PYTHON`), a release build unless `--debug`, `--release` or
`--profile` says otherwise, and copies the library to
`pydumpling/_native<suffix>` (the interpreter's `.abi3` suffix, or `.pyd` on
Windows) through a temporary file and a rename, so an interrupted build
never leaves a partial extension; it removes other builds of `_native`
that would shadow it. `RUSTFLAGS` and extra cargo arguments pass through,
which is how instrumented and profile-guided builds are made. The release
profile is `opt-level = 3`, fat LTO, `codegen-units = 1`.

The crate uses PyO3 0.30.1 with `abi3-py312` (one binary for CPython 3.12
and later), `#![forbid(unsafe_code)]`, and `stacker` for the thread's real
stack bounds. The built extension is git-ignored; the published wheel and
CI stay pure Python, and CI never builds the extension.

Supported: CPython 3.12 to 3.15, where the suite has run in both modes.
3.11 always uses the Python walk (`values.py` does not import the extension
there): on 3.11 C calls count toward the Python recursion limit, so a call
handed back from native code would overflow earlier than the Python walk
alone, at any limit. From 3.12 they count only toward CPython's own C
recursion limit (see "Accepted, documented differences"). Free-threaded
builds (3.13t, 3.14t) are not covered by abi3 and use the Python walk. The
extension has been built and tested on macOS arm64 only.

## When a context captures natively

A context gets a walker at the end of `SnapshotContext.__init__`
(`ValuesMixin._start_walker`) when the extension imported (CPython 3.12+,
built, and `PYDUMPLING_NATIVE` was not `"0"` when `values` was imported),
`PYDUMPLING_NATIVE` is not `"0"` when the context is created, Python does
not run under `-b` (`sys.flags.bytes_warning`), and the class has a
`_stock_walk`, which `context.py` records for `SnapshotContext` only;
subclasses inherit it and face the per-call check. Under `-b` comparing
bytes with str warns, and the walker's own dict and set lookups and
inserts, which a call it hands back makes the Python walk repeat, would
warn where the Python walk does not, or twice; the flag is fixed for the
life of the process.

Before every call the walker checks the context without running any of its
code (see "The class check"). If the class or state is one only the Python
walk reproduces, the walker *retires*: it writes every copy it holds, sets
`_walker = None`, and the context captures in Python for good. That
happens for:

- a class along the MRO holding, for any walk method (`_NATIVE_WALK`),
  attribute hook (`_STATE_ACCESS`), state name (`_NATIVE_STATE`) or copy
  hook (`_COPY_PROTOCOL`), something other than the stock class's: an
  override, a property in place of a state attribute other than `_walker`
  (see the accepted differences), a `_walker` slot, a
  reducer of its own, `__replace__` (which `@dataclass` adds from 3.13),
  `__class__`, `__getattr__`;
- an instance attribute named like a walk method or copy hook;
- a metaclass other than `type` whose own classes define `__copy__`,
  `__replace__`, `__getattr__` or `__getattribute__`, or have a dict key
  that is not an exact str; a class dict key that is not an exact str;
- no plain `__dict__` getset, one that refuses the instance (such as
  another class's planted there, before 3.15; 3.15 accepts it), or an
  instance dict that is not an exact dict or holds a non-str key;
- a missing state attribute; options not exactly `DumpOptions`, whose instance
  dict (a dict subclass assigned as `__dict__` too, which attribute lookup
  reads as a dict) holds a key other than an exact str or lacks a budget, or
  whose class has an MRO other than at import, a `__getattribute__`, or under a
  budget's name anything but `None` or an exact int (each call reads every
  budget, the Python walk only those it uses, so neither may run code); a
  budget that is not `None` or a positive exact int; a code text limit below 9;
  `_unlimited` not `False`; `_truncation` not an exact `Counter`, or with a key
  other than an exact str; `_namespaces` not an exact set; `_layouts` not an
  exact dict; `_nodes` not an exact non-negative int; `objects` or `_sources`
  not exact dicts, or one dict for both;
- a key in `objects` that is not an id (an exact int in address range), or
  one in `_sources` not of a plain kind (exact str, int, float, complex,
  bool, bytes, `None`, or an exact tuple of those).

Why these, where it is not obvious:

- Copy hooks are the names `copy` and `pickle` consult. A hook of the
  class's own may build a state from the instance dict, where the walker
  comes after `objects` (missing unwritten copies), or give a shallow copy
  the walker itself; the Python walk's memo is always complete when copied.
  `object.__reduce_ex__` reads `__getnewargs_ex__`/`__getnewargs__` before
  the state; `copy.replace` (3.13+) looks `__replace__` up through the
  metaclass, on the class `obj.__class__` names (which a class attribute,
  property or plain class can answer in place of `type(obj)`, so
  `__class__` counts); `__getattr__` can answer `__deepcopy__`.
- One dict for `objects` and `_sources`: the Python walk writes every
  original over its own copy there.
- Unlimited snapshots: probing each original with the serializer
  (`_serializable`, a `dill.dumps` per value) dominates their cost and runs
  application code.
- The code text limit: the Python walk names, and so may count as
  truncated, every placeholder it builds, including one for each exact
  tuple and frozenset; the walker builds those only when something refers
  back to them. With a limit of at least 9 (`len("frozenset")`) neither
  walk counts anything there, and a write can name a pending placeholder
  without a Python helper, which no hook can then see.

A single call goes to the Python walk (a *hand-back*), and the context
stays native, when its arguments are not in a shape the walker reads
without running code (listed below); when the walk reaches a point where
the Python walk would run application code (listed under "The
transaction"); when the Python walk would come within 64 frames of the
recursion limit, any `RecursionError` is raised, or the native stack
budget runs out; and when Python code is needed while a profiler, tracer
or monitoring tool is set. `Walker.handbacks` counts them. These argument
shapes hand a call back before the walk starts (`arguments.rs`), and the
Python walk then raises or converts as it would:

- no positional argument after the context (as when the value or source
  is passed by keyword), or more than two;
- a depth, positional or by keyword, that is not an exact int (a bool is
  not) or exceeds 2**62 in magnitude, or a depth given both ways;
- a keyword name that is not an exact str, or not valid UTF-8 (a lone
  surrogate);
- a keyword other than `depth` for `value`, or other than `depth`,
  `priority`, `counted` and `container` for `_mapping`;
- a `priority` that is not an exact tuple, list, set or frozenset of
  exact str names;
- `counted` other than `True` or `False`;
- a `container` other than `dict` or `FakeDict`.

## Module map

    lib.rs          module `pydumpling._native`: Statics, Walker, Dispatch
    dispatch.rs     Dispatch descriptor: routes calls to walker or Python
    arguments.rs    the argument shapes the walker reads natively
    walker.rs       Walker pyclass, State, per-call refresh of the state
    shadow.rs       per-call class check; finds the instance dict
    transaction.rs  transaction, hand-back, retire, headroom, stack, hooks
    walk.rs         value(): exact scalars, containers, FakeClass objects
    copies.rs       _scalar, _fill_copy, _filled_slot, cached _layout calls
    mapping.rs      _mapping, _attribute_mapping, _root_value
    keys.rs         key selection: preferred names, link order, limit + 1
    text.rs         budgets, tallies, _remember, text fast paths
    types.rs        per-type facts from the MRO, cached for one call; plain
                    reads of a placeholder's reason and the copy classes'
                    instance dicts
    statics.rs      per-import table of types, base methods and sys hooks
    memo.rs         insertion-ordered identity memo with journaled rollback
    mirror.rs       sync of the memo with ctx.objects and ctx._sources
    flush.rs        writes unwritten copies into objects and _sources
    release.rs      finalizer writes, GC traverse and clear, owner and owed

`rust/tests/memo_model.rs` is a property test of `Memo`; `rust/bench/`
holds the benchmark harness.

## How Python hooks in

Call sites do not change: `frames.py`, `exceptions.py` and the tests call
`self.value(...)` and `self._mapping(...)`. At import, `values.py` builds
one `_native.Statics(globals(), _identity)` table and replaces
`ValuesMixin.value` and `ValuesMixin._mapping` with `_native.Dispatch`
descriptors over the Python functions.

`Dispatch.__get__` reads `_walker` with `object.__getattribute__` (no
class hook runs). If it is a `Walker`, it returns a real `types.MethodType`
binding the descriptor to the context; otherwise the Python function bound
as usual. Called on the class, `Dispatch` routes `SnapshotContext.value(ctx,
...)` and `SnapshotContext.value(self=ctx, ...)` the same way, because the
Python function run directly on a context whose walker holds unwritten
copies would read a stale `objects`. The descriptor adds no Python frame,
so the Python walk overflows its recursion limit exactly where it would
without the extension. `__doc__`, `__wrapped__`, `__name__`, pickling (as
`getattr(owner, name)`) and the rest follow the function. Why not instance
attributes (`self.value = walker.value`): a throwaway context
(`SnapshotContext(o).value(x)`) died while its bound method ran, and a
strong walker-to-context reference would make a cycle that keeps originals
alive until a full collection.

`_start_walker` creates `_Walker(_STATICS, stock)` and sets it with
`object.__setattr__`, last in `__init__` (see "Release"). `_Walker` is a
`Walker` subclass with `__slots__ = ()` and `__del__ =
_native.Walker.release`. The finalizer is `__del__` on a Python subclass
because PyO3 sets no `tp_finalize` for a pyclass; for the same reason the
walker cannot avoid CPython refusing it on a spent C stack.

`ValuesMixin.__getstate__` reads `_walker` with `object.__getattribute__`,
calls `walker.flush()`, removes `_walker` with `object.__delattr__`, builds
`object.__getstate__(self)`, puts the walker back with
`object.__setattr__`, and returns a detached copy of the state
(`_detached`). Going through `object.*` builds the default state as for a
context that never had a walker, so the class's hooks run as often as in
the Python walk (for example a `_walker` slot with `__getattr__`). The
state is detached in both walks, so a subclass that edits it, or a
`__setstate__` that adopts it, behaves alike. A copy or pickle therefore
holds every copy and captures in Python, sharing `objects` with the
original as a copy of a Python-walk context does. `Walker.__reduce__`
flushes too and pickles and copies the walker as `None`.

## One call

`Dispatch` takes `try_borrow_mut` on the walker; a walker already borrowed
refuses the call with `RuntimeError("this SnapshotContext is already
capturing: a native context captures one call at a time, so reentrant or
concurrent capture through it is unsupported")`. That one check covers
re-entry and use from two threads at once (see "Accepted, documented
differences"). A copy made during a native call fails in `Walker.flush`,
also on `try_borrow_mut`, with `RuntimeError("this SnapshotContext is
already capturing: cannot copy a native transaction")`. Why not run a
re-entrant call in Python: the outer call's copies are not yet anywhere
the Python walk could share them, so it would read a memo missing them.

`State::run` then arms the owner reference (before 3.14 only; see
"Release"), runs `refresh` (the class check, reading the state from the
instance dict, and syncing the memo with the dicts), parses the arguments
(a shape it cannot read hands back), and runs the call as a transaction.
A class or state that only the Python walk reproduces retires the walker;
an exception raised during `refresh`, such as one a key's own comparison
raises while a sync writes, goes to the caller, and the walker stays.

### The class check

`shadow.rs` reads each class dict through the `type.__dict__` descriptors,
never through a metaclass, and takes a class attribute raw, without its
`__get__`. For every name in `_NATIVE_WALK + _STATE_ACCESS + _NATIVE_STATE
+ _COPY_PROTOCOL` it compares what the MRO yields with what
`values._walk_methods` recorded on the stock class. The result is cached
on the walker, so each new context computes it at its first call, and
revalidated on every call: the MRO tuple by identity, then only the
classes in front of each name's owner, skipping immutable builtin classes,
plus the metaclass names. The instance dict is scanned every call;
its length is no mutation token. A class patched after its contexts were
created is therefore seen at the next call, and nothing reads the context
through a patched `__getattribute__` first.

Once the class passes, the walker reads `options`, the budgets,
`_truncation`, `_namespaces`, `objects`, `_sources`, `_nodes`,
`_node_marker` and `_layouts` from the instance dict, and writes `_nodes`
back there, as stock attribute access would. A class attribute of any of
those names would be what the Python walk reads, which is why the check
covers state names too.

Nothing else is cached across calls: budgets, counters, the node count and
per-type facts are reread every call, because Python code runs between
calls (an exception's `__reduce_ex__`, say) and may change a class's
`__bases__` or the context. Within a call, native code runs no application
code, so per-type facts (`types.rs`) are computed once per type and call;
code that runs inside a call anyway (see "Hooks and signals") and changes
a class is seen from the next call (see "Accepted, documented
differences").

### The transaction

`memo.rs` journals every change between `begin` and `commit`; the
transaction also saves the node count, a copy of `_truncation`, the list of
in-place replacements (`dirty`) and the namespace ids it added. On success
it commits the memo, writes `_nodes`, and, in eager mode, flushes. On a
Python exception (`Halt::Py`) it commits as on success and raises, because
the Python walk keeps whatever it did before an error. On `Halt::Bail` it
rolls everything back exactly, order included, and hands the call back. A
`RecursionError` from anywhere becomes `Bail`: only the Python walk has
the Python stack the overflow depends on. `except Exception` clauses of the
Python walk are reproduced by `unless_exception`, which also lets
`RecursionError` bail.

The hand-back flushes every unwritten copy into the dicts this call synced
with, replaces `_walker` with a `HandBack` marker, runs the Python function
with the original arguments, restores the walker, and resyncs with whatever
the Python walk left (adopting the copies it appended and any depth
placeholder it replaced in place). While the marker is set, the context's
calls go to Python, and a copy of the context calls `HandBack.flush`,
which switches the walker to eager mode afterwards.

Every place where the Python walk would run application code is a
hand-back point reached before any of it runs; handing back afterwards ran
that code twice. The points are:

- a mapping that is neither a dict, a dict subclass, a `frozendict` (3.15)
  nor a `FrameLocalsProxy` (3.13+), whose own `items()` is application
  code. Dict subclasses are read through `dict.items`, frame proxies
  through `items()`, which runs none. Frame proxies are read natively only
  if `sys._getframe` was allowed when `values` was imported
  (`_frame_locals_type`); if an audit hook refused it then, the walker does
  not know the proxy type, and every `FrameLocalsProxy` namespace (function
  locals on 3.13+) hands back, while dict namespaces such as globals stay
  native;
- a captured value whose class has, along its MRO, a class dict key other
  than an exact str, or a getset not named `__dict__` planted as
  `__dict__` (the same in the context's own class retires it; see "When a
  context captures natively");
- a type whose `__qualname__` is a str subclass;
- a memo hit in `_mapping` on an entry that is not an exact `dict`,
  `FakeDict`, `TruncatedValue`, builtin scalar or builtin container: the
  Python walk's `isinstance` reads its `__class__`;
- copying a `FakeClass` whose instance dict is not an exact dict of exact
  str keys, or whose class holds, along its MRO, a `__repr__` other than
  the one it had at import or a `__getattribute__` (the Python walk calls
  `FakeClass.__repr__` as the class then has it);
- any object while `FakeClass` has an MRO other than the one it had at
  import, checked before anything runs: given other bases, it builds its
  copies through theirs, and 3.12 to 3.14 refuse the instance-dict
  descriptor it had at import;
- a memo hit in `_mapping` on a placeholder whose `reason` the walker cannot
  read without running code, as an exact str: one whose class holds, along its
  MRO, `reason`, a `__getattribute__` but `object`'s, or a class dict key other
  than an exact str, whose instance dict holds a key other than an exact str,
  or that holds no `reason`, or one of another type (the Python walk then
  raises or runs the hook once);
- an item view yielding anything but exact 2-tuples;
- filtering the private attribute of a captured snapshot copy, or reading
  its own type name, from a dict whose keys are not all exact str;
- a truncation tally that is not an exact int below 2**63 - 1, checked
  before the walker or a Python helper adds to it;
- adding a copy that is not a plain key to a set or dict, which may run
  `FakeContainer.__hash__`, while a tracing hook is set.

### Recursion and the native stack

The Python walk's result depends on the recursion limit (`_root_value`
recovers from an overflow), so the walker hands back wherever the Python
walk could come close to it. At the start of a call one
`sys._getframe(min((limit - 97) / 2, 1024))` that fails proves the stack
shallow enough for the first 32 walk levels, since a visible frame counts
at most twice toward the limit (3.13+ runs `__init__` under a hidden shim
frame). Otherwise, and whenever the walk needs more, the walker recurses
into `values._headroom(n)` and lets `RecursionError` answer (doubling, then
binary search), so the interpreter counts every frame, hidden or not. Each
walk level must leave 64 frames of margin for helpers, generators and hash
methods. The common case raises no audit event. An audit hook that refuses
`sys._getframe` during a call only forces the probe; one that refused it
at import also turned off the frame-locals fast path (above).

The native stack budget is `min(256 KiB, (remaining - 64 KiB) / 4)`, from
`stacker::remaining_stack()` at the start of the call. A fixed 256 KiB
crashed with SIGBUS on small thread stacks; the reserve covers the
hand-back itself, and three quarters of the rest are left for collections
and finalizers that run between two checks.

### Hooks and signals

The walker runs Python code in a few places: the headroom probe and
pydumpling's own helpers (`_text`, `_code_text`, `_bounded_bytes`,
`_layout`, `FakeClass.__repr__`, and `FakeContainer.__hash__` through a
set or dict insert). Before each it asks `sys.monitoring._all_events()`
(about 10 ns), which shows every profile or trace function, including one
set from C without an object as yappi does, and every tool's global events,
from any thread; without `_all_events` it asks `sys.getprofile` and
`sys.gettrace`. The six tool ids, for a tool with only per-code events, are
asked once per call; asking them every time cost about 15% on calls that
truncate many strings. If a hook is set, the call hands back, so the hook
runs exactly where the Python walk runs it (what it then sees is listed
under the accepted differences). Strings that fit need no helper: an exact
str whose length times four is within the limit, or an ASCII one
(`str.isascii` is C) no longer than the limit, is kept as `_bounded_text`
would keep it.

Signal handlers and collections (with their callbacks and finalizers) run
only at the walker's signal check, every 4096 nodes so Ctrl-C interrupts a
long capture, or inside its Python calls; an exception they raise ends the
call as it would end the Python walk's. Why not hand back whenever a signal
handler is installed: that turned the walker off in every `asyncio.run`
program and every server with a SIGTERM handler.

## The memo across calls

The Python walk memoizes each copy in `ctx.objects[id(original)]` and
keeps the original alive in `ctx._sources[id(original)]`. The walker keeps
its own `Memo<Entry>` (address to copy and original, insertion-ordered,
with a hash map index) and writes the dicts lazily.

Why lazy: an eager prototype that wrote every call's copies to
`ctx.objects` and looked earlier ones up there halved the capture speed-up
(7.1x to 3.5x) and raised peak RSS above the Python walk's, only to honor
code outside the walk changing that internal dict, which nothing in
pydumpling does. Replacing a dict is still honored. Exporting the whole memo
at each hand-back was also rejected: with 300,000 entries in the dicts, a
hand-back that wrote them all again took 21-100 ms (run medians 23-49 ms),
where one that writes only the entries made since the last write takes
3-5 µs, the handed-back call included (CPython 3.13, release build; each
hand-back followed a one-item call).

The fields are documented on `State` in `walker.rs`. In short: `mirror`
holds the dicts as of the last sync; `mirrored` counts the memo entries,
oldest first, that `objects` holds in the same order (later ones are
*unwritten*); `sources_mirrored` and `sources_last` are the length and
newest key of `_sources` at the last sync or write; `dirty` lists
mirrored entries the walk replaced in place since (a depth placeholder
captured later as a root); `truncated` holds keys whose copy is a depth
placeholder the Python walk may replace; `foreign` says either dict holds
a key the walker may not look up; `eager` says each call writes its copies
when it commits; `owed` holds writes into `_sources` that a release which
wrote only `objects` left out (in practice before 3.14).

### Sync

Each call's `refresh` syncs (`mirror.rs`). The walker checks whether
either dict was replaced, whether a length changed, and whether the key
ending the part of each dict it last saw is still where it saw it, which
catches removals masked by as many additions. It checks the ending keys at
every sync, before the call copies anything, since a later write cannot
tell whether Python changed a dict before or after the walker's copies.
That check costs nothing while neither dict holds an entry the walker
synced with, and up to about 0.1 us per call once either does; capture and
save times are unchanged in the benchmark.

If Python code replaced a dict, changed `objects`, or removed entries while the
walker held unwritten copies, the walker first writes them into the dicts it
mirrored (see "Writes"), then takes its memo from `objects`. A dict Python
replaced, the sync lets go of once it has written there the copies it had yet
to write, before it reads the dicts that replaced it, so it takes in what that
dict's finalizers write into `objects`, as they run at once without the walker;
a dict such a finalizer puts on the context in place of one it reads, the
walker reads from the next call (see "Accepted, documented differences").
Otherwise a replaced dict or a moved ending key (a shorter dict has none where
the walker saw it) rebuilds the memo from `objects` with the originals
`_sources` holds, letting go of the originals Python removed so their ids can
be reused as in the Python walk, and copies Python appended to `objects` (a
hand-back's, say) are adopted from the tail; if that leaves the memo shorter
than `objects`, as when Python put back an entry a removal the walker did not
notice took out, the memo is rebuilt too. Entries Python only appended to
`_sources`, as `exception()` keeps builtin exceptions there, need no write: the
walker checks their keys and takes the new length. In eager mode, and at the
first sync after a hand-back, depth placeholders Python replaced in place are
adopted, each with the original `_sources` then holds for it: a hand-back
leaves the walker unchecked until a sync rechecks, so a resync that reads
nothing, or stops before or in its recheck, leaves it to the next call's. The
walker notes every `TruncatedValue` it makes or finds in `objects`, of a
subclass too, which the Python walk's `isinstance` takes for one (the walker
reads that from the type's MRO, which runs no code), and adopts whatever
`objects` holds in place of one whose class still derives from
`TruncatedValue`. The walker returned each placeholder, and application code
may have changed it or its class since, so of one still in place it reads
`reason` only where that runs no code: an exact `TruncatedValue`'s, through the
instance-dict descriptor the class had at import, from a dict of exact str keys
while the class holds along its MRO no `reason` or `__getattribute__` but
`object`'s, and only exact str keys. It lets go of one whose reason reads so as
another than `"max_depth"`, as the Python walk replaces only depth
placeholders, and of one application code gave a class not derived from
`TruncatedValue`; any other, it keeps checking, and it never raises for one. It
checks them last, once the sync has taken in all it found and counted what each
dict holds. A placeholder the recheck lets go of may have finalizers that write
into the dicts, as they ran at once when the Python walk replaced it: when
either dict's length then moved, the sync runs again and takes those writes in
(an overwrite of an entry that is not a noted placeholder it does not notice,
as with any such overwrite). It reruns its recheck at most as many times as the
recursion limit: the Python walk runs such finalizers nested, so the recursion
limit ends a chain of them, except on 3.12, where CPython's trashcan frees the
placeholders one after another and the chain runs to its end; past that, the
walker stays unchecked and the next sync carries on (see "Accepted, documented
differences"). It counts as checked only once a recheck returns with neither
length moved. So a recheck cut short, which takes `MemoryError`, CPython's C
recursion limit, or the `__eq__` of a key that hashes like an id and that a
removal the walker did not notice let in (see "Accepted, documented
differences"), leaves the walker as one that returned would, but still
unchecked, so that the next sync rechecks, with the placeholder it was checking
and those it had yet to check still noted. Every key the walker reads is
checked as it reads it, without running code: a key in `objects` that is not an
id, or a `_sources` key not of a plain kind, sets `foreign`, and the context
retires once the walker has written what it holds.

### Writes

The walker writes (`flush.rs`) on a hand-back, a retire, `Walker.flush`
(from `__getstate__`, a copy or a pickle), a sync that found the dicts
changed, an eager commit, and a release that holds copies it has yet to
write or owes `_sources` (a release with nothing to write neither checks
the dicts nor rebuilds the memo). Every write first checks where
Python left the dicts, and the keys Python appended to `_sources`, except
the one a sync makes right after checking them itself; an eager commit
checks too, since Python code the walk ran, such as a collection's
callbacks, may have changed them since the sync. A write:

1. builds pending placeholders: an exact tuple or frozenset met again
   while being filled gets its `TruncatedValue` only if something referred
   back to it;
2. writes what a release owes `_sources` (see "Release"), then the copies
   the walk replaced in place, over the entries they replace, then each
   unwritten memo entry in memo order, each original before its copy, so
   that a write that fails never leaves a copy whose original is held
   nowhere;
3. where Python code changed `objects` since the walker last wrote it,
   puts a new copy in only where `objects` holds none for that id: Python
   wrote its own later, which the Python walk keeps;
4. rebuilds the memo from `objects` if Python changed `objects` or removed
   originals from `_sources`, or if a copy met an id Python had added.

`owed` and `dirty` are cleared only once the whole write succeeds, like
memo entries past `mirrored`, so a failed write leaves everything to the
next one, and writing any entry again puts it where it was.

Why the walker's entries simply follow Python's: keeping the Python walk's
order when Python code changed the dicts between calls took a merge that
moved Python's later entries behind the walker's, only where every key
compared as a builtin, and in-place rules for originals Python kept or
removed: about 250 lines of Rust and most of the accepted differences, for
an order nothing in pydumpling reads. pydumpling never reads `_sources`,
reads `objects` only through the walk, which looks copies up by id
(`_root_value` reads its order only past the length it found when it
started, after every copy a hand-back's write put there), and never
copies or pickles a context. Why check only the keys Python
appended: checking every key at every write would be a pass over
`_sources` for each builtin exception `exception()` keeps there, quadratic
in an exception chain; with default options that doubled the time and was
slower than the Python walk.

### Eager mode

A shallow copy shares `objects` and `_sources` with its original, and its
Python walk may replace a depth placeholder in place. So once the context
has been copied (`Walker.flush`, or a copy during a hand-back), and once
the walker's finalizer has run, the walker writes each call's copies when
the call commits, after rechecking the dicts, and rechecks its depth
placeholders at every sync. The commit's check costs about 0.1 us per call,
8% on a loop of small calls after a copy.

### Release

The walker writes lazily, so when it goes, `objects` and `_sources` may
lack copies the Python walk had put there at once. `_Walker.__del__` is the
native `Walker.release` (`release.rs`). CPython runs it as reference
counting frees the walker, and, for a walker in a reference cycle (a
context and a captured object that refers back to it), before the collector
clears any object of the cycle and before it checks what that made
reachable again. Copies written into a dict that outlives the cycle
survive it intact, and originals written into a `_sources` kept elsewhere
keep the cycle alive, as the Python walk's dicts did. `release` raises
nothing, and does nothing to a walker that is in use (borrowed by a
running call).

The finalizer sets eager mode (a finalizer runs once, so a walker brought
back must not hold unwritten copies again), and writes into each mirrored
dict that something besides the walker holds: both if `_sources` is held,
only `objects` (below) if only that is, nothing otherwise. It reads
reference counts without calling Python: `sys.getrefcount` is a call,
which fails where the C stack budget is spent, and the finalizer then
wrote nothing. It subtracts the walker's own references (the mirror's, and
the memo's when the walk captured the context and so the dict). `freed` is
a reference count of one: CPython calls a finalizer from deallocation at
one, while the collector holds the walker as it finalizes it.

Why native: a Python `__del__` pushes a frame, and failed before it wrote
anything at the recursion limit, with a signal pending, or under a tracer
that raised. Why only the finalizer writes, never `Drop`: when the
collector frees the walker, the copies may already be cleared, and an
exception may be unwinding, which a call into Python would replace
(`SystemError`, and crashes on 3.15). A walker of the plain `Walker` class,
which pydumpling never makes, has no finalizer and writes nothing when
freed.

Why `_walker` is the instance dict's last key: an instance dict releases
its values in its class's shared-key order, which the class's first
instance fixes, on 3.12 to 3.15 alike. A walker released before `objects`
and `_sources` would find them still held and write every unwritten copy
into dicts about to be freed. On 3.13, with a fresh class whose first
instance set `_walker` first, freeing a context that had captured a list
of 50,000 distinct ints took 2.5-3 ms instead of 0.2 ms (0.5 ms in the
Python walk; 22-45 ms under `tracemalloc`), and peak RSS rose by 13-15 MB
over the capture, against 4 MB with `_walker` last and 12 MB in the
Python walk. Released last, it finds them held by itself alone and writes
nothing. An instance releases slots before its dict, which is why a
`_walker` slot retires the context.

Before 3.14 the collector leaves untracked a dict that holds nothing it
tracks, as the walker's dicts are until it writes them; such a dict is not
among the objects a collection frees, so what goes into it lives on until a
later collection. `Statics.untracked_dicts` (from `gc.is_tracked({})`)
detects this. There, each call arms an `Owner`: a weak reference to the
context of the walker's latest call whose callback notes that the context
was freed. The collector clears, without a callback, weak references that
only dying objects hold, so a context the walker was not told was freed is
taken to die with it, and its reference to each dict is not counted as a
holder; the consequences are listed under the accepted differences. From
3.14 every dict is tracked and nothing is armed.

When only `objects` is held, the walker writes the originals into a spare
copy of `_sources` that is dropped, so the dying dict keeps nothing alive,
and records each write it made there in `owed`, in order, with its
original. Its next write,
should the context live on and be used, makes those writes first, as they
would have been made then; `__traverse__` visits the owed originals, so
the collector sees them. Before 3.14 this is the case above; from 3.14 the
walker owes `_sources` only when nothing but the walker holds the
`_sources` it mirrored, as after Python replaced it. Afterwards the walker
takes `_sources` as it last saw it before the release (its length and
newest key, not the spare's).

A release whose write fails empties the memo and its replacements and
forgets `mirrored`, so the next sync takes the copies `objects` holds as
Python's. A failed full write leaves `owed` as it was. A release that
wrote only `objects` owes every write it made into the spare, which begin
with what it already owed; if it failed among those, it still owes all of
them.

### Retire

`retire` flushes into the mirrored dicts, without reading the context
again (so a replaced `_sources` loses nothing), clears the memo, the
mirror and its view of `_sources` (whose newest key may be the very key
that retired the context), sets `_walker = None` with
`object.__setattr__`, and runs the call in Python. The walker is then
usually freed, with nothing left to write; one that Python code still
holds keeps none of the context's originals, nor `objects`, `_sources` or
their keys, alive (it still holds the `_truncation` and `_namespaces` it
last read).

## Invariants and why

- Native code never runs application code, but for the key comparisons listed
  under the accepted differences (a dict key not of a plain kind whose `__eq__`
  runs when the walker writes or looks up an id it collides with, an element of
  `_namespaces` that hashes like the id of a dict it captures, and a key other
  than a str planted in a class dict of `DumpOptions` that hashes like a
  budget's name), and for the finalizers and weak reference callbacks of
  objects whose last reference the walker lets go of: mostly objects the Python
  walk lets go of too, but at other times (a dict Python replaced, at the next
  sync; an original or copy the memo drops), and also originals the Python walk
  keeps, when a write in the walker's finalizer is cut short (see the accepted
  differences). Types are compared by address, class attributes are read
  through `type.__dict__` descriptors, containers through their builtin base's
  unbound methods, and keys are hashed or compared only when they are of a
  plain kind. Anything else hands back before it runs, so application hooks run
  exactly as often as without the extension. Neither a call nor a release
  imports anything, since `builtins.__import__` may be the application's: every
  module-level object the walker or its finalizer needs (including the answer
  of `gc.is_tracked({})` and `collections.Counter`) is resolved when `Statics`
  is built.
- The order of memo entries, node counts and truncation counters is part of the
  snapshot, and so is object sharing (the pickle memo shows it). The walker
  follows the Python walk step for step, builds every copy as its class's
  constructor does while the class holds no hook of its own, and uses the same
  reason objects (`_RECURSIVE_IMMUTABLE`, the node marker) so placeholders
  pickle to the same bytes.
- A call either commits what it did or rolls back to the exact state it
  started from; the Python walk then runs from that state.
- `objects` and `_sources` end up holding what the Python walk would
  leave in them whenever Python code other than code reading internals
  between calls can see them, and in its order unless Python code changed
  them between calls, but for the accepted differences below. Each
  original goes in before its copy, so a write that fails never leaves a
  copy whose original is held nowhere, and a new object at a reused id
  never gets an old copy. The walker never removes or moves Python's
  entries.
- The walker reads the context only after the class check, and only
  through its instance dict, `object.__getattribute__` and
  `object.__setattr__`. It holds the context only during a call; between
  calls it refers to it only through what its memo, `owed`, the dicts it
  mirrors, the newest key of `_sources`, the `_truncation`, `_namespaces`
  and node marker it last read and the types it last met keep and, before
  3.14, a weak reference. `__traverse__` visits all of them (the weak
  reference too, through which the collector does not reach the context),
  so a context in a cycle through any of them is collected as in the
  Python walk.
- The `Statics` table is built from the `values` module's own globals at
  import and leaked for the life of the process: a reloaded or vendored
  package gets its own table and copies into its own classes, and old
  snapshots keep theirs.
- `memo::push` grows vectors into a fresh buffer of twice the capacity
  instead of `realloc`: macOS caches the large blocks a chain of `realloc`
  frees and does not reuse them, so each big capture would leave megabytes
  resident.
- Every source file stays at most 400 lines, and the crate forbids
  `unsafe`; reference counts are read through PyO3's deprecated safe
  reader, which needs no Python call (see "Release").

## Accepted, documented differences

Each of these is a difference from the Python walk that remains by design.

- Lazy writes. Until the context is first copied (with `copy`, `pickle` or
  `__getstate__`, or during a hand-back) or the walker's finalizer has run,
  the walker writes its copies lazily. Code other than the Python walk,
  `copy` and `pickle` that reads `objects` or `_sources` between calls sees
  fewer entries, or a depth placeholder the walker has since replaced in
  place; so does a reducer that leaves the walker out of the state it
  builds from the instance dict. A walker removed from its context while
  something else holds it writes only when it is released.
- Changes to the dicts between calls. The walker notices additions,
  removals and replaced dicts at every call and write, by count and by the
  key ending the part of each dict it last saw. Where Python code changed
  them while the walker held copies it had yet to write, the walker writes
  those after what Python added, where the Python walk wrote them first;
  so it does after an entry Python appended to `_sources` alone, as
  `exception()` does with each builtin exception it keeps, the one change
  pydumpling itself makes. The dicts then hold what the Python walk's
  hold, as `==` compares their keys, in another order, which pydumpling
  never reads (see "Writes"). What they hold differs where:
  - Python overwrote an entry the walker already wrote: the walker does
    not notice, and keeps returning its own copy until a noticed removal,
    a replaced dict, or a write after Python changed `objects` makes it
    take its memo from `objects` again, or, for a placeholder it still
    notes whose class still derives from `TruncatedValue`, until its next
    recheck takes up Python's entry (see "Sync");
    likewise, an original whose `_sources` entry Python overwrote stays
    alive in the memo until then or the walker's release, where the
    Python walk lets go of it;
  - Python wrote or removed an entry of `_sources` at an id where the
    walker holds an original it has yet to write (one whose copy the walk
    replaced in place included), or owes one: the walker's original goes
    in over Python's value, under Python's key, or back in, where the
    Python walk, having written first, keeps Python's change. Noticing
    that would mean writing every call's copies at once, about 65% more
    per call. Python's own entry in `objects` at such an id stays, under
    its own key, and the walker takes it up; a key there equal to the id
    but not an int (a float, say) thus stays where the Python walk keeps
    the int it wrote first, and then retires the context;
  - Python removed or overwrote an entry of `objects`, which the walker
    had written, that the walk has since replaced in place (a depth
    placeholder captured later as a root): the walker's replacement goes
    in over it, or back in (a placeholder replaced before the walker
    wrote it is a new entry, and Python's stays, as in the Python walk);
  - `objects.clear()` lets go of the copies the walker wrote, as in the
    Python walk, but the walker's next write puts those it had yet to
    write into the emptied dict, and it keeps returning them, where the
    Python walk copies those originals afresh; after `_sources.clear()`
    both walks keep returning their copies, and the walker's next write
    puts the originals it had yet to write back into `_sources`, where the
    Python walk's stays empty;
  - a removal leaves the ending entry where it was (by id in `objects`,
    by key in `_sources`), as when Python removes that entry and puts its
    own back, or removes an earlier entry, adds one, then removes the
    walker's last entry and puts it back after its addition. The walker
    may then return a removed copy, count nodes differently, keep a
    removed original alive until a rebuild or its release, or copy afresh
    an original whose id Python gave an entry of its own in the same
    change and write that copy over Python's entry; a key not of a
    plain kind let in that way it never reads, so the context stays on
    the walker, and the key's comparison may run (below).
- A key of either dict not of a plain kind (an int subclass, a frozenset)
  may run its `__eq__` when the walker writes an id it collides with, or,
  for a key an unnoticed removal let in, looks one up; an exception raised
  there goes to the call, copy or pickle that made the walker write, and
  to each later one until the key is gone. For a dict Python replaced
  since, which only the walker then holds, `ctx._walker = None` ends that.
  In the finalizer, which cannot raise, it ends the write: the walker lets
  go of its unwritten copies and their originals, which the Python walk,
  having written each at once, still holds. Every copy the walker wrote
  keeps its original, and Python's entries stay. A key in `objects` that
  is not an id (anything but an exact int in address range, such as a str
  or a float equal to an id), or a key in `_sources` not of a plain kind,
  retires the context once the walker, having read it, has written what
  it holds. One that a write between calls (a copy's, say) found retires
  it at the next call, unless Python removed it before then and that
  call notices the removal, which leaves the context on the walker.
- An element of `_namespaces` that hashes like the id of a dict a call
  captures runs its `__eq__` when the walker looks that id up, and again
  in the Python walk when the walker hands the call back after it. A key
  other than a str planted in a class dict of `DumpOptions` (through
  `gc.get_referents`: `setattr` adds only str keys) that hashes like a
  budget's name runs its `__eq__` at each call, as the walker looks every
  budget's name up along the class's MRO, where the Python walk reads a
  budget only when the call uses it, through CPython's attribute cache.
- A chain of finalizers of depth placeholders, each writing a new
  placeholder over a noted one as a recheck lets go of it, is followed for
  as many rechecks per sync as the recursion limit, and then at each later
  sync, where the Python walk, which runs such a chain nested, stops it at
  the recursion limit (the `RecursionError` a finalizer raises is ignored),
  except on 3.12, where CPython's trashcan frees the chain's placeholders
  one after another: there it follows the chain to its end, and never
  returns from an endless one.
- A placeholder a recheck lets go of, as above, stays let go of: one whose
  `reason` application code changed to another, or whose class it changed
  to one not derived from `TruncatedValue`. If the code changes it back,
  or that class answers `__class__` with `TruncatedValue`, which the
  Python walk's `isinstance` reads, the walker does not notice when the
  Python walk replaces it in place, and keeps returning the placeholder,
  where the Python walk returns the replacement; nor does it note a
  placeholder of such a class that it finds in `objects`.
- Replacing `_sources` releases the originals only it held at the next
  call, not at once: a finalizer of one runs inside that call, where
  capturing through the context raises `RuntimeError`, and its id is not
  free before.
- Cycles. An object that only the walker's unwritten copies keep alive, in
  a reference cycle the collector frees while something else keeps the
  dict the copy goes into, is written there by the finalizer and survives
  intact with all it refers to, as in the Python walk, where the dict held
  it all along (an original in `_sources`, or a copy in `objects` that is
  or refers to the captured object). But the collector has by then run
  their own finalizers, cleared weak references to them (all of them before
  3.15, from 3.15 those with a callback), and run the callbacks of those
  that live on, such as `weakref.finalize` cleanups and the pruning of weak
  containers.
- Before 3.14 only, because the walker takes a context it was not told was
  freed for one dying with it, and that context's reference to each dict
  for one going with the cycle, when the collector frees the walker:
  - a dict Python took out of the context or replaced after its last call,
    and the dicts of a living context whose walker Python took off it, lack
    the copies the walker had yet to write;
  - a context another finalizer or `gc.DEBUG_SAVEALL` brings back lacks
    them, unless it captures again before it goes for good;
  - when only `objects` is kept, the originals go into neither dict: the
    walker owes them, and writes them at its next write once a context that
    lives on is used again; a walker Python takes off the context first
    lets go of them, and a `_sources` Python puts in place of the context's
    before that write lacks them, as it lacks the walker's unwritten
    copies; a new object may then take one's id and get its copy;
  - the cycle takes a second collection to free when another object of it
    holds `_sources`, or holds `objects` while a copy the walker writes
    there refers back into the cycle (only a captured `TruncatedValue`,
    which is its own copy, can); when something outside the cycle holds
    the walker's weak reference to the context (from
    `weakref.getweakrefs`), so the collector runs its callback and the
    walker takes the context for freed; or when a finalizer of the cycle
    makes the walker write: one that captures through the context after
    the walker's finalizer ran, or one that, before or after it, copies or
    pickles the context or calls it so that the call is handed back. The
    walker then writes into an untracked dict, which keeps what it holds
    alive until the next collection. Why not write only into
    dicts something outside the cycle holds: the copy or pickle that
    finalizer makes would then lack the walker's copies.

  pydumpling itself never takes the dicts or the walker out of a context.
- A spent C stack. There CPython refuses to call the walker's finalizer,
  on every version, and, from a little higher up (one C level on 3.12 and
  3.13, several on 3.14 and later), the `dict.__reversed__` the walker's
  write calls before it writes whenever either dict held entries at its
  last sync or Python added some since (as `exception()` does to
  `_sources`). A walker the collector frees there, or on 3.12 one reference
  counting frees there (from 3.13 CPython defers that deallocation until
  the stack is shallower), loses the copies it had yet to write, though the
  dicts are kept elsewhere; a refusal of a call after the writes loses
  nothing. CPython never runs a refused finalizer again, so a walker that
  lives on, as one `gc.DEBUG_SAVEALL` keeps, also loses the copies of its
  later calls it has yet to write when it goes. Reading the newest keys
  without `__reversed__` would cost a scan per write and still leave the
  refusal.
- Near CPython's C recursion limit a call through the walker can raise
  `RecursionError` where the Python walk returns. Python code reaches that
  limit under a recursion limit raised past it when its recursion makes a C
  call at each level (`list(map(f, ...))`, a property, a C-wrapped
  function; plain Python-to-Python recursion stops at the Python limit in
  both walks); at the default limit from a callback deep in recursion that
  runs in C (a `__repr__` or `__eq__` of an item of deeply nested
  containers); and from 3.14, where CPython bounds the C stack by its size,
  in a thread with a small stack. Before it can hand back, the walker makes
  nested C calls the Python walk does not (the descriptor call, reading
  `_walker`, the sync's reads, and on a hand-back its write), and the
  hand-back runs the Python walk beneath them, which may capture part of
  the value before it raises. Measured shifts, which depend on the value
  and the stack's layout: in `list(map(...))` recursion under a raised
  limit, about one level on 3.12 and 3.13 and two to five from 3.14 (for
  `save_dumping`, one on 3.14 and none elsewhere); in nested `repr` or
  `==` at the default limit, two or three levels on 3.12 and 3.13, about
  twenty on 3.14 and about ten on 3.15 (for `save_dumping`, none on 3.12
  and 3.13 and three to six from 3.14); in a thread with a 512 KiB stack,
  up to four levels from 3.14. Nothing short of a Python descriptor avoids
  this. On 3.12 and 3.13 the limit can also move the other way: a call the
  walker completes itself on a value holding a dict can return one or two
  levels deeper than the Python walk, whose `_mapping` resumes generators
  (`dict.fromkeys`, `itertools.islice`) that CPython counts as C levels.
- Copies made outside the context class's own protocol (through `copyreg`, a
  pickler's `reducer_override` or `dispatch_table`, the `copy` module's
  dispatch tables, or a by-hand copy of the instance dict) read the instance
  dict as it stands. A deep copy or pickle meets `objects` before the
  walker, which writes too late for it, so it holds only the copies written
  so far; a shallow one shares the walker and with it the memo: capturing
  through one context while the other's call runs raises `RuntimeError`, and
  once either retires the two may diverge. One made during a hand-back
  shares the `HandBack` instead and captures in Python from then on, sharing
  only the dicts as they stood, so it misses the copies the original's
  walker writes lazily afterwards. A copy hook given to the context class
  after its last call is seen at its next call, so a copy in between is one
  of these. Copy contexts with `copy`, `pickle` or `__getstate__`, or give
  the class its hook before using it.
- Code that runs inside a native call (a finalizer, a collection callback,
  an audit hook, or a signal handler at one of the walker's signal checks)
  and captures through the same context, or copies it, gets `RuntimeError`
  where the Python walk would capture; so does using one context from two
  threads at once (another thread can run while the walker runs Python
  code), instead of interleaving the captures. Capturing gets the same
  "already capturing" message in both cases (see "One call"). Set
  `PYDUMPLING_NATIVE=0` for such code.
- Code that runs inside a native call (a signal handler at a signal
  check, a collection's callbacks and finalizers, or another thread while
  the walker runs one of pydumpling's helpers) and changes a class, such as
  its `__bases__` or a `__dict__` descriptor, is seen by the walker from
  the next call: it reads each type's facts once per call, where the
  Python walk rereads them, apart from the layouts both cache per context,
  for every value.
- Code that runs inside a native call (as above, or a finalizer of a dict
  Python replaced, which the walker lets go of at the next call's sync, or
  of a placeholder a recheck lets go of) and replaces `objects` or
  `_sources` is seen by the walker from the next call: it captures the rest
  of the call against the dicts it read and writes the call's copies there,
  where the Python walk puts the copies it makes afterwards into the new
  dicts. A later call then captures those values afresh and counts their
  nodes again.
- Under a profiler, tracer or `sys.monitoring` tool, a call that needs
  Python code runs in the Python walk; one that does not stays native, and
  the hook sees none of it, as with any C accelerator. `_all_events` is
  interpreter-wide, so a hook on another thread sends such calls to Python
  too. A hook set inside one of pydumpling's helpers sees the rest of that
  helper; the call is handed back if it needs Python code again, and
  otherwise commits natively. A tool that takes an id during a call with
  only per-code events is seen from the next call; on 3.12 and 3.13, one
  that freed its id and kept only per-code events is seen by no check, and
  sees the stack probe and helpers the walker calls.
- A context captured as a value (one a dumped frame holds, or one passed
  to `value`) shows its `_walker` attribute and only the copies written so
  far: none before the walker first writes (on a copy, a hand-back, or a
  call after Python changed the dicts), then those of its earlier
  calls; the Python walk's copy also holds those of the capturing call.
  Captured through itself, it shows its node count as that call began,
  where the Python walk shows the count it had reached (two nodes on
  through `value`, three through a frame namespace); captured by another
  context, the counts match. A self capture handed back to the Python
  walk, as under a profiler, shows the `HandBack` marker as `_walker` and
  is otherwise as in the Python walk.
- A call that passes the value or source by keyword runs in the Python
  walk: correct, without the speed-up. pydumpling passes them positionally.
- `_walker` is the walker's own attribute, set with `object.__setattr__`: a
  class attribute of that name, such as a property, is unsupported; a slot
  of that name makes the context capture in Python.
- The walker's table is built when `values` is imported: patching its globals
  afterwards (`FakeClass`, `TruncatedValue`, `DumpOptions`, `_headroom`) is not
  seen by any walker, and the walker never calls `_plain_key`. A call the
  walker hands back runs in the Python walk, which sees them, so one context
  can show a patch in some calls and not in others (under a tracer, an
  `OrderedDict` is handed back). Patching the context's walk methods is seen.
  To patch the module's helpers, set `PYDUMPLING_NATIVE=0`. Methods patched
  onto `collections.Counter`, such as `__missing__`, are not run by the
  walker's own counts, which write the tally dict directly. Hooks patched onto
  the copy classes (`TruncatedValue`, `FakeClass`, and `FakeDict`, `FakeList`,
  `FakeTuple`, `FakeSet`, `FakeFrozenSet`, `FakeStr`, `FakeBytes`, `FakeInt`,
  `FakeFloat`: `__new__`, `__init__`, `__setattr__`, `__getattribute__`, a
  `__dict__` descriptor, another `_type_key`) do not run, nor are seen, when
  the walker builds a copy, with the builtin base's `__new__`
  (`object.__new__`, `dict.__new__`, ...), the instance-dict descriptor and the
  `_type_key` the class had at import. The Python walk calls the class, which
  runs `__new__` and `__init__`, sets a placeholder's fields through
  `__setattr__`, and reads a copy's `__dict__`, a list or set copy's `append`
  or `add`, through its `__getattribute__` (a `FakeClass`'s once, and twice
  more for an object with an instance dict); so those hooks run there alone,
  and a planted `__dict__` descriptor sends the copy's attributes where it
  points. Hooks that read a captured `FakeClass` (its `__repr__`, a
  `__getattribute__`) or the options (a `__getattribute__` or property on
  `DumpOptions`) do run as in the Python walk: the walker hands such a call
  back, or retires the context.
- Introspection of the entry points: once the extension is imported,
  `SnapshotContext.value` and `_mapping` are `Dispatch` descriptors for
  every context, whatever `PYDUMPLING_NATIVE` says when it is created, so
  `inspect.isfunction` is false, `__module__` is `pydumpling._native`,
  `inspect.getsourcefile` raises `TypeError` (`getsource` and `signature`
  work through `__wrapped__`), and `mock.create_autospec`, of the
  attribute or of the class, sees `(*args, **kwargs)`. A native context's
  bound method reads the same way and forwards the descriptor's
  `__wrapped__`, which a pure context's bound method lacks. Called as
  `ctx.value.__wrapped__(ctx, x)` while the walker has copies it has yet to
  write, it misses them for `x` alone: `x` is copied afresh, one more node
  is counted and the new copy replaces the walker's for later calls, while
  the values nested in `x` reach the walker and get its copies.
- From a stack deeper than `min((limit - 97) / 2, 1024)` frames, counting
  the calling frame (451 at the default limit of 1000), each call the
  walker starts raises one `sys._getframe` audit event, from the start
  check; a call handed back before it starts (the value by keyword, a
  non-int depth) raises none, nor does the Python walk.
- If an audit hook refuses `sys._getframe` when `values` is imported,
  frame namespaces (`FrameLocalsProxy`, 3.13+) capture in the Python walk
  for the life of the process (see "The transaction").
- A str-subclass keyword *name* in `**kwargs` has its `__hash__` run
  before the walker hands the call back, where the Python walk runs none:
  once while PyO3 collects `**kwargs`, and through a native context's
  bound method once more, as `Dispatch` has no vectorcall and CPython
  builds a keyword dict for it. A call through the class
  (`SnapshotContext.value(ctx, ...)`) reaches `Dispatch` on any context
  once the extension is imported, so it hashes once on a pure context
  too, and a receiver passed there as `self=` runs one more `__eq__`.

## Behaviours of the reference walk both walks keep

The native walk reproduces these byte for byte, or hands the call back, so
fixing one means changing both walks together: a non-dict namespace's own
`items()` view is iterated up to three times, and a one-shot view loses
entries in later passes; after a `RecursionError` in a frame variable,
`_root_value` forgets the half-built copies but keeps their node count and
counters; instances whose `__dict__` is a member descriptor
(`SimpleNamespace`, modules) lose their attributes; the first capture of an
exception materializes its `__dict__`, so a second capture differs; a
str-subclass `__qualname__` runs its own `__radd__` and slicing (the walker
hands such a call back); from 3.13.6 and 3.14 `reprlib` writes an int past
`sys.get_int_max_str_digits()` with its address, so the message of an
exception whose arguments hold one differs between dumps, and before
that, where `reprlib` raises, the message is unavailable.

## Testing

The whole suite runs with the native walker and with `PYDUMPLING_NATIVE=0`
on CPython 3.12 to 3.15; 3.11 runs the Python walk only.
`tests/test_native_parity.py` and `tests/test_native_differences.py` skip
when the extension is not built or `PYDUMPLING_NATIVE=0`, and the native cases of `tests/test_native_import.py`
skip when it is not built. CI, which never builds it, therefore checks
only the Python walk and the import guard; run the rest locally after
`python rust/build_native.py`.

The parity oracle is byte identity. `test_walk_parity` draws, per seed,
random `DumpOptions`, a random object graph (nested containers, subclasses,
slots, named tuples, enums, cycles, shared references, long strings, big
ints, `OrderedDict`s with non-plain keys, frame-like namespaces with
priorities) and a random call stack and sequence of capture operations, and
runs it on a native and a pure context over the same live objects. The
pickles, the truncation counters in key order, the node counts and the
application hooks run must match exactly; there is no tolerance, as values
are copied, not computed. A first, discarded pure run materializes lazy
state such as an exception's `__dict__`. Some cases run a few frames below
the recursion limit, starting both walks at the same depth. While the tests
run, `TruncatedValue`, `FakeClass` and NaN `FakeFloat` copies hash by
content, so sets iterate alike, and, so that no output carries an address,
the harness lifts the int digit limit from 3.13 and drops the address from
`reprlib`'s fallback for a failing `repr`. `PYDUMPLING_PARITY_SEEDS` sets
the number of seeds (300 by default); `test_dump_parity` compares whole
dumps, metadata included, from `save_dumping` and `dump_current_traceback`
for each of 20 seeds. A seed reproduces under the same
`PYTHONHASHSEED`; a failure prints the seed, the options, the first byte
at which the pickles differ and the reproducing command. Outcomes that do
not pickle compare by type and parts; `test_parity_check_detects_differences`
checks that the comparison itself catches perturbations.

Beyond the random cases, each design decision above has a targeted regression
test: hand-back points with hook counts equal to the Python walk's, class and
state changes between calls, copies and pickles, hooks and signals, re-entry,
recursion and small thread stacks (in child processes), dict changes between
calls, no import during a call or a release, and the release paths per CPython
version (fresh classes, since the release order depends on a class's first
instance; cycles collected where no Python frame fits; a builtin in place of
`_Walker.__del__` to check that nothing writes when the finalizer does not).
Accepted differences are pinned in both walks, so that changing either walk's
behaviour fails a test; those no other test covers are in
`tests/test_native_differences.py`. Three are not pinned in full: of the
"Before 3.14 only" consequences, only the owed writes and the second collection
that a weak reference to the context kept elsewhere needs (the tests of the
lost copies run from 3.14, where nothing is lost); of the C recursion limit,
only the direction of the shifts (both directions on 3.12 and 3.13), as their
size depends on the value and the stack's layout; and the spent C stack,
reached only on some versions and layouts (below). A context's instance dict
with a colliding key retires the context, and no test counts the Python walk's
hooks there: its own `__eq__` count changes over its first runs as `LOAD_ATTR`
specializes.

A mutation run on 3.13 checks that the rules fail a test when reverted: 191
mutants of `mirror.rs`, `flush.rs`, `release.rs`, `transaction.rs`,
`State::remember`, the builds of `TruncatedValue` and `FakeClass` copies,
the plain reads of a placeholder's `reason` and the subclass check in
`types.rs`, and the `_mapping` memo hit on one (a negated or dropped
condition, a changed comparison, a skipped or moved statement or call). 177
fail a test. Of the 14 that pass, ten change nothing observable: each
changes a check another one already implies, repeats a flush, rebuild or
adoption that finds nothing new, keeps state that is reset before it is
read, takes in a list equal to the one it has, or notes an entry that is no
placeholder, which the next recheck lets go of. Three differ only after a
failure no test provokes: a placeholder build that overflows the C stack,
or a sync or write cut short. Failures of that kind can be provoked on
purpose (a key an unnoticed removal let in that raises from `__eq__`, the C
recursion limit beneath nested comparisons), and tests provoke them where a
rule of the recheck depends on them. The last, arming the owner reference
from 3.14 too, fails nine tests on 3.14 and 3.15. A test can stop guarding
its rule without failing (a later helper of the same name in its module can
shadow the one it uses), so rerun the mutants a changed test covers. A
context let go guards the traversal of owed originals, a rule a revert does
not make fail outright: it fails only if the walker hides them from the
collector.

The collection tests are order-independent: the `DEBUG_SAVEALL` case
collects first and finds its context in `gc.garbage` by id, and the test of
a context freed inside another finalizer collects first and disables
automatic collection, since the collector finalizes that object before the
walker only within one generation. Child interpreters get the repository
root on `PYTHONPATH`, so they import the checkout from any directory. The
C-stack test reaches CPython's refusal only on some versions and stack
layouts (3.13 for a collection in the deepest frame, 3.15 for some
environment sizes) and otherwise passes as the Python walk does; it skips
where the main thread's stack overflows first (3.12 and 3.13 with less than
about 7.5 MB).

Rust tests: unit tests in `memo.rs` and `text.rs`, doc-tests in
`memo.rs`, and `rust/tests/memo_model.rs`, a `proptest` model of `Memo`
against Python dict semantics with transactions (`cargo test
--manifest-path rust/Cargo.toml`). Coverage: build with
`RUSTFLAGS="-C instrument-coverage"` through `build_native.py`, run the
suite and the parity harness, merge the `.profraw` files with
`llvm-profdata` and report with `llvm-cov`. The target is 95% of lines,
with unreachable defensive branches documented if it is missed.

## Performance

Capture is about 3.5 times as fast as the current Python walk, and a full
`save_dumping` about 2.1 times (geometric means over six workloads). A
throwaway context that captures a scalar or a short list of scalars is
slower natively: the per-call class check and state reading cost more than
the walk they save.

Environment: CPython 3.13, Apple M5 Pro (macOS, arm64),
`PYTHONHASHSEED=0`, the six workloads of `rust/bench/workloads.py`: a
typical small failure, two heavy ones (one with value budgets off), a deep
stack, a trace through `argparse` and `json` with whole files embedded, and
a stack over 20 large module namespaces. Snapshots were byte-identical
across implementations throughout.

| Speed-up | Capture | save_dumping |
| :-- | --: | --: |
| PGO build (fat LTO + PGO) vs original Python walk | 7.88x | 2.91x |
| Current build vs original Python walk | 7.4x | 3.0x |
| Current build vs current Python walk | 3.5x | 2.1x |

The first two rows compare sums of per-workload medians (PGO build against
original: capture 73.22 ms to 9.29 ms, save 96.37 ms to 33.07 ms); the last
is the geometric mean of per-workload ratios. The PGO build is an earlier
version of the walker, before the per-call class check, state reading and
dict checks; PGO made its capture about 7% faster than the same code without
PGO, and `save_dumping` no faster. The current build is a fat-LTO release
build without PGO, which with those checks reaches 7.4x against the original
walk (median of 3 rounds, measured on a checkout of this branch alone). SIMD has no target here; the walk is pointer chasing through Python
objects.

The original Python walk is the package before the walker: `ed7ae81`'s
`values.py`, measured together with local changes, uncommitted at the
time, outside the value walk (metadata, embedded sources and saving). An
`ed7ae81` export, as below, therefore reproduces that baseline only
approximately.

Small values. A memo hit costs about 1.15 us natively against 0.05 us in
Python (a longer MRO or a metaclass adds to it), and a throwaway context
that captures a scalar or a short list of scalars is faster in Python.
Creating a context costs about 2.2 us natively and 1.8 us in Python.
Creating one and capturing a single value costs, natively against Python:
an int 5.2 us against 2.1 us, `[1, 2, 3]` 5.3 us against 3.6 us,
`[1, {"a": "b"}, (2.5,)]` 6.0 us against 7.1 us, a 20-key dict 7.7 us
against 20.4 us (medians of three interleaved rounds of 20,000 contexts).
The walker wins once a value holds containers or many items.

To reproduce (results go to `rust/bench/results/`, which is git-ignored):

    python rust/build_native.py
    mkdir -p /tmp/legacy
    git archive ed7ae81 pydumpling | tar -x -C /tmp/legacy
    python rust/bench/bench_walk.py --impl legacy \
        --legacy-root /tmp/legacy --rounds 3 \
        --json rust/bench/results/legacy.json
    python rust/bench/bench_walk.py --impl pure --rounds 3 \
        --json rust/bench/results/pure.json
    python rust/bench/bench_walk.py --impl native --rounds 3 \
        --json rust/bench/results/native.json
    python rust/bench/bench_walk.py --compare \
        rust/bench/results/pure.json rust/bench/results/native.json
    python rust/bench/bench_walk.py --compare \
        rust/bench/results/legacy.json rust/bench/results/native.json

Each workload runs in a fresh subprocess per round and is timed for the
full `save_dumping(strict=True)` and for capture alone, at least
`--min-runs` (15) runs and `--min-time` (0.5 s) each; peak RSS is
`ru_maxrss`. `--compare BASE OTHER` prints speed-ups (BASE / OTHER) per
workload and as sums and geometric means, and whether the capture pickles,
node counts and truncation counters match. `--impl native` fails if a
context has no walker. See `--help` for `--workloads` and `--hash-seed`.
A PGO build passes `-Cprofile-generate=DIR` through `RUSTFLAGS`, runs the
benchmark and the tests, merges with `llvm-profdata merge`, and rebuilds
with `-Cprofile-use=FILE`.

## Open problems

- Packaging. `build_native.py` builds in place and the wheel stays pure
  Python. Shipping the extension needs maturin or cibuildwheel and
  per-platform abi3 wheels: a release decision.
- Platforms. Only macOS arm64 has been built and tested; Linux and Windows
  builds are unchecked, and CI exercises only the Python walk.
- Free-threaded CPython. abi3 does not cover 3.13t and 3.14t, which use
  the Python walk. Running `build_native.py` on such an interpreter has not
  been tried.
- Stack budget (see "Recursion and the native stack"): deep
  `max_depth=None` captures beyond roughly 90-220 levels run in Python, as
  does any walk level fewer than 64 frames from the recursion limit.
- Per-call overhead (see "Performance"): a memo hit costs about twenty
  times as much as in Python, so small contexts lose.
- The lazy memo is the source of most of the remaining complexity in
  `mirror.rs`, `flush.rs` and `release.rs` and of most accepted
  differences. Moving to a single Python-side memo, written through, would
  remove much of it (an estimated 400-500 lines of Rust). A review
  measured the cost with eager mode as a proxy (CPython 3.13, one machine,
  median of 3 rounds): the capture speed-up against the current Python
  walk falls from 7.05x to 4.25x as a sum and from 3.50x to 2.78x as a
  geometric mean, and the peak RSS of `b2_heavy_nolimits` rises from 65.5
  to 79.2 MB, above the Python walk's 72.2 MB. Removing the release and
  finalizer writes while keeping the lazy memo would bring back lost
  copies in collected cycles.
- Before 3.14, `Owner`, `flush_objects` and `owed` (see "Release") exist
  only so that a context the collector frees with its walker gets nothing
  written into dicts the collector does not track, which would keep the
  cycle alive until a later collection. Dropping them would remove about
  150-200 lines and the "Before 3.14 only" differences, at the cost of
  such cycles being freed a collection later. A review measured that cost
  on 3.13 for 300 such contexts of 200 items each: the first collection
  took about 73 ms instead of about 15 ms, and about 17 MB stayed until
  the second; 9 cases of 6 collected-cycle tests on 3.12 and 3.13 would
  change. Not reproduced here.
