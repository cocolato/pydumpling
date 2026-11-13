# Native value walker: design

This crate builds `pydumpling._native`, an optional native implementation of
the bounded value walk in `pydumpling/snapshot/values.py`. The pure-Python
walk there is the reference and the fallback: the native walk must produce a
snapshot that pickles to the same bytes, run application code exactly as
often, and raise where the Python walk raises. It works on the context's own
state: it looks copies up in `ctx.objects` and writes them there, as the
Python walk does, and keeps no memo of its own. This document describes the
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
stack bounds. The built extension is git-ignored, and the published wheel
stays pure Python.

The script builds with `--locked`, as CI's clippy and tests do, so a
`Cargo.toml` changed without `Cargo.lock` fails the build until the lock is
updated (`cargo update -p <crate>`, or any cargo command without the flag),
and passes cargo a fingerprint of the crate's sources (`Cargo.toml`, `Cargo.lock`, `build.rs`, `src/`), which the
extension carries as `pydumpling._native.fingerprint`. A build stays in the
package across checkouts and edits of the crate, and one built from other
sources still imports and captures as the crate no longer does, so
`tests/conftest.py` compares the fingerprint with the checkout's and refuses
to run the suite on a stale build. An extension that lacks a name the crate
reads from `values` when it is imported (one built for another version of
that module) is not used: `values` warns and captures in Python.

CI's `native` job builds the extension and runs the whole suite on it, on
Linux for CPython 3.12 to 3.15 and on Windows and macOS for 3.14; its `rust`
job runs `cargo fmt --check`, clippy with `-D warnings` and `cargo test`.
The `tests` job stays without the extension and checks the Python walk.

Supported: CPython 3.12 to 3.15, where the suite has run in both modes.
3.11 always uses the Python walk (`values.py` does not import the extension
there): on 3.11 C calls count toward the Python recursion limit, so a call
handed back from native code would overflow earlier than the Python walk
alone, at any limit. From 3.12 they count only toward CPython's own C
recursion limit (see "Accepted, documented differences"). Free-threaded
builds (3.13t, 3.14t) are not covered by abi3 and use the Python walk. The
extension was developed, and its benchmarks run, on macOS arm64 only.

## When a context captures natively

A context gets a walker at the end of `SnapshotContext.__init__`
(`ValuesMixin._start_walker`) when the extension imported, `PYDUMPLING_NATIVE`
is not `"0"` when the context is created, and the class has a `_stock_walk`,
which `context.py` records for `SnapshotContext` only; subclasses inherit it
and face the per-call check. `values` imports the extension on CPython 3.12+
when it is built, `PYDUMPLING_NATIVE` is not `"0"`, and Python does not run
under `-b` (`sys.flags.bytes_warning`, fixed for the life of the process):
there comparing bytes with str warns, and the walker's own dict and set
lookups and inserts, which a call it hands back makes the Python walk
repeat, would warn where the Python walk does not, or twice.

Before every call the walker checks the context without running any of its
code (see "The class check"). If the class or state is one only the Python
walk reproduces, the walker *retires*: it sets `_walker = None`, and the
context captures in Python for good. That happens for:

- a class along the MRO holding, for any walk method (`_NATIVE_WALK`),
  attribute hook (`_STATE_ACCESS`) or state name (`_NATIVE_STATE`),
  something other than the stock class's: an override, a property in place
  of a state attribute other than `_walker` (see the accepted differences),
  a `_walker` slot;
- an instance attribute named like a walk method;
- a class dict key that is not an exact str;
- no plain `__dict__` getset, one that refuses the instance (such as
  another class's planted there, before 3.15; 3.15 accepts it), or an
  instance dict that is not an exact dict or holds a non-str key;
- a missing state attribute; options not exactly `DumpOptions`, whose instance
  dict (a dict subclass assigned as `__dict__` too, which attribute lookup
  reads as a dict) holds a key other than an exact str or lacks a budget, or
  whose class has an MRO other than at import, a `__getattribute__`, or under a
  budget's name anything but `None` or an exact int (each call reads every
  budget, the Python walk only those it uses, so neither may run code); a
  budget that is not `None` or a positive exact int; `_unlimited` not `False`;
  `_truncation` not an exact `Counter`, or with a key other than an exact str;
  `_namespaces` not an exact set; `_layouts` not an exact dict; `_nodes` not
  an exact non-negative int; `objects` or `_sources` not exact dicts, or one
  dict for both.

Why these, where it is not obvious:

- One dict for `objects` and `_sources`: the Python walk writes every
  original over its own copy there.
- Unlimited snapshots: probing each original with the serializer
  (`_serializable`, a `dill.dumps` per value) dominates their cost and runs
  application code.

How a context is copied does not matter: the walker holds nothing of the
context between calls, so a shallow copy, which shares `objects` and
`_sources`, shares the walker too, whatever hook made it, and a pickle or
deep copy, which has dicts of its own, holds `None` in its place
(`Walker.__reduce__`) and captures in Python.

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
    walker.rs       Walker pyclass, State, per-call refresh of the state,
                    GC traverse and clear
    shadow.rs       per-call class check; finds the instance dict
    transaction.rs  transaction, rollback, hand-back, retire, headroom,
                    stack, hooks
    walk.rs         value(): exact scalars, containers, FakeClass objects
    copies.rs       _scalar, _fill_copy, _filled_slot, cached _layout calls
    mapping.rs      _mapping, _attribute_mapping, _root_value
    keys.rs         key selection: preferred names, link order, limit + 1
    text.rs         budgets, tallies, _remember, text fast paths
    types.rs        per-type facts from the MRO, cached for one call; plain
                    reads of a placeholder's reason and the copy classes'
                    instance dicts
    statics.rs      per-import table of types, base methods and sys hooks
    address.rs      maps keyed by object address

`rust/bench/` holds the benchmark harness.

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
...)` and `SnapshotContext.value(self=ctx, ...)` the same way. The
descriptor adds no Python frame, so the Python walk overflows its recursion
limit exactly where it would without the extension. `__doc__`,
`__wrapped__`, `__name__`, pickling (as `getattr(owner, name)`) and the rest
follow the function. Why not instance attributes (`self.value =
walker.value`): a throwaway context (`SnapshotContext(o).value(x)`) died
while its bound method ran, and a strong walker-to-context reference would
make a cycle that keeps originals alive until a full collection.

`_start_walker` creates `_native.Walker(_STATICS, stock)` and sets it with
`object.__setattr__`. Nothing else in Python knows the walker: the context
has no `__getstate__`, finalizer or copy hook for it. `Walker.__reduce__`
pickles and deep-copies it as `None`.

## One call

`Dispatch` takes `try_borrow_mut` on the walker; a walker already borrowed
refuses the call with `RuntimeError("this SnapshotContext is already
capturing: a native context captures one call at a time, so reentrant or
concurrent capture through it is unsupported")`. That one check covers
re-entry and use from two threads at once (see "Accepted, documented
differences"). Why not run a re-entrant call in Python: the outer call
keeps its node count, and what a hand-back would undo, natively until it
ends, so the Python walk would count from a stale `_nodes` and the outer
call would write its own count over the inner one's.

`State::run` then runs `refresh` (the class check, and reading the state
from the instance dict), parses the arguments (a shape it cannot read hands
back), and runs the call as a transaction. A class or state that only the
Python walk reproduces retires the walker. When the call ends, the walker
lets go of the context, its instance dict, `objects` and `_sources`.

### The class check

`shadow.rs` reads each class dict through the `type.__dict__` descriptors,
never through a metaclass, and takes a class attribute raw, without its
`__get__`. For every name in `_NATIVE_WALK + _STATE_ACCESS + _NATIVE_STATE`
it compares what the MRO yields with what `values._walk_methods` recorded on
the stock class. The result is cached on the walker, so each new context
computes it at its first call, and revalidated on every call: the MRO tuple
by identity, then only the classes in front of each name's owner, skipping
immutable builtin classes. The instance dict is scanned every call; its
length is no mutation token. A class patched after its contexts were
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

The walker writes each copy into `objects`, and its original into
`_sources`, as it makes it (`State::remember`), and counts nodes natively.
When a call starts, the transaction notes how many entries each dict holds
and saves the node count, a copy of `_truncation` and, as the call goes, the
namespace ids it adds and each copy it replaces in place (a depth
placeholder that `_mapping` meets again as a root). On success it writes
`_nodes`. On a Python exception (`Halt::Py`) it does the same and raises,
because the Python walk keeps whatever it did before an error. On
`Halt::Bail` it rolls back and hands the call back: each replaced copy goes
back where it was, and the entries the call added, which a dict keeps after
its older ones, are removed from both dicts, newest first (`dict.__reversed__`
gives them without a scan). The Python walk then starts from the dicts as
the call found them. A `RecursionError` from anywhere becomes `Bail`: only
the Python walk has the Python stack the overflow depends on. `except
Exception` clauses of the Python walk are reproduced by `unless_exception`,
which also lets `RecursionError` bail.

The hand-back sets `_walker` to `None`, runs the Python function with the
original arguments, and puts the walker back. While it is `None`, the
context's calls go to Python, and a copy made of the context then captures
in Python from then on.

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

## The memo

The Python walk memoizes each copy in `ctx.objects[id(original)]` and keeps
the original alive in `ctx._sources[id(original)]`. So does the walker: it
reads both dicts from the instance dict at the start of every call, looks a
value up in `objects` before copying it, and writes the copy and then the
original as the Python walk's `_remember` does, each under an int object of
its own, as that function's two `id()` calls make them: a walk that
captures the dicts themselves copies every key as a value, and counts a
node for each. An exact tuple or frozenset gets its `TruncatedValue`
placeholder while it is filled and its copy afterwards, as in the Python
walk. Whatever Python code does to the dicts between calls (adding,
overwriting or removing entries, emptying or replacing a dict) the next
call finds, since it reads nothing else.

Why not a memo of the walker's own, written to the dicts lazily: an earlier
version had one, which made capture about a fifth faster (see
"Performance") and cost a sync of the memo with the dicts at every call, a
write of the pending copies before any Python code could read them (a
hand-back, a copy, a pickle), a finalizer to write them when the context
died in a reference cycle, and about 1,200 lines of Rust; most of the
differences from the Python walk that the design then accepted came from
it.

## Invariants and why

- Native code never runs application code, but for the key comparisons listed
  under the accepted differences (a key of `objects` or `_sources` not of a
  plain kind that hashes like an id, an element of `_namespaces` that hashes
  like the id of a dict the call captures, and a key other than a str planted
  in a class dict of `DumpOptions` that hashes like a budget's name), and for
  the finalizers and weak reference callbacks of objects whose last reference
  a dict write or removal lets go of, as the Python walk's own writes do.
  Types are compared by address, class attributes are read through
  `type.__dict__` descriptors, containers through their builtin base's unbound
  methods, and keys other than ids are hashed or compared only when they are
  of a plain kind. Anything else hands back before it runs, so application
  hooks run exactly as often as without the extension. A call imports nothing,
  since `builtins.__import__` may be the application's: every module-level
  object the walker needs is resolved when `Statics` is built.
- The order of memo entries, node counts and truncation counters is part of the
  snapshot, and so is object sharing (the pickle memo shows it). The walker
  follows the Python walk step for step, builds every copy as its class's
  constructor does while the class holds no hook of its own, and uses the same
  reason objects (`_RECURSIVE_IMMUTABLE`, the node marker) so placeholders
  pickle to the same bytes.
- A call either commits what it did or rolls back to the exact state it
  started from; the Python walk then runs from that state.
- After every call `objects` and `_sources` hold what the Python walk would
  leave in them, in its order: the walker makes the Python walk's lookups and
  writes on the dicts themselves, the copy before its original and each under
  a key object of its own, as `_remember` writes them, and a hand-back takes
  out exactly what the call put in. The one exception is a walk that captures
  the context itself, which finds the walker there (see the accepted
  differences).
- The walker reads the context only after the class check, and only
  through its instance dict, `object.__getattribute__` and
  `object.__setattr__`. It holds the context, its instance dict, `objects`
  and `_sources` only during a call; between calls it keeps the
  `_truncation`, `_namespaces`, node marker and string budget it last read
  and what it learned of the types it last met. `__traverse__` visits all
  of them, so a context in a cycle through any of them is collected as in
  the Python walk.
- The `Statics` table is built from the `values` module's own globals at
  import and leaked for the life of the process: a reloaded or vendored
  package gets its own table and copies into its own classes, and old
  snapshots keep theirs.
- `address::push` grows vectors into a fresh buffer of twice the capacity
  instead of `realloc`: macOS caches the large blocks a chain of `realloc`
  frees and does not reuse them, so each big capture would leave megabytes
  resident.
- Every source file stays at most 400 lines, and the crate forbids
  `unsafe`.

## Accepted, documented differences

Each of these is a difference from the Python walk that remains by design.

- A key of `objects` or `_sources` not of a plain kind (an int subclass, an
  object with its own `__hash__`) that hashes like an id runs its `__eq__`
  when the walker looks that id up or writes it, as when the Python walk
  does; and again in the Python walk when the walker hands the call back
  after it, and when the hand-back takes the id out of a dict again, which
  for `_sources` is the only lookup either walk makes. An exception it
  raises while a hand-back takes entries out goes to the caller, where the
  Python walk would have run. pydumpling writes only ids there.
- An element of `_namespaces` that hashes like the id of a dict a call
  captures runs its `__eq__` when the walker looks that id up, and again
  in the Python walk when the walker hands the call back after it. A key
  other than a str planted in a class dict of `DumpOptions` (through
  `gc.get_referents`: `setattr` adds only str keys) that hashes like a
  budget's name runs its `__eq__` at each call, as the walker looks every
  budget's name up along the class's MRO, where the Python walk reads a
  budget only when the call uses it, through CPython's attribute cache.
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
  `_walker` and the context's state, and on a hand-back taking its entries
  out of the dicts), and the hand-back runs the Python walk beneath them,
  which may capture part of the value before it raises. Shifts measured
  while the walker still kept a memo of its own, and not again since, which
  depend on the value and the stack's layout: in `list(map(...))` recursion under a raised
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
- A shallow copy of a native context shares its walker, as it shares
  `objects` and `_sources`: capturing through one while the other's call
  runs raises `RuntimeError`, as re-entry does (next), and
  `Walker.handbacks` counts the calls of both. A pickle or deep copy holds
  `None` under `_walker` in its instance dict, where the copy of a context
  that captures in Python has no such entry, and captures in Python; so
  does a shallow copy made while the Python walk takes a call from the
  walker, during which the context itself holds `None`.
- Code that runs inside a native call (a finalizer, a collection callback,
  an audit hook, or a signal handler at one of the walker's signal checks)
  and captures through the same context gets `RuntimeError` where the
  Python walk would capture; so does using one context from two threads at
  once (another thread can run while the walker runs Python code), instead
  of interleaving the captures (see "One call"). Set `PYDUMPLING_NATIVE=0`
  for such code. Such code that reads `ctx._nodes` finds the count as the
  call began, and a count it writes there is overwritten when the call
  ends: the walker counts natively and writes `_nodes` once.
- Code that runs inside a native call (a signal handler at a signal
  check, a collection's callbacks and finalizers, or another thread while
  the walker runs one of pydumpling's helpers) and changes a class, such as
  its `__bases__` or a `__dict__` descriptor, is seen by the walker from
  the next call: it reads each type's facts once per call, where the
  Python walk rereads them, apart from the layouts both cache per context,
  for every value.
- Code that runs inside a native call (as above) and replaces `objects` or
  `_sources` is seen by the walker from the next call: it captures the rest
  of the call against the dicts it read when the call began and writes the
  call's copies there, where the Python walk puts the copies it makes
  afterwards into the new dicts. A later call then captures those values
  afresh and counts their nodes again.
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
  to `value`) shows its `_walker` attribute, an unsupported-type
  `FakeClass`. Captured through itself, it shows its node count as that
  call began, where the Python walk shows the count it had reached (two
  nodes on through `value`, three through a frame namespace); captured by
  another context, the counts match. A self capture handed back to the
  Python walk, as under a profiler, shows `None` as `_walker` and is
  otherwise as in the Python walk. The capturing context spends a node and
  an `unsupported_type` count on the walker's copy, which takes an entry in
  each of its dicts, and a node on the count it shows unless the walk has
  met that int already, as the Python walk does on its own, other count.
  Under a node budget, the captures that follow such a one can therefore
  be cut off a value sooner or later than in the Python walk. pydumpling
  itself captures no frame that holds a context.
- A call that passes the value or source by keyword runs in the Python
  walk: correct, without the speed-up. pydumpling passes them positionally.
- `_walker` is the walker's own attribute, set with `object.__setattr__`: a
  class attribute of that name, such as a property, is unsupported; a slot
  of that name makes the context capture in Python, and then holds `None`
  where a context that never had a walker leaves it unset.
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
  `__wrapped__`, which a pure context's bound method lacks.
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
when `values` did not import the extension (not built, `PYDUMPLING_NATIVE=0`
or `python -b`), and the native cases of `tests/test_native_import.py` and
`tests/test_context_copy.py` skip then too. CI's `native` job builds the
extension and runs them all; without it, as in the `tests` job, the suite
checks the Python walk, the import guard and the fallback from an extension
that does not match `values`. `tests/conftest.py` refuses to run on an
extension built from other sources than the checkout's (see "Build and
supported range").

The parity oracle is byte identity. `test_walk_parity` draws, per seed,
random `DumpOptions`, a random object graph (nested containers, subclasses,
slots, named tuples, enums, cycles, shared references, long strings, big
ints, `OrderedDict`s with non-plain keys, frame-like namespaces with
priorities) and a random call stack and sequence of capture operations, and
runs it on a native and a pure context over the same live objects. The
pickles, the truncation counters in key order, the node counts, the kinds
of what `objects` and `_sources` end up holding, in their order, and the
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
test: hand-back points with hook counts equal to the Python walk's, a hand-back
that starts from the dicts as the call found them, class and state changes
between calls, copies and pickles, hooks and signals, re-entry, recursion and
small thread stacks (in child processes), dict changes between calls, a walk
that captures the dicts themselves, no import during a call, and the collection
of a context that a class the walker met refers to. Accepted differences are pinned in both walks, so that changing
either walk's behaviour fails a test; those no other test covers are in
`tests/test_native_differences.py`. One is not pinned in full: of the C
recursion limit, only the direction of the shifts (both directions on 3.12 and
3.13), as their size depends on the value and the stack's layout. One rule has
no test: that a hand-back takes the ids the call added out of `_namespaces`.
The Python walk adds them again, but for the id of the dict the walker built
for a snapshot copy's attributes, and the dict the Python walk builds in its
place gets the freed one's address in every run tried, so the id is the same
either way. A context's
instance dict with a colliding key retires the context, and no test counts the
Python walk's hooks there: its own `__eq__` count changes over its first runs as
`LOAD_ATTR` specializes.

A mutation check on 3.13 covers the rules of the memo: no removal of a call's
entries on a hand-back, a replaced copy not put back, the type cache hidden
from the collector, the walker not put back after a hand-back, no placeholder
while an immutable container fills, an out-of-nodes check that ignores the
memo, an original not kept in `_sources`, the node count not written, one key
object for both dicts, and the journal of namespace ids not reset when a call
starts. Each of the ten fails a test. Two more pass, for the rule without a
test above: the ids a call added not taken out on a hand-back, and not
journaled. A wider run (191 mutants) predates the removal
of the walker's own memo, which most of it covered; the builds of copies and
the plain reads of a placeholder's `reason` have not been mutated since. A
test can stop guarding its rule without failing (a later helper of the same
name in its module can shadow the one it uses), so rerun the mutants a changed
test covers.

Child interpreters get the repository root on `PYTHONPATH`, so they import
the checkout from any directory.

Rust tests: unit tests in `address.rs` and `text.rs` (`cargo test
--manifest-path rust/Cargo.toml`). Coverage: build with
`RUSTFLAGS="-C instrument-coverage"` through `build_native.py`, run the
suite and the parity harness, merge the `.profraw` files with
`llvm-profdata` and report with `llvm-cov`. It has not been measured since
the walker's own memo was removed.

## Performance

Capture is about 2.9 times as fast as the Python walk, and a full
`save_dumping` about twice (geometric means over six workloads). A
throwaway context that captures a scalar or a short list of scalars is
slower natively: the per-call class check and state reading cost more than
the walk they save.

Environment: CPython 3.13, Apple M5 Pro (macOS, arm64),
`PYTHONHASHSEED=0`, the six workloads of `rust/bench/workloads.py`: a
typical small failure, two heavy ones (one with value budgets off), a deep
stack, a trace through `argparse` and `json` with whole files embedded, and
a stack over 20 large module namespaces. A fat-LTO release build without
PGO, on a copy of this branch alone, median of 3 rounds, every build in one
sitting; from one sitting to the next the Python walk's own times moved by
up to 6%. Snapshots were byte-identical across implementations throughout.

| Speed-up over the Python walk | Capture | save_dumping |
| :-- | --: | --: |
| Geometric mean of per-workload ratios | 2.87x | 1.96x |
| Sum of per-workload medians | 3.97x | 2.39x |

The sums are 73.78 ms to 18.58 ms for capture and 98.90 ms to 41.37 ms for
`save_dumping`; per workload, capture is 1.5 to 5.9 times as fast. Peak RSS
over the workloads is 72.6 MB natively and 73.9 MB in Python.

What the memo in the context's own dicts costs. The walker that kept a memo
of its own, measured in the same sitting, reached 3.43x and 2.08x as
geometric means, 7.16x and 2.91x as sums, and 60.2 MB. The difference is a
dict lookup, two dict writes and two key objects per value, so it shows most
on a large, flat graph: `b2_heavy_nolimits` (152,643 nodes) captures in
14.6 ms, against 6.5 ms then and 60.7 ms in Python. The second key object,
which keeps a walk over the dicts themselves in step with the Python walk,
is about 3% of the geometric mean (2.96x with one key object for both
dicts) and 4 MB on that workload. An exact tuple or frozenset costs about
0.28 us, against 0.09 us then and 1.15 us in Python: the walker now builds
its placeholder at once, as the Python walk does, where it built one only
when something referred back to the tuple.

Not measured again: the speed-up over the Python walk as it was before this
work (`ed7ae81`), which was 7.4x for capture and 3.0x for `save_dumping`
with the walker's own memo, and a PGO build, which made that walker's
capture about 7% faster and `save_dumping` no faster. SIMD has no target
here; the walk is pointer chasing through Python objects.

Small values. A memo hit costs about 1.1 us natively against 0.05 us in
Python (a longer MRO or a metaclass adds to it), and a throwaway context
that captures a scalar or a short list of scalars is faster in Python.
Creating a context costs about 2.3 us natively and 1.75 us in Python.
Creating one and capturing a single value costs, natively against Python:
an int 4.9 us against 2.05 us, `[1, 2, 3]` 5.2 us against 3.5 us,
`[1, {"a": "b"}, (2.5,)]` 6.5 us against 6.9 us, a 20-key dict 8.0 us
against 20.6 us (medians of five rounds of 20,000 contexts, two runs, before
`_sources` got a key object of its own).
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
- Platforms. The extension has been built and run locally on macOS arm64
  only; Linux and Windows are left to CI's `native` job.
- Free-threaded CPython. abi3 does not cover 3.13t and 3.14t, which use
  the Python walk. Running `build_native.py` on such an interpreter has not
  been tried.
- Stack budget (see "Recursion and the native stack"): deep
  `max_depth=None` captures beyond roughly 90-220 levels run in Python, as
  does any walk level fewer than 64 frames from the recursion limit.
- Per-call overhead (see "Performance"): a memo hit costs about twenty
  times as much as in Python, so small contexts lose.
- Placeholders of exact tuples and frozensets (see "Performance"). Building
  one only when something refers back to the container would bring a
  tuple's cost near a list's, about 0.09 us, for a marker in `objects`
  while the container fills and a pass that names the unfilled ones when
  an error ends the call.
