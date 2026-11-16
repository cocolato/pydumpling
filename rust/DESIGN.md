# Native value walker: design

This crate builds `pydumpling._native`, an optional native implementation of
the bounded value walk in `pydumpling/snapshot/values.py`. The Python walk
there is the reference and the fallback. This document describes the design
as it stands, for whoever changes it.

## Contract

As approved by the owner (2026-11-15): wherever the Python walk does not
reach the recursion limit, a native capture writes the same snapshot (pickle
bytes, truncation counters and their order, node counts, `objects` and
`_sources` contents and order) and runs the same application code the same
number of times. There is no promise about where recursion overflows: near
the limit the native walk may capture more, never less. CPython's own C
recursion and stack limit is excepted, narrowed as far as it goes: a capture
that starts with too little C stack runs entirely in the Python walk; native
code reaches Python code only through the C entry points the Python walk
uses; what remains is that a Python-mode subtree starts one interpreter
entry deeper, and on 3.14+ application code the native walk runs has the
native frames beneath it ("The C recursion limit" states the remainder with
measured shifts). There is no promise about what tracers, profilers,
coverage, `sys.monitoring` or audit hooks see of the walk's internals, nor
how often pydumpling's own helpers run.

The walker undoes nothing to keep it: it runs application code where the
Python walk runs it, and sends a node it does not model to the Python walk
in place before that node has done anything. It works on the context's own
state and keeps no memo of its own. The differences under "Accepted
differences" go beyond the contract and wait for the owner's ruling.

## Scope

Only the bounded capture walk is native: `ValuesMixin.value`, `_mapping` and
the helpers they call (`_attribute_mapping`, `_root_value`, `_fill_copy`,
`_filled_slot`, `_scalar`, `_layout`, `_attributes`, `_remember`,
`_has_nodes`, `_omit`, `_type_name`, `_text`, `_code_text`,
`_bounded_text`, `_bounded_bytes`, `context._utf8_prefix`), and
`FakeClass.__repr__` as the walk calls it. The walker does what each does,
in the same order, with the same tallies, and calls none of them. The
writer, reduction, the debugger, the report and the CLI stay in Python: the
value walk dominates capture, which runs while the failing process waits,
and the rest is I/O bound or built on `pdb`.

## Build and supported range

    python rust/build_native.py [--debug] [cargo build arguments...]

The script runs `cargo build --lib --locked` for the running interpreter
(`PYO3_PYTHON`), in release unless `--debug`, `--release` or `--profile`
says otherwise, and installs the library as `pydumpling/_native<suffix>`
through a temporary file and a rename, removing other builds that would
shadow it. `RUSTFLAGS` and extra cargo arguments pass through, for
instrumented and profile-guided builds. The release profile is
`opt-level = 3`, fat LTO, `codegen-units = 1`. The extension carries a
fingerprint of the crate's sources (`pydumpling._native.fingerprint`), and
`tests/conftest.py` refuses to run the suite on a build from other sources.
An extension that lacks a name the crate reads from `values`, or has other
entries, is not used: `values` warns and captures in Python.

The crate uses PyO3 0.30.1 with `abi3-py312`, `stacker` for the thread's
stack bounds and `#![forbid(unsafe_code)]`, and needs Rust 1.88. The
extension is git-ignored; the wheel stays pure Python. CI's `native` job
builds it and runs the suite on Linux for 3.12 to 3.15 and on Windows and
macOS for 3.14; the `rust` job runs `cargo fmt --check`, clippy with
`-D warnings` and `cargo test`; the `tests` job checks the Python walk
without the extension, 3.11 included.

Supported: CPython 3.12 to 3.15. 3.11 runs the Python walk only, because
the extension is built for the stable ABI of 3.12: on 3.11 C calls count
toward the Python recursion limit itself, which the C level the walker's
entry adds beneath application code and Python-mode nodes ("The C
recursion limit") would move, and the contract excepts only the C limit.
Free-threaded builds run the Python walk too: a stable-ABI extension does
not fit their object layout and the walker relies on the GIL, so `values`
does not import it where `Py_GIL_DISABLED` is set and `build_native.py`
refuses to build there. The extension has been developed and benchmarked
on macOS arm64 only.

## Choosing the walk

**Per process.** `values` imports the extension when it is itself imported,
on 3.12+, if the build is not free-threaded, `PYDUMPLING_NATIVE` is not
`"0"` (read then only), Python does not run under `-b`, and `sysconfig`
loads. Under `-b` comparing bytes with str warns, and a warning raised by
the walker's own dict and set operations would be attributed to the line
that called the walker rather than to the Python walk's comparison. Without
the extension every context captures in the Python walk: pure mode.

**Per class.** `ValuesMixin._native_walk` is `False`. When the extension
loaded, `values._stock`, called at the end of `context.py`, sets it `True`
on `SnapshotContext` and records that class's walk attributes as
`_STOCK_WALK`. `_choose_walk` judges every other class once, from
`ValuesMixin.__init_subclass__`, or from `SnapshotContext.__init__` at the
first instance of a class whose bases' `__init_subclass__` never reached
`ValuesMixin`'s (`_native_walk_class` names the class judged). For each
name in `_NATIVE_WALK` (17 walk methods), `_STATE_ACCESS`
(`__getattribute__`, `__getattr__`, `__setattr__`, `__delattr__`) and
`_NATIVE_STATE` (ten state names), it compares by identity the raw class
attribute the MRO yields, read through `type.__dict__` without `__get__`
or a metaclass hook. A class that differs in any of them (an override, a
property in place of a state attribute, a slot of a state name), or along
whose MRO a class dict holds a key that is not an exact str, gets
`_native_walk = False` and captures in Python, which runs what it changed.
A class that sets `_native_walk = False` itself keeps it, as do its
subclasses, even one that restores the stock attributes; any other value
it sets is replaced by the verdict. Nothing checks the class again. A
subclass or an instance with `_native_walk = False` is a pure context in a
native process, with exactly the Python walk's frames; tests compare the
walks that way.

**Per instance.** An instance whose own `_native_walk` is `False` captures
in Python for good. An entry sets it, *retiring* the context, when the
call meets, before it has done anything, an argument or state only the
Python walk reproduces: an argument shape pydumpling's callers never pass
(a depth that is not an exact int or exceeds 2**62 in magnitude, a
`priority` that is not an exact tuple, list, set or frozenset of exact
strs, `counted` other than `True` or `False`, a `container` other than
`dict` or `FakeDict`); a class not judged, whose `_native_walk_class` is
not the class itself (an instance unpickled, made by `object.__new__` or
given by assigning `__class__` a class created past
`ValuesMixin.__init_subclass__`, which no `__init__` judged); `objects`,
`_nodes` or `options` missing, `objects` not an exact dict, or `_nodes`
not an exact int from 0 to 2**64 - 1; options the walker does not read
("State reads"), or a budget missing or not `None` or a positive exact
int; `_unlimited` other than `False` (`value` offers a call where it is
any false value). Other state of another kind does not retire (a
`_truncation` that is not an exact `Counter` of exact str keys and int
tallies, `_sources` that is not an exact dict or is `objects` itself,
`_namespaces` not an exact set, `_layouts` not an exact dict, an unusual
`_code_text_limit`): the walker runs its code in place.

**Per call.** A call goes to the Python walk, and the context stays native,
when its entry finds before the call has done anything that less than
192 KiB of native stack is left ("The native stack"), that `_mapping` was
given a source other than a dict, a dict subclass, a `frozendict` (3.15) or
a `FrameLocalsProxy` (3.13+), whose own `items()`, read up to three times,
is application code, or that the value is one Python mode takes (below).
The entry returns `_IN_PYTHON` and the method runs the call itself, in the
same frame. Frame proxies are read natively only if `sys._getframe` was
allowed when `values` was imported.

**Unlimited contexts** are never offered: `value` reads `_unlimited` before
the flag, and `_mapping` offers a call only while `_unlimited is False`.
Probing each original with `dill.dumps` dominates their cost, and the
walker would run it beneath a C call the Python walk does not make. Nothing
is written to such a context in either mode: storing `_native_walk` on an
instance adds the name to the class's shared instance keys (3.13+), after
which a load of the flag, on any context in the process, takes 6.0 to
8.5 ns instead of 2.5 to 3.0 ns (3.13 and 3.14, 2026-11-16).

### Python mode

Python mode is how the walker, once a call has done something, sends one
node and everything under it to the Python walk without leaving the call.
`State::python_value` sets `_native_walk = False` on the instance, writes
its node count to `_nodes`, calls `ValuesMixin.value(ctx, value, depth)`
with the arguments the Python walk passes there, reads `_nodes` back and
drops its cached type facts. Then, on every exit, it puts the flag back:
where it read `True` it removes the instance's entry, so the context reads
its class's flag again, and any other value it read it sets back. So an
instance's own `True` is not kept; on a class judged `False`, where
nothing in pydumpling sets one, the context then captures in Python.
While the flag is false every call in the subtree runs the reference body:
no native frames, no extra Python frames, no C stack per level, and no
child returns to the walker. `ValuesMixin.value` is the one Python
function the walker calls. A node goes there, decided from the stack or
its type before it has done anything, when it is:

- met past the native stack budget, since the native walk's recursion
  takes C stack per level and the Python walk's none;
- an object copied as `FakeClass` while `FakeClass` has another MRO than
  at import: it would build its copies through other bases, and 3.12 to
  3.14 refuse the instance-dict descriptor it had then;
- a `FakeClass` instance while `FakeClass` holds, along its MRO, another
  `__repr__` than at import, which the Python walk calls as it now is;
- a value that is not an exact scalar, met after the call has done
  something, on a context whose `_unlimited` is not `False`.

At an entry the same nodes return `_IN_PYTHON` instead. `_STATICS.fallbacks`
counts, process-wide, every call an entry sent to the Python walk and every
node run in Python mode.

## One call

### How Python hooks in

Call sites do not change: `frames.py`, `exceptions.py` and the tests call
`self.value(...)` and `self._mapping(...)`, the plain functions of
`ValuesMixin`. At import `values` builds one table,
`_STATICS = _native.Statics(globals(), _identity)`, and binds its entries as
`_native_value` and `_native_mapping` (and as `_native_stock` the method
with which `_stock` names the composed class to the table):

    def value(self, value, depth=0):
        cached = self.objects.get(id(value), _UNMEMOIZED)
        if cached is not _UNMEMOIZED:
            return cached
        if not self._has_nodes():
            self._truncation["max_nodes"] += 1
            return self._node_marker
        self._nodes += 1
        ...  # TruncatedValue, None, bool, float, complex, int, str, bytes
        if self._unlimited:
            ...  # the serializer probe
        elif self._native_walk:
            captured = _native_value(self, value, depth)
            if captured is not _IN_PYTHON:
                return captured
        ...  # the reference walk, unchanged

`_mapping` starts with `if self._native_walk and self._unlimited is
False:` and the same call; the walker does its memo logic. `value` keeps its
memo lookup, node check and count and its exact scalars before the entry,
in the reference order, as each costs less in Python than the entry; the
entry takes back the node `value` counted and counts it as the walk counts
its own. The entry is a C call in the method's frame, so no path adds a
Python frame: a call sent to Python, a pure context and a subclass's
`value` calling `super().value` reach the recursion limit where the Python
walk alone would.

An entry first checks that the context's class was judged: the class
`_stock` named, or one whose `_native_walk_class` names itself, read
through the MRO, or from the class's own dict under a metaclass other than
`type`; otherwise it retires the context ("Per instance"). On a class not
judged, that read runs the code of a key or descriptor planted under the
name, which the Python walk never reads ("Accepted differences"). It
returns the copy, raises what the Python walk would raise, or returns
`_IN_PYTHON` before the call has done anything. When an exception ends a
call, the walker writes `_nodes` and raises, keeping what it did, as the
Python walk does. `_root_value` recovers a `RecursionError` under a root
mapping's variable, `_mapping` re-raises it rather than counting
`concurrent_mutation`, and `unless_exception` reproduces `except Exception`.

### State reads

The walker reads the context's state with attribute access, as the Python
walk does: `objects`, `_nodes`, `options` and the four budgets when the call
starts; `_sources`, `_truncation`, `_namespaces`, `_node_marker`,
`_layouts`, `_code_text_limit` and `_unlimited` where the call first needs
each, which is where the Python walk first reads it. Each is read once per
call, and only `_nodes` again: after Python code, and for a captured dict
that holds a `_nodes` key while the context does not hold the walk's count
(below). On a judged class these reads run no code.

A captured dict may be the context's own instance dict ("Capturing again
from inside a call"), whose `_nodes` the Python walk reads as the running
count. Only a dict that holds the interned str `"_nodes"` (an attribute
name, or a literal) as a key while the context does not hold the walk's
count, with the very object the context holds there (read as an
attribute), can be. For such a dict the walker writes its count to the
context, which the Python walk holds there at every step, and reads the
dict's entry again by iterating its storage, which runs no code: only the
context's own dict then holds the very int written. The walker never
reads `ctx.__dict__` for this. Reading it makes the instance dict exist
(3.12 to 3.15), after which attribute stores on that context take about
twice as long, and on 3.12 loads too: when an object with a `_nodes`
attribute made an earlier build read it, later calls of a small value on
that context took 1.2 to 1.33 times as long natively as in the Python
walk on 3.12 and 3.14 (2026-11-16). The walker reads it only to retire an
instance of a class not judged, whose flag it writes there.

The budgets are read as attributes of options whose type passes a check
made once per type (for `DumpOptions` at import, and kept for up to 64
other types at a time, past which the table forgets them all and judges
each again): along its MRO, read through `type.__dict__`, every class dict
holds only exact str keys, no class but `object` holds `__getattribute__`
or `__getattr__`, `DumpOptions` is there, and no class in front of it
holds a budget's name. Any other options object retires the context.

### Python code inside a call

Where the Python walk runs Python code that is not pydumpling's, the walker
makes the same operation once, in place (`State::application`), through the
C entry point the Python walk goes through (a descriptor's `__get__`, a
builtin method, `PyObject_GetItem`, `PyNumber_Add`, `operator.iadd`,
`PyObject_SetItem`, `str.encode`):

- a truncation tally on another kind of `_truncation`: subscript, in-place
  add and store (an exact `Counter` of str keys and int tallies is counted
  in the dict);
- `_sources` other than an exact dict: `__setitem__`; `_namespaces` other
  than an exact set: `in` and `add`; `_layouts` other than an exact dict:
  `get` and `__setitem__`; an unusual `_code_text_limit`: `_bounded_text`'s
  slice, `len`, `encode` and comparison;
- a class along whose MRO a class dict holds a key that is not an exact
  str: `_attributes`' `.get("__dict__")` per class dict, `_type_name`'s
  `type.__dict__["__module__"].__get__`, and `_layout`'s pass with the same
  set `in`/`add` and `name == "_fields"`, so such a key's `__hash__` and
  `__eq__` run as often;
- a getset other than the instance dict's planted as `__dict__`: its
  `__get__`;
- a `__qualname__` that is a str subclass: `module + "." + name`, whose
  `__radd__` runs, and where the name is then not an exact str,
  `"<%s snapshot>" % name` and the code-text bound, with its own code;
- an own copy's attributes with a key that is not an exact str:
  `attrs.get(key)`; a slot or field name that is not plain: the store into
  the copy's namespace, and every store there after it, the type name's
  included, which may compare with it;
- a `_layouts` entry `_layout` did not write: `layout[1]`, `layout[2]`,
  iterating its members, `key, value = item`, `member.__get__(value, cls)`,
  each slot's store, `namespace.update(zip(fields, copy))` and the type
  name's store;
- a memo hit in `_mapping` on an entry other than a dict, `FakeDict` or a
  builtin scalar or container: `isinstance(cached, TruncatedValue)` and
  `cached.reason != "max_depth"`;
- leaving a snapshot copy's private attribute out of a dict with a key that
  is not plain: the dict insert, which hashes it, and every insert after
  it, which may compare with it;
- the signal check, every 4,096 nodes.

Around each, and around Python mode, the walker writes its node count to
`_nodes` before and reads it back after, and drops its per-type facts (the
layout read from `_layouts` among them), `FakeClass`'s state and whether
`_truncation`'s keys are plain, which the next value of each type reads
anew. It also writes `_nodes` before capturing the context's own instance
dict, which holds it.
Everything else the walker calls is C: descriptors' `__get__`, builtin
types' unbound methods (`dict.items`, `set.__iter__`, `int.bit_length`, a
builtin base's `__new__`), `slice`, `zip`, `operator.iadd` and the
abstract object protocol. Through them it also reaches Python code it
does not sync around: `FakeContainer.__hash__`, pydumpling's own, wherever
a set or dict hashes a copy of a container subclass instance (a member of
a set copy, a key of a dict copy), as the Python walk does at that step;
and the finalizers and weak reference callbacks of what the walker lets
go of, which the Python walk may let go of at another step ("Hooks and
signals"). A call imports nothing (`builtins.__import__` may be the
application's): `Statics` resolves every module-level object at import.

### Capturing again from inside a call

Nothing refuses a capture through a context while one of its calls runs.
Code the walk runs at a synced point (application code in place, Python
mode, a signal handler or a collection there or at the signal check) may
capture through the same context or a shallow copy sharing its dicts, and
another thread can run only while Python code runs inside the call. Such a
capture counts on from the synced `_nodes`, the outer call on from its
count, and it finds in `objects` every copy the outer call made, which the
walker writes there as it makes it. One from Python code between two
synced points (a finalizer, or a signal handler, a collection or another
thread while `FakeContainer.__hash__` runs) sees a stale count ("Accepted
differences").

### The memo

The walker looks a value up in `ctx.objects` before copying it, and writes
the copy to `objects` and the original to `_sources`, as `_remember` does,
each under an int object of its own as that function's two `id()` calls
make them (a walk that captures the dicts themselves copies every key). An
exact tuple or frozenset gets its `TruncatedValue` placeholder while it is
filled. What Python code does to the dicts the walk finds at its next
lookup; a dict replaced on the context is found by the next call.

## Recursion and the stack

### The native stack

The walker reads neither the recursion limit nor the caller's depth, and
pushes no Python frame per level. Its C stack is bounded instead, from
`stacker::remaining_stack()` at the entry: `min(1536 KiB, (remaining -
128 KiB) / 2)`. A node met past that budget goes to Python mode. The
reserve and half of the rest are left for what runs beneath the walk:
application code, Python mode, collections, finalizers, and from 3.14
CPython's own C stack checks. A budget under 32 KiB, that is less than
192 KiB left, sends the call to the Python walk without retiring the
context. Cold paths (Python mode, the signal check, a state attribute's
first read) are out of line, so their locals take no room in every level's
frame.

On macOS arm64 (release build, `max_depth=None`, 2026-11-16) the budget
holds 1,424 levels of nested lists or tuples, 722 of dicts and 595 of
objects on the main thread on 3.13 and 3.14, and on a 512 KiB thread 182,
92 and 76 on 3.13 and 180, 91 and 75 on 3.14.

### Near the limit

Where the Python walk overflows, the native walk may capture more. In the
gate (2026-11-16, 3.13; 3.14 the same unless named), `_mapping` of a
2,000-level list on the main thread captures all 2,003 nodes natively, the
levels past the budget in Python mode, against 992 and a `'recursion'`
placeholder in the Python walk; on a 512 KiB thread 1,172 nodes (3.14:
1,169) against 991, both with the placeholder. A capture in an
`except RecursionError` handler 30 frames below the limit captures the same
339 nodes (3.14: 338) in both walks.

"Never less" is checked by a provisional definition the owner has yet to
approve. Where the Python walk reached the limit: (i) no root variable both
walks captured (an entry of a mapping captured at depth 0, of a captured
exception's state, or of a captured frame's namespaces) reaches more
`TruncatedValue('recursion')` placeholders natively; (ii) the native walk
keeps at least as many originals in `objects`. Not nodes: the Python walk's
`_nodes` also counts the copies `_root_value` drops after an overflow
(parity seed 1279 on 3.13: 1,029 nodes natively against 1,955, while
`objects` keeps 1,029 against 39). Under a binding `max_nodes`, capturing
more of an earlier variable can leave a later one with fewer entries and a
`max_nodes` cut the Python walk lacks, which the definition allows: with
`max_nodes=150` 25 frames below the limit (`near_limit_nodes`), the native
walk keeps a 41-level list whole and cuts the next variable at 150 nodes,
where the Python walk gives the list a `'recursion'` placeholder and keeps
the next variable whole (86 nodes). Of 3,000 parity seeds on 3.13
(`PYTHONHASHSEED` 0, and again with 1), 245 reach the limit in the Python
walk, 34 of them show that shift, and none violates (i) or (ii).

### The C recursion limit

What remains of the contract's carve-out, measured on macOS arm64 (release
build) on 2026-11-16, and for 3.12 and 3.15 also on 2026-11-15 with the
same results.

On 3.12 and 3.13 CPython counts C levels. The entry is a C call, so
application code the walker runs in place starts one C level deeper than in
the Python walk (two for a tally that is not an exact int: PyO3's safe API
has no in-place add, so it goes through `operator.iadd`), and a Python-mode
node two deeper (the entry and the call back into Python). The Python
walk's `_mapping` resumes generators (`dict.fromkeys` over one,
`itertools.islice`) that CPython counts too, which outweighs the entry's
level where the value holds a dict or an object. Beneath nested `==` at the
default limit, the deepest nesting at which each walk captures the value
whole (3.12 / 3.13):

    captured                            Python        native        shift
    [a dict, an object]                 9,989/9,990   9,992/9,993   3 later
    a root namespace, string bounded    9,989/9,990   9,993/9,994   4 later
    [a dict, an object whose class
      name runs a __radd__]             9,989/9,990   9,991/9,992   2 later
    [a dict, a long string],
      a tally that is not an exact int  9,989/9,990   9,990/9,991   1 later
    [a long string], the same tally     9,991/9,992   9,990/9,991   1 sooner
    [a dict, a FakeClass copy] after
      FakeClass.__repr__ was replaced
      (a Python-mode node)              9,989/9,990   9,987/9,988   2 sooner

From 3.14 CPython measures the C stack instead; those captures start with
less than 192 KiB left and run in the Python walk from their start, so no
form shifts on 3.14 or 3.15.

In a capture that starts with more stack, application code the walker runs
in place, and a Python-mode subtree, have up to the stack budget of native
frames beneath them, where the Python walk's recursion takes no C stack.
On a 512 KiB thread, 150 levels of lists deep (about 175 KiB of native
frames), with a `__qualname__`'s `__radd__` there taking the `repr()` of a
K-level nested list, the largest K that completes:

    Python   Python walk   native   past it
    3.12     1,584         1,086    the process crashes (SIGBUS)
    3.13     1,584         1,086    the process crashes (SIGBUS)
    3.14     2,206         1,455    RecursionError (stack overflow)
    3.15     2,083         1,382    RecursionError (stack overflow)

So on 3.12 and 3.13, which do not check the C stack, a native capture
crashes for K from 1,087 to 1,584 where the Python walk completes. Python
code reaches the C limit when its recursion makes a C call per level
(`list(map(f, ...))`, a property, a C-wrapped function), from a callback
deep in recursion that runs in C (a `__repr__` or `__eq__` of nested
containers), and from 3.14 on a thread with a small stack. Other shapes
have not been measured. A smaller stack share for the walker would narrow
this and send more of a deep value to Python mode.

## Hooks and signals

During a native call a profiler, tracer, coverage or `sys.monitoring` tool
sees only the application code the walker runs in place, Python-mode
subtrees, and `FakeContainer.__hash__` where a set or dict hashes a
container copy, as in the Python walk: the walk is one C call from `value`
or `_mapping`. `cProfile` attributes its time to that call, coverage shows
the walk's lines in `values.py` and `context.py` as not run, and `pdb`
cannot step into it. An audit hook sees one `builtins.id` event per native
call, from `value`'s memo lookup, where the Python walk raises one per
`id()` (47 for `[1, "two", (3.0,), {"k": [None]}]`). To profile, trace,
measure or debug pydumpling's own walk, set `PYDUMPLING_NATIVE=0` before
`values` is imported, or `_native_walk = False` on the context class or
instance.

CPython runs pending signal handlers, and from 3.12 the cyclic collector,
only where it checks for pending work, which Python code does at each call
and loop, or in `PyErr_CheckSignals`. Inside a native call that is the
walker's signal check, every 4,096 nodes, so Ctrl-C interrupts a long
capture, and any Python code that runs in the call. At the signal check,
around application code and in Python mode the walker syncs `_nodes` and
reads type facts anew after; inside `FakeContainer.__hash__` or a
finalizer it does neither, so a handler or a collection's finalizers that
run there see a stale count, and a class they change may be seen late
("Accepted differences"). An exception a handler raises ends the call as
it would end the Python walk's. Sending a call to Python whenever a signal
handler is installed would turn the walker off in every `asyncio.run`
program and every server with a SIGTERM handler.

## Invariants

- Native code runs application code only where the Python walk runs it,
  once, through the same C entry point, with `_nodes` synced around it,
  except for the finalizers and weak reference callbacks of objects the
  walker lets go of, the key comparisons of objects planted in `objects`,
  `_sources`, `_namespaces` or `_layouts`, a key or descriptor planted
  under `_native_walk_class` on a class not judged, and what a signal
  handler or a collection runs while `FakeContainer.__hash__` runs. Types
  are compared by address, class attributes read through `type.__dict__`
  descriptors, containers through their builtin base's unbound methods,
  and keys other than ids hashed or compared natively only when of a plain
  kind.
- The order of memo entries, node counts and truncation counters is part of
  the snapshot, as is object sharing. The walker follows the Python walk
  step for step, builds each copy as its class's constructor does while the
  class holds no hook of its own, and uses the same reason objects
  (`_RECURSIVE_IMMUTABLE`, the node marker).
- No native object per context and no memo of the walker's own: between
  calls the table keeps only the options types it judged (64 at most) and
  the context class `_stock` named, so a context is collected as in the
  Python walk. A memo of its own would need a sync with the dicts at every
  call and its writes flushed before any Python code reads them. The table
  is built from the `values` module's own globals and kept for the
  process; a reloaded or vendored package gets its own.
- `address::push` grows a vector into a fresh buffer of twice the capacity
  instead of `realloc`, whose freed large blocks macOS keeps resident.
- Every source file stays within 400 lines; the crate forbids `unsafe`.

## Accepted differences

Each goes beyond the approved contract and waits for the owner ("Open
problems").

- **The walk is chosen when a context class is created** (or at its first
  instance). A walk method or state attribute changed on the class
  afterwards, or a walk method set on an instance, is not seen by a native
  context: a method patched onto `SnapshotContext` after import runs under
  `PYDUMPLING_NATIVE=0` only. A state property or attribute hook patched on
  later runs at the walker's reads, once per call per attribute, where the
  Python walk runs it at every read; so does the `__eq__` of a key planted
  in the instance dict that hashes like a state name. `value`'s memo
  lookup, node check and exact scalars stay in Python, so changes there
  run. A subclass of a class judged `False` captures in Python even if it
  restores the stock attributes. To have a change run, set
  `PYDUMPLING_NATIVE=0` or `_native_walk = False`, or subclass after it.
  To tell whether a class was judged, pydumpling reads its private
  `_native_walk_class` from it, natively only: at the first
  `SnapshotContext.__init__` of a class created past
  `ValuesMixin.__init_subclass__`, which then writes the name, and at a
  native entry for an instance that reached such a class without
  `__init__`. A key planted in the dict of such a class, or of a base ahead
  of `SnapshotContext`, that hashes like the name runs its `__eq__` there,
  once to three times, and a descriptor defined under the name its
  `__get__`, once (at an entry, under the metaclass `type` only). The
  Python walk never reads the name; the capture is the same.
- **Options are judged once per type, budgets read once per call.** A
  property or `__getattribute__` patched onto `DumpOptions` or a subclass
  after its verdict, or a key planted in its class dict that hashes like a
  budget's name, runs once per call per budget natively, where the Python
  walk runs it at each budget read; both take a property's answer. The
  verdict on `DumpOptions` lasts for the process, and on a subclass while
  the table keeps it ("State reads"). A call that never needs a budget
  still reads it natively.
- **State is read once per call.** Code that runs inside a native call
  (application code it runs, a Python-mode node, a finalizer, a signal
  handler, another thread while it runs Python code) and rebinds state
  other than `_nodes` (`objects`, `_sources`, `_truncation`, `_namespaces`,
  `_layouts`, `options`, a budget) after the call read it, or rebinds the
  context's `__class__`, is seen from the next call; a Python-mode node
  reads it anew. Replaced `objects` or `_sources` dicts get the rest of the
  call's copies, where the Python walk puts them in the new dicts, so a
  later call captures those values again and counts their nodes again.
  Changes to the same dicts are seen at once. `_nodes` rebound to an exact
  int from 0 to 2**64 - 1 is read back at once; rebound to anything else
  (a negative int or a larger one, a float, an int subclass), the walker
  counts on from its own count and writes it there at its next sync, where
  the Python walk counts on from that value.
- **The count is stale between synced points.** Code that runs there runs
  as often as in the Python walk but sees `_nodes` as of the last sync: a
  finalizer or weak reference callback run as the walker lets go of a value
  (the Python walk can let it go at another step); the `__eq__` of an
  object planted in `_namespaces`, or of a key of `objects`, `_sources` or
  `_layouts` not of a plain kind, that hashes like an id the walker looks
  up; and a signal handler, a collection's finalizers and callbacks, or
  another thread that runs while `FakeContainer.__hash__` runs. A capture
  through the same context from there starts from that count, and the
  walker then writes its own count over the one it left, so its nodes
  count toward no budget. pydumpling writes only ids there. Such a key of
  `_layouts` also runs less often: the walker looks a class's layout up
  once and keeps it until it next syncs, where the Python walk looks it up
  for every value that needs it. For three instances of one tuple subclass
  it runs twice natively, at the lookup and the insert for the first, and
  4 times in the Python walk, which then looks it up for each other one.
  Telling such a key apart takes a pass over `_layouts`; made once per
  call, it cost 7 times the Python walk's time for a value on a context
  holding 2,000 layouts, and 1.13 times for a 2,000-frame stack whose
  frames each hold an instance of a record type of their own
  (`call_live_subclass`, `record_stack_d2000`, 3.13). A context's instance
  dict that application code replaced with one whose `_nodes` key is an
  equal str, not the interned one, holds the count of the last sync when
  the walker captures it from inside a call, where the Python walk's copy
  holds the running count. The walker looks for that key by identity only,
  which keeps a str comparison off every key of every mapping it captures.
- **A class changed mid-call may be seen late.** One changed by
  application code the walker runs is seen at once, as the walker then
  drops its type facts. One changed between synced points (its
  `__bases__`, a `__dict__` descriptor), by a finalizer or weak reference
  callback or by a collection or signal handler that runs while
  `FakeContainer.__hash__` runs, is seen, for a type whose facts the call
  holds, from the next synced point or call; the Python walk rereads a
  type's facts for every value (both walks cache layouts per context).
- **The table is built at import.** `values`' globals patched afterwards
  (`FakeClass`, `TruncatedValue`, `DumpOptions`) are not seen by the
  walker, which never calls `_plain_key` either; a Python-mode node sees
  them. Methods patched onto `collections.Counter`, such as `__missing__`,
  do not run for counts on an exact `Counter`, which the walker writes to
  the dict. Hooks patched onto the copy classes (`TruncatedValue`,
  `FakeClass` and the `Fake*` containers and scalars: `__new__`,
  `__init__`, `__setattr__`, `__getattribute__`, a `__dict__` descriptor,
  another `_type_key`) neither run nor are seen when the walker builds a
  copy with the builtin base's `__new__` and the descriptor and `_type_key`
  of import time; the Python walk calls the class. Hooks that read a
  captured `FakeClass` (`__repr__`, `__getattribute__`) run as in Python.
- **A context captured as a value** (one a captured frame holds, or one
  passed to `value`) shows `_native_walk = False` in its instance dict
  where the Python walk never writes it: on a retired context for good, and
  on a context while one of its nodes runs in Python mode, after which the
  walker removes the entry where it read `True` and otherwise sets back
  the value it read ("Python mode"). A copy or pickle of the context made
  there keeps the flag and captures in Python for good. An unlimited
  context holds no flag in either mode. Either write adds the name to the
  class's shared instance keys (3.13+), so from then on, in that process,
  each flag load on any context costs 3 to 6 ns more, in either walk.
- **Unpacking a `_layouts` entry Python code wrote.** For an item that is
  not an exact pair, the walker raises the interpreter's errors but names a
  non-iterable item's type by its `__name__`, where the interpreter prints
  `tp_name` (with the module for a static extension type) cut at 200
  characters, and decides whether the item is iterable by looking
  `__iter__` and `__getitem__` up in its classes' dicts, which runs the
  `__eq__` of a key there that is not an exact str and hashes alike.

Within the contract: the C recursion limit, what hooks see, and capturing
more near the recursion limit. One more changes only the cost: if an audit
hook refuses `sys._getframe` when `values` is imported, every
`FrameLocalsProxy` namespace runs in the Python walk for the process.

## Behaviours of the reference walk both walks keep

The native walk reproduces these byte for byte, so fixing one means
changing both: a non-dict namespace's own `items()` view is iterated up to
three times, and a one-shot view loses entries in later passes; after a
`RecursionError` in a frame variable, `_root_value` forgets the half-built
copies but keeps their node count and counters; instances whose `__dict__`
is a member descriptor (`SimpleNamespace`, modules) lose their attributes;
the first capture of an exception materializes its `__dict__`, so a second
capture differs; a str-subclass `__qualname__` runs its own `__radd__` and
slicing; from 3.13.6 and 3.14 `reprlib` writes an int past
`sys.get_int_max_str_digits()` with its address, so the message of an
exception whose arguments hold one differs between dumps.

## Module map

    lib.rs        module `pydumpling._native`: `Statics`, `fingerprint`
    walker.rs     the Statics handle and its entries `value` and `mapping`:
                  argument checks, the judged check, retiring, the stack
                  budget; the sync of `_nodes` around Python code
                  (`application`), Python mode (`python_value`); the
                  fallback counter
    statics.rs    per-import table of types, base methods, descriptors;
                  whether a context's class was judged
    state.rs      State: the context state a call reads, at its start or
                  when first needed; inserts into a dict that may hold
                  application keys; the options verdict
    walk.rs       value(): the memo, exact scalars, containers, the signal
                  check
    objects.rs    objects copied as FakeClass, _attributes,
                  FakeClass.__repr__
    copies.rs     _scalar, _fill_copy, the copy classes as at import
    layouts.rs    _layout, _filled_slot
    mapping.rs    _mapping, _attribute_mapping, _root_value
    keys.rs       key selection (preferred names, link order, limit + 1);
                  `key, value = item`
    text.rs       budgets, tallies, _remember, _omit, _type_name, the bounds
                  of text, code text and bytes
    types.rs      per-type facts from the MRO, cached until Python code
                  runs; the class dicts along an MRO; FakeClass as it is now
    address.rs    maps keyed by object address; a vector push macOS reuses

The crate has 3,472 lines (`wc -l rust/src/*.rs`). `rust/bench/` holds
`never_slower.py`, the gate; `bench_walk.py`, whole-workload timings; and
`workloads.py`, the six workloads both use.

## Testing

The suite runs with the native walker and with `PYDUMPLING_NATIVE=0` on
CPython 3.12 to 3.15, and 3.11 runs it on the Python walk. The native tests
skip when `values` did not import the extension.

The parity oracle is byte identity. `test_walk_parity` draws, per seed,
random `DumpOptions`, a random object graph (nested containers, subclasses,
slots, named tuples, enums, cycles, shared references, long strings, big
ints, `OrderedDict`s with non-plain keys, frame-like namespaces with
priorities) and a random stack and sequence of captures, and runs it on a
native and a pure context over the same objects. Pickles, truncation
counters in key order, node counts, the kinds of what `objects` and
`_sources` hold in order, and the application hooks run must match. Most
seeds ending in 3 run 120 to 400 frames from the recursion limit; deep
chains (seeds ending in 9, and some others) pass the stack budget or reach
the limit, and where the Python walk reached it the test checks "never
less" (`_captures_less`) instead of bytes. `PYDUMPLING_PARITY_SEEDS` sets
the number of seeds (300 by default); a seed reproduces under the same
`PYTHONHASHSEED`, and a failure prints the command. `test_dump_parity`
compares whole dumps for 20 seeds, and
`test_parity_check_detects_differences` checks the comparison itself.

Each decision above also has a targeted test against the Python walk's
bytes and hooks: each operation run in place, each node run in Python mode,
captures from inside a call, the native bounds of text and bytes, each
budget at 1, changes between calls, recursion and small thread stacks in
child processes, and that no path imports or adds a Python frame. Repeated
captures of sliced text and bytes keep no more blocks allocated natively
than in Python, and capturing objects with a `_nodes` attribute leaves
the context's instance dict unmade. The behaviour of each accepted
difference is pinned in both walks, in `tests/test_native_differences.py`
where no other test covers it, but one path: a class changed by a
finalizer that runs as the walker lets go of a value, which needs a value
only the walker still holds, and so another thread; the same change from
inside `FakeContainer.__hash__` is pinned. Costs are measured, not pinned,
and of the C recursion limit only the direction per form is pinned.
`tests/test_native_compat.py` checks in both modes that one state dumps
to the same bytes twice, that a native dump loads and reports without the
extension, that a context holds the same attributes in both modes, that a
budget of `sys.maxsize` or more bounds no mapping, and that a context
class's creation reaches an `__init_subclass__` later in its MRO.

Rust unit tests are in `address.rs` and `text.rs`. For coverage, build with
`RUSTFLAGS="-C instrument-coverage"` through `build_native.py`, run the
suite and the parity harness, and report with `llvm-profdata` and
`llvm-cov`. Neither coverage nor the mutation check (last run before the
memo was removed) has been run on this layout.

## Performance

### The gate

The owner's requirement is that a native capture is never slower than the
Python walk; `rust/bench/never_slower.py` checks it. Each scenario runs in
fresh child processes, one per mode, native (the extension must load) and
pure (`PYDUMPLING_NATIVE=0`), with `PYTHONPATH` at the tree and
`PYTHONHASHSEED=0`, alternating which runs first. A child builds the
scenario once to hash its content (snapshot pickle, truncation counters in
order, node count, kinds of what `objects` and `_sources` hold) and once to
time it (median microseconds per op over at least 20 runs and 1 s).

A scenario is judged on the median native/pure ratio of its rounds
(`--rounds`, 3): at most 0.99 passes, at least 1.05 fails, and in between
it runs alone for 12 more rounds in ABBA order and fails if the median of
all its rounds exceeds 1.02 (`--tolerance`). Its content must match too;
in a near-limit scenario (`~`) a round whose pure child reached the limit
is held to "never less" instead, and reported as shifted when it shows the
allowed shift. `--baseline DIR` compares this tree's pure walk with
another tree's, both children pure, by the same rule, near-limit scenarios
on content. Numbers that count come from an isolated copy: `--isolate DIR`
copies the checkout with its built extension (without `.git`, `.venv`,
`rust/target`, `_review`, `rust/migration`, `rust/bench/results` or
caches) and runs the copy's own script, and every report records a digest
of the tree it ran from.

The 51 scenarios cover per-call costs, one of them on a context whose
`_layouts` holds 2,000 classes; `_mapping` of a frame namespace from 20 to
950 frames deep; `from_frame` and `dump_current_traceback`; a 2,000-frame
stack whose frames each hold an instance of a named tuple type of their
own; seven tracing and profiling hooks; option and context subclasses;
unlimited contexts; a 2,000-level list and a 300-object chain on the main
thread and a 512 KiB thread; a capture in `except RecursionError`;
`near_limit_nodes`; five dense frames of about 300 non-builtin values each
(dataclasses, enums, datetimes and str subclasses, namedtuples,
`OrderedDict`s, `defaultdict`s and `Mapping`s), for the type-fact cache; a
`max_depth=100` frame on both stacks, for Python mode; and the six
workloads of `workloads.py`.

### Native against pure

Gate run of 2026-11-16, 00:52 to 01:13 UTC, from an isolated copy (tree
`cfb27f5659fd1945`): macOS 27 arm64, Apple M5 Pro, fat-LTO release build
without PGO, CPython 3.13.18 and 3.14.10, load average 0.9 to 1.5. All 49
scenarios pass on 3.13 and 48 on 3.14, where `call_live_bytes` failed the
sweep at 1.052 but gives 1.003 alone over 15 rounds (0.989 in a later run);
both modes run the same Python code for exact bytes. Contents match but
where the native walk captures more near the limit. Native/pure:

                                             3.13          3.14
    six workloads, geometric mean            0.32          0.36
      per workload                           0.16-0.68     0.19-0.71
      b2, no budgets, ms native / pure       13.6 / 57.0   13.8 / 48.6
    _mapping of a typical frame, any depth   0.16          0.19-0.20
    from_frame, 20 / 700 frames deep         0.42 / 0.47   0.45 / 0.51
    dump_current_traceback, 20 / 700         0.68 / 0.70   0.69 / 0.71
    no hook                                  0.42          0.50
    settrace / setprofile / cProfile         0.27/0.17/0.14  0.27/0.18/0.15
    sys.monitoring / coverage sysmon, ctrace 0.38/0.41/0.16  0.44/0.49/0.17
    dense frames                             0.18-0.25     0.21-0.30
    options subclasses                       0.16-0.17     0.20-0.22
    context subclass, unlimited (Python)     0.996-1.006   1.004-1.007
    new context and a small dict             0.48          0.51
    _mapping({}) on a live context           0.27          0.27
    other per-call, answered before entry    0.975-1.012   0.981-1.052
    2,000-level list, main / 512 KiB thread  0.74 / 0.96   0.73 / 0.92
    300-object chain, main / 512 KiB thread  0.19 / 0.81   0.22 / 0.81
    max_depth=100 frame, main / 512 KiB      0.18 / 0.21   0.20 / 0.24
    except RecursionError                    0.31          0.35
    near_limit_nodes                         0.35          0.44

After the review's fixes 20 scenarios (the per-call ones, the workloads,
`dense_scalar`, options and context subclasses) were run again from an
isolated copy (tree `07c65e18b9473edb`, 06:38 to 06:51 UTC, load 4.0 to
5.6): all pass on both versions, the workloads' geometric means are 0.32
and 0.36, and every other ratio is within 0.02 of the table.

After the second round, 28 scenarios (the per-call ones, the two added for
`_layouts`, the dense frames, `hooks_none`, `mapping_d20`, `from_frame_*`,
`dump_current_*` and the workloads) were run from an isolated copy (tree
`85a2ae4fa973dfd0`, 08:19 to 08:33 UTC, load 3.7 to 6.8): all pass on both
versions. `call_live_subclass` gives 0.70 and 0.86, `record_stack_d2000`
0.56 and 0.63, the workloads' geometric means 0.32 and 0.37, and no other
ratio is more than 0.03 above the table.

### Pure mode against main

Run of 2026-11-16, 01:13 to 02:07 UTC, same copy, `--baseline` at main
(`ed7ae81`, from `git archive`), load average 1.3 to 1.7: 43 of 49
scenarios pass on 3.13 and 44 on 3.14, and all contents match main's (the
branch's one intended change to the Python walk, that an `islice` bound
past `sys.maxsize` no longer raises, shows in no scenario). Each FAIL was
rerun alone and split by source with scratch trees: (a) HEAD against main,
the PR's committed Python changes; (b) the uncommitted walker hunks
against (a); (c) the owner's uncommitted hunks against (b). Medians of 15
rounds, or of 3 at 1.05 or more:

    scenario                 Python  alone  (a)    (b)    (c)
    call_new_int             3.13    1.326  1.017  0.993  1.311
                             3.14    1.301  1.008  0.991  1.320
    call_new_dict            3.13    1.120  1.012  0.994  1.101
                             3.14    1.110  1.000  0.997  1.113
    workload_d_stdlib_...    3.13    1.181  1.002  0.999  1.162
                             3.14    1.189  1.000  0.997  1.180
    near_limit_nodes         3.13    1.021  0.998  1.016  1.022
                             3.14    1.024  0.998  1.000  1.023
    hooks_setprofile         3.13    1.021  1.005  0.999  1.019
    mapping_d950             3.13    1.022  1.011  0.999  1.005
    call_live_mapping        3.14    1.019 (passes alone)

Most of it is (c): a new context reads `options.is_unlimited` once more for
the command line's byte allowance (`SnapshotContext()` takes 2.42 us
against 1.84 us on 3.13, 2.59 against 1.97 on 3.14), and source embedding
reads each file again per capture instead of through `linecache`. With the
allowance derived from `_code_text_limit` and the lines read kept in
`linecache`, those scenarios come to 1.000 to 1.018 of main, but for
`hooks_setprofile` on 3.13 at 1.022, whose unchanged control gave 1.018 in
the same period (load 2.8 to 4.1).

(a) cost `mapping_d950` 1.1% on 3.13: HEAD bounded `islice` with
`min(limit + 1, sys.maxsize)`, a call per mapping, which the tree has
replaced with a `ValueError` handler that costs nothing unless raised.

The current tree was rerun against main (isolated copy
`06a373342c873f0b`, 03:45 to 04:22 UTC, load 2.5 to 10). As it stands:
`mapping_*` and `call_live_mapping` 1.003 to 1.009 on 3.13 and 1.016 to
1.024 on 3.14 (four fail in one run, two in another), on 3.13 and 3.14
`near_limit_nodes` 1.042 and 1.039 and `hooks_setprofile` 1.022 and 1.015,
and the (c) scenarios as above. With
the owner's two hunks changed as above, all of those pass on both versions
but `call_live_mapping` on 3.14 (1.028, then 1.030) and `near_limit_nodes`
on 3.13 (1.023, then 1.014). An A/A control (04:09 to 04:19, load 3.8 to
4.9) gave 0.986 to 1.006 for main against a copy of itself, where the tree
as it stands gave 1.008 to 1.019 for `mapping_d20`, `mapping_d900` and
`call_live_mapping` and 1.037 to 1.038 for `near_limit_nodes`.

What pure mode itself costs against main is the walker's hook in the
Python walk: a load of `_native_walk` per `value` of a value that is not an
exact scalar, a load and an `_unlimited is False` test per `_mapping`, and
the `try` around `islice`. Counted with `sys.monitoring`, capturing a
typical frame on a new context runs 37,228 bytecode instructions on main
and 37,594 on the tree with the owner's hunks changed as above (1.0% more;
37,724 as the tree stands); in time, up to about 1.5% on 3.13 and 3% on
3.14 for calls dominated by `_mapping`'s entry. Removing it would take a
second copy of the walk body or an extra Python frame, which the design
rules out.

### Reproducing

Not measured on this layout: `save_dumping` as a whole, the native walk
against main's walk directly, and a PGO build. Results go to the
git-ignored `rust/bench/results/`:

    python rust/build_native.py
    python rust/bench/never_slower.py --isolate /tmp/gate313
    python rust/bench/never_slower.py --isolate /tmp/gate314 \
        --python /path/to/python3.14
    mkdir -p /tmp/main && git archive main | tar -x -C /tmp/main
    python rust/bench/never_slower.py --isolate /tmp/base \
        --baseline /tmp/main
    python rust/bench/bench_walk.py --impl native --rounds 3 \
        --json rust/bench/results/native.json

`--scenarios` takes names or glob patterns and `--list` lists them.
`bench_walk.py` times `save_dumping(strict=True)` and capture alone per
workload in a fresh subprocess, for `--impl legacy`, `pure` or `native`;
see its `--help`. A PGO build passes `-Cprofile-generate=DIR` through
`RUSTFLAGS`, runs the benchmark and the tests, merges with
`llvm-profdata merge` and rebuilds with `-Cprofile-use=FILE`.

## Open problems

- For the owner: the accepted differences; the "never less" definition;
  what pure mode costs against main (up to about 3% on 3.14 for calls
  dominated by `_mapping`'s entry, over the 2% allowed); and the gate's
  rule, under which one noisy sweep at 1.05 fails a scenario that passes
  alone.
- The owner's uncommitted hunks cost both modes about 0.6 us per new
  context, and source embedding rereads files ("Pure mode against main").
- Past the stack budget a deep value runs in Python mode at the Python
  walk's cost, and application code the walker runs has up to the budget
  of native frames beneath it ("The C recursion limit").
- The entry's C level and `operator.iadd`'s on 3.12 and 3.13: an in-place
  add through `PyNumber_InPlaceAdd` would take one back; PyO3's safe API
  has none.
- Python code the walker does not sync around (`FakeContainer.__hash__`,
  finalizers) sees a stale count and may see a class change late; syncing
  around each hash of a container copy would cost it a `_nodes` write and
  read. The read of `ctx.__dict__` for a `"_nodes"` key ("State reads")
  could be skipped in most cases by first comparing the key's value with
  `ctx._nodes`.
- Placeholders of exact tuples and frozensets: building one only when
  something refers back to the container would bring a tuple's cost near a
  list's.
- Packaging needs maturin or cibuildwheel and per-platform abi3 wheels, a
  release decision; the extension has run locally on macOS arm64 only.
- Free-threaded CPython runs the Python walk; the walker would need to work
  without the GIL.
