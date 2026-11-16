"""The accepted differences between the native walker and the Python walk
that ``rust/DESIGN.md`` lists, and that no other module pins, each against the
Python walk.  A change to either walk here changes that entry too.  Unless a
test says otherwise, both walks behave the same on CPython 3.12 to 3.15.
"""

import contextlib
import copy
import gc
import json
import os
import pickle
import reprlib
import subprocess
import sys
from collections import Counter, OrderedDict

import pytest

from pydumpling.fake_types import (
    FakeBytes,
    FakeClass,
    FakeDict,
    FakeFloat,
    FakeFrozenSet,
    FakeInt,
    FakeList,
    FakeSet,
    FakeStr,
    FakeTuple,
    TruncatedValue,
)
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot import values as values_module

if (
    getattr(values_module, "_native", None) is None
    or os.environ.get("PYDUMPLING_NATIVE") == "0"
):
    pytest.skip(
        "the native walker is unavailable: pydumpling._native is not built, "
        "or PYDUMPLING_NATIVE=0",
        allow_module_level=True,
    )

from tests.test_native_parity import (  # noqa: E402
    _STATE_NAMES,
    _armed,
    _called,
    _check_walker,
    _child_environment,
    _fallbacks,
    _logged_capture,
    _logged_hooks,
    _logged_property,
    _mode,
)

# --- Helpers patched after import --------------------------------------------
# Helpers patched over the ``values`` module, onto ``collections.Counter``, or
# onto the classes of the copies.
#
# The native walker's table is built when ``pydumpling.snapshot.values`` is
# imported, so a module global patched afterwards is seen by no walker: a
# ``TruncatedValue`` or ``FakeClass`` subclass set as the global is the class
# of the Python walk's placeholders and copies and never of the walker's, and a
# patched ``_plain_key`` runs in the Python walk alone, as the walker never
# calls it. A node the walker runs in the Python walk in place sees the patch.
# The walker writes the truncation tally with dict calls, so a
# ``__missing__`` patched onto ``Counter`` runs once for each reason the Python
# walk first counts and never for the walker's counts, which come out the same.
# Hooks patched onto ``TruncatedValue`` or ``FakeClass`` to build a copy
# (``__new__``, ``__init__``, ``__setattr__``, ``__getattribute__``, a
# ``__dict__`` descriptor) run in the Python walk alone: the walker builds both
# through the instance-dict descriptor each had at import, so a ``__dict__``
# descriptor planted on ``FakeClass`` neither runs nor decides where the copy's
# attributes go. (A ``__repr__`` patched onto ``FakeClass``, which names a
# captured ``FakeClass``, the walker leaves to the Python walk, which runs it;
# ``test_native_parity`` pins that.)


_BOTH_WALKS = pytest.mark.parametrize("native", [False, True], ids=["python", "native"])


class _Instance:
    def __init__(self):
        self.field = 1


def _capture(native, options, value):
    """Capture ``value`` on a fresh context of the given walk, which must
    run no node in the Python walk."""
    with _mode(native):
        context = SnapshotContext(options)
        _check_walker(context, native)
        before = _fallbacks()
        captured = context.value(value)
    assert _fallbacks() == before
    return context, captured


@_BOTH_WALKS
def test_patched_truncated_value(monkeypatch, native):
    """A recursive tuple's placeholder is of the patched class in the Python
    walk and a ``TruncatedValue`` in the walker."""

    class Placeholder(TruncatedValue):
        pass

    monkeypatch.setattr(values_module, "TruncatedValue", Placeholder)
    inner = []
    outer = (inner,)
    inner.append(outer)
    _, captured = _capture(native, DumpOptions(), outer)
    placeholder = captured[0][0]
    assert type(placeholder) is (TruncatedValue if native else Placeholder)
    assert vars(placeholder) == {
        "reason": values_module._RECURSIVE_IMMUTABLE,
        "type_name": "tuple",
    }


@_BOTH_WALKS
def test_patched_fake_class(monkeypatch, native):
    """An instance's copy is of the patched class in the Python walk and a
    ``FakeClass`` in the walker."""

    class Copy(FakeClass):
        pass

    monkeypatch.setattr(values_module, "FakeClass", Copy)
    _, captured = _capture(native, DumpOptions(), _Instance())
    assert type(captured) is (FakeClass if native else Copy)
    assert vars(captured) == {
        "field": 1,
        "_FakeClass__repr": f"<{__name__}._Instance snapshot>",
    }


@_BOTH_WALKS
def test_patched_plain_key(monkeypatch, native):
    """The Python walk asks the patched ``_plain_key`` about each key of an
    ``OrderedDict``; the walker never calls it."""
    calls = []
    plain_key = values_module._plain_key

    def recorded(key):
        calls.append(key)
        return plain_key(key)

    monkeypatch.setattr(values_module, "_plain_key", recorded)
    _capture(native, DumpOptions(), OrderedDict(a=1, b=2))
    assert calls == ([] if native else ["a", "b"])


@_BOTH_WALKS
def test_patched_counter_missing(monkeypatch, native):
    """A ``__missing__`` patched onto ``Counter`` runs for the first count of
    each reason in the Python walk and for none in the walker."""
    missing = []

    def recorded(counter, key):
        missing.append(key)
        return 0

    monkeypatch.setattr(Counter, "__missing__", recorded)
    options = DumpOptions(max_items=2, max_depth=1)
    context, _ = _capture(native, options, [[1], [2], 3])
    assert missing == ([] if native else ["max_depth", "max_items"])
    assert dict(context._truncation) == {"max_depth": 2, "max_items": 1}


@_BOTH_WALKS
def test_hooks_patched_onto_the_copy_classes(monkeypatch, native):
    """Attribute hooks patched onto ``TruncatedValue`` and ``FakeClass`` run
    in the Python walk, which builds each with its ``__init__`` and fills a
    ``FakeClass`` through ``__dict__``, and not at all in the walker, which
    builds both through the instance-dict descriptor each had at import."""
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_depth=2))
    _check_walker(context, native)
    calls = []
    for cls in (TruncatedValue, FakeClass):

        def get(copy, name, cls=cls):
            calls.append((cls.__name__, "get", name))
            return object.__getattribute__(copy, name)

        def put(copy, name, value, cls=cls):
            calls.append((cls.__name__, "set", name))
            object.__setattr__(copy, name, value)

        monkeypatch.setattr(cls, "__getattribute__", get)
        monkeypatch.setattr(cls, "__setattr__", put)
    before = _fallbacks()
    context.value([[{"k": 1}], _Instance()])  # the dict is too deep
    assert _fallbacks() == before
    assert calls == (
        []
        if native
        else [
            ("TruncatedValue", "set", "reason"),
            ("TruncatedValue", "set", "type_name"),
            ("FakeClass", "get", "__dict__"),
            ("FakeClass", "set", "_FakeClass__repr"),
            ("FakeClass", "get", "__dict__"),
            ("FakeClass", "get", "__dict__"),
        ]
    )


_COPY_CLASSES = (
    FakeDict,
    FakeList,
    FakeTuple,
    FakeSet,
    FakeFrozenSet,
    FakeStr,
    FakeBytes,
    FakeInt,
    FakeFloat,
)


def _subclass_instances():
    """An instance, with an attribute, of a subclass of each builtin a copy
    class stands for."""
    made = []
    for base, value in (
        (dict, {"k": 1}),
        (list, [1]),
        (tuple, (1,)),
        (set, {1}),
        (frozenset, {1}),
        (str, "s"),
        (bytes, b"b"),
        (int, 7),
        (float, 1.5),
    ):
        instance = type(f"_{base.__name__.title()}", (base,), {})(value)
        instance.field = 1
        made.append(instance)
    return made


@_BOTH_WALKS
def test_hooks_patched_onto_the_container_and_scalar_copy_classes(monkeypatch, native):
    """``__init__`` and ``__getattribute__`` patched onto the copy classes
    of container and scalar subclasses run in the Python walk, which builds
    each copy by calling its class and reads it through attribute access,
    and not at all in the walker, which builds each with its builtin base's
    ``__new__`` and reads its instance dict through the descriptor the
    class had at import."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    calls = set()
    for cls in _COPY_CLASSES:

        def init(copy, *args, cls=cls):
            calls.add((cls.__name__, "__init__"))

        def get(copy, name, cls=cls):
            calls.add((cls.__name__, name))
            return object.__getattribute__(copy, name)

        monkeypatch.setattr(cls, "__init__", init)
        monkeypatch.setattr(cls, "__getattribute__", get)
    before = _fallbacks()
    context.value(_subclass_instances())
    assert _fallbacks() == before
    names = sorted(cls.__name__ for cls in _COPY_CLASSES)
    pure = {(name, hook) for name in names for hook in ("__init__", "__dict__")}
    pure |= {("FakeList", "append"), ("FakeSet", "add")}
    assert calls == (set() if native else pure)


_NEW_SCRIPT = """
from pydumpling import fake_types
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext, values
from tests.test_native_differences import _subclass_instances

class Instance:
    def __init__(self):
        self.field = 1

names = (
    "TruncatedValue", "FakeClass", "FakeDict", "FakeList", "FakeTuple",
    "FakeSet", "FakeFrozenSet", "FakeStr", "FakeBytes", "FakeInt", "FakeFloat",
)
calls = set()
for name in names:
    cls = getattr(fake_types, name)
    base = next(base for base in cls.__mro__ if base.__module__ == "builtins")
    def new(klass, *args, name=name, base=base, **kwargs):
        calls.add(name)
        return object.__new__(klass) if base is object else base.__new__(klass, *args)
    cls.__new__ = new
found = []
for native in (False, True):
    context = SnapshotContext(DumpOptions(max_depth=2))
    if not native:
        context._native_walk = False
    calls.clear()
    before = values._STATICS.fallbacks
    context.value([[{"k": 1}], Instance(), *_subclass_instances()])
    assert values._STATICS.fallbacks == before
    found.append(",".join(sorted(calls)) or "-")
print(*found)
"""


def test_new_patched_onto_the_copy_classes():
    """A ``__new__`` patched onto the copy classes runs in the Python walk
    only. Run in a child interpreter: a ``__new__`` set and then deleted
    leaves the class refusing its constructor's arguments."""
    process = subprocess.run(
        [sys.executable, "-c", _NEW_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.split() == [
        "FakeBytes,FakeClass,FakeDict,FakeFloat,FakeFrozenSet,FakeInt,FakeList,"
        "FakeSet,FakeStr,FakeTuple,TruncatedValue",
        "-",
    ]


@_BOTH_WALKS
def test_a_dict_descriptor_planted_on_fake_class(native):
    """A ``__dict__`` descriptor planted in ``FakeClass``'s class dict after
    import, here one answering a dict of its own, runs each time the Python
    walk reads a ``FakeClass``'s ``__dict__``, so the copy's attributes go
    into the dict it answers; the walker builds the copy through the
    instance-dict descriptor the class had at import, never runs it, and
    keeps the attributes."""
    real = vars(FakeClass.__mro__[1])["__dict__"]
    calls = []

    def elsewhere(copy):
        calls.append("FakeClass.__dict__")
        return {}

    namespace = gc.get_referents(vars(FakeClass))[0]
    namespace["__dict__"] = property(elsewhere)
    FakeClass._planted = None  # resets the attribute cache
    del FakeClass._planted
    try:
        _, captured = _capture(native, DumpOptions(), _Instance())
    finally:
        del namespace["__dict__"]
        FakeClass._planted = None
        del FakeClass._planted
    representation = f"<{_Instance.__module__}._Instance snapshot>"
    assert (calls, real.__get__(captured)) == (
        ([], {"_FakeClass__repr": representation, "field": 1})
        if native
        else (["FakeClass.__dict__"] * 3, {"_FakeClass__repr": representation})
    )


# --- The walk is chosen when the class is created -----------------------------
# Whether a context captures natively is decided when its class is created
# (``ValuesMixin.__init_subclass__``, or the first ``SnapshotContext.__init__``
# of a class created past it). A walk method patched onto its class afterwards,
# or set on the instance, is not seen by a native context: the Python walk runs
# it, the walker its own. The walker bounds strings and fills ``_layouts`` in
# Rust, where the Python walk looks ``_text``, ``_code_text``,
# ``_bounded_bytes`` and ``_layout`` up on the context, so no such hook runs for
# them. The walker reads the context's state with attribute access, once per
# call for each attribute: ``objects``, ``_nodes`` and ``options`` as it starts,
# the rest where the Python walk first reads it. So a state property or an
# attribute hook patched onto the class later runs there, a part of what the
# Python walk runs. The walker decides once per options type whether it reads
# an options object's budgets with attribute access once per call
# (``DumpOptions`` at import); a hook given to ``DumpOptions`` later runs there.


def test_a_walk_method_patched_onto_the_class_later(monkeypatch):
    """A walk method patched onto the context class after it was created
    runs in the Python walk, and not at all for a native context, which the
    walker keeps."""
    found = {}
    for native in (False, True):
        reasons = []
        omit = SnapshotContext._omit

        def recorded(context, reason, value=None, reasons=reasons, omit=omit):
            reasons.append(reason)
            return omit(context, reason, value)

        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=1))
        with monkeypatch.context() as patch:
            patch.setattr(SnapshotContext, "_omit", recorded)
            found[native] = (pickle.dumps(context.value([[1]]), 5), reasons)
        _check_walker(context, native)
    assert found[True][0] == found[False][0]
    assert (found[False][1], found[True][1]) == (["max_depth"], [])


def test_a_walk_method_set_on_an_instance():
    """A walk method set on a context instance runs in the Python walk,
    which looks it up on the instance, and not at all in the walker."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=1))
        reasons = []
        omit = context._omit

        def recorded(reason, value=None, reasons=reasons, omit=omit):
            reasons.append(reason)
            return omit(reason, value)

        context._omit = recorded
        found[native] = (pickle.dumps(context.value([[1]]), 5), reasons)
        _check_walker(context, native)
    assert found[True][0] == found[False][0]
    assert (found[False][1], found[True][1]) == (["max_depth"], [])


class _PureContext(SnapshotContext):
    """A context of the Python walk, whose instance dict holds no flag."""

    _native_walk = False


@pytest.mark.parametrize("how", ["retired", "in python in place"])
def test_a_context_captured_as_a_value_shows_its_walk_flag(monkeypatch, how):
    """A context the walker retired holds ``_native_walk = False`` in its
    instance dict for good, and one whose node the walker runs in the Python
    walk in place holds it until that node is done, so captured as a value
    there it shows the flag, where a context of the Python walk holds none.
    (An unlimited context holds none in either mode; ``test_native_compat``
    pins that.)"""
    shown, left = {}, {}
    for native, cls in ((True, SnapshotContext), (False, _PureContext)):
        context = cls(DumpOptions())
        _check_walker(context, native)
        if how == "retired":
            context.value([1], 1.0)  # a depth that is not an exact int
            copied = SnapshotContext(DumpOptions()).value(context)
        else:
            # While ``FakeClass.__repr__`` is replaced, the walker runs an
            # exact ``FakeClass`` in the Python walk in place.
            held = FakeClass("<held>", {"context": context})
            with monkeypatch.context() as patch:
                patch.setattr(FakeClass, "__repr__", lambda self: "<replaced>")
                copied = vars(context.value([held])[0])["context"]
        shown[native] = vars(copied).get("_native_walk", "unset")
        left[native] = vars(context).get("_native_walk", "unset")
    assert shown == {True: False, False: "unset"}
    if how == "retired":
        assert left == {True: False, False: "unset"}
    else:
        assert left == {True: "unset", False: "unset"}


def test_a_context_copied_inside_a_node_run_in_python(monkeypatch):
    """A copy or a pickle of the context made by application code inside a
    node the walker runs in the Python walk in place carries the instance's
    ``_native_walk = False``, and captures in Python for good; the context
    itself goes back to the walker after that node. A copy made by the
    Python walk's application code holds no flag."""
    found = {}
    for native, cls in ((True, SnapshotContext), (False, _PureContext)):
        context = cls(DumpOptions())
        _check_walker(context, native)
        made = []

        def replaced(fake, context=context, made=made):
            made.append(copy.copy(context))
            made.append(pickle.loads(pickle.dumps(context)))
            return "<replaced>"

        with monkeypatch.context() as patch:
            patch.setattr(FakeClass, "__repr__", replaced)
            context.value([FakeClass("<held>", {"a": 1})])
        assert len(made) == 2
        found[native] = [vars(copied).get("_native_walk", "unset") for copied in made]
        assert "_native_walk" not in vars(context)
    assert found == {True: [False, False], False: ["unset", "unset"]}


class _SwapName(str):
    """A ``__qualname__`` whose ``__radd__``, which naming its class runs,
    rebinds ``_SwapName.context`` to ``_Redacting``."""

    context = None

    def __radd__(self, other):
        _SwapName.context.__class__ = _Redacting
        return str.__add__(other, str(self))


class _Redacting(SnapshotContext):
    def _text(self, value):
        return "<redacted>"


def test_a_class_rebound_inside_a_native_call():
    """Application code the walker runs that rebinds the context's
    ``__class__`` to a subclass overriding a walk method is seen from the
    next call: the rest of the call keeps the stock walk, where the Python
    walk calls the override at once."""
    named = type("Named", (), {"__qualname__": _SwapName("Named")})
    found = {}
    for native, cls in ((True, SnapshotContext), (False, _PureContext)):
        context = _SwapName.context = cls(DumpOptions())
        _check_walker(context, native)
        try:
            first = context.value([named(), "kept"])[1]
        finally:
            _SwapName.context = None
        found[native] = (first, context.value(["next"])[0])
    assert found == {True: ("kept", "<redacted>"), False: ("<redacted>", "<redacted>")}


@pytest.mark.parametrize(
    "attributes",
    [
        *[
            pytest.param(
                lambda log, name=name: {name: _logged_property(name, log)}, id=name
            )
            for name in _STATE_NAMES
        ],
        *[
            pytest.param(
                lambda log, hook=hook: {hook: _logged_hooks(log)[hook]}, id=hook
            )
            for hook in ("__getattribute__", "__setattr__")
        ],
    ],
)
def test_state_access_patched_onto_the_class_later(attributes):
    """A state property or an attribute hook patched onto a context class
    after a first call: the snapshots are the same, and the native context
    runs it where ``value`` or ``_mapping`` touch the state around the
    walker, and where the walker reads it, once per call: a part of what the
    Python walk runs."""
    outcomes = _logged_capture(attributes, "patched")
    assert outcomes[True][0] == outcomes[False][0]
    native, pure = (Counter(outcomes[mode][1]) for mode in (True, False))
    assert native <= pure
    assert pure, "the Python walk never ran it"


class _Hashing:
    """A key that hashes like ``hashed`` and logs each comparison that
    reaches its ``__eq__``, which finds it unequal."""

    def __init__(self, hashed):
        self.hashed = hashed

    def __hash__(self):
        return self.hashed

    def __eq__(self, other):
        _called("planted key __eq__")
        return False


@contextlib.contextmanager
def _planted_in_dump_options():
    """A key other than a str in ``DumpOptions``' class dict, where it
    hashes like ``max_items`` and comes ahead of it in its probe order."""
    namespace = gc.get_referents(vars(DumpOptions))[0]
    planted = _Hashing(hash("max_items"))
    kept = namespace.pop("max_items")
    namespace[planted] = None
    namespace["max_items"] = kept
    # Setting and deleting an attribute resets the attribute caches.
    DumpOptions._planted = None
    del DumpOptions._planted
    try:
        yield
    finally:
        del namespace[planted]
        DumpOptions._planted = None
        del DumpOptions._planted


@pytest.mark.parametrize("hook", ["__getattribute__", "a property", "a planted key"])
def test_hooks_patched_onto_dump_options_later(monkeypatch, hook):
    """A ``__getattribute__`` or a budget's property patched onto
    ``DumpOptions`` after import, or a key planted in its class dict that
    hashes like a budget's name, runs as often or less often in the walker
    than in the Python walk, which runs it at each read of a budget (the
    key's ``__eq__`` at each lookup the attribute cache misses). The walker
    judged ``DumpOptions`` at import and reads the budgets with attribute
    access, once per call. The snapshots are the same."""

    def items(options):
        _called("DumpOptions.max_items")
        return 1

    def get(options, name):
        _called(f"DumpOptions.__getattribute__ {name}")
        return object.__getattribute__(options, name)

    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        context.value(None)
        with contextlib.ExitStack() as stack:
            patch = stack.enter_context(monkeypatch.context())
            if hook == "a property":
                patch.setattr(DumpOptions, "max_items", property(items))
            elif hook == "__getattribute__":
                patch.setattr(DumpOptions, "__getattribute__", get)
            else:
                stack.enter_context(_planted_in_dump_options())
            with _armed() as hooks:
                captured = context.value([[1, 2, 3]])
        found[native] = (pickle.dumps(captured, protocol=5), list(hooks))
        _check_walker(context, native)
    native, pure = (Counter(found[mode][1]) for mode in (True, False))
    assert native <= pure
    assert pure, "the Python walk never ran it"
    assert found[True][0] == found[False][0]


# --- Over-long ints ----------------------------------------------------------
# An int past ``sys.get_int_max_str_digits()`` that an exception message
# formats with ``reprlib`` gives both walks the same message: from 3.13.6 and
# 3.14, whose ``reprlib`` writes such an int as a placeholder with its address,
# equal ints held by different objects give different messages; before, where
# ``reprlib`` raises, the message is unavailable and each formatting counts
# ``exception_format``.


class _AppError(Exception):
    """An exception the walk formats from its arguments with ``reprlib``."""


def _messages(native, held):
    """The messages of exceptions that each hold one of ``held``, directly and
    in a list the walk copies, and the truncation counts, from one context."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
        _check_walker(context, native)
        before = _fallbacks()
        messages = [context.exception(_AppError(big, [big]))["message"] for big in held]
        assert _fallbacks() == before
    return messages, dict(context._truncation)


def test_over_long_int_in_an_exception_message():
    """Both walks write the message ``reprlib`` gives: where it has a
    fallback for an int too long to convert (3.13.6 and 3.14 on), the int's
    address, twice, so equal ints held by different objects give different
    messages; elsewhere none, as ``reprlib`` raises."""
    held = [10**700 for _ in range(2)]
    assert held[0] is not held[1]
    saved = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    try:
        pure = _messages(False, held)
        native = _messages(True, held)
        try:
            reprlib.repr(held[0])
            addressed = True
        except ValueError:
            addressed = False
    finally:
        sys.set_int_max_str_digits(saved)
    assert native == pure
    messages, truncation = native
    if addressed:
        for big, message in zip(held, messages):
            address = f"{id(big):#x}"
            text = f"<int instance with roughly 701 digits (limit at 640) at {address}>"
            assert message == f"({text}, [{text}])"
        assert messages[0] != messages[1]
        assert truncation == {}
    else:
        assert messages == ["<exception message unavailable>"] * 2
        assert truncation == {"exception_format": 4}


# --- Application code between two sync points --------------------------------
# The walker keeps a call's node count in Rust. It writes the count to the
# context's ``_nodes`` before application code it runs where the Python walk
# runs it, and before it reads the context's own instance dict, and reads it
# back after, so a capture through the same context from there counts on from
# the walk's count and the walk from its. A collection or a signal handler runs
# inside a native call only at the walker's signal check, which is synced the
# same way. Application code that runs anywhere else inside a native call is
# not: a finalizer or weakref callback that runs as the walker lets go of the
# last reference to a value, or the ``__eq__`` of an object planted in the
# context's own ``_namespaces`` (or of a key planted in ``objects``) that the
# walker's lookup meets. A capture through the same context from there starts
# from the count of the last sync, and the walk then writes its own count over
# the one that capture left, so that capture's nodes count toward no budget.
# Such code runs as often as in the Python walk; only the count differs.


class _Tally(int):
    """A truncation tally whose ``+`` runs ``_Tally.hook`` first."""

    hook = None

    def __add__(self, other):
        _Tally.hook()
        return int(self) + other


def _finalized_inside(native):
    """Capture a namespace, two entries at most, whose long string's tally
    drops its third entry, an object whose finalizer captures three nodes
    through the context once the walk lets the selection go. Returns the
    ``_nodes`` the finalizer saw before and after its capture, and the
    context's count after the call."""
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_string_bytes=8, max_items=2))
    _check_walker(context, native)
    seen = []

    class Finalized:
        def __del__(self):
            seen.append(context._nodes)
            context.value([1.5, 2.5])
            seen.append(context._nodes)

    namespace = {"text": "x" * 100, "other": 1, "finalized": Finalized()}
    _Tally.hook = lambda: namespace.pop("finalized", None) and None
    context._truncation["max_string_bytes"] = _Tally(0)
    before = _fallbacks()
    try:
        context._mapping(namespace)
    finally:
        _Tally.hook = None
    assert _fallbacks() == before
    return seen, context._nodes


def test_a_finalizer_capturing_inside_a_native_call():
    """The Python walk lets the selection go as ``_mapping`` returns, and
    the finalizer sees the call's whole count and adds its three nodes to
    it. The walker lets it go inside the call, after the tally's sync: the
    finalizer sees the count of that sync, and the walk's count replaces
    what its capture added."""
    (pure, pure_nodes), (native, native_nodes) = (
        _finalized_inside(native) for native in (False, True)
    )
    walked = pure_nodes - 3
    assert pure == [walked, walked + 3]
    assert native_nodes == walked
    assert native == [native[0], native[0] + 3]
    assert native[0] < walked


class _Planted:
    """An object that hashes like ``hashed`` and records the context's node
    count at each comparison that reaches its ``__eq__``, which finds it
    unequal."""

    def __init__(self, hashed, context):
        self.hashed, self.context, self.seen = hashed, context, []

    def __hash__(self):
        return self.hashed

    def __eq__(self, other):
        self.seen.append(self.context._nodes)
        return False


def test_an_object_planted_in_namespaces():
    """An element of ``_namespaces`` that hashes like the id of a dict the
    call captures is compared with that id as often by the walker as by the
    Python walk. It sees the count of the walker's last sync, the list
    ``value`` counted before the walker's entry, where the Python walk's
    comparison sees the count it reached: the list and the dict. How often
    one lookup meets it depends on the id's bits, so each lookup's count is
    measured first."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        _check_walker(context, native)
        context.value([1])
        start = context._nodes
        source = {"a": 1}
        planted = _Planted(hash(id(source)), context)
        context._namespaces.add(planted)
        assert id(source) not in context._namespaces
        per_lookup = len(planted.seen)
        assert per_lookup >= 1
        planted.seen.clear()
        context.value([source])
        assert len(planted.seen) == per_lookup
        found[native] = {count - start for count in planted.seen}
    assert found == {False: {2}, True: {1}}


# --- CPython's C recursion limit ---------------------------------------------
# The contract excepts CPython's C recursion limit. The Python walk calls its
# methods from Python, at no C level. The walker bounds strings and names
# types in Rust, and reaches application code only through the C entry points
# the Python walk goes through at that step, but on 3.12 and 3.13, where
# CPython counts C levels, one level deeper, as the call into the walker is a
# C call (two for a tally that is not an exact int, added through
# ``operator.iadd``). The Python walk's ``_mapping`` resumes generators that
# CPython counts too, which outweighs that wherever the value holds a dict or
# an object; in a value that holds neither, a non-int tally overflows one
# level sooner than in the Python walk. A node the walker runs in the Python
# walk (an exact ``FakeClass`` once ``FakeClass.__repr__`` was replaced, or
# one below the walker's native stack budget) runs ``ValuesMixin.value`` from
# C: the call into the walker and the call back into Python are two levels
# the Python walk does not spend, so beneath nested comparisons that leave
# fewer, that node raises ``RecursionError`` where the Python walk captures
# it. From 3.14, where CPython measures the C stack, the walker's budget
# sends a call made near the end of the stack to the Python walk from its
# start, and the limit comes no sooner. How far the limit moves on 3.12 and
# 3.13 depends on the version and the value, so only the direction is pinned.

_C_RECURSION_SCRIPT = """
import pickle, sys
from pydumpling.fake_types import FakeClass, TruncatedValue
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext, values

form = sys.argv[1]


class QualName(str):
    def __radd__(self, other):
        return str.__add__(other, str(self))


Named = type("Named", (), {})
if form == "application":
    # Naming the class runs the ``__radd__`` of its ``__qualname__``.
    Named.__qualname__ = QualName("Outer.Named")
value = [{"a": "b"}, Named()]
if form == "python":
    # The walker runs an exact ``FakeClass`` in the Python walk once
    # ``FakeClass.__repr__`` is replaced.
    value[1] = SnapshotContext(DumpOptions()).value(Named())
    assert type(value[1]) is FakeClass
    FakeClass.__repr__ = lambda self: "<replaced>"
# A root namespace whose string is too long to keep whole.
namespace = {"a": "x" * 5000}
outcome = []


class Probe:
    # Compared at the bottom of two nested lists, each level of which holds
    # one more C call of the comparison.
    __hash__ = None

    def __eq__(self, other):
        try:
            if form == "bounded":
                outcome.append(context._mapping(namespace))
            else:
                outcome.append(context.value(value))
        except RecursionError:
            outcome.append("RecursionError")
        return True


left, right = [[Probe()]], [[Probe()]]


def capture_beneath(native, levels):
    global context
    context = SnapshotContext(DumpOptions())
    if not native:
        context._native_walk = False
    while len(left) <= levels:
        left.append([left[-1]])
        right.append([right[-1]])
    outcome.clear()
    try:
        left[levels] == right[levels]
    except RecursionError:
        return "overflowed"
    # Pickled up here: pickling a placeholder runs Python code, which would
    # overflow beneath the nesting.
    captured = outcome[0]
    return captured if isinstance(captured, str) else pickle.dumps(captured, 4)


def kind(outcome):
    if isinstance(outcome, str):
        return outcome
    captured = pickle.loads(outcome)
    if isinstance(captured, dict) and isinstance(captured["a"], TruncatedValue):
        return captured["a"].reason
    return "captured"


whole = capture_beneath(False, 0)
assert capture_beneath(True, 0) == whole
# The deepest nesting beneath which the Python walk captures the whole value.
low, high = 0, 1
while capture_beneath(False, high) == whole:
    low, high = high, 2 * high
while high - low > 1:
    middle = (low + high) // 2
    if capture_beneath(False, middle) == whole:
        low = middle
    else:
        high = middle
before = values._STATICS.fallbacks
there = capture_beneath(True, low)
sent = values._STATICS.fallbacks - before
shallower = capture_beneath(True, low // 2)
print("whole" if there == whole else kind(there), shallower == whole, sent)
"""


@pytest.mark.parametrize("form", ["bounded", "application", "python"])
def test_the_c_recursion_limit_through_the_walker(form):
    """Beneath nested comparisons at the default recursion limit, at the
    deepest nesting where the Python walk alone captures the whole value,
    the walker captures it whole too, itself, for a root namespace whose
    string it bounds and for a value whose capture runs application code
    (the ``__radd__`` of a ``__qualname__``). A value holding an exact
    ``FakeClass`` after ``FakeClass.__repr__`` was replaced, which node the
    walker runs in the Python walk, raises ``RecursionError`` there on 3.12
    and 3.13; from 3.14 it is captured whole, as is every form, by a call
    the walker's stack budget sends to the Python walk. At half that depth
    every form is captured alike. The deepest nesting is found anew in each
    child interpreter."""
    process = subprocess.run(
        [sys.executable, "-c", _C_RECURSION_SCRIPT, form],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    there, shallower, sent = process.stdout.split()
    if sys.version_info >= (3, 14):
        assert (there, shallower) == ("whole", "True")
        assert int(sent) >= 1
    elif form == "python":
        assert (there, shallower) == ("RecursionError", "True")
        assert int(sent) >= 1
    else:
        assert (there, shallower, sent) == ("whole", "True", "0")


_DEEPER_SCRIPT = """
import pickle
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext, values

value = {"a": "b"}
outcome = []

class Probe:
    __hash__ = None
    def __eq__(self, other):
        try:
            outcome.append(pickle.dumps(context.value(value), 4))
        except RecursionError:
            outcome.append("RecursionError")
        return True

left, right = [[Probe()]], [[Probe()]]

def capture_beneath(native, levels):
    global context
    context = SnapshotContext(DumpOptions())
    if not native:
        context._native_walk = False
    while len(left) <= levels:
        left.append([left[-1]])
        right.append([right[-1]])
    outcome.clear()
    try:
        left[levels] == right[levels]
    except RecursionError:
        return "overflowed"
    return outcome[0]

whole = capture_beneath(False, 0)
low, high = 0, 1
while capture_beneath(False, high) == whole:
    low, high = high, 2 * high
while high - low > 1:
    middle = (low + high) // 2
    if capture_beneath(False, middle) == whole:
        low = middle
    else:
        high = middle
python = capture_beneath(False, low + 1)
before = values._STATICS.fallbacks
native = capture_beneath(True, low + 1)
print(
    python if isinstance(python, str) else "captured",
    "whole" if native == whole else native,
    values._STATICS.fallbacks - before,
)
"""


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="from 3.14 CPython measures the C stack"
)
def test_the_c_recursion_limit_can_come_later_through_the_walker():
    """On 3.12 and 3.13, beneath nested comparisons at the default limit, a
    call the walker completes itself on a dict captures it whole one level
    below the deepest nesting where the Python walk does, whose ``_mapping``
    resumes generators that CPython counts as C levels."""
    process = subprocess.run(
        [sys.executable, "-c", _DEEPER_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.split() == ["RecursionError", "whole", "0"]


# --- Audit events --------------------------------------------------------------
# An audit hook sees fewer events from a native call. The Python walk raises
# ``builtins.id`` at each ``id()``, ``_layout``'s included; the walker reads
# ids in Rust. A native call raises only the event of ``value``'s own memo
# lookup, which runs before the walker. An audit hook cannot be removed, so the
# counts are taken in a child.

_AUDIT_COUNT_SCRIPT = """
import collections, json, sys
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot import values as walk

events = []
armed = [False]
sys.addaudithook(lambda event, args: armed[0] and events.append(event))

class Items(list):
    pass

class Text(str):
    pass

values = {
    "plain": [1, "two", (3.0,), {"k": [None]}],
    "subclasses": [Items([1]), Text("x"), {"k": Items()}],
}
outcome = {}
for native in (True, False):
    for name, value in values.items():
        context = SnapshotContext(DumpOptions())
        if not native:
            context._native_walk = False
        before = walk._STATICS.fallbacks
        events.clear()
        armed[0] = True
        context.value(value)
        armed[0] = False
        outcome[name + (" native" if native else " python")] = [
            dict(collections.Counter(events)),
            walk._STATICS.fallbacks - before,
        ]
print(json.dumps(outcome))
"""


def test_audit_events_of_a_capture():
    """A native call raises the one ``builtins.id`` event of ``value``'s
    memo lookup, where the Python walk raises one per ``id()``."""
    process = subprocess.run(
        [sys.executable, "-c", _AUDIT_COUNT_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {
        "plain native": [{"builtins.id": 1}, 0],
        "plain python": [{"builtins.id": 47}, 0],
        "subclasses native": [{"builtins.id": 1}, 0],
        "subclasses python": [{"builtins.id": 50}, 0],
    }
