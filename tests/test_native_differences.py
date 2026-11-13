"""The accepted differences between the native walker and the Python walk
that ``rust/DESIGN.md`` lists, and that no other module pins, each against the
Python walk.  A change to either walk here changes that entry too.  Unless a
test says otherwise, both walks behave the same on CPython 3.12 to 3.15.
"""

import contextlib
import copy
import copyreg
import gc
import inspect
import json
import os
import reprlib
import subprocess
import sys
from collections import Counter, OrderedDict
from unittest import mock

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
    _armed,
    _called,
    _check_walker,
    _child_environment,
    _collecting,
    _inside_the_walk,
    _mode,
    _traced,
)

# --- A shallow copy shares the walker ----------------------------------------
# A shallow copy of a native context, made with ``copy.copy``, through
# ``copyreg`` or by hand from the instance dict, holds the original's walker,
# as it shares ``objects`` and ``_sources``; a pickle or deep copy holds
# ``None`` in its place and captures in Python. The walker captures one call
# at a time, so a capture through the copy while the original's native call
# runs raises ``RuntimeError`` where the Python walk captures. Both walks
# behave so on every supported version.


_UNTRACED = pytest.mark.skipif(
    _traced(), reason="a tracer hands every call back to the Python walk"
)


def _restore(cls, state):
    """An instance of ``cls`` whose instance dict is updated from ``state``."""
    instance = object.__new__(cls)
    instance.__dict__.update(state)
    return instance


def _reduce(context):
    """Reduce a context from a copy of its instance dict, bypassing its class."""
    return _restore, (type(context), dict(vars(context)))


@contextlib.contextmanager
def _registered():
    """Copy contexts through ``_reduce`` from ``copyreg``."""
    copyreg.pickle(SnapshotContext, _reduce)
    try:
        yield
    finally:
        del copyreg.dispatch_table[SnapshotContext]


def _shallow_copy(context, how):
    """A shallow copy of ``context`` sharing the values of its instance dict."""
    if how == "copy":
        return copy.copy(context)
    if how == "copyreg":
        with _registered():
            return copy.copy(context)
    return _restore(type(context), dict(vars(context)))


def _walking(context):
    """Whether a walk on ``context`` runs: its native walker is busy, or a
    frame of the Python walk is on the stack."""
    if _inside_the_walk(context):
        return True
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename != values_module.__file__:
        frame = frame.f_back
    return frame is not None


@_UNTRACED
@pytest.mark.parametrize("how", ["copy", "copyreg", "by_hand"])
def test_a_shallow_copy_shares_the_walker(how):
    """A shallow copy holds the original's walker natively, and no walker
    in Python; it shares ``objects`` either way.  A capture through it from a collection
    callback while the original's native call runs raises RuntimeError;
    while the original's Python walk runs it captures, and outside a walk
    it captures in both."""
    graph = [[i, str(i)] for i in range(20000)]
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
        _check_walker(context, native)
        twin = _shallow_copy(context, how)
        assert twin.objects is context.objects
        if native:
            assert vars(twin)["_walker"] is vars(context)["_walker"]
        else:
            assert "_walker" not in vars(twin)
        outcomes = []

        def callback(phase, info, context=context, twin=twin, outcomes=outcomes):
            if phase != "start" or len(outcomes) >= 50:
                return
            inside = _walking(context)
            try:
                outcomes.append((inside, twin.value(["inner", 2.5])))
            except RuntimeError as error:
                outcomes.append((inside, error))

        with _collecting(callback):
            context.value(graph)
        inside = [outcome for within, outcome in outcomes if within]
        outside = [outcome for within, outcome in outcomes if not within]
        assert inside, "no collection ran during the walk"
        assert all(outcome == ["inner", 2.5] for outcome in outside)
        if native:
            assert context._walker.handbacks == 0
            assert all(
                isinstance(error, RuntimeError) and "already capturing" in str(error)
                for error in inside
            )
        else:
            assert all(outcome == ["inner", 2.5] for outcome in inside)


# --- A class changed during a native call ------------------------------------
# A class changed inside a native call is seen by the walker from the next
# call.
#
# A collection runs inside a native call only at the walker's signal check,
# once per 4096 nodes, and its callbacks may change a class there. Here a
# callback rebases a class whose ``__dict__`` is a property onto one with the
# real instance-dict descriptor, so its instances' attributes become readable.
# The Python walk rereads a type's MRO for every value: the instances it
# captures after the collection carry their attributes, and only those before
# it count as ``unsupported_type``. The walker reads each type's facts once per
# call: every instance in that call is captured without attributes and counted,
# and the next call reads the attributes. The same holds on CPython 3.12 to
# 3.15.


_COUNT = 6000


def _rebased_mid_call(native):
    """Capture ``_COUNT`` instances while a collection past node 4096 rebases
    their class, then three new instances in a second call.  Returns each
    copy's ``x`` attribute (None where it has none) for both calls, the
    ``unsupported_type`` tally after the first, and whether the class changed
    during the first call."""

    class WithDict:
        pass

    class Planted:
        @property
        def __dict__(self):
            return {}

    class Child(Planted):
        pass

    def instances(count):
        made = [Child() for _ in range(count)]
        for index, instance in enumerate(made):
            instance.x = index
        return made

    first = instances(_COUNT)
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
    _check_walker(context, native)
    rebased = []

    def rebase(phase, info):
        if phase != "start" or rebased:
            return
        if _inside_the_walk(context) if native else context._nodes >= 4096:
            Child.__bases__ = (WithDict,)
            rebased.append(True)

    with _collecting(rebase):
        captured = context.value(first)
    tally = context._truncation["unsupported_type"]
    if native:
        assert context._walker.handbacks == 0
    later = context.value(instances(3))
    return (
        [vars(copy).get("x") for copy in captured],
        tally,
        [vars(copy).get("x") for copy in later],
        bool(rebased),
    )


def test_class_rebased_by_a_collection_mid_call():
    """The Python walk sees the new base from the next value; the walker
    sees it from the next call."""
    xs, tally, later, rebased = _rebased_mid_call(False)
    assert rebased, "no collection ran during the Python walk"
    hidden = xs.count(None)
    assert 0 < hidden < _COUNT
    assert xs == [None] * hidden + list(range(hidden, _COUNT))
    assert tally == hidden
    assert later == [0, 1, 2]

    xs, tally, later, rebased = _rebased_mid_call(True)
    assert rebased, "no collection ran during the native walk"
    assert xs == [None] * _COUNT
    assert tally == _COUNT
    assert later == [0, 1, 2]


# --- A context captured as a value -------------------------------------------
# A context captured as a value shows the native walker.
#
# A native context keeps its walker in its instance dict as ``_walker`` and
# writes ``_nodes`` only when a call commits. So a native context captured as a
# value, by itself (through ``value`` or a frame that holds it) or by another
# context, has a ``_walker`` attribute, an unsupported-type ``FakeClass``, and,
# captured by itself, the node count as the call began. The Python walk's copy
# has no ``_walker`` and the count the walk had reached when it selected the
# instance dict's entries: two nodes past the call's start through ``value``,
# three through a frame namespace. Both show every copy made so far, the
# capturing call's own included, and, captured by another context, the count
# its last call left. A self capture handed back to the Python walk, as one is
# under a profiler, shows ``None`` as ``_walker``, which the context holds
# while the Python walk takes the call, and otherwise what the Python walk
# shows.


_WALKER = {"_FakeClass__repr": "<pydumpling._native.Walker snapshot>"}


def _held_by_a_frame(context):
    """Capture this frame, whose one variable is ``context``, through it."""
    head = context.from_frame(sys._getframe())
    return head.tb_frame.f_locals["context"]


def _captured(native, by):
    """A context, after one capture, and the copy of it that capturing it
    ``by`` some route makes, with its earlier originals' ids and its node
    count just before."""
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_frames=1))
        other = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    _check_walker(other, native)
    data = [1, "a"]
    context.value(data)
    nodes = context._nodes
    if by == "itself":
        copy = context.value(context)
    elif by == "its own frame":
        copy = _held_by_a_frame(context)
    else:
        copy = other.value(context)
    return context, vars(copy), [id(data), id(data[0]), id(data[1])], nodes


def _assert_memo(context, attrs, earlier, own):
    """The copy holds every copy made so far, the capturing call's too."""
    objects, sources = list(attrs["objects"]), list(attrs["_sources"])
    if own:
        assert objects[:3] == sources[:3] == earlier
        assert {id(context), id(vars(context))} <= set(objects)
    else:
        assert objects == sources == earlier


_ROUTES = pytest.mark.parametrize(
    "by, counted", [("itself", 2), ("its own frame", 3), ("another context", 0)]
)


@_ROUTES
def test_the_native_copy_shows_the_walker_and_the_count_as_the_call_began(by, counted):
    if _traced():
        pytest.skip("a tracer hands the capture back to the Python walk")
    context, attrs, earlier, nodes = _captured(True, by)
    assert "_walker" in attrs and vars(attrs["_walker"]) == _WALKER
    _assert_memo(context, attrs, earlier, bool(counted))
    assert attrs["_nodes"] == nodes
    assert context._walker.handbacks == 0


@_ROUTES
def test_the_python_copy_shows_the_running_count(by, counted):
    context, attrs, earlier, nodes = _captured(False, by)
    assert "_walker" not in attrs
    _assert_memo(context, attrs, earlier, bool(counted))
    assert attrs["_nodes"] == nodes + counted


@pytest.mark.parametrize("by, counted", [("itself", 2), ("its own frame", 3)])
def test_a_handed_back_self_capture_shows_no_walker(by, counted):
    previous = sys.getprofile()
    sys.setprofile(lambda frame, event, arg: None)
    try:
        context, attrs, earlier, nodes = _captured(True, by)
    finally:
        sys.setprofile(previous)
    assert context._walker.handbacks >= 1
    assert "_walker" in attrs and attrs["_walker"] is None
    _assert_memo(context, attrs, earlier, True)
    assert attrs["_nodes"] == nodes + counted


# --- Helpers patched after import --------------------------------------------
# Helpers patched over the ``values`` module, onto ``collections.Counter``, or
# onto the classes of the copies.
#
# The native walker's table is built when ``pydumpling.snapshot.values`` is
# imported, so a module global patched afterwards is seen by no walker, not
# even one built after the patch: a ``TruncatedValue`` or ``FakeClass``
# subclass set as the global is the class of the Python walk's placeholders and
# copies and never of the walker's, and a patched ``_plain_key`` runs in the
# Python walk alone, as the walker never calls it. A call the walker hands back
# runs in the Python walk and sees the patch, as an ``OrderedDict`` does under
# a tracer. A walk method patched onto the context class is seen: the walker
# leaves that context to the Python walk, which runs it. The walker writes the
# truncation tally with dict calls, so a ``__missing__`` patched onto
# ``Counter`` runs once for each reason the Python walk first counts and never
# for the walker's counts, which come out the same. Hooks patched onto
# ``TruncatedValue`` or ``FakeClass`` to build a copy (``__new__``,
# ``__init__``, ``__setattr__``, ``__getattribute__``, a ``__dict__``
# descriptor) run in the Python walk alone: the walker builds both through the
# instance-dict descriptor each had at import, so a ``__dict__`` descriptor
# planted on ``FakeClass`` neither runs nor decides where the copy's attributes
# go. (Hooks that read a captured ``FakeClass`` or the options, a ``__repr__``
# or ``__getattribute__``, the walker leaves to the Python walk, which runs
# them; ``test_native_parity`` pins that.)


_BOTH_WALKS = pytest.mark.parametrize("native", [False, True], ids=["python", "native"])


class _Instance:
    def __init__(self):
        self.field = 1


def _capture(native, options, value):
    """Capture ``value`` on a fresh context of the given walk, which must
    commit without handing the call back."""
    with _mode(native):
        context = SnapshotContext(options)
        _check_walker(context, native)
        captured = context.value(value)
    if native:
        assert context._walker.handbacks == 0
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
    if native and _traced():
        pytest.skip("under a tracer the walker hands an OrderedDict back")
    calls = []
    plain_key = values_module._plain_key

    def recorded(key):
        calls.append(key)
        return plain_key(key)

    monkeypatch.setattr(values_module, "_plain_key", recorded)
    _capture(native, DumpOptions(), OrderedDict(a=1, b=2))
    assert calls == ([] if native else ["a", "b"])


@_BOTH_WALKS
def test_patched_walk_method(monkeypatch, native):
    """A walk method patched onto the context class runs in both walks, and
    the walker leaves the context to the Python walk."""
    reasons = []
    omit = SnapshotContext._omit

    def recorded(context, reason, value=None):
        reasons.append(reason)
        return omit(context, reason, value)

    with _mode(native):
        context = SnapshotContext(DumpOptions(max_depth=1))
        _check_walker(context, native)
        monkeypatch.setattr(SnapshotContext, "_omit", recorded)
        context.value([[1]])
    assert reasons == ["max_depth"]
    _check_walker(context, False)


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
    context.value([[{"k": 1}], _Instance()])  # the dict is too deep
    if native:
        assert context._walker.handbacks == 0
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
    context.value(_subclass_instances())
    if native:
        assert context._walker.handbacks == 0
    names = sorted(cls.__name__ for cls in _COPY_CLASSES)
    pure = {(name, hook) for name in names for hook in ("__init__", "__dict__")}
    pure |= {("FakeList", "append"), ("FakeSet", "add")}
    assert calls == (set() if native else pure)


_NEW_SCRIPT = """
import os
from pydumpling import fake_types
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
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
    os.environ["PYDUMPLING_NATIVE"] = "1" if native else "0"
    context = SnapshotContext(DumpOptions(max_depth=2))
    assert (vars(context).get("_walker") is not None) is native
    calls.clear()
    context.value([[{"k": 1}], Instance(), *_subclass_instances()])
    assert not native or context._walker.handbacks == 0
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


# --- Introspection of the entry points ---------------------------------------
# Introspection of the native walker's entry points and of the functions they
# wrap.
#
# Once ``pydumpling._native`` is imported, ``SnapshotContext.value`` and
# ``SnapshotContext._mapping`` are ``Dispatch`` descriptors for every context,
# whatever ``PYDUMPLING_NATIVE`` says when the context is built; only
# ``PYDUMPLING_NATIVE=0`` at import leaves the plain functions the Python walk
# has. A descriptor is no function, its ``__module__`` is
# ``pydumpling._native``, ``inspect.getsourcefile`` raises ``TypeError``,
# ``inspect.getsource`` and ``inspect.signature`` follow ``__wrapped__`` to the
# function, and ``mock.create_autospec`` sees ``(*args, **kwargs)``, on the
# attribute and on an autospec of the class. A native context's bound method
# shows its descriptor the same way and forwards the descriptor's
# ``__wrapped__``, the unbound function, which a pure context's bound method
# lacks. Called directly on a native context whose walker still holds its
# copies, that function reads ``objects`` without them: it copies the outermost
# value afresh, counts one more node and puts its copy in place of the walker's
# for later calls, while the values nested in it reach the walker again through
# ``self.value`` and get the walker's copies. Once a copy of the context has
# made the walker write its copies, the function finds them, as it always does
# on a pure context.


_NAMES = ("value", "_mapping")


_FACTS_SCRIPT = """
import inspect, json
from unittest import mock
from pydumpling.snapshot import SnapshotContext

def source_file(attribute):
    try:
        return inspect.getsourcefile(attribute)
    except TypeError as error:
        return type(error).__name__

spec = mock.create_autospec(SnapshotContext)
facts = {}
for name in ("value", "_mapping"):
    attribute = getattr(SnapshotContext, name)
    facts[name] = {
        "type": type(attribute).__name__,
        "isfunction": inspect.isfunction(attribute),
        "module": attribute.__module__,
        "sourcefile": source_file(attribute),
        "source": inspect.getsource(attribute),
        "signature": str(inspect.signature(attribute)),
        "autospec": str(inspect.signature(mock.create_autospec(attribute))),
        "class autospec": str(inspect.signature(getattr(spec, name))),
    }
print(json.dumps(facts))
"""


def _class_attribute_facts(native):
    """What introspection reports on the class attributes of a child
    interpreter that imports pydumpling with or without the walker."""
    environment = dict(_child_environment(), PYDUMPLING_NATIVE="1" if native else "0")
    process = subprocess.run(
        [sys.executable, "-B", "-c", _FACTS_SCRIPT],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def test_class_attribute_introspection():
    """The class attribute introspects as the function under the Python walk
    and as a descriptor around it under the native walker."""
    python = _class_attribute_facts(native=False)
    native = _class_attribute_facts(native=True)
    for name in _NAMES:
        function = vars(values_module.ValuesMixin)[name].__wrapped__
        signature = inspect.signature(function)
        unbound = signature.replace(parameters=list(signature.parameters.values())[1:])
        assert python[name] == {
            "type": "function",
            "isfunction": True,
            "module": "pydumpling.snapshot.values",
            "sourcefile": values_module.__file__,
            "source": inspect.getsource(function),
            "signature": str(signature),
            "autospec": str(signature),
            "class autospec": str(unbound),
        }
        assert native[name] == dict(
            python[name],
            type="Dispatch",
            isfunction=False,
            module="pydumpling._native",
            sourcefile="TypeError",
            autospec="(*args, **kwargs)",
            **{"class autospec": "(*args, **kwargs)"},
        )


def test_class_attribute_whatever_the_walk():
    """A context built for either walk finds the descriptor on its class."""
    for name in _NAMES:
        descriptor = vars(values_module.ValuesMixin)[name]
        assert type(descriptor) is values_module._native.Dispatch
        for native in (True, False):
            with _mode(native):
                context = SnapshotContext(DumpOptions())
            _check_walker(context, native)
            assert getattr(type(context), name) is descriptor
            assert not inspect.isfunction(getattr(type(context), name))


def test_bound_method_introspection():
    """A native context's bound method wraps the descriptor and forwards its
    ``__wrapped__``; a pure context's wraps the function, which has none."""
    for name in _NAMES:
        descriptor = vars(values_module.ValuesMixin)[name]
        function = descriptor.__wrapped__
        for native in (True, False):
            with _mode(native):
                context = SnapshotContext(DumpOptions())
            _check_walker(context, native)
            bound = getattr(context, name)
            signature = str(inspect.signature(bound))
            assert str(inspect.signature(mock.create_autospec(bound))) == signature
            if native:
                assert bound.__func__ is descriptor
                assert bound.__module__ == "pydumpling._native"
                with pytest.raises(TypeError):
                    inspect.getsourcefile(bound)
                assert bound.__wrapped__ is function
                assert inspect.unwrap(bound) is function
            else:
                assert bound.__func__ is function
                assert bound.__module__ == "pydumpling.snapshot.values"
                assert inspect.getsourcefile(bound) == values_module.__file__
                assert not hasattr(bound, "__wrapped__")
                assert inspect.unwrap(bound) is bound


# --- The frame audit event on a deep stack -----------------------------------
# At the start of each call it runs, the native walker calls
# ``sys._getframe(min((limit - 97) // 2, 1024))``, which raises a
# ``sys._getframe`` audit event only when it succeeds. From a stack of more
# visible frames than that, counting the frame that makes the call (201, 451
# and 1024 at recursion limits of 500, 1000 and 4000), each native ``value`` or
# ``_mapping`` call raises exactly one event, including a call whose headroom
# probe then recurses into ``values._headroom``; from that many frames or fewer
# it raises none. A call the walker hands to Python before it starts, such as
# one passing the value by keyword, raises none, and the Python walk raises
# none at either depth. Audit hooks cannot be removed, so each limit runs in a
# child interpreter.


_SCRIPT = """
import json, os, sys
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

limit, threshold = int(sys.argv[1]), int(sys.argv[2])
sys.setrecursionlimit(limit)
events, armed = [], [False]

def audit(event, arguments):
    if armed[0] and event == "sys._getframe":
        events.append(event)

sys.addaudithook(audit)

def stack():
    frame, count = sys._getframe(1), 0
    while frame is not None:
        count, frame = count + 1, frame.f_back
    return count

def call_at(frames, here, method, *args, **kwargs):
    if here < frames:
        return call_at(frames, here + 1, method, *args, **kwargs)
    assert stack() == frames, (stack(), frames)
    events.clear()
    armed[0] = True
    try:
        method(*args, **kwargs)
    finally:
        armed[0] = False
    return len(events)

deep = None
for _ in range(150):
    deep = [deep]
outcomes = {}
for native in (True, False):
    os.environ["PYDUMPLING_NATIVE"] = "1" if native else "0"
    for position, frames in (("at", threshold), ("deeper", threshold + 1)):
        context = SnapshotContext(DumpOptions(max_depth=None, max_nodes=None))
        walker = vars(context).get("_walker")
        events_per_call = [
            call_at(frames, stack() + 1, context.value, {"key": [1, "two", (3.0,)]}),
            call_at(frames, stack() + 1, context._mapping, {"deep": deep}),
            call_at(frames, stack() + 1, context.value, value=[4]),
        ]
        handbacks = None if walker is None else walker.handbacks
        mode = "native" if native else "python"
        outcomes.setdefault(mode, {})[position] = [events_per_call, handbacks]
print(json.dumps(outcomes))
"""


@pytest.mark.parametrize(
    ("limit", "threshold"), [(500, 201), (1000, 451), (4000, 1024)]
)
def test_deep_stack_raises_one_frame_audit_event_per_native_call(limit, threshold):
    """Only the native walk, and only from beyond the threshold, raises it."""
    process = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(limit), str(threshold)],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {
        "native": {"at": [[0, 0, 0], 1], "deeper": [[1, 1, 0], 1]},
        "python": {"at": [[0, 0, 0], None], "deeper": [[0, 0, 0], None]},
    }


# --- Keyword names of a str subclass, and over-long ints ---------------------
# Two documented behaviours around the native walker, pinned against the Python
# walk. A keyword name that is a str subclass makes the walker hand the call to
# the Python walk after the name's own ``__hash__`` has run. Every call that
# reaches the ``Dispatch`` descriptor hashes each name once, while PyO3
# collects ``**kwargs``: a call on a native context, and a call through the
# class, ``SnapshotContext.value(ctx, ...)``, on any context once
# ``pydumpling._native`` is imported. A call through a native context's bound
# method, such as ``ctx.value(...)``, hashes each name once more, while CPython
# builds the keyword dict that ``Dispatch``, which has no vectorcall, takes. A
# receiver passed through the class as ``self=`` is looked up and removed by
# the descriptor, one ``__eq__`` more than the parameter match it replaces;
# every other name is compared with the parameters alike. With
# ``PYDUMPLING_NATIVE=0`` at import no name is hashed. An int past
# ``sys.get_int_max_str_digits()`` that an exception message formats with
# ``reprlib`` gives both walks the same message: from 3.13.6 and 3.14, whose
# ``reprlib`` writes such an int as a placeholder with its address, equal ints
# held by different objects give different messages; before, where ``reprlib``
# raises, the message is unavailable and each formatting counts
# ``exception_format``.


_HASH = "_Name.__hash__"


_EQ = "_Name.__eq__"


# Hooks run per call form: on a native context, on a pure one in this process,
# and on a pure one in a process that never imported the extension.
_KEYWORD_HOOKS = {
    "bound": ([_HASH] * 2 + [_EQ] * 3, [_EQ] * 3, [_EQ] * 3),
    "class": ([_HASH] + [_EQ] * 3, [_HASH] + [_EQ] * 3, [_EQ] * 3),
    "mapping": ([_HASH] * 2 + [_EQ] * 5, [_EQ] * 5, [_EQ] * 5),
    "receiver": ([_HASH] + [_EQ] * 2, [_HASH] + [_EQ] * 2, [_EQ]),
}


class _Name(str):
    """A keyword name whose hash and comparison are application code."""

    def __hash__(self):
        _called(_HASH)
        return str.__hash__(self)

    def __eq__(self, other):
        _called(_EQ)
        return str.__eq__(self, other)


def _keyword_call(form, context, data):
    """The call ``form`` names, with a ``_Name`` keyword, built before any
    hook is collected, as building the keyword dict hashes the name."""
    if form == "bound":
        names = {_Name("depth"): 1}
        return lambda: context.value(data, **names)
    if form == "class":
        names = {_Name("depth"): 1}
        return lambda: SnapshotContext.value(context, data, **names)
    if form == "mapping":
        names = {_Name("counted"): True}
        return lambda: context._mapping({"key": data}, **names)
    names = {_Name("self"): context, "value": data}
    return lambda: SnapshotContext.value(**names)


@pytest.mark.parametrize("form", list(_KEYWORD_HOOKS))
def test_str_subclass_keyword_name(form):
    """The walker hands the call back with the result the Python walk gives;
    the hooks run differ from a pure context's only through a native
    context's bound method, which hashes each name twice."""
    outcomes = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
            _check_walker(context, native)
            call = _keyword_call(form, context, [1, "two"])
            with _armed() as hooks:
                result = call()
            outcomes[native] = (result, list(hooks))
            if native:
                assert context._walker.handbacks == 1
    native_hooks, pure_hooks, _ = _KEYWORD_HOOKS[form]
    assert outcomes[True][0] == outcomes[False][0]
    assert outcomes[False][1] == pure_hooks
    assert outcomes[True][1] == native_hooks


_UNLOADED_SCRIPT = """
import json
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot import values

hooks = []

class Name(str):
    def __hash__(self):
        hooks.append("_Name.__hash__")
        return str.__hash__(self)

    def __eq__(self, other):
        hooks.append("_Name.__eq__")
        return str.__eq__(self, other)

context = SnapshotContext(DumpOptions())
calls = {
    "bound": (context.value, ([1],), {Name("depth"): 1}),
    "class": (SnapshotContext.value, (context, [1]), {Name("depth"): 1}),
    "mapping": (context._mapping, ({"key": [1]},), {Name("counted"): True}),
    "receiver": (SnapshotContext.value, (), {Name("self"): context, "value": [1]}),
}
seen = {}
for form, (function, args, names) in calls.items():
    hooks.clear()
    function(*args, **names)
    seen[form] = hooks[:]
print(json.dumps([values._native is None, seen]))
"""


def test_str_subclass_keyword_name_without_the_extension():
    """With ``PYDUMPLING_NATIVE=0`` at import, no call form hashes a name."""
    process = subprocess.run(
        [sys.executable, "-c", _UNLOADED_SCRIPT],
        capture_output=True,
        text=True,
        env=dict(_child_environment(), PYDUMPLING_NATIVE="0"),
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    unloaded, seen = json.loads(process.stdout)
    assert unloaded
    assert seen == {form: hooks[2] for form, hooks in _KEYWORD_HOOKS.items()}


class _AppError(Exception):
    """An exception the walk formats from its arguments with ``reprlib``."""


def _messages(native, held):
    """The messages of exceptions that each hold one of ``held``, directly and
    in a list the walk copies, and the truncation counts, from one context."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
        _check_walker(context, native)
        messages = [context.exception(_AppError(big, [big]))["message"] for big in held]
        if native:
            assert context._walker.handbacks == 0
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


# --- Keys that hash like what the walker looks up ----------------------------
# The walker looks up the id of each dict a call captures in ``_namespaces``,
# and each budget's name in ``DumpOptions``' class dicts, at each call. An
# element of ``_namespaces`` that hashes like such an id has its ``__eq__`` run
# by the walker, and again by the Python walk when the walker hands the call
# back after it. A key other than a str planted in a class dict of
# ``DumpOptions`` (``setattr`` adds only str keys), hashing like a budget's
# name, has its ``__eq__`` run at each call, where the Python walk reads a
# budget only when the call needs it, and then through CPython's attribute
# cache.


class _Hashing:
    """A key that hashes like ``hashed`` and counts the comparisons that
    reach its ``__eq__``, which finds it unequal."""

    def __init__(self, hashed):
        self.hashed = hashed
        self.compared = 0

    def __hash__(self):
        return self.hashed

    def __eq__(self, other):
        self.compared += 1
        return False


class _QualName(str):
    """A ``__qualname__`` that is a str subclass."""


class _Odd:
    """A class on whose str-subclass ``__qualname__`` the walker hands a
    call back."""


_Odd.__qualname__ = _QualName("_Odd")


def test_a_namespaces_element_is_compared_again_after_a_hand_back():
    """An element of ``_namespaces`` that hashes like the id of a dict the
    call captures is compared with that id by the walker, and again by the
    Python walk the walker hands the call back to after it, where the
    Python walk alone compares it once."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        _check_walker(context, native)
        source = {"a": 1}
        planted = _Hashing(hash(id(source)))
        context._namespaces.add(planted)
        context.value([source, _Odd()])
        if native:
            assert context._walker.handbacks == 1
        found[native] = planted.compared
    assert found == {False: 1, True: 2}


def test_a_key_planted_in_dump_options_class_dict_is_compared_at_each_call():
    """A key other than a str, planted in ``DumpOptions``' class dict where
    it hashes like ``max_items``: the walker, which reads every budget at
    each call, runs its ``__eq__`` at each call, where the Python walk,
    which reads ``max_items`` only for a container, runs it not at all."""
    contexts = {}
    for native in (False, True):
        with _mode(native):
            contexts[native] = SnapshotContext(DumpOptions())
        _check_walker(contexts[native], native)
    namespace = gc.get_referents(vars(DumpOptions))[0]
    planted = _Hashing(hash("max_items"))
    kept = namespace.pop("max_items")
    namespace[planted] = None  # ahead of ``max_items`` in its probe order
    namespace["max_items"] = kept
    DumpOptions._planted = None  # resets the attribute cache
    del DumpOptions._planted
    found = {}
    try:
        for native, context in contexts.items():
            planted.compared = 0
            context.value(None)
            context.value(None)
            found[native] = planted.compared
    finally:
        del namespace[planted]
        DumpOptions._planted = None
        del DumpOptions._planted
    assert found == {False: 0, True: 2}
    assert contexts[True]._walker.handbacks == 0


# --- CPython's C recursion limit ---------------------------------------------
# Near CPython's C recursion limit a call the walker hands to the Python walk
# can raise ``RecursionError`` where the Python walk alone returns: the
# descriptor call and the walker's own C calls lie beneath the Python walk it
# hands back to. On 3.12 and 3.13, where CPython counts C levels, a call the
# walker completes itself can instead reach one level deeper than the Python
# walk, whose ``_mapping`` resumes generators that CPython counts. By how many
# levels the limit moves depends on the version, the value and the stack's
# layout, so only the direction is pinned.

_C_RECURSION_SCRIPT = """
import os, pickle, sys
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

form = sys.argv[1]


class QualName(str):
    pass


# The walker hands a call to the Python walk when it reaches an instance of
# a class whose ``__qualname__`` is a str subclass. A value the walker
# captures itself may fit deeper than the Python walk on 3.12 and 3.13,
# whose ``_mapping`` makes counted C calls the walker does not make.
Named = type("Named", (), {})
Named.__qualname__ = QualName("Outer.Named")
value = [{"a": "b"}, Named()]
outcome = []


class Probe:
    # Compared at the bottom of two nested lists, each level of which holds
    # one more C call of the comparison.
    __hash__ = None

    def __eq__(self, other):
        try:
            if form == "keyword":
                captured = context.value(value=value)
            else:
                captured = context.value(value)
            outcome.append(pickle.dumps(captured, 4))
        except RecursionError:
            outcome.append("RecursionError")
        return True


left, right = [[Probe()]], [[Probe()]]


def capture_beneath(native, levels):
    global context
    os.environ["PYDUMPLING_NATIVE"] = "1" if native else "0"
    context = SnapshotContext(DumpOptions())
    assert (vars(context).get("_walker") is not None) is native
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
there = capture_beneath(True, low)
shallower = capture_beneath(True, low // 2)
print(
    there if isinstance(there, str) else "captured",
    shallower == whole,
    vars(context)["_walker"].handbacks,
)
"""


@pytest.mark.parametrize("form", ["keyword", "handed back"])
def test_the_c_recursion_limit_comes_sooner_through_the_walker(form):
    """Beneath nested comparisons at the default recursion limit, a call
    the walker hands to the Python walk, at once for a keyword argument or
    partway for a class whose ``__qualname__`` is a str subclass, raises
    ``RecursionError`` at the deepest nesting where the Python walk alone
    captures the whole value; at half that depth both walks capture alike.
    Each level is one C level where CPython counts them (3.12 and 3.13) and
    the same C stack where it measures the stack (from 3.14), so the
    deepest nesting is found anew in each child interpreter."""
    process = subprocess.run(
        [sys.executable, "-c", _C_RECURSION_SCRIPT, form.split()[0]],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.split() == ["RecursionError", "True", "1"]


_DEEPER_SCRIPT = """
import os, pickle
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

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
    os.environ["PYDUMPLING_NATIVE"] = "1" if native else "0"
    context = SnapshotContext(DumpOptions())
    assert (vars(context).get("_walker") is not None) is native
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
native = capture_beneath(True, low + 1)
print(
    python if isinstance(python, str) else "captured",
    "whole" if native == whole else native,
    vars(context)["_walker"].handbacks,
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
