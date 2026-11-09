"""The accepted differences between the native walker and the Python walk
that ``rust/DESIGN.md`` lists, and that no other module pins, each against the
Python walk.  A change to either walk here changes that entry too.  Unless a
test says otherwise, both walks behave the same on CPython 3.12 to 3.15.
"""

import contextlib
import copy
import copyreg
import inspect
import io
import json
import os
import pickle
import reprlib
import subprocess
import sys
from collections import Counter, OrderedDict
from unittest import mock

import pytest

from pydumpling.fake_types import FakeClass, TruncatedValue
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

# --- Python code that clears or overwrites the dicts between calls -----------
# Changes Python code makes to ``objects`` and ``_sources`` between calls while
# the native walker holds copies it has yet to write, pinned against the Python
# walk. ``objects.clear()`` lets go of the copies the walker already wrote, as
# the Python walk does, but the walker's next write puts the copies it had yet
# to write into the emptied dict, and it keeps returning them, where the Python
# walk leaves the dict empty and copies those originals afresh. After
# ``_sources.clear()`` both walks keep returning their copies, and the walker's
# next write puts the originals it had yet to write back into ``_sources``,
# where the Python walk's stays empty. An entry the walker already wrote, which
# Python code then overwrites in place, goes unnoticed and stays: the walker
# keeps returning its own copy where the Python walk returns Python's, until a
# noticed removal, a replaced dict, or a write after Python changed ``objects``
# makes it take its memo from ``objects`` again.


_ALL = ["written", "written[0]", "handed", "unwritten", "unwritten[0]"]
_UNWRITTEN = ["unwritten", "unwritten[0]"]


def _after_clear(native, cleared):
    """Empties ``cleared`` after the walker wrote ``written`` and before it
    wrote ``unwritten``, then copies the context, which writes.

    Returns the names of the originals whose ids key ``objects`` and
    ``_sources`` then, and of the roots that capture to their first copy
    again.  A keyword call is handed back to the Python walk, so the walker
    writes every copy it holds before it.
    """
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    written, handed, unwritten = [[]], [], [[]]
    nodes = {
        "written": written,
        "written[0]": written[0],
        "handed": handed,
        "unwritten": unwritten,
        "unwritten[0]": unwritten[0],
    }
    names = {id(node): name for name, node in nodes.items()}
    first = {"written": context.value(written)}
    context.value(value=handed)
    first["unwritten"] = context.value(unwritten)
    getattr(context, cleared).clear()
    copy.copy(context)
    assert all(source is nodes[names[key]] for key, source in context._sources.items())
    objects = [names[key] for key in context.objects]
    sources = [names[key] for key in context._sources]
    again = {name: context.value(nodes[name]) for name in first}
    assert again == first
    return objects, sources, [name for name in first if again[name] is first[name]]


@pytest.mark.parametrize(
    ("cleared", "pure", "native"),
    [
        ("objects", ([], _ALL, []), (_UNWRITTEN, _ALL, ["unwritten"])),
        (
            "_sources",
            (_ALL, [], ["written", "unwritten"]),
            (_ALL, _UNWRITTEN, ["written", "unwritten"]),
        ),
    ],
)
def test_clear_keeps_the_entries_the_walker_has_yet_to_write(cleared, pure, native):
    """``clear()`` leaves the walker's unwritten entries in the emptied dict,
    and only those; the Python walk keeps nothing there."""
    assert _after_clear(False, cleared) == pure
    assert _after_clear(True, cleared) == native


def _after_overwrite(native, written_by):
    """What capturing ``original`` returns after Python code overwrites the
    copy the walker wrote for it: alone, inside a new list, what ``objects``
    holds for it after a copy of the context, and alone again once Python
    code removed another entry.  ``"first copy"`` stands for the copy the
    first capture returned."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    original, other = [[]], [[]]
    first = context.value(original)
    context.value(other)
    if written_by == "copy":
        copy.copy(context)
    else:
        context.value(value="hand-back")
    context.objects[id(original)] = "<python>"
    seen = [context.value(original), context.value([original])[0]]
    copy.copy(context)
    seen.append(context.objects[id(original)])
    del context.objects[id(other)]
    seen.append(context.value(original))
    return ["first copy" if item is first else item for item in seen]


@pytest.mark.parametrize("written_by", ["copy", "hand-back"])
def test_an_entry_python_overwrites_in_place(written_by):
    """An overwritten entry the walker already wrote, whether a copy made it
    write each call's copies at once or a hand-back made it write once, is
    neither noticed nor undone until a noticed removal rebuilds its memo."""
    assert _after_overwrite(False, written_by) == ["<python>"] * 4
    assert _after_overwrite(True, written_by) == [
        "first copy",
        "first copy",
        "<python>",
        "<python>",
    ]


# --- Copies made outside the context's copy protocol -------------------------
# Copies of a native context made outside its class's own copy protocol.
#
# A copy made through ``copyreg``, a pickler's ``reducer_override`` or
# ``dispatch_table``, or a by-hand copy of the instance dict reads the instance
# dict as it stands, where ``_walker`` comes after ``objects``. A pickle or
# deep copy meets ``objects`` first, so it holds only the copies the walker has
# written so far; copying ``_walker`` then writes the rest into the original,
# too late for the copy, and leaves ``None`` in it. A shallow copy shares the
# walker itself, so a capture through it while the original's native call runs
# raises ``RuntimeError`` where the Python walk captures. The Python walk's
# ``objects`` is complete whenever it is copied. Both walks behave so on every
# supported version. ``rust/DESIGN.md`` lists this among the accepted
# differences, so a change to either walk here changes that entry too.


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


class _Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        return _reduce(obj) if type(obj) is SnapshotContext else NotImplemented


@contextlib.contextmanager
def _registered():
    """Copy contexts through ``_reduce`` from ``copyreg``."""
    copyreg.pickle(SnapshotContext, _reduce)
    try:
        yield
    finally:
        del copyreg.dispatch_table[SnapshotContext]


def _deep_copy(context, how):
    """A pickle or deep copy of ``context`` reduced by ``_reduce``."""
    if how == "deepcopy":
        with _registered():
            return copy.deepcopy(context)
    buffer = io.BytesIO()
    if how == "reducer_override":
        _Pickler(buffer, protocol=5).dump(context)
    else:
        pickler = pickle.Pickler(buffer, protocol=5)
        pickler.dispatch_table = {SnapshotContext: _reduce}
        pickler.dump(context)
    return pickle.loads(buffer.getvalue())


def _shallow_copy(context, how):
    """A shallow copy of ``context`` sharing the values of its instance dict."""
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
@pytest.mark.parametrize("how", ["reducer_override", "dispatch_table", "deepcopy"])
def test_a_deep_copy_holds_only_the_copies_written_so_far(how):
    """A pickle or deep copy reduced from the instance dict holds natively
    only the copies the walker wrote before it: here those of the calls up
    to one that passes its value by keyword, which goes to the Python walk
    after the walker writes its copies, and not those of the native call
    after it.  The Python walk's copy holds them all, as the original holds
    them natively once the copy is made."""
    held = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        _check_walker(context, native)
        context.value([1, "a", (2, [3])])
        context.value(value={"b": [4]})
        context.value([[5], {"c": (6,)}])
        written = list(context.objects)
        copied = _deep_copy(context, how)
        held[native] = (written, list(copied.objects), list(context.objects))
        assert len(copied._sources) == len(copied.objects)
        if native:
            assert context._walker.handbacks == 1
            assert vars(copied)["_walker"] is None
        else:
            assert "_walker" not in vars(copied)
    written, copied, complete = held[False]
    assert copied == written == complete
    written, copied, after = held[True]
    assert copied == written == after[: len(written)]
    assert 0 < len(written) < len(after) == len(complete)


@_UNTRACED
@pytest.mark.parametrize("how", ["copyreg", "by_hand"])
def test_a_shallow_copy_shares_the_walker(how):
    """A shallow copy built from a copy of the instance dict holds the
    original's walker natively, and no walker in Python; it shares
    ``objects`` either way.  A capture through it from a collection
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
# A context captured as a value shows the native walker's own state.
#
# A native context keeps its walker in its instance dict as ``_walker``, writes
# its copies into ``objects`` and ``_sources`` lazily, and writes ``_nodes``
# only when a call commits. So a native context captured as a value, by itself
# (through ``value`` or a frame that holds it) or by another context, has a
# ``_walker`` attribute, an unsupported-type ``FakeClass``, and only the copies
# written so far: none before the context is first copied, then only those of
# its earlier calls. Captured by itself, it also has the node count as the call
# began. The Python walk's copy has no ``_walker``, every copy made so far, the
# capturing call's own included, and the count the walk had reached when it
# selected the instance dict's entries: two nodes past the call's start through
# ``value``, three through a frame namespace. Captured by another context, both
# walks show the count its last call left. A self capture handed back to the
# Python walk, as one is under a profiler, shows the ``HandBack`` marker as
# ``_walker`` and otherwise what the Python walk shows.


_WALKER = {"_FakeClass__repr": "<pydumpling.snapshot.values._Walker snapshot>"}


_HAND_BACK = {"_FakeClass__repr": "<pydumpling._native.HandBack snapshot>"}


def _held_by_a_frame(context):
    """Capture this frame, whose one variable is ``context``, through it."""
    head = context.from_frame(sys._getframe())
    return head.tb_frame.f_locals["context"]


def _captured(native, by, flushed):
    """A context, after one capture and maybe a copy, and the copy of it that
    capturing it ``by`` some route makes, with its earlier originals' ids and
    its node count just before."""
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_frames=1))
        other = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    _check_walker(other, native)
    data = [1, "a"]
    context.value(data)
    if flushed:
        context.__getstate__()
    nodes = context._nodes
    if by == "itself":
        copy = context.value(context)
    elif by == "its own frame":
        copy = _held_by_a_frame(context)
    else:
        copy = other.value(context)
    return context, vars(copy), [id(data), id(data[0]), id(data[1])], nodes


def _assert_running(context, attrs, earlier, nodes, counted):
    """The copy holds every copy made so far and the count the walk reached."""
    objects, sources = list(attrs["objects"]), list(attrs["_sources"])
    if counted:
        assert objects[:3] == sources[:3] == earlier
        assert {id(context), id(vars(context))} <= set(objects)
    else:
        assert objects == sources == earlier
    assert attrs["_nodes"] == nodes + counted


@pytest.mark.parametrize("flushed", [False, True], ids=["lazy", "copied"])
@pytest.mark.parametrize("by", ["itself", "its own frame", "another context"])
def test_the_native_copy_shows_the_walkers_state(by, flushed):
    if _traced():
        pytest.skip("a tracer hands the capture back to the Python walk")
    context, attrs, earlier, nodes = _captured(True, by, flushed)
    assert "_walker" in attrs and vars(attrs["_walker"]) == _WALKER
    written = earlier if flushed else []
    assert list(attrs["objects"]) == list(attrs["_sources"]) == written
    assert attrs["_nodes"] == nodes
    assert context._walker.handbacks == 0


@pytest.mark.parametrize("flushed", [False, True], ids=["lazy", "copied"])
@pytest.mark.parametrize(
    "by, counted", [("itself", 2), ("its own frame", 3), ("another context", 0)]
)
def test_the_python_copy_shows_the_running_walk(by, counted, flushed):
    context, attrs, earlier, nodes = _captured(False, by, flushed)
    assert "_walker" not in attrs
    _assert_running(context, attrs, earlier, nodes, counted)


@pytest.mark.parametrize("by, counted", [("itself", 2), ("its own frame", 3)])
def test_a_handed_back_self_capture_shows_the_marker(by, counted):
    previous = sys.getprofile()
    sys.setprofile(lambda frame, event, arg: None)
    try:
        context, attrs, earlier, nodes = _captured(True, by, False)
    finally:
        sys.setprofile(previous)
    assert context._walker.handbacks >= 1
    assert "_walker" in attrs and vars(attrs["_walker"]) == _HAND_BACK
    _assert_running(context, attrs, earlier, nodes, counted)


# --- Helpers patched after import --------------------------------------------
# Helpers patched over the ``values`` module or onto ``collections.Counter``.
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
# for the walker's counts, which come out the same.


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


@pytest.mark.parametrize("copied", [False, True], ids=["unwritten", "written"])
def test_wrapped_function_on_a_native_context(copied):
    """The wrapped function, called directly, misses the copies the walker has
    yet to write, and finds them once a copy of the context wrote them."""
    for native in (True, False):
        nested = [2]
        original = [1, nested, {"key": 3}]
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        _check_walker(context, native)
        captured = context.value(original)
        if copied:
            copy.copy(context)
        if native:
            assert context._walker.handbacks == 0
            wrapped = context.value.__wrapped__
        else:
            wrapped = SnapshotContext.value.__wrapped__
        stale = native and not copied
        assert (id(original) in context.objects) is not stale
        nodes = context._nodes
        again = wrapped(context, original)
        if stale:
            assert again is not captured
            assert again == captured
            assert again[1] is captured[1] and again[2] is captured[2]
            assert context._nodes == nodes + 1
        else:
            assert again is captured
            assert context._nodes == nodes
        assert context.objects[id(original)] is again
        assert context.value(original) is again


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
