import builtins
import os
import pickle
import sys
import traceback
import tracemalloc
from collections import Counter, OrderedDict, defaultdict, namedtuple
from dataclasses import FrozenInstanceError
from enum import IntEnum, StrEnum
from types import SimpleNamespace

import pytest

from pydumpling import dump_current_traceback, load_dumpling, save_dumping
from pydumpling.fake_types import (
    FakeBytes,
    FakeClass,
    FakeDict,
    FakeFloat,
    FakeInt,
    FakeList,
    FakeStr,
    FakeTuple,
    TruncatedValue,
)
from pydumpling.options import DumpOptions
from pydumpling.pydumpling import _capture_frame
from pydumpling.snapshot import (
    _CODE_BYTES_LIMIT,
    _CODE_TEXT_LIMIT,
    SnapshotContext,
)


def make_stack(count):
    code = (lambda: None).__code__
    shared = {"answer": 42}
    globals_ = {"shared": shared}
    frames = []
    previous = None
    for index in range(count):
        frame = SimpleNamespace(
            f_code=code,
            f_locals={"index": index, "shared": shared},
            f_globals=globals_,
            f_builtins={},
            f_lineno=index + 1,
            f_lasti=0,
            f_back=previous,
        )
        frames.append(frame)
        previous = frame
    traceback = None
    for frame in reversed(frames):
        traceback = SimpleNamespace(
            tb_frame=frame,
            tb_lineno=frame.f_lineno,
            tb_lasti=0,
            tb_next=traceback,
        )
    return frames, traceback


def traceback_frames(traceback):
    result = []
    while traceback is not None:
        result.append(traceback.tb_frame)
        traceback = traceback.tb_next
    return result


def test_options_defaults_validation_and_unlimited():
    assert DumpOptions().max_frames == 64
    assert DumpOptions.unlimited().is_unlimited
    for value in (0, -1, True, "5"):
        with pytest.raises(ValueError, match="positive integer"):
            DumpOptions(max_nodes=value)
    with pytest.raises(FrozenInstanceError):
        DumpOptions().max_frames = 5


def test_deep_stack_retains_failure_frames_and_shared_identity():
    _, traceback = make_stack(1100)
    context = SnapshotContext(DumpOptions())
    restored = pickle.loads(pickle.dumps(context.from_traceback(traceback)))
    frames = traceback_frames(restored)
    assert len(frames) == 64
    assert frames[0].f_lineno == 1037
    assert frames[-1].f_lineno == 1100
    assert frames[-1].f_locals["index"] == 1099
    assert frames[0].f_back is None
    assert all(inner.f_back is outer for outer, inner in zip(frames, frames[1:]))
    assert all(frame.f_code is frames[0].f_code for frame in frames)
    assert frames[0].f_locals["shared"] is frames[-1].f_locals["shared"]
    assert context.metadata()["truncation"]["max_frames"] == 1036


def test_from_frame_includes_supplied_call_site_and_minimal_has_no_variables():
    frames, _ = make_stack(5)
    options = DumpOptions(max_frames=2)
    captured = traceback_frames(SnapshotContext(options).from_frame(frames[-1]))
    assert [frame.f_lineno for frame in captured] == [4, 5]
    context = SnapshotContext(options, minimal=True)
    minimal = traceback_frames(context.from_frame(frames[-1]))
    assert all(frame.f_locals == frame.f_globals == {} for frame in minimal)
    assert context.metadata()["truncation"]["minimal_snapshot"] == 1


def test_long_function_retains_fault_line_and_columns_after_code_budget():
    namespace = {}
    source = (
        "def failure():\n"
        + "".join("    value = %s\n" % index for index in range(200))
        + "    return value / 0\n"
    )
    exec(compile(source, "<bounded-code-test>", "exec"), namespace)
    try:
        namespace["failure"]()
    except ZeroDivisionError:
        original = sys.exc_info()[2]
        context = SnapshotContext(DumpOptions(max_frames=1, max_items=3))
        saved = pickle.loads(pickle.dumps(context.from_traceback(original)))
        expected = traceback.extract_tb(original)[-1]
        actual = traceback.extract_tb(saved)[-1]
    assert (actual.filename, actual.lineno, actual.colno, actual.end_colno) == (
        expected.filename,
        expected.lineno,
        expected.colno,
        expected.end_colno,
    )


def test_value_budgets_preserve_cycles_and_bound_nested_data():
    options = DumpOptions(max_depth=2, max_items=3, max_string_bytes=8)
    context = SnapshotContext(options)
    cycle = []
    cycle.append(cycle)
    captured = context.value(cycle)
    assert captured[0] is captured
    assert len(context.value(list(range(10000)))) == 3
    assert len(context.value("饺子" * 10000).encode("utf-8")) <= 8
    assert isinstance(context.value([[[[42]]]])[0][0], TruncatedValue)
    limited = SnapshotContext(DumpOptions(max_nodes=3))
    assert len(limited.value(list(range(10000)))) <= 3
    assert limited.metadata()["truncation"]["max_nodes"]
    code = (lambda arg, *args, **kwargs: None).__code__
    clipped = SnapshotContext(DumpOptions(max_items=1)).code(code)
    assert clipped.co_argcount == 1
    assert not clipped.co_flags & (4 | 8)


@pytest.mark.parametrize("field", ["f_locals", "f_globals"])
def test_depth_truncated_value_is_recaptured_as_frame_mapping(field, capsys):
    from pydumpling.helpers import print_snapshot_notice

    frames, traceback = make_stack(1)
    source = {"important": "value"}
    setattr(frames[0], field, source)
    context = SnapshotContext(DumpOptions(max_depth=2))
    context.value({"a": {"b": source}})
    before = context.metadata()["truncation"]["max_depth"]
    restored = pickle.loads(pickle.dumps(context.from_traceback(traceback)))
    assert getattr(restored.tb_frame, field) == source
    assert context.metadata()["truncation"]["max_depth"] == before
    print_snapshot_notice({"snapshot_meta": context.metadata()})
    assert "max_depth=1" in capsys.readouterr().err


def test_recapturing_depth_truncated_mapping_respects_remaining_budget():
    context = SnapshotContext(DumpOptions(max_depth=2, max_nodes=5))
    source = {"important": "value"}
    context.value({"a": {"b": source}})
    assert context._mapping(source) == {}
    assert context.metadata()["truncation"] == {"max_depth": 1, "max_nodes": 1}


def test_bounded_capture_never_calls_application_serialization_or_repr_hooks():
    calls = []

    class Hostile:
        def __init__(self):
            self.answer = 42

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("application repr called")

        def __reduce_ex__(self, protocol):
            calls.append("reduce")
            raise AssertionError("application reduce called")

    value = Hostile()
    captured = SnapshotContext(DumpOptions()).value({"value": value, "alias": value})
    restored = pickle.loads(pickle.dumps(captured))
    assert isinstance(restored["value"], FakeClass)
    assert restored["value"].answer == 42
    assert restored["value"] is restored["alias"]

    class HostileGlobals(dict):
        def items(self):
            calls.append("items")
            raise AssertionError("application mapping items called")

    namespace = HostileGlobals()
    exec("import sys\nframe = sys._getframe()\n", namespace)
    context = SnapshotContext(DumpOptions(max_frames=1))
    assert context.from_frame(namespace["frame"]).tb_frame.f_globals
    assert calls == []


def test_oserror_winerror_survives_snapshot_and_serialization():
    import dill

    error = OSError(2, "missing", "source", 2, "destination")
    error.winerror = 2
    saved = SnapshotContext(DumpOptions()).exception(error)["exc_value"]
    for serializer in (pickle, dill):
        restored = serializer.loads(serializer.dumps(saved))
        assert restored.winerror == 2
        assert str(restored) == str(error)


def test_exception_fallback_keeps_custom_message_and_group_children():
    class CustomError(Exception):
        def __str__(self):
            raise AssertionError("application str called")

    error = CustomError("useful message")
    error.generator = (number for number in range(3))
    context = SnapshotContext(DumpOptions())
    extra = context.exception(error)
    assert "useful message" in "".join(extra["formatted"])
    group = ExceptionGroup("batch", [ValueError("first"), error])
    extra = context.exception(group)
    assert "first" in "".join(extra["formatted"])
    assert "useful message" in "".join(extra["formatted"])
    pickle.dumps(extra)
    outer = RuntimeError("outer")

    class FalseyError(Exception):
        def __bool__(self):
            raise AssertionError("application bool called")

    outer.__cause__ = FalseyError("inner")
    outer.add_note("diagnostic note")
    summary = "".join(SnapshotContext(DumpOptions()).exception(outer)["formatted"])
    assert all(text in summary for text in ("outer", "inner", "diagnostic note"))
    exhausted = SnapshotContext(DumpOptions(max_nodes=1))
    exhausted.value([42])
    assert (
        exhausted.exception(ValueError("failure reason"))["message"] == "failure reason"
    )


def test_unlimited_preserves_serializable_objects_and_pickle_degrades_lambdas():
    class Original:
        pass

    value = Original()
    assert SnapshotContext(DumpOptions.unlimited()).value(value) is value
    captured = SnapshotContext(DumpOptions.unlimited(), serializer="pickle").value(
        lambda: 42
    )
    assert isinstance(captured, FakeClass)
    pickle.dumps(captured)


def test_reducing_snapshot_values_preserves_placeholders_and_representations():
    omitted = TruncatedValue("max_depth", "original.Type")
    value = FakeClass("<original.Type snapshot>", {"omitted": omitted})
    value.self = value
    reduced = SnapshotContext(DumpOptions(max_depth=2)).value(value)
    assert repr(reduced) == repr(value)
    assert reduced.omitted is omitted
    assert reduced.self is reduced


def test_container_subclasses_keep_entries_type_and_attributes():
    import dill

    class Registry(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.owner = "team-a"

    class SlottedList(list):
        __slots__ = ("owner",)

    record = namedtuple("Record", "x y")
    ordered = OrderedDict(a=1, b=2, c=3)
    ordered.move_to_end("a")
    slotted = SlottedList([1, 2])
    slotted.owner = "team-b"
    status = os.stat(__file__)
    context = SnapshotContext(DumpOptions())
    captured = context.value(
        {
            "counter": Counter("aab"),
            "ordered": ordered,
            "default": defaultdict(list, items=[1]),
            "record": record(1, 2),
            "status": status,
            "registry": Registry(entries=2),
            "slotted": slotted,
            "unset": SlottedList([3]),
        }
    )
    assert type(captured["counter"]) is FakeDict
    assert repr(captured["counter"]) == "collections.Counter({'a': 2, 'b': 1})"
    # dict.items would replay insertion order; an LRU cache needs link order.
    assert list(captured["ordered"]) == ["b", "c", "a"]
    assert captured["default"] == {"items": [1]}
    assert "default_factory" in vars(captured["default"])
    # Named fields and struct sequence fields outside the tuple stay attributes.
    assert type(captured["record"]) is FakeTuple
    assert captured["record"] == (1, 2) and captured["record"].x == 1
    assert captured["status"] == tuple(status)
    assert captured["status"].st_mtime == status.st_mtime
    assert captured["registry"] == {"entries": 2}
    assert captured["registry"].owner == "team-a"
    assert type(captured["slotted"]) is FakeList
    assert captured["slotted"].owner == "team-b"
    assert "owner" not in vars(captured["unset"])
    # Nothing is lost except default_factory, a class, which is never captured.
    assert context.metadata()["truncation"] == {"unsupported_type": 1}
    for restored in (
        pickle.loads(pickle.dumps(captured)),
        dill.loads(dill.dumps(captured)),
        # A reduction stage captures the copies again.
        SnapshotContext(DumpOptions()).value(captured),
    ):
        assert repr(restored["counter"]) == repr(captured["counter"])
        assert restored["record"].y == 2
        assert restored["status"].st_size == status.st_size
        assert restored["registry"].owner == "team-a"


@pytest.mark.parametrize(
    "options, reason",
    [
        (DumpOptions(max_items=1), "max_items"),
        (DumpOptions(max_items=None, max_nodes=3), "max_nodes"),
    ],
)
def test_ordered_dict_budget_bounds_temporary_memory(options, reason):
    value = OrderedDict((i, i + 20000) for i in range(20000))
    value.move_to_end(0)
    context = SnapshotContext(options)
    # Measure capture allocations, excluding the already-existing application data.
    tracemalloc.start()
    try:
        captured = context.value(value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert list(captured.items()) == [(1, 20001)]
    assert context.metadata()["truncation"][reason] == 1
    assert peak < 512 * 1024


def test_ordered_dict_with_hooked_keys_keeps_the_frame(tmp_path):
    class Key:
        armed = False

        def __hash__(self):
            if Key.armed:
                raise AssertionError("application __hash__ called")
            return 1

    path = tmp_path / "ordered.dump"

    def business():
        important = 42
        ordered = OrderedDict([(Key(), "first"), ("plain", 1)])
        ordered.move_to_end("plain", last=False)
        Key.armed = True
        try:
            dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
        finally:
            Key.armed = False
        return important, ordered

    business()
    result = load_dumpling(path)
    assert result["snapshot_meta"]["attempts"] == []
    variables = result["traceback"].tb_frame.f_locals
    assert variables["important"] == 42
    # CPython's OrderedDict iterator would hash the key, so the entries keep
    # insertion order instead of link order, and the snapshot says so.
    assert list(variables["ordered"].values()) == ["first", 1]
    assert result["snapshot_meta"]["truncation"]["mapping_order"] == 1


def test_ordered_dict_with_plain_tuple_keys_keeps_link_order():
    ordered = OrderedDict([((1, "a"), 1), ((2, "b"), 2)])
    ordered.move_to_end((1, "a"))
    context = SnapshotContext(DumpOptions())
    assert list(context.value(ordered)) == [(2, "b"), (1, "a")]
    assert "mapping_order" not in context.metadata()["truncation"]


def test_ordered_dict_budget_preserves_priority_and_memoized_keys():
    value = OrderedDict(a=None, b=None, c=None)
    value.move_to_end("a")
    context = SnapshotContext(DumpOptions(max_nodes=6))
    for key in value:
        context.value(key)
    context.value(None)
    captured = context._mapping(value, priority=("c",))
    assert list(captured.items()) == [("c", None), ("b", None), ("a", None)]
    assert context.metadata()["truncation"] == {}


@pytest.mark.parametrize(
    "options, expected, reason",
    [
        (DumpOptions(max_items=1), {"a": 1}, "max_items"),
        (DumpOptions(max_nodes=3), {"a": 1, "b": 2}, "max_nodes"),
    ],
)
def test_container_slots_respect_budgets(options, expected, reason):
    class SlottedList(list):
        __slots__ = ("a", "b", "c", "unset")

    value = SlottedList()
    value.a, value.b, value.c = 1, 2, 3
    context = SnapshotContext(options)
    captured = context.value(value)
    attributes = vars(captured).copy()
    attributes.pop("_FakeContainer__type")
    assert attributes == expected
    assert context.metadata()["truncation"] == {reason: 1}


def test_container_slots_share_item_budget_with_instance_attributes():
    class SlottedList(list):
        __slots__ = ("a", "__dict__")

    value = SlottedList()
    value.a, value.owner = 1, "team-a"
    context = SnapshotContext(DumpOptions(max_items=1))
    captured = context.value(value)
    assert captured.owner == "team-a"
    assert not hasattr(captured, "a")
    assert context.metadata()["truncation"] == {"max_items": 1}


def test_container_copies_with_cycles_or_shadowing_attributes_serialize(tmp_path):
    import dill

    class EasyDict(dict):
        """Mirrors every key as an attribute, as config objects often do."""

        def __init__(self, **entries):
            super().__init__(**entries)
            self.__dict__.update(entries)

    class Looped(list):
        pass

    class LoopedSet(set):
        __hash__ = object.__hash__

    path = tmp_path / "cycles.dump"

    def business():
        important = 42
        config = EasyDict(items=[1], name="cfg")
        looped = Looped()
        looped.append(looped)
        looped_set = LoopedSet()
        looped_set.add(looped_set)
        looped_set.owner = "team-a"
        dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
        return important, config, looped, looped_set

    business()
    result = load_dumpling(path)
    # One such value must not cost the frame: both serializers succeed, so the
    # writer never falls back to a minimal snapshot.
    assert result["snapshot_meta"]["attempts"] == []
    variables = result["traceback"].tb_frame.f_locals
    assert variables["important"] == 42
    # ``items`` shadows dict.items on the copy, as it does on the original.
    assert variables["config"] == {"items": [1], "name": "cfg"}
    assert variables["config"].items == [1]
    assert variables["looped"][0] is variables["looped"]
    for restored in (variables, pickle.loads(pickle.dumps(variables))):
        (member,) = restored["looped_set"]
        assert member is restored["looped_set"]
        assert member.owner == "team-a"
    assert dill.loads(dill.dumps(variables["config"])).name == "cfg"


@pytest.mark.parametrize("name", ["__reduce_ex__", "__getstate__", "items"])
def test_attribute_named_like_a_pickling_hook_keeps_the_snapshot(tmp_path, name):
    class Holder:
        pass

    class Registry(dict):
        pass

    class Label(str):
        pass

    path = tmp_path / "hooks.dump"

    def business():
        important = 42
        values = [Holder(), Registry(entry=1), Label("alice")]
        for value in values:
            setattr(value, name, "captured")
        dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
        return important, values

    business()
    result = load_dumpling(path)
    # pickle and dill look these names up on the copy, where the captured
    # attribute would replace the method; neither writer may fail.
    assert result["snapshot_meta"]["attempts"] == []
    variables = result["traceback"].tb_frame.f_locals
    assert variables["important"] == 42
    holder, registry, label = variables["values"]
    assert registry == {"entry": 1} and label == "alice"
    for copy in (holder, registry, label):
        assert vars(copy)[name] == "captured"


def test_container_subclass_copies_stay_hashable():
    class FrozenMapping(dict):
        __slots__ = ()

        def __hash__(self):
            return hash(frozenset(self.items()))

    class Odd(tuple):
        def __hash__(self):
            return 1

    mapping = FrozenMapping(a=1)
    context = SnapshotContext(DumpOptions())
    # A plain dict copy, or a tuple copy holding a list, would be unhashable.
    captured = context.value(
        {"members": {mapping, Odd(([1],))}, "keys": {(mapping,): "value"}}
    )
    member = next(item for item in captured["members"] if type(item) is FakeDict)
    assert member == {"a": 1}
    assert ([1],) in [tuple(item) for item in captured["members"]]
    ((key, value),) = captured["keys"].items()
    assert key[0] is member and value == "value"
    assert context.metadata()["truncation"] == {}
    restored = pickle.loads(pickle.dumps(captured))
    assert len(restored["members"]) == 2


def test_immutable_copies_are_found_by_equal_plain_values():
    point = namedtuple("Point", "x y")
    tags = type("Tags", (frozenset,), {})
    captured = SnapshotContext(DumpOptions()).value(
        {"by_point": {point(1, 2): "hit"}, "tags": {tags({"a"})}}
    )
    # A copy equals the plain value, so lookups with one must find it.
    assert captured["by_point"][(1, 2)] == "hit"
    assert frozenset({"a"}) in captured["tags"]
    restored = pickle.loads(pickle.dumps(captured))
    assert restored["by_point"][(1, 2)] == "hit"


def test_attribute_names_are_names_not_string_data():
    class Registry(dict):
        def __init__(self):
            super().__init__(entries=1)
            self.owner_team = "platform"

    holder = _MetadataProbe()
    holder.long_attribute_name = "value"
    context = SnapshotContext(DumpOptions(max_string_bytes=4))
    registry, plain = context.value([Registry(), holder])
    # Values obey the string budget; the names that reach them do not.
    assert registry.owner_team == "plat"
    assert plain.long_attribute_name == "valu"


def test_captured_copies_survive_a_reduction_stage_unchanged():
    holder = _MetadataProbe()
    holder.answer = 42
    first = SnapshotContext(DumpOptions())
    copies = first.value([Counter(a=1), holder])
    # A reduction stage captures the copies again with halved budgets.
    again = SnapshotContext(DumpOptions(max_string_bytes=4))
    counter, plain = again.value(copies)
    # Bookkeeping keys are restored, never converted as data.
    assert vars(counter) == {"_FakeContainer__type": "collections.Counter"}
    assert repr(plain) == repr(copies[1]) and plain.answer == 42
    assert "_Fak" not in vars(plain)
    assert again.metadata()["truncation"] == {}


def test_subclass_attribute_hides_a_base_slot_as_in_attribute_lookup():
    class Base(list):
        __slots__ = ("label",)

    class Child(Base):
        @property
        def label(self):
            return "property"

    value = Child([1])
    Base.label.__set__(value, "slot")
    captured = SnapshotContext(DumpOptions()).value(value)
    # The original's ``label`` is the property, so the slot is not its value.
    assert value.label == "property"
    assert "label" not in vars(captured)


def test_many_instances_of_one_container_type_keep_their_fields():
    record = namedtuple("Record", "x")
    captured = SnapshotContext(DumpOptions()).value([record(i) for i in range(50)])
    assert [item.x for item in captured] == list(range(50))


def test_container_copies_pretty_print_across_lines():
    import pprint

    captured = SnapshotContext(DumpOptions()).value(
        Counter({f"key{index}": index for index in range(8)})
    )
    lines = pprint.pformat(captured, width=40).splitlines()
    # pdb's ``pp`` keeps the layout and the original type name.
    assert len(lines) == 8
    assert lines[0].startswith("collections.Counter({")


class _Name(str):
    pass


class _Blob(bytes):
    pass


class _Count(int):
    pass


class _Ratio(float):
    __slots__ = ("unit", "unset")


class _Color(IntEnum):
    RED = 1


class _Mode(StrEnum):
    FAST = "fast"


def test_scalar_subclasses_keep_value_type_and_attributes():
    import dill

    name = _Name("alice")
    name.owner = "team-a"
    name.self = name
    ratio = _Ratio(0.5)
    ratio.unit = "s"
    context = SnapshotContext(DumpOptions())
    captured = context.value(
        {
            "name": name,
            "blob": _Blob(b"xy"),
            "count": _Count(7),
            "ratio": ratio,
            "color": _Color.RED,
            "mode": _Mode.FAST,
            "by_color": {_Color.RED: "red"},
        }
    )
    expected = {
        "name": (FakeStr, "alice", "_Name"),
        "blob": (FakeBytes, b"xy", "_Blob"),
        "count": (FakeInt, 7, "_Count"),
        "ratio": (FakeFloat, 0.5, "_Ratio"),
        "color": (FakeInt, 1, "_Color"),
        "mode": (FakeStr, "fast", "_Mode"),
    }
    for key, (copy_type, plain, type_name) in expected.items():
        assert type(captured[key]) is copy_type
        assert captured[key] == plain
        assert repr(captured[key]) == f"{__name__}.{type_name}({plain!r})"
    assert captured["name"].owner == "team-a"
    assert captured["name"].self is captured["name"]
    # A filled slot becomes an attribute; an unset one holds nothing.
    assert captured["ratio"].unit == "s"
    assert "unset" not in vars(captured["ratio"])
    assert (captured["color"]._name_, captured["color"]._value_) == ("RED", 1)
    assert (captured["mode"]._name_, captured["mode"]._value_) == ("FAST", "fast")
    # Copies hash like the plain value they equal, so lookups still work.
    assert captured["by_color"][1] == "red"
    # Only each enum member's class, a type, is not captured.
    assert context.metadata()["truncation"] == {"unsupported_type": 2}
    for restored in (
        pickle.loads(pickle.dumps(captured)),
        dill.loads(dill.dumps(captured)),
        # A reduction stage captures the copies again.
        SnapshotContext(DumpOptions()).value(captured),
    ):
        for key, (copy_type, plain, _) in expected.items():
            assert type(restored[key]) is copy_type
            assert repr(restored[key]) == repr(captured[key])
        assert restored["name"].self is restored["name"]
        assert restored["name"].owner == "team-a"
        assert restored["color"]._name_ == "RED"
        assert restored["by_color"][1] == "red"


def test_scalar_copies_pretty_print_across_lines():
    import pprint

    captured = SnapshotContext(DumpOptions()).value(_Name("word " * 20))
    lines = pprint.pformat(captured, width=40).splitlines()
    # pdb's ``pp`` splits a long copy as it would a str, under its type name.
    assert len(lines) > 1
    assert lines[0].startswith(f"{__name__}._Name('word ")
    assert lines[-1].endswith("')")


def test_scalar_subclasses_have_the_budgets_of_exact_values():
    context = SnapshotContext(DumpOptions(max_string_bytes=4))
    name, exact, wide, blob, count, small = context.value(
        [
            _Name("abcdef"),
            _Name("abcd"),
            _Name("饺子"),
            _Blob(b"abcdef"),
            _Count(1 << 40),
            _Count(1 << 20),
        ]
    )
    assert (name, exact, wide, blob, small) == ("abcd", "abcd", "饺", b"abcd", 1 << 20)
    assert type(wide) is FakeStr and type(blob) is FakeBytes
    assert repr(count) == f"<{__name__}._Count omitted: max_string_bytes>"
    assert context.metadata()["truncation"] == {"max_string_bytes": 4}
    # The value is a leaf, kept at any depth like an exact one.
    assert SnapshotContext(DumpOptions(max_depth=1)).value([_Name("deep")]) == ["deep"]
    # Only the budget is copied, never the whole string.
    huge = _Name("x" * (16 << 20))
    tracemalloc.start()
    try:
        SnapshotContext(DumpOptions()).value(huge)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 512 * 1024


def test_scalar_subclass_slots_follow_the_depth_budget():
    class Text(str):
        __slots__ = ("child",)

    root = node = Text("level0")
    for index in range(1, 1000):
        node.child = Text(f"level{index}")
        node = node.child
    context = SnapshotContext(DumpOptions(max_depth=3))
    captured = context.value({"root": root})["root"]
    # Each value is kept like a plain string's; the slot chain is nested data
    # and stops at the depth budget instead of overflowing the interpreter.
    assert captured == "level0" and repr(captured).endswith("Text('level0')")
    assert captured.child == "level1" and captured.child.child == "level2"
    assert "child" not in vars(captured.child.child)
    assert context.metadata()["truncation"] == {"max_depth": 1}


@pytest.mark.parametrize("max_depth", [1, 5])
def test_str_subclass_attribute_name_runs_no_comparison(tmp_path, max_depth):
    class AttributeName(str):
        def __eq__(self, other):
            raise ValueError("application __eq__ called")

        def __ne__(self, other):
            raise ValueError("application __ne__ called")

        __hash__ = str.__hash__

    class Label(str):
        pass

    path = tmp_path / "name.dump"

    def business():
        important = 42
        label = Label("x")
        # setattr keeps the str subclass as the attribute's name.
        setattr(label, AttributeName("owner"), 42)
        options = DumpOptions(max_frames=1, max_depth=max_depth)
        dump_current_traceback(path, options=options, strict=True)
        return important, label

    business()
    result = load_dumpling(path)
    # At the depth boundary the attribute is dropped, below it kept; neither
    # path may call the name's comparison and fail the whole capture.
    assert result["snapshot_meta"]["attempts"] == []
    variables = result["traceback"].tb_frame.f_locals
    assert variables["important"] == 42 and variables["label"] == "x"
    if max_depth == 1:
        assert "owner" not in vars(variables["label"])
        assert result["snapshot_meta"]["truncation"]["max_depth"]
    else:
        assert vars(variables["label"])["owner"] == 42


def test_scalar_subclasses_survive_a_dump_without_application_hooks(tmp_path):
    calls = []

    def hook(name):
        def called(self, *args):
            calls.append(name)
            raise AssertionError(f"application {name} called")

        return called

    class Text(str):
        __getitem__, __len__, __iter__ = hook("getitem"), hook("len"), hook("iter")
        __str__, __repr__, __format__ = hook("str"), hook("repr"), hook("format")
        __eq__, __hash__ = hook("eq"), hook("hash")

    class Data(bytes):
        __getitem__, __len__, __bytes__ = hook("getitem"), hook("len"), hook("bytes")

    class Number(int):
        __int__, __index__, __float__ = hook("int"), hook("index"), hook("float")
        bit_length, __repr__ = hook("bit_length"), hook("repr")

    class Fraction(float):
        __float__, __int__, __repr__ = hook("float"), hook("int"), hook("repr")

    class Failure(Exception):
        __str__ = hook("exception str")

    path = tmp_path / "scalars.dump"
    text = Text("word " * 20)
    text.owner = "team-a"
    # Captured as frame locals.
    _data, _number, _fraction = Data(b"xy"), Number(7), Fraction(0.5)
    _color, _mode = _Color.RED, _Mode.FAST
    try:
        raise Failure(Text("bad input"))
    except Failure:
        calls.clear()
        save_dumping(path, options=DumpOptions(max_frames=1), strict=True)
    assert calls == []
    result = load_dumpling(path)
    # A snapshot of scalar copies needs nothing more than plain pickle.
    assert result["dump_type"] == "PICKLE"
    assert result["snapshot_meta"]["attempts"] == []
    variables = result["traceback"].tb_frame.f_locals
    assert variables["text"] == "word " * 20 and variables["text"].owner == "team-a"
    assert variables["_data"] == b"xy" and variables["_fraction"] == 0.5
    assert variables["_number"] == 7
    assert repr(variables["_number"]).endswith(".Number(7)")
    assert (variables["_color"], variables["_color"]._name_) == (1, "RED")
    assert (variables["_mode"], variables["_mode"]._name_) == ("fast", "FAST")
    # As str(exc) would, the message is the argument's text.
    assert result["exc_extra"]["message"] == "bad input"


def test_plain_dicts_keep_keys_named_like_snapshot_bookkeeping(tmp_path):
    path = tmp_path / "business.dump"

    def business():
        payload = {"_FakeClass__repr": "r", "_FakeContainer__type": "t", "answer": 42}
        dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
        return payload

    business()
    result = load_dumpling(path)
    # Only a snapshot copy's own attributes are bookkeeping; business keys stay.
    assert result["traceback"].tb_frame.f_locals["payload"] == {
        "_FakeClass__repr": "r",
        "_FakeContainer__type": "t",
        "answer": 42,
    }
    assert result["snapshot_meta"]["attempts"] == []


def test_global_names_used_by_the_frame_come_first(tmp_path):
    path = tmp_path / "priority.dump"
    namespace = {"__name__": "business"}
    namespace.update((f"unused{index}", index) for index in range(5))
    namespace.update(
        zeta=1,
        path=path,
        dump_current_traceback=dump_current_traceback,
        DumpOptions=DumpOptions,
    )
    exec(
        "def business():\n"
        "    dump_current_traceback(\n"
        "        path, options=DumpOptions(max_frames=1, max_items=5), strict=True\n"
        "    )\n"
        "    return zeta\n",
        namespace,
    )
    namespace["business"]()
    names = set(load_dumpling(path)["traceback"].tb_frame.f_globals)
    # With room for five globals, the names the code reads win over the rest,
    # and ``__builtins__`` is never captured.
    assert names == {
        "__name__",
        "dump_current_traceback",
        "path",
        "DumpOptions",
        "zeta",
    }


@pytest.mark.skipif(
    not hasattr(builtins, "frozendict"), reason="frozendict is new in Python 3.15"
)
def test_frozendict_is_captured_as_a_mapping():
    import dill

    class Settings(builtins.frozendict):
        def items(self):
            raise AssertionError("application items called")

    context = SnapshotContext(DumpOptions())
    captured = context.value(
        {
            "plain": builtins.frozendict(a=1, nested=[2]),
            "settings": Settings(b=2),
            "members": {builtins.frozendict(k=1)},
        }
    )
    # Every copy is a FakeDict, so the dump also loads before Python 3.15.
    assert type(captured["plain"]) is FakeDict
    assert repr(captured["plain"]) == "frozendict({'a': 1, 'nested': [2]})"
    # Entries are read through frozendict.items, never the subclass override.
    assert repr(captured["settings"]).endswith("Settings({'b': 2})")
    (member,) = captured["members"]
    assert type(member) is FakeDict and member == {"k": 1}
    assert context.metadata()["truncation"] == {}
    for serializer in (pickle, dill):
        restored = serializer.loads(serializer.dumps(captured))
        assert repr(restored["plain"]) == repr(captured["plain"])


def test_deep_value_does_not_cost_the_rest_of_the_frame(tmp_path):
    path = tmp_path / "recursion.dump"

    def business():
        deep = current = []
        for _ in range(3000):
            nested = []
            current.append(nested)
            current = nested
        important = 42
        dump_current_traceback(
            path, options=DumpOptions(max_frames=1, max_depth=None), strict=True
        )
        return deep, important

    business()
    result = load_dumpling(path)
    metadata = result["snapshot_meta"]
    # (dill may still give up on the deep list; pickle then writes it.)
    assert all("serializer" in attempt for attempt in metadata["attempts"])
    assert metadata["truncation"]["recursion"] == 1
    assert "concurrent_mutation" not in metadata["truncation"]
    variables = result["traceback"].tb_frame.f_locals
    # The deep value is summarized; the other locals stay.
    assert variables["deep"].reason == "recursion"
    assert variables["important"] == 42


def test_recursion_overflow_in_a_nested_mapping_is_not_a_concurrent_mutation():
    deep = current = {}
    for _ in range(3000):
        current["next"] = current = {}
    context = SnapshotContext(DumpOptions(max_depth=None))
    captured = context.value({"deep": deep, "kept": 1})
    # The root mapping recovers; nothing is misfiled as concurrent mutation.
    assert captured["kept"] == 1 and captured["deep"].reason == "recursion"
    truncation = context.metadata()["truncation"]
    assert truncation["recursion"] == 1
    assert "concurrent_mutation" not in truncation


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows cannot remove the working directory"
)
def test_capture_survives_a_deleted_working_directory(tmp_path, monkeypatch):
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    namespace = {}
    exec(
        compile(
            "import sys\ndef f():\n    return sys._getframe()\n", "relative.py", "exec"
        ),
        namespace,
    )
    frame = namespace["f"]()
    os.rmdir(gone)
    path = tmp_path / "relative.dump"
    _capture_frame(frame, path, options=DumpOptions(max_frames=1))
    code = load_dumpling(path)["traceback"].tb_frame.f_code
    # A relative path beats losing the whole snapshot.
    assert code.co_filename == "relative.py"


class _MetadataProbe:
    pass


def a_very_long_function_name_for_metadata():
    return None


def test_code_metadata_is_not_limited_by_max_string_bytes():
    context = SnapshotContext(DumpOptions(max_string_bytes=2))
    code = context.code(a_very_long_function_name_for_metadata.__code__)
    assert code.co_name == "a_very_long_function_name_for_metadata"
    assert code.co_filename.endswith("test_snapshot.py")
    assert "code_metadata" not in context.metadata()["truncation"]
    # Application strings still honour the configured budget.
    assert context.value("abcdef") == "ab"
    assert context.metadata()["truncation"]["max_string_bytes"] == 1
    assert repr(context.value(_MetadataProbe())) == (
        f"<{__name__}._MetadataProbe snapshot>"
    )


def test_code_metadata_has_its_own_visible_bounds():
    code = (lambda: None).__code__
    oversized_text = code.replace(co_filename="f" * (_CODE_TEXT_LIMIT + 10))
    oversized_bytes = code.replace(co_code=b"\x00" * (_CODE_BYTES_LIMIT + 2))
    for original in (oversized_text, oversized_bytes):
        context = SnapshotContext(DumpOptions(max_string_bytes=8))
        context.code(original)
        truncation = context.metadata()["truncation"]
        # Bounded by the metadata limit, reported, and not charged to the
        # application's string budget.
        assert truncation["code_metadata"] == 1
        assert "max_string_bytes" not in truncation
        # Unlimited capture disables every budget, including these.
        unlimited = SnapshotContext(DumpOptions.unlimited())
        captured = unlimited.code(original)
        assert captured.co_filename.endswith(original.co_filename)
        assert captured.co_code == original.co_code
        assert "code_metadata" not in unlimited.metadata()["truncation"]


def test_capture_runs_no_application_hooks(tmp_path):
    calls = []

    class Meta(type):
        def __getattribute__(cls, name):
            if name == "__mro__":
                calls.append("metaclass __mro__")
            return type.__getattribute__(cls, name)

        # Type checks compare by identity: ``x in tuple`` would call __eq__
        # and a set lookup would call __hash__.
        def __eq__(cls, other):
            calls.append("metaclass __eq__")
            return type.__eq__(cls, other)

        def __hash__(cls):
            calls.append("metaclass __hash__")
            return type.__hash__(cls)

    class Proxy(metaclass=Meta):
        def __init__(self):
            self.answer = 42

        @property
        def __class__(self):
            calls.append("__class__")
            raise AssertionError("application __class__ read")

    class Items(list, metaclass=Meta):
        __slots__ = ("owner",)

        def __iter__(self):
            calls.append("__iter__")
            return iter(self[:1])

    class Planted:
        __dict__ = Proxy()

    class HostileError(Exception):
        @property
        def __class__(self):
            calls.append("exception __class__")
            raise AssertionError("application __class__ read")

    path = tmp_path / "hostile.dump"
    items = Items([1, 2, 3])
    items.owner = "team-a"
    _proxy, _planted = Proxy(), Planted()  # Captured as frame locals.
    try:
        raise HostileError("hostile-boom")
    except HostileError:
        calls.clear()
        save_dumping(path, options=DumpOptions(max_frames=1), strict=True)
    assert calls == []
    result = load_dumpling(path)
    variables = result["traceback"].tb_frame.f_locals
    assert variables["_proxy"].answer == 42
    # Entries are read through the base type, so the override cannot hide any.
    assert variables["items"] == [1, 2, 3] and variables["items"].owner == "team-a"
    planted = variables["_planted"]
    assert vars(planted) == {"_FakeClass__repr": repr(planted)}
    assert "hostile-boom" in "".join(result["exc_extra"]["formatted"])


def test_frame_variable_names_survive_a_tiny_string_budget(tmp_path):
    path = tmp_path / "names.dump"

    def business():
        application_string = "value"
        dump_current_traceback(
            path,
            options=DumpOptions(max_frames=1, max_string_bytes=4),
            strict=True,
        )
        return application_string

    business()
    variables = load_dumpling(path)["traceback"].tb_frame.f_locals
    # The name is an identifier; the value is application data.
    assert variables["application_string"] == "valu"
    assert "appl" not in variables


def test_global_names_survive_a_local_holding_globals(tmp_path):
    path = tmp_path / "globals.dump"
    namespace = {
        "dump_current_traceback": dump_current_traceback,
        "DumpOptions": DumpOptions,
        "path": path,
    }
    exec(
        "application_global = 'value'\n"
        "def business():\n"
        "    namespace = globals()\n"
        "    dump_current_traceback(\n"
        "        path, options=DumpOptions(max_frames=1, max_string_bytes=4),\n"
        "        strict=True,\n"
        "    )\n",
        namespace,
    )
    namespace["business"]()
    frame = load_dumpling(path)["traceback"].tb_frame
    # The local reaches the namespace first; its names must still survive.
    assert frame.f_globals["application_global"] == "valu"
    assert "appl" not in frame.f_globals
    assert frame.f_locals["namespace"] is frame.f_globals


def test_group_check_never_reads_an_overridden_class(monkeypatch):
    import pydumpling.snapshot as snapshot_module

    class HostileError(Exception):
        @property
        def __class__(self):
            raise AssertionError("application __class__ read")

    monkeypatch.setattr(
        snapshot_module,
        "_BUILTIN_EXCEPTIONS",
        {**snapshot_module._BUILTIN_EXCEPTIONS, id(HostileError): HostileError},
    )
    context = SnapshotContext(DumpOptions())
    extra = context.exception(HostileError("hostile-boom"))
    assert "hostile-boom" in "".join(extra["formatted"])
    # Reading __class__ while copying the exception would be swallowed into
    # exception_state.  (Only the stdlib formatter reads the copy's class,
    # which a real builtin exception cannot override.)
    assert "exception_state" not in context.metadata()["truncation"]
