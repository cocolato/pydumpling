import pickle
import sys
import traceback
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from pydumpling.fake_types import FakeClass, TruncatedValue
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext


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
