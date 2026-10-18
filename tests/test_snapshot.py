import os
import pickle
import sys
import traceback
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from pydumpling import dump_current_traceback, load_dumpling
from pydumpling.fake_types import FakeClass, TruncatedValue
from pydumpling.options import DumpOptions
from pydumpling.pydumpling import _capture_frame
from pydumpling.snapshot import _CODE_BYTES_LIMIT, _CODE_TEXT_LIMIT, SnapshotContext


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
