"""Round trips and failures that previously produced corrupt or misleading dumps."""

import gzip
import inspect
import os
import subprocess
import sys
import traceback

import pytest

from pydumpling import DumpOptions, dump_current_traceback, load_dumpling, save_dumping
from pydumpling.pydumpling import _capture_frame


def test_exception_round_trip(tmp_path):
    def crash():
        shared = {"answer": 42}
        left = right = shared
        numerator = 7
        return numerator / 0, left, right

    path = tmp_path / "exception.dump"
    try:
        crash()
    except ZeroDivisionError:
        assert (
            save_dumping(path, options=DumpOptions(max_frames=2), strict=True) is None
        )
    result = load_dumpling(path)
    tb = result["traceback"]
    assert tb.tb_next.tb_frame.f_code.co_name == "crash"
    variables = tb.tb_next.tb_frame.f_locals
    assert variables["numerator"] == 7
    assert variables["left"] is variables["right"]
    assert isinstance(result["exc_extra"]["exc_value"], ZeroDivisionError)
    assert tb.tb_next.tb_frame.f_back is tb.tb_frame
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "scenario", ["missing", "rename", "import", "syntax", "bounded", "minimal"]
)
def test_builtin_exception_diagnostics_survive_save_load_and_print(
    tmp_path, monkeypatch, capsys, scenario
):
    from pydumpling.helpers import print_traceback_and_except

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "diagnostic.dump"
    options = DumpOptions(max_frames=1)
    if scenario == "bounded":
        options = DumpOptions(
            max_frames=1,
            max_depth=1,
            max_items=1,
            max_nodes=1,
            max_string_bytes=32,
        )
    elif scenario == "minimal":
        options = DumpOptions(max_frames=1, max_dump_bytes=8192)
    # Force the minimal mode through the real serialized byte budget.
    _values = (
        {str(index): "x" * 4096 + str(index) for index in range(100)}
        if scenario == "minimal"
        else {}
    )
    try:
        if scenario == "rename":
            os.rename("missing-source.txt", "missing-destination.txt")
        elif scenario == "import":
            exec("from os import __pydumpling_missing_symbol")
        elif scenario == "syntax":
            compile("result = (\n", "broken.py", "exec")
        else:
            filename = (
                "oversized-" + "x" * 300 if scenario == "bounded" else "missing.txt"
            )
            with open(filename):
                pytest.fail("missing file unexpectedly existed")
    except (OSError, ImportError, SyntaxError) as error:
        if scenario == "import":
            error.__dict__.update({"field" + str(index): index for index in range(100)})
        expected_message = str(error)
        expected_display = "".join(traceback.format_exception_only(type(error), error))
        expected = {
            field: getattr(error, field)
            for field in (
                "filename",
                "filename2",
                "winerror",
                "name",
                "path",
                "lineno",
                "offset",
                "end_lineno",
                "end_offset",
                "text",
            )
            if getattr(error, field, None) is not None
        }
        save_dumping(path, options=options, strict=True)
    result = load_dumpling(path)
    extra = result["exc_extra"]
    if scenario == "minimal":
        assert result["snapshot_meta"]["truncation"]["minimal_snapshot"]
        assert "missing.txt" in "".join(extra["formatted"])
    else:
        saved = extra["exc_value"]
        for field, value in expected.items():
            if scenario == "bounded" and type(value) is str:
                value = value.encode("utf-8")[:32].decode("utf-8")
            assert getattr(saved, field) == value
        if scenario == "bounded":
            assert result["snapshot_meta"]["truncation"]["max_string_bytes"]
        else:
            assert str(saved) == expected_message
    print_traceback_and_except(result)
    output = capsys.readouterr().out
    if scenario == "bounded":
        assert "oversized-" + "x" * 22 in output
    else:
        assert expected_display in output


@pytest.mark.parametrize("minimal", [False, True])
@pytest.mark.parametrize("kind", ["value", "syntax", "group"])
def test_unrenderable_exception_survives_save_load_and_print(
    tmp_path, capsys, minimal, kind
):
    from pydumpling.helpers import print_traceback_and_except

    previous_limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    path = tmp_path / "unrenderable.dump"

    def crash():
        _marker = "preserved-local"
        _values = {str(i): "x" * 4096 + str(i) for i in range(100)} if minimal else {}
        error = ValueError(10**5000)
        if kind == "syntax":
            error = SyntaxError("bad syntax", ("broken.py", 10**5000, 1, "x"))
        elif kind == "group":
            error = ExceptionGroup("batch", [error, RuntimeError("useful sibling")])
        raise error

    try:
        try:
            crash()
        except Exception:
            save_dumping(
                path,
                options=DumpOptions(
                    max_frames=1, max_dump_bytes=8192 if minimal else 16 * 1024 * 1024
                ),
                strict=True,
            )
        restored = load_dumpling(path)
        frame = restored["traceback"].tb_frame
        assert frame.f_code.co_name == "crash"
        truncation = restored["snapshot_meta"]["truncation"]
        assert truncation["exception_format"]
        if minimal:
            assert truncation["serialization_fallback"] == 1
        else:
            assert frame.f_locals["_marker"] == "preserved-local"
            assert "serialization_fallback" not in truncation
        print_traceback_and_except(restored)
        output = capsys.readouterr().out
        assert "crash" in output
        assert ("SyntaxError" if kind == "syntax" else "ValueError") in output
        if kind == "group":
            assert "useful sibling" in output
    finally:
        sys.set_int_max_str_digits(previous_limit)


def test_current_stack_includes_caller_and_default_file_is_in_cwd(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def business():
        marker = "business-local"
        dump_current_traceback(options=DumpOptions(max_frames=1), strict=True)
        return marker

    assert business() == "business-local"
    (path,) = list(tmp_path.glob("*.dump"))
    result = load_dumpling(path)
    frame = result["traceback"].tb_frame
    assert frame.f_code.co_name == "business"
    assert frame.f_locals["marker"] == "business-local"
    assert result["traceback"].tb_next is None
    assert result["snapshot_meta"]["truncation"]


def test_partial_dill_failure_restarts_pickle(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    live = {"answer": 42}
    captures = []
    original = writer.SnapshotContext.from_frame

    def capture(context, frame):
        captures.append(frame)
        return original(context, frame)

    def partial_write(payload, output, **kwargs):
        output.write(b"BROKEN")
        live["answer"] = 99
        raise TypeError("dill failed after writing")

    monkeypatch.setattr(writer.SnapshotContext, "from_frame", capture)
    monkeypatch.setattr(writer.dill, "dump", partial_write)
    path = tmp_path / "fallback.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    assert gzip.decompress(path.read_bytes()).startswith(b"\x80")
    restored = load_dumpling(path)
    assert restored["dump_type"] == "PICKLE"
    assert restored["traceback"].tb_frame.f_locals["live"]["answer"] == 42
    assert len(captures) == 1
    assert restored["snapshot_meta"]["attempts"] == [
        {"stage": "full", "serializer": "dill", "error": "TypeError"}
    ]


@pytest.mark.parametrize("failure", ["serialization", "replace", "budget"])
def test_failure_preserves_existing_output(tmp_path, monkeypatch, failure):
    import pydumpling.pydumpling as writer

    path = tmp_path / "existing.dump"
    path.write_bytes(b"existing snapshot")
    settings = DumpOptions(max_frames=1)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    if failure == "serialization":
        monkeypatch.setattr(writer.dill, "dump", fail)
        monkeypatch.setattr(writer.pickle, "dump", fail)
    elif failure == "replace":
        monkeypatch.setattr(writer.os, "replace", fail)
    else:
        settings = DumpOptions(max_frames=1, max_dump_bytes=1)
    with pytest.raises((OSError, ValueError)):
        dump_current_traceback(path, options=settings, strict=True)
    assert path.read_bytes() == b"existing snapshot"
    assert list(tmp_path.iterdir()) == [path]
    with pytest.warns(RuntimeWarning):
        dump_current_traceback(path, options=settings)
    assert path.read_bytes() == b"existing snapshot"


def test_byte_budget_produces_valid_reduced_snapshot(tmp_path):
    def business():
        values = {str(i): "x" * 4096 for i in range(100)}
        return _capture_frame(
            sys._getframe(),
            tmp_path / "small.dump",
            options=DumpOptions(max_frames=1, max_dump_bytes=8192),
        ), values

    path, _ = business()
    assert len(gzip.decompress(path.read_bytes())) <= 8192
    result = load_dumpling(path)
    assert result["traceback"].tb_frame.f_code.co_name == "business"
    assert result["snapshot_meta"]["truncation"]["serialization_fallback"] == 1


def test_byte_budget_reduction_preserves_values_and_original_capture(
    tmp_path, monkeypatch
):
    import pydumpling.pydumpling as writer

    live = {"answer": 42}
    original = writer._serialize

    def serialize(payload, serializer, limit):
        live["answer"] = 99
        return original(payload, serializer, limit)

    monkeypatch.setattr(writer, "_serialize", serialize)

    def business():
        marker = live
        alias = marker
        values = {str(i): "x" * 4096 + str(i) for i in range(100)}
        return (
            _capture_frame(
                sys._getframe(),
                tmp_path / "reduced.dump",
                options=DumpOptions(max_frames=1, max_dump_bytes=180000),
            ),
            values,
            alias,
        )

    path, _, _ = business()
    result = load_dumpling(path)
    variables = result["traceback"].tb_frame.f_locals
    assert variables["marker"]["answer"] == 42
    assert variables["marker"] is variables["alias"]
    assert len(variables["values"]) == 50
    assert len(gzip.decompress(path.read_bytes())) <= 180000
    metadata = result["snapshot_meta"]
    assert "minimal_snapshot" not in metadata["truncation"]
    assert metadata["truncation"]["max_string_bytes"] >= 100
    assert metadata["limits"]["max_items"] == 100
    assert metadata["effective_limits"]["max_items"] == 50
    assert metadata["attempts"] == [
        {"stage": stage, "serializer": serializer, "error": "DumpTooLargeError"}
        for stage in ("full", "globals", "outer")
        for serializer in ("dill", "pickle")
    ]
    assert result["format_version"] == 1


def test_capture_failure_still_attempts_minimal_snapshot(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    original = writer.SnapshotContext.from_frame
    modes = []

    def capture(context, frame):
        modes.append(context.minimal)
        if not context.minimal:
            raise RuntimeError("capture failed")
        return original(context, frame)

    monkeypatch.setattr(writer.SnapshotContext, "from_frame", capture)
    path = tmp_path / "minimal.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    result = load_dumpling(path)
    assert modes == [False, True]
    assert result["snapshot_meta"]["attempts"] == [
        {"stage": "full", "error": "RuntimeError"}
    ]
    assert result["snapshot_meta"]["truncation"]["minimal_snapshot"] == 1


def test_reduction_failure_falls_back_without_recapturing(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    serialize = writer._serialize
    reduce = writer.reduce_snapshot
    capture = writer.SnapshotContext.from_frame
    captures = []

    def capture_once(context, frame):
        captures.append(context.minimal)
        return capture(context, frame)

    def exceed_budget(payload, serializer, limit):
        if "effective_limits" not in payload["snapshot_meta"]:
            raise writer.DumpTooLargeError("injected byte limit")
        return serialize(payload, serializer, limit)

    def fail_reduction(payload, options, *, stage):
        if stage == "reduced":
            raise RuntimeError("injected reduction failure")
        return reduce(payload, options, stage=stage)

    monkeypatch.setattr(writer, "_serialize", exceed_budget)
    monkeypatch.setattr(writer, "reduce_snapshot", fail_reduction)
    monkeypatch.setattr(writer.SnapshotContext, "from_frame", capture_once)
    path = tmp_path / "reduction-fallback.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    result = load_dumpling(path)
    assert captures == [False]
    assert result["traceback"].tb_frame.f_locals == {}
    assert result["snapshot_meta"]["attempts"][-1] == {
        "stage": "reduced",
        "error": "RuntimeError",
    }


def test_invalid_capture_and_missing_exception_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="active exception"):
        save_dumping(tmp_path / "missing.dump", strict=True)
    with pytest.warns(RuntimeWarning, match="active exception"):
        save_dumping(tmp_path / "missing.dump")
    assert not (tmp_path / "missing.dump").exists()
    with pytest.raises(ValueError, match="frame"):
        _capture_frame(None, tmp_path / "missing.dump")
    with pytest.raises(TypeError, match="DumpOptions"):
        dump_current_traceback(tmp_path / "missing.dump", options={}, strict=True)


@pytest.mark.parametrize("bulky_scope", ["globals", "outer", "code"])
def test_byte_budget_keeps_fault_locals_before_other_scopes(tmp_path, bulky_scope):
    from types import SimpleNamespace

    from pydumpling.pydumpling import _write_snapshot
    from pydumpling.snapshot import SnapshotContext

    bulk = {str(i): str(i) + "x" * 4000 for i in range(30)}
    outer = SimpleNamespace(
        f_code=(lambda: None).__code__,
        f_locals=bulk if bulky_scope == "outer" else {},
        f_globals=bulk if bulky_scope == "globals" else {},
        f_lineno=1,
        f_lasti=0,
        f_back=None,
    )
    request = {"user": {"id": 42}}
    inner = SimpleNamespace(
        f_code=(lambda request: None).__code__,
        f_locals={"request": request, "alias": request},
        f_globals=outer.f_globals,
        f_lineno=1,
        f_lasti=0,
        f_back=outer,
    )
    options = DumpOptions(max_frames=2, max_dump_bytes=8192)
    context = SnapshotContext(options)
    captured = context.from_frame(inner)
    if bulky_scope == "code":
        captured.tb_next.tb_frame.f_code.co_consts = (b"x" * 20000,)
    payload = {
        "traceback": captured,
        "version": "0.2.0",
        "snapshot_meta": context.metadata(),
    }
    path = _write_snapshot(tmp_path / "priority.dump", lambda minimal: payload, options)
    restored = load_dumpling(path)
    frame = restored["traceback"].tb_next.tb_frame
    assert frame.f_locals["request"]["user"]["id"] == 42
    assert frame.f_locals["request"] is frame.f_locals["alias"]
    assert frame.f_back is restored["traceback"].tb_frame
    assert not frame.f_globals
    if bulky_scope == "outer":
        assert not frame.f_back.f_locals
    if bulky_scope == "code":
        assert frame.f_code.co_consts == ()
        assert restored["snapshot_meta"]["truncation"]["code_payload_omitted"]
    assert "minimal_snapshot" not in restored["snapshot_meta"]["truncation"]
    # Projection must not mutate the original captured graph.
    if bulky_scope == "code":
        assert captured.tb_next.tb_frame.f_code.co_consts == (b"x" * 20000,)
    else:
        assert (
            captured.tb_frame.f_globals
            if bulky_scope == "globals"
            else captured.tb_frame.f_locals
        )


@pytest.mark.parametrize("force_minimal", [False, True])
def test_large_exception_summary_does_not_prevent_saving(
    tmp_path, capsys, force_minimal
):
    from pydumpling.helpers import print_traceback_and_except

    def crash():
        _marker = "keep-local"
        _values = (
            {str(i): str(i) + "x" * 4000 for i in range(100)} if force_minimal else {}
        )
        raise ExceptionGroup(
            "batch", [ValueError(str(i) + "错" * 4000) for i in range(30)]
        )

    path = tmp_path / "summary.dump"
    try:
        crash()
    except Exception:
        save_dumping(
            path, options=DumpOptions(max_frames=1, max_dump_bytes=8192), strict=True
        )
    restored = load_dumpling(path)
    assert len(gzip.decompress(path.read_bytes())) <= 8192
    assert restored["traceback"].tb_frame.f_code.co_name == "crash"
    assert restored["snapshot_meta"]["truncation"]["exception_summary"]
    if force_minimal:
        assert restored["snapshot_meta"]["truncation"]["minimal_snapshot"]
    else:
        assert restored["traceback"].tb_frame.f_locals["_marker"] == "keep-local"
        summary = traceback.extract_tb(restored["traceback"])[-1]
        assert summary.name == "crash"
        assert summary.lineno > 0 and summary.colno is not None
    extra = restored["exc_extra"]
    fields = [extra["type_name"], extra["message"], *extra["formatted"]]
    assert sum(len(text.encode("utf-8")) for text in fields) <= 2048
    assert len("".join(restored["exc_extra"]["formatted"]).encode("utf-8")) <= 2048
    print_traceback_and_except(restored)
    output = capsys.readouterr().out
    assert "ExceptionGroup" in output and "ValueError" in output
    assert "omitted" in output


def test_exception_with_unserializable_attribute_still_saves(tmp_path, capsys):
    from pydumpling.helpers import print_traceback_and_except

    class FalseyError(Exception):
        def __bool__(self):
            return False

    path = tmp_path / "custom.dump"
    try:
        error = FalseyError("diagnostic message")
        error.generator = (x for x in range(3))
        raise error
    except FalseyError:
        save_dumping(path, options=DumpOptions(max_frames=1), strict=True)
    print_traceback_and_except(load_dumpling(path))
    output = capsys.readouterr().out
    assert "FalseyError" in output
    assert "diagnostic message" in output


def test_explicit_exception_info_and_unlimited_snapshot(tmp_path):
    path = tmp_path / "explicit.dump"
    # An unlimited application snapshot should not capture pytest's own graph.
    script = (
        "import sys\n"
        "from pydumpling import DumpOptions, save_dumping\n"
        "try:\n"
        "    raise ValueError('explicit')\n"
        "except ValueError:\n"
        "    info = sys.exc_info()\n"
        "save_dumping(sys.argv[1], info, options=DumpOptions.unlimited(), strict=True)\n"
    )
    subprocess.run([sys.executable, "-c", script, str(path)], check=True, timeout=15)
    assert str(load_dumpling(path)["exc_extra"]["exc_value"]) == "explicit"


def test_frame_capture_does_not_skip_input(tmp_path):
    frame = inspect.currentframe()
    try:
        path = _capture_frame(
            frame, tmp_path / "frame.dump", options=DumpOptions(max_frames=1)
        )
        assert (
            load_dumpling(path)["traceback"].tb_frame.f_code.co_name
            == "test_frame_capture_does_not_skip_input"
        )
    finally:
        del frame
