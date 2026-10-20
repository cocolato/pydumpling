"""Round trips and failures that previously produced corrupt or misleading dumps."""

import gzip
import inspect
import os
import pickle
import re
import subprocess
import sys
import threading
import traceback
from datetime import datetime

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


# Kept early in the module: a later test also drives dill into a failed
# write, and a leaked buffer there aborts the run before this assertion.
def test_failed_dill_attempt_leaves_no_exported_buffer(tmp_path, monkeypatch):
    import gc

    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def crash():
        _values = {str(index): "x" * 4096 + str(index) for index in range(100)}
        raise ValueError("boom")

    try:
        crash()
    except ValueError:
        # Too small for either serializer, so dill, tried last, fails too.
        save_dumping(
            tmp_path / "small.dump",
            options=DumpOptions(max_frames=1, max_dump_bytes=8192),
            strict=True,
        )
    gc.collect()
    # Without clearing the failed frames, collecting dill's kept error frees a
    # still-exported buffer and Python reports an unraisable BufferError.
    assert unraisable == []


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
    captured = capsys.readouterr()
    if scenario == "bounded":
        # Application data still obeys the budget...
        assert "oversized-" + "x" * 22 in captured.out
        # ...while the frame header keeps the real file and function names.
        assert "in test_builtin_exception_diagnostics_survive" in captured.out
    else:
        assert expected_display in captured.out


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


def test_partial_pickle_failure_restarts_dill(tmp_path, monkeypatch):
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
        raise TypeError("pickle failed after writing")

    monkeypatch.setattr(writer.SnapshotContext, "from_frame", capture)
    # A bounded snapshot is written with C pickle first, dill second.
    monkeypatch.setattr(writer.pickle, "dump", partial_write)
    path = tmp_path / "fallback.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    assert gzip.decompress(path.read_bytes()).startswith(b"\x80")
    restored = load_dumpling(path)
    assert restored["dump_type"] == "DILL"
    assert restored["traceback"].tb_frame.f_locals["live"]["answer"] == 42
    assert len(captures) == 1
    assert restored["snapshot_meta"]["attempts"] == [
        {"stage": "full", "serializer": "pickle", "error": "TypeError"}
    ]


# The threading module's globals hold itertools objects, whose pickle support
# Python 3.12+ deprecates, and, under coverage, a tracer dill cannot locate.
@pytest.mark.filterwarnings("ignore:Pickle, copy, and deepcopy:DeprecationWarning")
@pytest.mark.filterwarnings("ignore::dill.PicklingWarning")
def test_unlimited_probe_uses_the_serializer_that_writes(tmp_path, monkeypatch):
    import math
    import threading

    namespace = {
        "__name__": "__main__",
        "dump_current_traceback": dump_current_traceback,
        "math": math,
    }
    exec(
        "def configured(values=(index for index in range(3))):\n"
        "    return values\n"
        "def business(path, options):\n"
        "    kept = 42\n"
        "    handler = configured\n"
        "    module = math\n"
        "    dump_current_traceback(path, options=options, strict=True)\n",
        namespace,
    )
    # pickle finds this function by name, but dill writes __main__ functions
    # by value and fails on the generator default; pickle cannot write the
    # module.  Accepting the function on a pickle probe fails both writers.
    monkeypatch.setattr(
        sys.modules["__main__"], "configured", namespace["configured"], raising=False
    )
    path = tmp_path / "probe.dump"
    # A fresh thread keeps the unlimited capture free of test-runner objects.
    thread = threading.Thread(
        target=namespace["business"], args=(path, DumpOptions.unlimited())
    )
    thread.start()
    thread.join()
    restored = load_dumpling(path, max_bytes=None)
    innermost = restored["traceback"]
    while innermost.tb_next is not None:
        innermost = innermost.tb_next
    assert restored["snapshot_meta"]["attempts"] == []
    assert innermost.tb_frame.f_locals["kept"] == 42
    assert innermost.tb_frame.f_locals["module"] is math


def test_bounded_snapshot_of_awkward_objects_is_written_with_pickle(tmp_path):
    import threading

    path = tmp_path / "awkward.dump"

    def business():
        class Local:
            pass

        callback = lambda value: value  # noqa: E731 - pickle rejects it
        generator = (index for index in range(3))
        lock = threading.Lock()
        instance = Local()
        try:
            raise ValueError(callback)
        except ValueError:
            save_dumping(path, options=DumpOptions(max_frames=1), strict=True)
        return generator, lock, instance

    business()
    restored = load_dumpling(path)
    # Capture leaves no original object behind, so fast pickle always fits.
    assert restored["dump_type"] == "PICKLE"
    assert restored["snapshot_meta"]["attempts"] == []


def test_dumps_are_compressed_at_level_six(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    levels = []
    gzip_file = writer.gzip.GzipFile

    def recording(*args, **kwargs):
        levels.append(kwargs.get("compresslevel", 9))
        return gzip_file(*args, **kwargs)

    monkeypatch.setattr(writer.gzip, "GzipFile", recording)
    path = tmp_path / "level.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    assert levels == [6]


@pytest.mark.parametrize(
    "options, dump_type",
    [(DumpOptions(), "PICKLE"), (DumpOptions.unlimited(), "DILL")],
)
def test_serializer_order_follows_the_capture_mode(tmp_path, options, dump_type):
    import pydumpling.pydumpling as writer

    path = tmp_path / "order.dump"
    # Both serializers can write this payload, so the first one tried wins: a
    # bounded snapshot goes to fast pickle, unlimited original objects to dill.
    writer._write_snapshot(
        path, lambda minimal: {"snapshot_meta": {"truncation": {}}}, options
    )
    assert pickle.loads(gzip.decompress(path.read_bytes()))["dump_type"] == dump_type


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
    original = writer._stream_dump

    def stream_dump(payload, filename, serializer, limit):
        live["answer"] = 99
        return original(payload, filename, serializer, limit)

    monkeypatch.setattr(writer, "_stream_dump", stream_dump)

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
        for serializer in ("pickle", "dill")
    ]
    assert result["format_version"] == 2


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

    stream_dump = writer._stream_dump
    reduce = writer.reduce_snapshot
    capture = writer.SnapshotContext.from_frame
    captures = []

    def capture_once(context, frame):
        captures.append(context.minimal)
        return capture(context, frame)

    def exceed_budget(payload, filename, serializer, limit):
        if "effective_limits" not in payload["snapshot_meta"]:
            raise writer.DumpTooLargeError("injected byte limit")
        return stream_dump(payload, filename, serializer, limit)

    def fail_reduction(payload, options, *, stage):
        if stage == "reduced":
            raise RuntimeError("injected reduction failure")
        return reduce(payload, options, stage=stage)

    monkeypatch.setattr(writer, "_stream_dump", exceed_budget)
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


def test_tiny_string_budget_keeps_frame_identifiers(tmp_path):
    def crash():
        return 1 / 0

    path = tmp_path / "identifiers.dump"
    try:
        crash()
    except ZeroDivisionError:
        save_dumping(
            path, options=DumpOptions(max_frames=1, max_string_bytes=2), strict=True
        )
    result = load_dumpling(path)
    frame = result["traceback"].tb_frame
    # Code metadata is diagnostic structure, not a captured string value.
    assert frame.f_code.co_name == "crash"
    assert frame.f_code.co_filename.endswith("test_dump.py")
    # User-facing text still obeys the budget.
    assert result["snapshot_meta"]["truncation"]["max_string_bytes"]


def test_repeated_default_saves_never_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for _ in range(3):
        try:
            raise ValueError("example")
        except ValueError:
            save_dumping(strict=True)
    names = sorted(path.name for path in tmp_path.glob("*.dump"))
    assert len(names) == 3
    # Names carry the source line, a UTC timestamp, and the pid, so repeated
    # failures keep every snapshot.
    pattern = re.compile(
        rf"test_dump\.py-\d+-\d{{8}}T\d{{6}}\.\d{{3}}-{re.escape(str(os.getpid()))}"
        r"-\d{6}\.dump"
    )
    assert all(pattern.fullmatch(name) for name in names)


def test_default_filename_for_the_current_stack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dump_current_traceback(strict=True)
    assert len(list(tmp_path.glob("*.dump"))) == 1


def test_saved_metadata_records_process_context(tmp_path):
    path = tmp_path / "meta.dump"
    try:
        raise ValueError("example")
    except ValueError:
        save_dumping(path, strict=True)
    metadata = load_dumpling(path)["snapshot_meta"]
    assert metadata["pid"] == os.getpid()
    assert metadata["created_at"].endswith("Z")
    datetime.fromisoformat(metadata["created_at"])
    assert metadata["thread_name"] == threading.current_thread().name
    assert type(metadata["duration_ms"]) is int
    assert metadata["duration_ms"] >= 0
    assert metadata["cwd"] == os.getcwd()
    if len(sys.argv) <= 256:
        assert metadata["argv"] == sys.argv


def test_unwritable_target_fails_before_capturing(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    captures = []
    capture = writer.SnapshotContext.from_frame

    def record(context, frame):
        captures.append(context.minimal)
        return capture(context, frame)

    monkeypatch.setattr(writer.SnapshotContext, "from_frame", record)
    with pytest.raises(FileNotFoundError):
        dump_current_traceback(tmp_path / "missing" / "x.dump", strict=True)
    assert captures == []


def test_commit_failure_is_not_retried_as_a_serialization_failure(
    tmp_path, monkeypatch
):
    import pydumpling.pydumpling as writer

    serializers = []
    stream_dump = writer._stream_dump

    def record(payload, output, serializer, limit):
        serializers.append(serializer)
        return stream_dump(payload, output, serializer, limit)

    def refuse(source, target):
        raise PermissionError("read-only target")

    monkeypatch.setattr(writer, "_stream_dump", record)
    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError, match="read-only target"):
        dump_current_traceback(
            tmp_path / "x.dump", options=DumpOptions(max_frames=1), strict=True
        )
    assert serializers == ["pickle"]
    assert list(tmp_path.iterdir()) == []


def test_reduction_counts_argv_entries_once(tmp_path, monkeypatch):
    from pydumpling.snapshot.context import _ARGV_LIMIT

    monkeypatch.setattr(sys, "argv", ["prog"] * (_ARGV_LIMIT + 5))

    def business():
        values = ["x" * 4096 + str(index) for index in range(64)]
        return _capture_frame(
            sys._getframe(),
            tmp_path / "argv.dump",
            options=DumpOptions(max_frames=1, max_dump_bytes=60000),
        ), values

    path, _ = business()
    metadata = load_dumpling(path)["snapshot_meta"]
    assert "effective_limits" in metadata  # A value reduction ran.
    assert metadata["truncation"]["argv_entries"] == 5


LONG_BASENAMES = ["a" * 217 + ".py", "é" * 118 + ".py"]


@pytest.mark.parametrize("basename", LONG_BASENAMES, ids=["ascii", "multibyte"])
def test_generated_names_leave_room_for_the_temporary_name(basename):
    from pydumpling.pydumpling import _dump_target

    for lineno in (None, 1, 10**9):
        name = _dump_target(basename, lineno).name
        # mkstemp adds "." before and ".XXXXXXXX" after the dump name.
        assert len(name.encode("utf-8")) + len("..") + 8 <= 255
        assert name.startswith(basename[:20])
        name.encode("utf-8")  # Cut on a character boundary.


@pytest.mark.skipif(
    sys.platform == "win32", reason="MAX_PATH also bounds the full temporary path"
)
@pytest.mark.parametrize("basename", LONG_BASENAMES, ids=["ascii", "multibyte"])
def test_long_source_names_still_get_a_default_dump(tmp_path, monkeypatch, basename):
    monkeypatch.chdir(tmp_path)
    try:
        exec(compile("raise ValueError('long name')", basename, "exec"))
    except ValueError:
        save_dumping(strict=True)
    (path,) = tmp_path.glob("*.dump")
    code = load_dumpling(path)["traceback"].tb_next.tb_frame.f_code
    assert code.co_filename.endswith(basename)
