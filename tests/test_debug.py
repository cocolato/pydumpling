import gzip
import importlib
import inspect
import io
import pdb
import pickle
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import dill
import pytest

from pydumpling import debug_dumpling, load_dumpling
from pydumpling.debug_dumpling import DumpFormatError, mock_inspect
from pydumpling.fake_types import FakeCode, FakeFrame, FakeTraceback


def _legacy_positions():
    return iter(((1, 1, None, None),))


def make_dumpling():
    """A small historical-style snapshot without references to live code."""
    code = FakeCode.__new__(FakeCode)
    code.__dict__.update(
        co_filename=str(Path(__file__).resolve()),
        co_name="snapshot_frame",
        co_argcount=1,
        co_kwonlyargcount=0,
        co_posonlyargcount=0,
        co_varnames=("value",),
        co_flags=0,
        co_consts=(),
        co_code=b"\x00\x00",
        co_firstlineno=1,
        co_lnotab=b"",
        _co_lines=[(0, 2, 1)],
        _co_positions=((1, 1, None, None),),
    )
    if not hasattr(FakeCode, "co_positions"):
        code.co_positions = _legacy_positions
    frame = FakeFrame.__new__(FakeFrame)
    frame.__dict__.update(
        f_code=code,
        f_locals={"value": 42},
        f_globals={"__name__": "snapshot"},
        f_lineno=1,
        f_back=None,
        f_lasti=0,
        f_builtins={},
    )
    traceback = FakeTraceback()
    traceback.tb_frame = frame
    traceback.tb_lineno = 1
    traceback.tb_lasti = 0
    return {"traceback": traceback, "version": "0.1.6"}


def write_dumpling(tmp_path, dumpling=None, serializer=pickle.dumps):
    payload = serializer(make_dumpling() if dumpling is None else dumpling)
    filename = tmp_path / "snapshot.dump"
    filename.write_bytes(gzip.compress(payload))
    return filename, len(payload)


@pytest.mark.parametrize("serializer", [pickle.dumps, dill.dumps])
def test_load_legacy_serializers_and_read_limit(tmp_path, serializer):
    filename, size = write_dumpling(tmp_path, serializer=serializer)
    assert load_dumpling(filename, max_bytes=size)["traceback"].tb_frame.f_locals == {
        "value": 42
    }
    assert load_dumpling(filename, max_bytes=None)["version"] == "0.1.6"
    with pytest.raises(DumpFormatError, match="read limit"):
        load_dumpling(filename, max_bytes=size - 1)


@pytest.mark.parametrize("version", [-1, 2, True, "1", None])
def test_reject_unsupported_snapshot_format(tmp_path, version):
    snapshot = make_dumpling()
    snapshot["format_version"] = version
    filename, _ = write_dumpling(tmp_path, snapshot)
    with pytest.raises(DumpFormatError, match="snapshot format version"):
        load_dumpling(filename)


@pytest.mark.parametrize("version", ["0.0.0", "1.0.0", "2.0.0"])
def test_current_format_is_independent_of_package_version(tmp_path, version):
    snapshot = make_dumpling()
    snapshot.update(format_version=1, version=version)
    filename, _ = write_dumpling(tmp_path, snapshot)
    assert load_dumpling(filename)["traceback"].tb_frame.f_locals["value"] == 42


@pytest.mark.parametrize("format_version", [0, 1])
def test_package_version_must_still_be_well_formed(tmp_path, format_version):
    snapshot = make_dumpling()
    snapshot.update(format_version=format_version, version="invalid version")
    filename, _ = write_dumpling(tmp_path, snapshot)
    with pytest.raises(DumpFormatError, match="Invalid dumpling version"):
        load_dumpling(filename)


def test_explicit_legacy_format_keeps_historical_package_range(tmp_path):
    snapshot = make_dumpling()
    snapshot.update(format_version=0, version="1.0.0")
    filename, _ = write_dumpling(tmp_path, snapshot)
    with pytest.raises(DumpFormatError, match="Unsupported dumpling version"):
        load_dumpling(filename)


def test_pickle_fallback_starts_at_byte_zero(tmp_path, monkeypatch):
    loader = importlib.import_module("pydumpling.debug_dumpling")
    filename, _ = write_dumpling(tmp_path)

    def failed_dill(payload):
        stream = io.BytesIO(payload)
        stream.read(1)
        raise ValueError("failed after consuming input")

    monkeypatch.setattr(loader.dill, "loads", failed_dill)
    assert load_dumpling(filename)["traceback"].tb_frame.f_locals["value"] == 42


@pytest.mark.parametrize("max_bytes", [None, 64 * 1024 * 1024])
@pytest.mark.parametrize("failure", ["truncated", "plain_text", "deflate", "crc"])
def test_reject_invalid_gzip(tmp_path, failure, max_bytes):
    filename, _ = write_dumpling(tmp_path)
    payload = bytearray(filename.read_bytes())
    if failure == "truncated":
        payload = payload[:-1]
    elif failure == "plain_text":
        payload = b"not gzip"
    elif failure == "deflate":
        payload[10] = 0x07  # Invalid DEFLATE block type.
    else:
        payload[-8] ^= 1  # Corrupt the gzip checksum.
    filename.write_bytes(payload)
    with pytest.raises(DumpFormatError, match="gzip") as error:
        load_dumpling(filename, max_bytes=max_bytes)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "failure",
    ["version", "format", "frame", "traceback_cycle", "frame_cycle", "corrupt"],
)
def test_reject_invalid_snapshot(tmp_path, failure):
    dumpling = make_dumpling()
    if failure == "version":
        dumpling["version"] = "2.0.0"
    elif failure == "format":
        dumpling["dump_type"] = "UNKNOWN"
    elif failure == "frame":
        dumpling["traceback"].tb_frame = None
    elif failure == "traceback_cycle":
        dumpling["traceback"].tb_next = dumpling["traceback"]
    elif failure == "frame_cycle":
        frame = dumpling["traceback"].tb_frame
        frame.f_back = frame
    filename, _ = write_dumpling(tmp_path, dumpling)
    if failure == "corrupt":
        filename.write_bytes(gzip.compress(b"not a pickle"))
    with pytest.raises(DumpFormatError):
        load_dumpling(filename)


@pytest.mark.parametrize("finish", ["q", "c"])
def test_real_pdb_snapshot_and_inspect_restoration(tmp_path, finish):
    filename, _ = write_dumpling(tmp_path)
    output = io.StringIO()
    original = inspect.isframe, inspect.istraceback, inspect.iscode

    def post_mortem(traceback):
        debugger = pdb.Pdb(
            stdin=io.StringIO(f"p value\nargs\nwhere\nlist\n{finish}\n"),
            stdout=output,
            readrc=False,
        )
        debugger.reset()
        debugger.interaction(None, traceback)

    def trace(frame, event, arg):
        return trace

    original_trace = sys.gettrace()
    original_sigint = signal.getsignal(signal.SIGINT)
    try:
        sys.settrace(trace)
        local_trace = sys._getframe().f_trace
        debug_dumpling(filename, SimpleNamespace(post_mortem=post_mortem))
        assert sys.gettrace() is trace
        assert sys._getframe().f_trace is local_trace
        assert signal.getsignal(signal.SIGINT) is original_sigint
    finally:
        sys.settrace(original_trace)
    assert "42" in output.getvalue()
    assert "value = 42" in output.getvalue()
    assert "snapshot_frame" in output.getvalue()
    assert (inspect.isframe, inspect.istraceback, inspect.iscode) == original


def test_legacy_position_table_can_be_printed(tmp_path, capsys):
    from pydumpling.helpers import print_traceback_and_except

    dumpling = make_dumpling()
    frame = dumpling["traceback"].tb_frame
    frame.f_code.__dict__.pop("_co_positions", None)
    frame.f_code.__dict__.pop("co_positions", None)
    frame.f_code.co_code = b"\x00\x00" * 11
    frame.f_lasti = dumpling["traceback"].tb_lasti = 20
    filename, _ = write_dumpling(tmp_path, dumpling)
    print_traceback_and_except(load_dumpling(filename))
    assert "snapshot_frame" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error", [RuntimeError("debugger failed"), KeyboardInterrupt()]
)
def test_inspect_restored_when_debugger_raises(tmp_path, error):
    filename, _ = write_dumpling(tmp_path)
    original = inspect.isframe, inspect.istraceback, inspect.iscode

    def post_mortem(traceback):
        assert inspect.istraceback(traceback)
        with mock_inspect():
            assert inspect.isframe(traceback.tb_frame)
        sys.settrace(None)
        raise error

    def trace(frame, event, arg):
        return trace

    original_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(type(error)):
            debug_dumpling(filename, SimpleNamespace(post_mortem=post_mortem))
        assert sys.gettrace() is trace
    finally:
        sys.settrace(original_trace)
    assert (inspect.isframe, inspect.istraceback, inspect.iscode) == original


def test_failed_load_does_not_patch_inspect(tmp_path):
    original = inspect.isframe, inspect.istraceback, inspect.iscode
    with pytest.raises(FileNotFoundError):
        debug_dumpling(tmp_path / "missing.dump")
    assert (inspect.isframe, inspect.istraceback, inspect.iscode) == original
