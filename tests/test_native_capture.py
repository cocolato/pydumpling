"""Public writer and reader contracts for both installations of pydumpling."""

import json
import os
import subprocess
import sys

import pytest

from pydumpling import DumpOptions, dump_current_traceback, load_dumpling, save_dumping
from pydumpling.pydumpling import DumpTooLargeError
from pydumpling.snapshot import _capture


def _leaf(shared, ticket, mode, target, options):
    # The fields exercise a Python-only object inside an otherwise builtin graph.
    assert ticket.payload is shared
    if mode == "exception":
        try:
            raise LookupError("inventory empty")
        except LookupError as cause:
            raise ValueError("missing SKU") from cause
    dump_current_traceback(target, options=options, strict=True)


def _outer(shared, ticket, mode, target, options):
    try:
        _leaf(shared, ticket, mode, target, options)
    except ValueError as error:
        if mode != "exception" or type(error) is not ValueError:
            raise
        save_dumping(target, options=options, strict=True)


def _cli(reader, action, dump, commands=None):
    # A separately installed reader must never resolve the writer checkout.
    isolated = ["-I"] if reader != sys.executable else []
    proc = subprocess.run(
        [reader, *isolated, "-m", "pydumpling", action, str(dump)],
        text=True,
        input=commands,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize("mode", ["exception", "stack"])
def test_public_capture_round_trip(mode, tmp_path, monkeypatch):
    observed = []
    captures = []
    if engine := _capture.pydumpling_native:
        capture = engine.capture

        def record(context, program):
            captures.append(None)
            return capture(context, program)

        monkeypatch.setattr(engine, "capture", record)

    class Name(str):
        def __radd__(self, prefix):
            observed.append("name")
            return prefix + str(self)

    class Ticket:
        def __init__(self, payload):
            self.payload = payload

        @property
        def dangerous(self):
            observed.append("property")
            raise AssertionError("capture called a property")

    # Python's object-to-snapshot conversion observes the name once. A second
    # visit to the same object in another frame must reuse its captured copy.
    Ticket.__qualname__ = Name("Ticket")
    shared = {"answer": 42, "rows": list(range(20))}
    shared["self"] = shared
    ticket = Ticket(shared)
    dump = tmp_path / f"{mode}.dump"
    _outer(shared, ticket, mode, dump, DumpOptions(max_frames=2, max_items=8))

    if engine:
        assert len(captures) == 1
    assert observed == ["name"]
    result = load_dumpling(dump)
    outer = result["traceback"]
    inner = outer.tb_next
    assert inner.tb_next is None
    assert inner.tb_frame.f_back is outer.tb_frame
    variables = inner.tb_frame.f_locals
    assert variables["ticket"] is outer.tb_frame.f_locals["ticket"]
    assert variables["ticket"].payload is variables["shared"]
    assert variables["shared"] is outer.tb_frame.f_locals["shared"]
    assert variables["shared"]["self"] is variables["shared"]
    assert variables["shared"]["rows"] == list(range(8))
    assert result["snapshot_meta"]["truncation"]["max_items"] >= 1
    assert result["snapshot_meta"]["attempts"] == []

    readers = [sys.executable]
    if baseline := os.environ.get("PYDUMPLING_BASE_READER"):
        readers.append(baseline)
    for reader in readers:
        report = json.loads(_cli(reader, "--json", dump))
        assert [frame["qualname"] for frame in report["frames"]] == ["_outer", "_leaf"]
        assert report["frames"][-1]["lineno"] == inner.tb_lineno
        if mode == "exception":
            assert report["exception"]["message"] == "missing SKU"
            assert "LookupError: inventory empty" in report["exception"]["formatted"]
        else:
            assert report["exception"] is None

        output = _cli(
            reader,
            "--debug",
            dump,
            "bt\npp ('inner', ticket.payload is shared)\n"
            "up\np ('outer', shared['answer'])\nquit\n",
        )
        assert "_leaf()" in output and "_outer()" in output
        assert "('inner', True)" in output
        assert "('outer', 42)" in output


@pytest.mark.parametrize("mode", ["exception", "stack"])
@pytest.mark.parametrize("failure", ["budget", "permission"])
@pytest.mark.parametrize("strict", [False, True])
def test_failure_keeps_the_previous_dump(mode, failure, strict, tmp_path, request):
    target = tmp_path / "previous.dump"
    dump_current_traceback(target, options=DumpOptions(max_frames=1), strict=True)
    previous = target.read_bytes()
    options = DumpOptions(max_frames=1)
    if failure == "permission":
        previous_mode = tmp_path.stat().st_mode
        request.addfinalizer(lambda: tmp_path.chmod(previous_mode))
        tmp_path.chmod(0o500)
        if os.access(tmp_path, os.W_OK):
            pytest.skip("filesystem permissions allow the process to write here")
        expected = PermissionError
    else:
        options = DumpOptions(max_frames=1, max_dump_bytes=1)
        expected = DumpTooLargeError
    try:
        raise ValueError("original")
    except ValueError as original:
        behavior = pytest.raises(expected) if strict else pytest.warns(RuntimeWarning)
        with behavior:
            write = dump_current_traceback if mode == "stack" else save_dumping
            write(target, options=options, strict=strict)
        assert sys.exception() is original
    assert target.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("limit", ["max_nodes", "max_depth"])
def test_fault_frame_respects_shared_capture_budgets(limit, tmp_path):
    dump = tmp_path / "bounded.dump"
    options = DumpOptions(max_frames=1, **{limit: 3 if limit == "max_nodes" else 2})

    def capture(samples):
        dump_current_traceback(dump, options=options, strict=True)

    capture([101, 102, 103] if limit == "max_nodes" else [[["leaf"]]])
    result = load_dumpling(dump)
    assert result["snapshot_meta"]["truncation"][limit] >= 1
    saved = result["traceback"].tb_frame.f_locals["samples"]
    if limit == "max_nodes":
        assert saved == [101]
    else:
        assert saved[0].reason == "max_depth"


def test_unlimited_capture_uses_python_from_the_start(tmp_path):
    # The stack belongs to the small application alone. An unlimited capture
    # of a pytest stack would retain the test runner and all its plugins.
    app = """
import sys
from pathlib import Path
from pydumpling import DumpOptions, dump_current_traceback, load_dumpling, save_dumping
from pydumpling.snapshot import _capture

def unexpected(*args):
    import pytest
    pytest.fail("an unlimited capture entered the native engine")

if _capture.pydumpling_native is not None:
    _capture.pydumpling_native.capture = unexpected
out = Path(sys.argv[1])
options = DumpOptions.unlimited()
def business():
    amount = 42
    dump_current_traceback(out / "stack.dump", options=options, strict=True)
    try:
        raise ValueError("no stock")
    except ValueError:
        save_dumping(out / "exception.dump", options=options, strict=True)
business()
for name in ("stack", "exception"):
    result = load_dumpling(out / (name + ".dump"))
    assert result["snapshot_meta"]["attempts"] == []
    tb = result["traceback"]
    while tb.tb_next:
        tb = tb.tb_next
    assert tb.tb_frame.f_locals["amount"] == 42
"""
    subprocess.run([sys.executable, "-c", app, str(tmp_path)], check=True, timeout=30)
