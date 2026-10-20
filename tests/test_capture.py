"""Native integration plus acknowledgement and timeout contract regressions."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from pydumpling import DumpOptions, capture_pid, load_dumpling
from pydumpling.capture import CaptureInterruptedError, CaptureTimeoutError


def test_pid_input_and_capability_errors(tmp_path, monkeypatch):
    for pid in (0, -1, True, os.getpid()):
        with pytest.raises(ValueError, match="pid"):
            capture_pid(pid, tmp_path / "capture.dump")
    for timeout in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="timeout"):
            capture_pid(1, tmp_path / "capture.dump", timeout=timeout)
    monkeypatch.setattr(sys, "remote_exec", None, raising=False)
    with pytest.raises(RuntimeError, match="CPython 3.14"):
        capture_pid(1, tmp_path / "capture.dump")


def test_acknowledged_target_failure_cleans_request(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    paths = []

    def reject(*args, **kwargs):
        raise OSError("target cannot save")

    def remote(pid, script):
        paths.append(Path(script))
        exec(compile(Path(script).read_text(), script, "exec"), {})

    monkeypatch.setattr(writer, "_capture_frame", reject)
    monkeypatch.setattr(sys, "remote_exec", remote, raising=False)
    with pytest.raises(RuntimeError, match="target cannot save"):
        capture_pid(1, tmp_path / "capture.dump")
    assert not paths[0].parent.exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_timeout_retains_script_until_late_completion(
    tmp_path, monkeypatch, interrupted
):
    paths = []

    def remote(pid, script):
        paths.append(Path(script))
        if interrupted:
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "remote_exec", remote, raising=False)
    output = tmp_path / "late.dump"
    try:
        error_type = CaptureInterruptedError if interrupted else CaptureTimeoutError
        with pytest.raises(error_type, match="may still execute") as caught:
            capture_pid(1, output, timeout=0.01, options=DumpOptions(max_frames=1))
        directory = caught.value.request_directory
        assert paths[0].parent == directory and paths[0].exists()
        assert not output.exists()
        exec(compile(paths[0].read_text(), str(paths[0]), "exec"), {})
        assert load_dumpling(output)["traceback"].tb_frame is not None
        assert (directory / "status.json").is_file()
    finally:
        if paths:
            shutil.rmtree(paths[0].parent, ignore_errors=True)


@pytest.mark.skipif(
    not callable(getattr(sys, "remote_exec", None)),
    reason="native capture requires CPython 3.14+",
)
def test_native_pid_capture_preserves_business_frame(tmp_path):
    target = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import os, time\n"
                "def business():\n"
                "    marker = 'native-business-local'\n"
                "    print(os.getpid(), flush=True)\n"
                "    while True:\n"
                "        time.sleep(0.01)\n"
                "business()\n"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    target_pid = None
    try:
        # Windows venv executables launch a child that owns the Python runtime.
        target_pid = int(target.stdout.readline().strip())
        path = capture_pid(
            target_pid,
            tmp_path / "native.dump",
            options=DumpOptions(max_frames=1),
            timeout=10,
        )
        result = load_dumpling(path)
        frame = result["traceback"].tb_frame
        assert frame.f_code.co_name == "business"
        assert frame.f_locals["marker"] == "native-business-local"
        assert result["traceback"].tb_next is None
    finally:
        if target_pid is not None and target_pid != target.pid:
            os.kill(target_pid, signal.SIGTERM)
        target.terminate()
        target.communicate(timeout=10)


@pytest.mark.parametrize(
    "options, expected",
    [(None, {}), (DumpOptions(max_frames=1), {"max_frames": 1})],
)
def test_request_sends_only_options_that_differ_from_defaults(
    tmp_path, monkeypatch, options, expected
):
    import pydumpling.options as options_module

    received = []

    class TargetOptions(DumpOptions):
        # An older target rejects fields it does not know, such as new ones.
        def __init__(self, **kwargs):
            received.append(kwargs)
            super().__init__(**kwargs)

    def remote(pid, script):
        exec(compile(Path(script).read_text(), script, "exec"), {})

    monkeypatch.setattr(options_module, "DumpOptions", TargetOptions)
    monkeypatch.setattr(sys, "remote_exec", remote, raising=False)
    output = capture_pid(1, tmp_path / "request.dump", options=options, timeout=10)
    assert received == [expected]
    assert load_dumpling(output)["traceback"] is not None
