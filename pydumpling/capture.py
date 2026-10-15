"""Request bounded snapshots through CPython's public remote execution API."""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .options import DumpOptions
from .pydumpling import Pathish, _options


class CaptureTimeoutError(TimeoutError):
    """The request is still pending; its script must remain available."""

    def __init__(self, pid: int, directory: Path):
        self.request_directory = directory
        super().__init__(
            f"Timed out waiting for PID {pid}. The request may still execute. "
            f"Retained request files at {directory}; remove them only after "
            "status.json confirms completion or the target process has exited."
        )


class CaptureInterruptedError(KeyboardInterrupt):
    """Waiting was interrupted without cancelling the queued target script."""

    def __init__(self, pid: int, directory: Path):
        self.request_directory = directory
        super().__init__(
            f"Stopped waiting for PID {pid}; the request may still execute. "
            f"Retained request files at {directory}; remove them only after "
            "status.json confirms completion or the target process has exited."
        )


def _request_script(request: dict) -> str:
    # repr is applied only to our own JSON string, never to captured objects.
    return f"""import sys
_business_frame = sys._getframe(1)
import json
import os
_request = json.loads({json.dumps(request)!r})
_status = {{"request_id": _request["request_id"]}}
try:
    from pydumpling.options import DumpOptions
    from pydumpling.pydumpling import _capture_frame
    _output = _capture_frame(
        _business_frame,
        _request["output"],
        options=DumpOptions(**_request["options"]),
    )
    _status.update(status="success", output=str(_output))
except BaseException as _error:
    _status.update(
        status="error",
        error=type(_error).__name__ + ": " + str(_error)[:4096],
    )
finally:
    del _business_frame
    _temporary = _request["status"] + ".tmp"
    _fd = os.open(_temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(_fd, "w", encoding="utf-8") as _handle:
        json.dump(_status, _handle)
        _handle.flush()
        os.fsync(_handle.fileno())
    os.replace(_temporary, _request["status"])
"""


def capture_pid(
    pid: int,
    filename: Pathish,
    *,
    options: DumpOptions | None = None,
    timeout: float = 30.0,
) -> Path:
    """Save a running process's main-thread stack and wait for confirmation.

    Both interpreters must have compatible CPython 3.14+ runtimes. The target
    must be able to import pydumpling and access the same filesystem paths.
    A timeout stops waiting; it does not cancel remote execution.
    """
    if type(pid) is not int or pid <= 0 or pid == os.getpid():
        raise ValueError("pid must be the positive PID of another process")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")
    if sys.implementation.name != "cpython" or not callable(
        getattr(sys, "remote_exec", None)
    ):
        raise RuntimeError("PID capture requires CPython 3.14+ with sys.remote_exec")
    settings = _options(options)
    target = Path(filename).absolute()
    if not target.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {target.parent}")
    if target.exists() and not target.is_file():
        raise ValueError(f"Output is not a regular file: {target}")
    directory = Path(tempfile.mkdtemp(prefix="pydumpling-capture-"))
    status_path = directory / "status.json"
    script_path = directory / "capture.py"
    request_id = uuid.uuid4().hex
    request = {
        "request_id": request_id,
        "output": str(target),
        "status": str(status_path),
        "options": asdict(settings),
    }
    try:
        fd = os.open(script_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as script:
            script.write(_request_script(request))
        try:
            sys.remote_exec(pid, str(script_path))
        except KeyboardInterrupt:
            raise CaptureInterruptedError(pid, directory) from None
    except CaptureInterruptedError:
        raise
    except BaseException:
        shutil.rmtree(directory)
        raise

    try:
        return _await_capture(pid, target, directory, status_path, request_id, timeout)
    except KeyboardInterrupt as error:
        raise CaptureInterruptedError(pid, directory) from error


def _await_capture(
    pid: int,
    target: Path,
    directory: Path,
    status_path: Path,
    request_id: str,
    timeout: float,
) -> Path:
    deadline = time.monotonic() + timeout
    # Keep the script available until a completion acknowledgement is observed.
    while True:
        if status_path.is_file():
            try:
                with status_path.open(encoding="utf-8") as status_file:
                    status = json.load(status_file)
                if (
                    not isinstance(status, dict)
                    or status.get("request_id") != request_id
                ):
                    raise RuntimeError("Invalid capture completion response")
                if status.get("status") == "error":
                    raise RuntimeError(
                        f"PID {pid} could not save a snapshot: {status.get('error', 'unknown error')}"
                    )
                if (
                    status.get("status") != "success"
                    or status.get("output") != str(target)
                    or not target.is_file()
                ):
                    raise RuntimeError(
                        "Capture completion did not confirm the requested output"
                    )
                return target
            finally:
                shutil.rmtree(directory)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureTimeoutError(pid, directory)
        time.sleep(min(0.05, remaining))
