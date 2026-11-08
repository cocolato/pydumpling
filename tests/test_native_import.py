"""Importing the optional walker must not require access to Python frames."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pydumpling.snapshot import values as values_module

_IMPORT_SCRIPT = """
import sys

# dill itself inspects a frame during its first import; the regression is the
# additional frame access required only by pydumpling's optional extension.
import dill

def namespace():
    value = [1, 2]
    return sys._getframe().f_locals

source = namespace()

def deny(event, arguments):
    if event == "sys._getframe":
        raise ERROR("frame access denied")

sys.addaudithook(deny)
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

context = SnapshotContext(DumpOptions())
assert context.value([1, 2]) == [1, 2]
walker = context._walker
if NATIVE:
    assert walker is not None
    assert walker.handbacks == 0
else:
    assert walker is None
assert context._mapping(source) == {"value": [1, 2]}
if NATIVE and type(source) is not dict:
    assert walker.handbacks == 1
print("ok")
"""


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("error", ["RuntimeError", "PermissionError"])
def test_import_with_frame_access_denied(native, error):
    if native and values_module._native is None:
        pytest.skip("the optional native walker is unavailable")
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYDUMPLING_NATIVE="1" if native else "0")
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    script = _IMPORT_SCRIPT.replace("ERROR", error).replace("NATIVE", repr(native))
    process = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "ok"


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_frame_probe_preserves_base_exceptions(monkeypatch, error):
    def unavailable():
        raise error("interrupted")

    monkeypatch.setattr(values_module, "sys", SimpleNamespace(_getframe=unavailable))
    with pytest.raises(error, match="interrupted"):
        values_module._frame_locals_type()


def test_frame_probe_does_not_hide_locals_errors(monkeypatch):
    class Frame:
        @property
        def f_locals(self):
            raise RuntimeError("locals failed")

    monkeypatch.setattr(
        values_module, "sys", SimpleNamespace(_getframe=lambda: Frame())
    )
    with pytest.raises(RuntimeError, match="locals failed"):
        values_module._frame_locals_type()
