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

def namespace():
    value = [1, 2]
    return sys._getframe().f_locals

source = namespace()

# A module pydumpling first imports from outside itself may read a frame as it
# loads: dill does, and before 3.12 so does each collections.namedtuple call,
# which uuid makes on Linux by importing platform. Those load first; the
# regression is frame access by pydumpling's own code.
for name in PRELOAD:
    __import__(name)

def deny(event, arguments):
    if DENY and event == "sys._getframe":
        raise ERROR("frame access denied")

loaded = set(sys.modules)
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
    # Only frame access tells the walker the locals proxy's type; without it
    # the walker hands the proxy to the Python walk.
    assert walker.handbacks == int(DENY)

def outside(name):
    # Aliases such as multiprocessing's __mp_main__ load with their module.
    spec = getattr(sys.modules[name], "__spec__", None)
    return name.partition(".")[0] != "pydumpling" and spec and spec.name == name

print(*sorted(filter(outside, set(sys.modules) - loaded)))
print("ok")
"""


def _run_import_script(script, native):
    """Run ``script`` and return the modules from outside pydumpling it loaded."""
    # Built here, not passed in, so that a failure's traceback leaves the
    # environment out.
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYDUMPLING_NATIVE="1" if native else "0")
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    process = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    *_, outside, ok = process.stdout.splitlines()
    assert ok == "ok"
    return outside.split()


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("error", ["RuntimeError", "PermissionError"])
def test_import_with_frame_access_denied(native, error):
    if native and values_module._native is None:
        pytest.skip("the optional native walker is unavailable")
    script = _IMPORT_SCRIPT.replace("ERROR", error).replace("NATIVE", repr(native))
    # A run that allows frame access learns what pydumpling loads from outside
    # itself, which the run that denies it then loads first.
    allowed = script.replace("DENY", "False").replace("PRELOAD", "()")
    outside = _run_import_script(allowed, native)
    denied = script.replace("DENY", "True").replace("PRELOAD", repr(outside))
    _run_import_script(denied, native)


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


_MISMATCH_SCRIPT = """
import sys, types, warnings

native = types.ModuleType("pydumpling._native")

def statics(namespace, identity):
    raise KeyError("values has no _gone")

native.Statics = statics
sys.modules["pydumpling._native"] = native
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from pydumpling.snapshot import SnapshotContext, values
from pydumpling.options import DumpOptions

assert values._native is None
messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
assert len(messages) == 1 and "rust/build_native.py" in messages[0], messages
context = SnapshotContext(DumpOptions())
assert context._walker is None
assert context.value([1, {"a": 2}]) == [1, {"a": 2}]
print("ok")
"""


@pytest.mark.skipif(sys.version_info < (3, 12), reason="3.11 never imports it")
def test_an_extension_that_does_not_match_leaves_the_python_walk():
    """An extension built for another version of ``values``, which lacks a
    name it reads there, is not used: the package still imports, warns once,
    and captures in Python."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYDUMPLING_NATIVE", None)
    environment["PYTHONPATH"] = (
        str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    process = subprocess.run(
        [sys.executable, "-B", "-c", _MISMATCH_SCRIPT],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "ok"
