"""What a snapshot holds does not depend on where objects live or on which walk
took it: the same state dumps to the same bytes in each mode, a dump the native
walker wrote loads and reports without the extension, and a context holds the
same state in both modes."""

import gzip
import json
import os
import subprocess
import sys
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path

import pytest

from pydumpling import DumpOptions, save_dumping
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot import context as context_module
from pydumpling.snapshot import values as values_module

_ROOT = Path(__file__).resolve().parents[1]
_NATIVE = values_module._native is not None

_Pair = namedtuple("_Pair", "left right")


@dataclass
class _Order:
    number: int
    lines: list
    note: str = "none"


class _Slotted:
    __slots__ = ("first", "second")

    def __init__(self, first, second):
        self.first, self.second = first, second


class _Status(Enum):
    OPEN = "open"
    DONE = "done"


class _Level(IntEnum):
    LOW = 1
    HIGH = 2


class _Label(str):
    pass


class _Node:
    def __init__(self, after):
        self.after = after


def _raise(levels, state):
    """Raise ``levels`` calls down, each frame holding part of ``state``."""
    here = state[levels % len(state)]
    if levels:
        return _raise(levels - 1, state)
    raise ValueError("boom", here)


def _failure():
    """An exception raised under frames that hold values of every kind the
    walk copies, built afresh: no set holds a value hashed by address, so
    the copies do not depend on where the objects live."""
    label = _Label("tagged")
    label.owner = "team"
    chain = None
    for _ in range(12):
        chain = _Node(chain)
    rows = [{"id": i, "name": f"row-{i}", "tags": ["ab"] * (i % 3)} for i in range(40)]
    state = [
        rows,
        {"order": _Order(7, [1.5, -2, None]), "slotted": _Slotted(b"ab", (1, 2))},
        [_Pair(1, "x"), _Status.DONE, _Level.HIGH, label, 2**100, "é" * 3000],
        (OrderedDict(b=1, a=2), defaultdict(list, k=[3]), frozenset({1, 2, 3})),
        {"chain": chain, "bytes": bytearray(b"payload" * 900), "nested": [[[[[1]]]]]},
    ]
    try:
        _raise(4, state)
    except ValueError as error:
        return error


class _FrozenClock:
    """Stands in for the context module's ``datetime``: a fixed creation time."""

    @staticmethod
    def now(tz=None):
        return datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc)


def _dumped(path, error):
    """The decompressed payload of a dump of ``error``."""
    exc_info = (type(error), error, error.__traceback__)
    save_dumping(path, exc_info=exc_info, options=DumpOptions(), strict=True)
    return gzip.decompress(path.read_bytes())


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_the_same_state_dumps_to_the_same_bytes(native, tmp_path, monkeypatch):
    """Two dumps of one state, and one of the same state built again from new
    objects, hold the same bytes in either mode. Each state is dumped once
    first: the first capture of an exception creates its instance dict,
    which every later capture then holds. Besides the snapshot a dump holds
    when it was made and how long capture took, frozen here, and the process
    id, host, working directory, command line and thread name, which one
    process shares; its gzip header holds no time and no name."""
    if native and not _NATIVE:
        pytest.skip("the optional native walker is unavailable")
    if not native:
        monkeypatch.setattr(SnapshotContext, "_native_walk", False)
    assert SnapshotContext._native_walk is native
    monkeypatch.setattr(context_module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(context_module, "datetime", _FrozenClock)
    first, rebuilt = _failure(), _failure()
    errors = (first, first, first, rebuilt, rebuilt)
    payloads = [
        _dumped(tmp_path / f"{index}.dump", error) for index, error in enumerate(errors)
    ]
    assert payloads[1] == payloads[2] == payloads[4]
    assert b"row-39" in payloads[1]


_REPORT_SCRIPT = """
import io, json, sys
from pydumpling import load_dumpling
from pydumpling.report import dump_to_dict, print_traceback_and_except
from pydumpling.snapshot import values

dumpling = load_dumpling(sys.argv[1])
text = io.StringIO()
print_traceback_and_except(dumpling, file=text)
view = dump_to_dict(dumpling, include_globals=True)
print(json.dumps([values._native is not None, view, text.getvalue()], default=str))
"""


_BLOCKED = "import sys\nsys.modules['pydumpling._native'] = None\n"


def _report(path, native):
    """What a child interpreter in the given mode reads from the dump."""
    environment = dict(os.environ, PYTHONHASHSEED="0")
    environment["PYDUMPLING_NATIVE"] = "1" if native else "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_ROOT), os.environ.get("PYTHONPATH")))
    )
    # Without the extension: importing it fails, as where it is not built.
    script = _REPORT_SCRIPT if native else _BLOCKED + _REPORT_SCRIPT
    process = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path)],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


@pytest.mark.skipif(not _NATIVE, reason="the optional native walker is unavailable")
def test_a_native_dump_loads_and_reports_without_the_extension(tmp_path):
    """A dump the native walker wrote loads where the extension cannot be
    imported, and reports there exactly as where it is loaded."""
    assert SnapshotContext._native_walk
    path = tmp_path / "native.dump"
    _dumped(path, _failure())
    loaded, view, text = _report(path, native=True)
    assert loaded
    assert _report(path, native=False) == [False, view, text]
    assert "ValueError: ('boom'" in text
    assert view["frames"][-1]["qualname"] == "_raise"


_CONTEXT_STATE_SCRIPT = """
import json
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

found = {}
for name, options in (("limited", DumpOptions()), ("unlimited", DumpOptions.unlimited())):
    context = SnapshotContext(options)
    # Captures through both entries leave no flag behind either.
    context.value({"a": [1, 2], "b": object()})
    context._mapping({"k": [3, {"n": None}]})
    copied = SnapshotContext(DumpOptions()).value(context)
    found[name] = [
        list(vars(context)),
        vars(context).get("_native_walk", "unset"),
        list(vars(copied)),
        vars(copied).get("_native_walk", "unset"),
    ]
print(json.dumps(found))
"""


def test_a_context_holds_the_same_state_in_both_modes():
    """``SnapshotContext`` gives no context ``_native_walk`` of its own,
    limited or unlimited, with or without the extension, so a context
    captured as a value shows the same attributes in both modes."""
    found = []
    for native in ("1", "0"):
        environment = dict(os.environ, PYDUMPLING_NATIVE=native)
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(_ROOT), os.environ.get("PYTHONPATH")))
        )
        process = subprocess.run(
            [sys.executable, "-B", "-c", _CONTEXT_STATE_SCRIPT],
            cwd=_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert process.returncode == 0, process.stderr
        found.append(json.loads(process.stdout))
    assert found[0] == found[1]
    for name in ("limited", "unlimited"):
        assert found[0][name][1] == found[0][name][3] == "unset"
