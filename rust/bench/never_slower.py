"""Gate: a native capture is never slower than the pure-Python walk.

Every scenario runs in fresh child processes, one per mode: native
(``PYDUMPLING_NATIVE`` unset; the extension must load) and pure
(``PYDUMPLING_NATIVE=0``), with ``PYTHONPATH`` set to this checkout and
``PYTHONHASHSEED=0``.  The modes alternate which runs first, for ``--rounds``
rounds.  A child builds its scenario twice.  The first build captures once to
warm up and once for content, while copies hash by content and reprs carry no
addresses, as in tests/test_native_parity.py; the child then hashes the
pickled snapshot, its truncation counters in order, its node count and the
kinds of what ``objects`` and ``_sources`` hold.  The second build warms up
and is timed with ``time.perf_counter`` until ``--min-runs`` runs and
``--min-time`` seconds; the child reports the median microseconds per op.

Judging, per scenario, on the median native/pure ratio of its rounds: at
most 0.99 passes and at least 1.05 (or ``1 + --tolerance``, if larger)
FAILS.  In between, the scenario runs alone for ``--extra-rounds`` (12) more
rounds, the modes in ABBA order (native first, then pure first, ...), and
FAILS if the median over all its rounds exceeds ``1 + --tolerance``.  A
scenario also FAILS when the modes' content differs.  Scenarios marked
near_limit (``~``) are where the pure walk may reach the recursion limit,
and the native walk may then capture more.  A round whose pure child did not
reach it (no ``'recursion'`` tally, no ``RecursionError``) must have the same
content; one whose pure child did is held instead to "never less", a
definition the owner has yet to approve: (i) no root variable both children
captured (an entry of a mapping captured at depth 0, of the state of an
exception copy at the top, or of a captured frame's namespaces) reaches more
``TruncatedValue('recursion')`` placeholders natively, and (ii) the native
child keeps at least as many originals in ``objects`` (not nodes: the pure
walk's ``_nodes`` also counts the copies it dropped where it recovered an
overflow at a root variable).  A node budget the native walk spent on
an earlier variable may cut a later one: a root variable only the pure child
captured is allowed, and a round with one, or with more ``max_nodes`` cuts
natively, is counted as shifted.  The fallbacks column counts the calls the
native walker sent to the Python walk in a whole native child, warm-ups and
timed runs included (``_STATICS.fallbacks``).  The exit status is 1 when a
scenario fails and 2 on a harness error.

``--baseline DIR`` compares this checkout's pure walk with another tree's
instead (for example ``git archive main | tar -x -C DIR``): both children
run with ``PYDUMPLING_NATIVE=0``, one with ``PYTHONPATH`` at DIR, judged by
the same rule, near-limit scenarios on content as any other.

Numbers that count come from a copy of the checkout that nothing edits or
rebuilds meanwhile: ``--isolate DIR`` copies the checkout, its built
extension included (not ``.git``, ``.venv``, ``rust/target``, ``_review``,
``rust/migration``), into DIR and runs the copy's own script with the other
arguments.  Every report records a digest of the tree it ran from.

Example::

    .venv/bin/python rust/bench/never_slower.py --list
    .venv/bin/python rust/bench/never_slower.py --scenarios 'mapping_*' --rounds 1
    .venv/bin/python rust/bench/never_slower.py --python /path/to/python3.14
    .venv/bin/python rust/bench/never_slower.py --isolate /tmp/gate --rounds 3
    .venv/bin/python rust/bench/never_slower.py --baseline /tmp/main
"""

# ruff: noqa: F841

import argparse
import contextlib
import fnmatch
import functools
import gc
import gzip
import hashlib
import json
import os
import pickle
import platform
import reprlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from array import array
from collections import Counter, OrderedDict, defaultdict, namedtuple
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import NamedTuple

# Keep __pycache__ out of rust/bench when this file imports workloads.py.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULT = "never_slower-result: "
# Frames from _at_depth to run(): _descend's last level, the runner callback,
# Runner.__call__, _sample or _content, and run itself.
_RUN_OFFSET = 5
_HANDLER_MARGIN = 30  # how far below the recursion limit the handler captures
_NEAR_ROOM = 25  # frames left above near_limit_nodes's capture
_MISSING = object()
# A sweep's median ratio at or under which a scenario passes, and at or over
# which it fails, without extra rounds.
_CLEAR_PASS, _CLEAR_FAIL = 0.99, 1.05
# Left out of an --isolate copy: history, environments, builds, caches and
# the owner's review notes.
_NOT_COPIED = {
    ".": {".git", ".venv", "venv", "_review", ".code-review-graph", ".codex"},
    "rust": {"target", "migration"},
    "rust/bench": {"results"},
}
_NEVER_COPIED = {"__pycache__", ".pytest_cache", ".ruff_cache", ".coverage"}


# --------------------------------------------------------- values to capture

Point = namedtuple("Point", "x y")


class Level(IntEnum):
    LOW = 1
    HIGH = 2


class Label(str):
    """A str subclass; its copy keeps the type name and attributes."""


class Node:
    def __init__(self, after):
        self.after, self.value = after, 1


@dataclass
class Order:
    number: int
    name: str
    price: float


@dataclass(slots=True)
class Line:
    number: int
    sku: str
    quantity: float


Color = Enum("Color", [f"C{i}" for i in range(300)])
Rank = IntEnum("Rank", [f"R{i}" for i in range(300)])


class Table(Mapping):
    """A ``collections.abc.Mapping`` over a dict of its own."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


# Values of each dense scenario, 300 of each kind, built from an index.
_DENSE = {
    "dataclass": (
        lambda i: Order(i, f"order-{i}", i / 4),
        lambda i: Line(i, f"sku-{i}", i * 1.5),
    ),
    "enum": (lambda i: Color(i + 1), lambda i: Rank(i + 1)),
    "scalar": (
        lambda i: datetime(2026, 1, 1) + timedelta(minutes=i),
        lambda i: Label(f"label-{i}"),
    ),
    "namedtuple": (lambda i: Point(i, -i),),
    "mapping": (
        lambda i: OrderedDict(a=i, b=-i),
        lambda i: defaultdict(list, k=[i]),
        lambda i: Table({"a": i}),
    ),
}
_DENSE_KINDS = {
    "dataclass": "dataclass and slots dataclass instances",
    "enum": "Enum and IntEnum members",
    "scalar": "datetimes and str subclass instances",
    "namedtuple": "namedtuples",
    "mapping": "OrderedDicts, defaultdicts and collections.abc.Mappings",
}


def _typical_frame():
    """A frame with a dozen small locals: dicts, lists, str, ints."""
    config = {"host": "localhost", "port": 8080, "retries": [1, 2, 3]}
    rows = [{"id": i, "name": f"row-{i}"} for i in range(30)]
    title, attempt = "hello world", 3
    return sys._getframe()


def _helper_frame():
    """A frame of values whose capture in Python runs pydumpling's helpers."""
    ordered = OrderedDict(first=1, second=[2, 3], third="three")
    grouped = defaultdict(list, {"a": [1, 2], "b": [3]})
    counts, point, level = Counter("mississippi"), Point(3, 4), Level.HIGH
    label = Label("tagged")
    label.owner = "team"
    text, buffer = "x" * 10_000, bytearray(b"payload-" * 32)
    by_point = {Point(1, 2): "near", Point(5, 6): "far"}
    pairs = {("a", 1), ("b", 2), ("c", 3)}
    return sys._getframe()


# ------------------------------------------------------------ the child side


class Case(NamedTuple):
    """One measured operation.  run() returns (snapshot, context), or what
    snapshot() turns into that pair."""

    run: object
    inner: int = 1  # operations per run
    reset: object = None  # untimed, before every run
    snapshot: object = None
    root: bool = False  # the snapshot is a mapping captured at depth 0


class Skip(Exception):
    """The scenario cannot run under this interpreter."""


class _Box:
    """Slots only: a capture of the stack walks neither the box nor what it
    holds, so what a child keeps here does not change later captures."""

    __slots__ = ("capture",)


def _api():
    """pydumpling's capture API, imported only where a capture runs: no
    frame a scenario captures holds a pydumpling module, which differs."""
    from pydumpling import DumpOptions, dump_current_traceback
    from pydumpling.snapshot import SnapshotContext

    return DumpOptions, SnapshotContext, dump_current_traceback


def _depth():
    """How many Python frames are on the stack, the caller's included."""
    frame, count = sys._getframe(1), 0
    while frame is not None:
        frame, count = frame.f_back, count + 1
    return count


def _descend(levels, fn):
    return fn() if levels <= 0 else _descend(levels - 1, fn)


def _at_depth(depth, fn):
    """Call ``fn``, a runner call, so that each run() executes ``depth`` deep."""
    return _descend(depth - _depth() - _RUN_OFFSET, fn)


def _in_thread(stack_bytes, fn):
    """Run ``fn`` on a new thread with a native stack of ``stack_bytes``."""
    outcome = {}

    def target():
        try:
            outcome["result"] = fn()
        except BaseException as error:  # re-raised on the calling thread
            outcome["error"] = error

    previous = threading.stack_size(stack_bytes)
    try:
        thread = threading.Thread(target=target)
        thread.start()
    finally:
        threading.stack_size(previous)
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


@contextlib.contextmanager
def _normalised():
    """Captures independent of allocation, as the parity tests make them:
    copies hash by content (one set order), reprs carry no address and long
    ints repr whole."""
    from pydumpling.fake_types import FakeClass, FakeFloat, TruncatedValue

    def truncated_hash(self):
        return hash(("TruncatedValue", self.reason, self.type_name))

    def fake_class_hash(self):
        label = self.__dict__.get("_FakeClass__repr")
        return hash(("FakeClass", label if type(label) is str else None))

    def fake_float_hash(self):
        return 0x7FF8 if float.__ne__(self, self) else float.__hash__(self)

    original = reprlib.Repr.repr_instance

    def repr_instance(self, x, level):
        text, name = original(self, x, level), x.__class__.__name__
        at = f"<{name} instance at {id(x):#x}>"
        return f"<{name} instance>" if text == at else text

    patches = [
        (TruncatedValue, "__hash__", truncated_hash),
        (FakeClass, "__hash__", fake_class_hash),
        (FakeFloat, "__hash__", fake_float_hash),
        (reprlib.Repr, "repr_instance", repr_instance),
    ]
    saved = [(cls, name, vars(cls).get(name, _MISSING)) for cls, name, _ in patches]
    digits = sys.get_int_max_str_digits()
    for cls, name, value in patches:
        setattr(cls, name, value)
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(digits)
        for cls, name, value in saved:
            if value is _MISSING:
                delattr(cls, name)
            else:
                setattr(cls, name, value)


def _content(case):
    """Capture once to warm up, then once for content; nothing of the first
    capture is a local while the second runs.  (snapshot, context, root)."""
    with _normalised():
        if case.reset is not None:
            case.reset()
        case.run()
        if case.reset is not None:
            case.reset()
        result = case.run()
        snapshot, context = result if case.snapshot is None else case.snapshot(result)
        return snapshot, context, case.root


def _recursion_placeholders(root):
    """How many ``TruncatedValue('recursion')`` placeholders a captured value
    reaches, each counted once."""
    from pydumpling.fake_types import TruncatedValue

    count, seen, pending = 0, set(), [root]
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if type(item) is TruncatedValue:
            count += vars(item).get("reason") == "recursion"
            continue
        if isinstance(item, dict):
            pending += dict.keys(item)
            pending += dict.values(item)
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending += item
        try:
            state = object.__getattribute__(item, "__dict__")
        except AttributeError:
            continue
        if type(state) is dict:
            pending += state.values()
    return count


def _tops(snapshot):
    """A snapshot and the entries of a top tuple or dict."""
    if type(snapshot) is tuple:
        return [snapshot, *snapshot]
    if type(snapshot) is dict:
        return [snapshot, *dict.values(snapshot)]
    return [snapshot]


def _frame_copies(snapshot):
    """The frame copies of the tracebacks a snapshot holds at its top, or
    in a top tuple or dict, in chain order."""
    from pydumpling.fake_types import FakeFrame, FakeTraceback

    frames = []
    for entry in _tops(snapshot):
        while isinstance(entry, FakeTraceback):
            frame = vars(entry).get("tb_frame")
            while isinstance(frame, FakeFrame) and frame not in frames:
                frames.append(frame)
                frame = vars(frame).get("f_back")
            entry = vars(entry).get("tb_next")
    return frames


def _root_variables(snapshot, root):
    """Each root variable of a capture, by mapping and position, with the
    ``'recursion'`` placeholders it reaches: the entries of the snapshot if
    it is a mapping captured at depth 0, of the state of an exception copy
    that an exception capture at the top holds, and of the frame copies'
    namespaces."""
    roots = [("mapping", snapshot)] if root else []
    for position, entry in enumerate(_tops(snapshot)):
        saved = dict.get(entry, "exc_value") if type(entry) is dict else None
        if isinstance(saved, BaseException):
            roots.append((f"exception {position}", vars(saved)))
    for position, frame in enumerate(_frame_copies(snapshot)):
        for name in ("f_locals", "f_globals"):
            roots.append((f"frame {position} {name}", vars(frame).get(name)))
    found, seen = {}, set()
    for name, mapping in roots:
        # Frames of one module share their globals' copy.
        if isinstance(mapping, dict) and id(mapping) not in seen:
            seen.add(id(mapping))
            for position, item in enumerate(dict.values(mapping)):
                found[f"{name} {position}"] = _recursion_placeholders(item)
    return found


def _digest(snapshot, context, root=False, near_limit=False):
    """What the modes must agree on, as tests/test_native_parity.py compares,
    and near the limit, what "never less" compares."""
    state = {} if context is None else vars(context)
    memo = [
        [type(item).__name__ for item in dict.values(held)]
        if isinstance(held, dict)
        else None
        for held in (state.get("objects"), state.get("_sources"))
    ]
    counters = state.get("_truncation")
    truncation = None if counters is None else list(counters.items())
    record = (snapshot, truncation, state.get("_nodes"), memo)
    try:
        raw = pickle.dumps(record, protocol=5)
    except Exception:  # an unlimited capture may keep what only dill writes
        import dill

        raw = dill.dumps(record, protocol=5, byref=True)
    found = {
        "digest": hashlib.sha256(raw).hexdigest()[:16],
        "snapshot_bytes": len(raw),
        "nodes": state.get("_nodes"),
        "truncation": truncation,
    }
    if near_limit:
        found["roots"] = _root_variables(snapshot, root)
        found["kept"] = None if memo[0] is None else len(memo[0])
        # Whether the walk reached the recursion limit.
        raised = type(snapshot) is str and snapshot == "RecursionError"
        found["overflowed"] = raised or "recursion" in dict(truncation or ())
    return found


def _sample(case, min_runs, min_time):
    """Warm up, then time run() until both floors are met."""
    run, reset, clock = case.run, case.reset, time.perf_counter
    depth = _depth() + 1  # where run() executes
    samples = array("d")  # opaque to a capture of this frame
    for floor_runs, floor_time in ((3, 0.1), (min_runs, min_time)):
        del samples[:]
        total = 0.0
        while len(samples) < floor_runs or total < floor_time:
            if reset is not None:
                reset()
            begin = clock()
            run()
            samples.append(clock() - begin)
            total += samples[-1]
        gc.collect()
    q1, median, q3 = statistics.quantiles(sorted(samples), n=4, method="inclusive")
    return {
        "us_per_op": median / case.inner * 1e6,
        "iqr_pct": 100 * (q3 - q1) / median if median else 0.0,
        "runs": len(samples),
        "inner": case.inner,
        "op_depth": depth,
    }


class Runner:
    """Measures a scenario's cases in a child: content first, then time."""

    def __init__(self, scratch, min_runs, min_time):
        self.scratch, self.min_runs, self.min_time = scratch, min_runs, min_time

    def __call__(self, build):
        box = _Box()
        box.capture = _content(build())
        result = _sample(build(), self.min_runs, self.min_time)
        # Hashed by run_child on the main thread, after any hook is removed.
        result["capture"] = box.capture
        return result


def _capture_build(target, options, context_class, mapping=True):
    """Cases capturing ``target`` on a new context, as ``_mapping`` or value."""

    def build():
        def run():
            context = context_class(options)
            if mapping:
                return context._mapping(target), context
            return context.value(target), context

        return Case(run, root=mapping)

    return build


def _from_caller(context_class, options):
    """Capture the caller's stack; the context is no local of those frames."""
    context = context_class(options)
    return context.from_frame(sys._getframe(1)), context


def _loaded_dump(path):
    """A dump file's payload, less the metadata that differs every run."""
    import dill

    with gzip.open(path, "rb") as stream:
        payload = dill.load(stream)
    for key in ("created_at", "duration_ms", "pid"):
        payload["snapshot_meta"].pop(key)
    return payload, None


# ------------------------------------------------------------------ scenarios


def _mapping_at(depth, runner):
    DumpOptions, SnapshotContext, _ = _api()
    build = _capture_build(_typical_frame().f_locals, DumpOptions(), SnapshotContext)
    return _at_depth(depth, lambda: runner(build))


def _frame_at(depth, dump, runner):
    DumpOptions, SnapshotContext, dump_current_traceback = _api()
    options = DumpOptions()
    path = os.path.join(runner.scratch, "dump_current.dump")

    def run():
        dump_current_traceback(path, options=options, strict=True)
        return path

    def build():
        if dump:
            return Case(run, snapshot=_loaded_dump)
        return Case(lambda: _from_caller(SnapshotContext, options))

    return _at_depth(depth, lambda: runner(build))


def _frame_holding(**values):
    """A suspended generator and its frame, whose locals are ``values`` and
    whose globals hold only ``sys``, so a capture of the frame walks them
    alone.  The frame lives as long as the generator."""
    scope = {"__name__": "held", "sys": sys}
    exec(f"def held({', '.join(values)}):\n    yield sys._getframe()\n", scope)
    generator = scope["held"](**values)
    return generator, next(generator)


def _frame_capture(values, options, runner):
    """from_frame of a frame holding ``values`` (see ``_frame_holding``)."""
    _, SnapshotContext, _ = _api()
    generator, frame = _frame_holding(**values)

    def run():
        context = SnapshotContext(options)
        return context.from_frame(frame), context

    try:
        return runner(lambda: Case(run))
    finally:
        generator.close()


def _dense(kind, runner):
    """300 values of each kind in the frame, in locals that each hold a list
    of 100 (the default max_items), so each is captured, with what it holds
    up to the default max_depth."""
    DumpOptions, _, _ = _api()
    values = [make(i) for make in _DENSE[kind] for i in range(300)]
    chunks = [values[start : start + 100] for start in range(0, len(values), 100)]
    held = {f"chunk{index}": chunk for index, chunk in enumerate(chunks)}
    held["count"] = len(values)
    return _frame_capture(held, DumpOptions(), runner)


# The named tuple types of record_stack, one per frame, made when it runs.
_RECORD_TYPES = []


def _records(level, levels, fn):
    """Descend ``levels`` frames, each holding an instance of a named tuple
    type of its own, then call ``fn``."""
    record = _RECORD_TYPES[level](level, "x")
    return fn() if level + 1 >= levels else _records(level + 1, levels, fn)


def _record_stack(depth, runner):
    """from_frame of a stack ``depth`` deep, every frame captured, whose
    frames each hold an instance of a named tuple type of their own, as
    functions holding their own record types do: each frame's _mapping reads
    a layout ``_layouts`` lacks, beside those of the frames captured before
    it.  The recursion limit is raised to leave the walk room."""
    DumpOptions, SnapshotContext, _ = _api()
    options = DumpOptions(max_frames=depth + 100)
    _RECORD_TYPES[:] = [namedtuple(f"Record{i}", "level name") for i in range(depth)]

    def build():
        return Case(lambda: _from_caller(SnapshotContext, options))

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(depth + 1000)
    try:
        # _records' frames, then the runner callback, Runner.__call__,
        # _sample or _content, and run itself, which executes ``depth`` deep.
        levels = depth - _depth() - _RUN_OFFSET + 1
        return _records(0, levels, lambda: runner(build))
    finally:
        sys.setrecursionlimit(limit)


def _deep_frame(threaded, runner):
    """A frame of a 150-level list, dict chain and object chain, captured at
    max_depth=100: the native walk keeps each level natively on the main
    thread, and on a thread with a 512 KiB stack runs the levels past its
    stack budget in the Python walk in place; the Python walk stays well
    below the recursion limit."""
    DumpOptions, _, _ = _api()
    levels = range(150)
    values = {
        "nested": functools.reduce(lambda inner, _: [inner], levels, []),
        "chained": functools.reduce(lambda inner, _: {"next": inner}, levels, {}),
        "linked": functools.reduce(lambda after, _: Node(after), levels, None),
    }
    capture = functools.partial(_frame_capture, values, DumpOptions(max_depth=100))
    if threaded:
        return _in_thread(512 * 1024, lambda: capture(runner))
    return capture(runner)


def _near_nodes(runner):
    """Near the limit under a binding node budget: the pure walk overflows on
    ``a``, leaves a 'recursion' placeholder and keeps all of ``b``; the
    native walk keeps ``a`` whole, and the budget then cuts ``b``."""
    DumpOptions, SnapshotContext, _ = _api()
    # Ints past the small-int cache: each is a node of its own.
    ints = list(range(1000, 1050))
    nested = functools.reduce(lambda inner, _: [inner], range(40), ints)
    target = {"a": nested, "b": list(range(2000, 2060))}
    options = DumpOptions(max_depth=None, max_nodes=150)
    build = _capture_build(target, options, SnapshotContext)
    return _at_depth(sys.getrecursionlimit() - _NEAR_ROOM, lambda: runner(build))


def _install_hook(kind):
    """Install a trace, profile or monitoring hook; return (detail, remove)."""

    def noop(*args):
        return None

    if kind == "none":
        return None, noop
    if kind in ("settrace", "setprofile"):
        install = getattr(sys, kind)
        install(noop)
        return None, lambda: install(None)
    if kind == "cprofile":
        import cProfile

        profile = cProfile.Profile()
        profile.enable()
        return None, profile.disable
    if kind == "sysmon":
        monitoring = getattr(sys, "monitoring", None)
        if monitoring is None:
            raise Skip("sys.monitoring needs Python 3.12+")
        tool = next(i for i in (5, 4, 3, 0) if monitoring.get_tool(i) is None)
        event = monitoring.events.PY_START
        monitoring.use_tool_id(tool, "never_slower")
        monitoring.register_callback(tool, event, noop)
        monitoring.set_events(tool, event)
        return f"tool {tool}", lambda: monitoring.set_events(tool, 0)
    try:
        import coverage
    except ImportError:
        raise Skip("coverage is not importable") from None
    import pydumpling

    # The package the child imported, this checkout's or a baseline's.
    package = Path(pydumpling.__file__).resolve().parent
    cov = coverage.Coverage(data_file=None, source=[str(package)])
    cov.start()
    tracer = getattr(getattr(cov, "_collector", None), "tracer_name", None)
    return (tracer() if callable(tracer) else None), cov.stop


def _hooked(kind, runner):
    DumpOptions, SnapshotContext, _ = _api()
    build = _capture_build(_helper_frame().f_locals, DumpOptions(), SnapshotContext)
    detail, remove = _install_hook(kind)
    try:
        result = runner(build)
    finally:
        remove()
    result["hook"] = detail
    return result


def _configured(config, mapping, runner):
    from dataclasses import dataclass

    DumpOptions, SnapshotContext, _ = _api()

    @dataclass(frozen=True)
    class CustomOptions(DumpOptions):
        """An options subclass that changes nothing."""

    class OmitContext(SnapshotContext):
        def _omit(self, reason, value=None):
            return super()._omit(reason, value)

    options = {
        "options_subclass": CustomOptions,
        "context_subclass": DumpOptions,
        "unlimited": DumpOptions.unlimited,
    }[config]()
    context_class = OmitContext if config == "context_subclass" else SnapshotContext
    if mapping:
        target = _typical_frame().f_locals
    else:
        target = {f"k{i}": [i, f"v{i}"] for i in range(500)}
    return runner(_capture_build(target, options, context_class, mapping))


def _per_call(kind, runner):
    """Fixed costs per call, in batches: memo hit, _mapping({}), value() of a
    new int, float, short bytes or short str, and value() with no nodes left,
    on a live context, which each run gets afresh, and value() on a new
    context."""
    DumpOptions, SnapshotContext, _ = _api()
    options, live = DumpOptions(), [None]
    if kind == "out_of_nodes":
        options = DumpOptions(max_nodes=1)
    held = {"new_int": 12345, "new_dict": {"id": 7, "tags": ["a", "b"]}}.get(kind)
    held = [1, 2] if held is None else held
    # Distinct values, each a memo miss once: ints past the small-int cache.
    fresh = {
        "live_float": [100_000.5 + step for step in range(200)],
        "live_bytes": [b"bytes-%d" % step for step in range(200)],
        "live_str": [f"text-{step}" for step in range(200)],
    }.get(kind) or [100_000 + step for step in range(200)]

    def reset():
        live[0] = SnapshotContext(options)
        live[0].value(held)

    def memo_hit():
        context = live[0]
        for _ in range(199):
            context.value(held)
        return context.value(held), context

    def live_mapping():
        context = live[0]
        for _ in range(199):
            context._mapping({})
        return context._mapping({}), context

    def fresh_values():
        context = live[0]
        for value in fresh[:-1]:
            context.value(value)
        return context.value(fresh[-1]), context

    def new_context():
        for _ in range(49):
            SnapshotContext(options).value(held)
        context = SnapshotContext(options)
        return context.value(held), context

    if kind.startswith("new"):
        return runner(lambda: Case(new_context, inner=50))
    run = {"memo_hit": memo_hit, "live_mapping": live_mapping}.get(kind, fresh_values)
    return runner(lambda: Case(run, inner=200, reset=reset))


def _subclass_calls(runner):
    """value() of a new list subclass instance, 200 per run, on a live
    context whose ``_layouts`` already holds the layouts of 2,000 classes,
    as after capturing that many builtin subclass types; each run gets the
    context afresh, with a copy of those layouts."""
    DumpOptions, SnapshotContext, _ = _api()
    options, live = DumpOptions(), [None]
    kinds = [type(f"Kind{i}", (list,), {}) for i in range(2000)]
    filled = SnapshotContext(DumpOptions(max_items=None, max_nodes=None))
    filled.value([kind((i,)) for i, kind in enumerate(kinds)])

    class Row(list):
        """The class of the values each call captures."""

    # Ints past the small-int cache: each is a node of its own.
    rows = [Row((100_000 + step,)) for step in range(200)]

    def reset():
        live[0] = SnapshotContext(options)
        live[0]._layouts.update(filled._layouts)

    def run():
        context = live[0]
        for row in rows[:-1]:
            context.value(row)
        return context.value(rows[-1]), context

    return runner(lambda: Case(run, inner=200, reset=reset))


def _deep_data(kind, threaded, runner):
    DumpOptions, SnapshotContext, _ = _api()
    if kind == "list":  # [[[...]]], 2000 levels
        data = functools.reduce(lambda inner, _: [inner], range(2000), [])
    else:  # Node(Node(...)), 300 objects
        data = functools.reduce(lambda after, _: Node(after), range(300), None)
    holder = {"payload": data}
    build = _capture_build(holder, DumpOptions(max_depth=None), SnapshotContext)
    if threaded:
        return _in_thread(512 * 1024, lambda: runner(build))
    return runner(build)


def _runaway(level, stop, then):
    local = {"level": level, "path": [level % 7, level % 11]}
    try:
        return _runaway(level + 1, stop, then)
    except RecursionError as error:
        if level > stop:
            raise
        return then(error)


def _recursion_handler(runner):
    DumpOptions, SnapshotContext, _ = _api()
    options = DumpOptions()

    def then(error):
        def run():
            context = SnapshotContext(options)
            try:
                traceback = context.from_traceback(error.__traceback__)
                return (traceback, context.exception(error)), context
            except RecursionError:  # what save_dumping would reduce
                return "RecursionError", context

        return runner(lambda: Case(run))

    # _runaway(level) runs at depth _depth() + 1 + level.
    stop = sys.getrecursionlimit() - _HANDLER_MARGIN - _depth() - 1
    return _runaway(0, stop, then)


def _workload(name, runner):
    import workloads

    DumpOptions, SnapshotContext, _ = _api()
    _, _, factory, kwargs = {item[0]: item for item in workloads.registry()}[name]
    options = DumpOptions(**kwargs)

    def build():
        _, error, traceback = factory()

        def run():
            context = SnapshotContext(options)
            payload = {
                "traceback": context.from_traceback(traceback),
                "exc_extra": context.exception(error),
            }
            source = context.embedded_sources()
            if source:
                payload["source"] = source
            return payload, context

        return Case(run)

    return runner(build)


class Scenario(NamedTuple):
    name: str
    description: str
    run: object  # Runner -> result dict
    near_limit: bool = False  # the pure walk reaches the recursion limit
    env: tuple = ()  # (name, value) pairs for the child's environment
    baseline: bool = True  # runs on another tree's package (--baseline)


def _registry():
    import workloads

    scenarios, partial = [], functools.partial

    def add(name, description, run, near_limit=False, env=(), baseline=True):
        scenarios.append(Scenario(name, description, run, near_limit, env, baseline))

    for depth in (20, 500, 900, 950):
        what = f"_mapping(typical frame namespace), stack depth {depth}"
        add(f"mapping_d{depth}", what, partial(_mapping_at, depth), depth >= 950)
    for depth in (20, 700):
        what = f"stack depth {depth}, default options"
        add(f"from_frame_d{depth}", f"from_frame, {what}", partial(_frame_at, depth, 0))
        what = f"dump_current_traceback, {what}"
        add(f"dump_current_d{depth}", what, partial(_frame_at, depth, 1))
    hooks = ("none", "settrace", "setprofile", "cprofile", "sysmon")
    for kind in hooks + ("coverage_sysmon", "coverage_ctrace"):
        env = (("COVERAGE_CORE", kind[9:]),) if kind.startswith("coverage") else ()
        what = (
            f"_mapping(frame of values whose Python capture runs helpers), hook: {kind}"
        )
        add(f"hooks_{kind}", what, partial(_hooked, kind), env=env)
    for config in ("options_subclass", "context_subclass", "unlimited"):
        what = f"{config}: value(500-key dict)"
        add(f"{config}_dict", what, partial(_configured, config, False))
        what = f"{config}: _mapping(typical frame)"
        add(f"{config}_frame", what, partial(_configured, config, True))
    for kind, what in (
        ("memo_hit", "value() memo hit on a live context"),
        ("new_int", "new context + value(12345)"),
        ("new_dict", "new context + value(small dict)"),
        ("live_mapping", "_mapping({}) on a live context"),
        ("live_int", "value(new int) on a live context"),
        ("live_float", "value(new float) on a live context"),
        ("live_bytes", "value(new short bytes) on a live context"),
        ("live_str", "value(new short str) on a live context"),
        ("out_of_nodes", "value(new int) on a live context with no nodes left"),
    ):
        add(f"call_{kind}", what, partial(_per_call, kind))
    what = "value(new list subclass instance) on a live context, 2,000 _layouts"
    add("call_live_subclass", what, _subclass_calls)
    for kind, data in (("list", "2000-level list"), ("chain", "300-object chain")):
        for threaded in (False, True):
            where = "thread with a 512 KiB stack" if threaded else "main thread"
            what = f"max_depth=None, _mapping({{'payload': {data}}}), {where}"
            name = f"deep_{kind}_{'thread' if threaded else 'main'}"
            # The pure walk of the list overflows the default recursion limit.
            add(name, what, partial(_deep_data, kind, threaded), kind == "list")
    what = f"capture in except RecursionError, {_HANDLER_MARGIN} frames below limit"
    add("recursion_handler", what, _recursion_handler, True)
    what = (
        "max_nodes=150: _mapping({'a': 41-level list, 'b': 60 ints}), "
        f"{_NEAR_ROOM} frames below the limit"
    )
    add("near_limit_nodes", what, _near_nodes, True)
    for kind, values in _DENSE_KINDS.items():
        what = f"from_frame, frame of 300 each of {values}"
        add(f"dense_{kind}", what, partial(_dense, kind))
    what = "from_frame, max_frames=2100, stack depth 2000, a record type per frame"
    add("record_stack_d2000", what, partial(_record_stack, 2000))
    for threaded in (False, True):
        what = "from_frame, max_depth=100, frame of 150-level list, dicts, objects"
        where = ", thread with a 512 KiB stack" if threaded else ""
        name = "deep_frame_d100" + ("_thread" if threaded else "")
        add(name, what + where, partial(_deep_frame, threaded))
    for name, description, _, _ in workloads.registry():
        what = f"capture only: {description}"
        add(f"workload_{name}", what, partial(_workload, name))
    return {scenario.name: scenario for scenario in scenarios}


SCENARIOS = _registry()


def run_child(args):
    runner = Runner(args.scratch, args.min_runs, args.min_time)
    try:
        result = SCENARIOS[args.child].run(runner)
    except Skip as reason:
        result = {"skipped": str(reason)}
    capture = result.pop("capture", None)
    if capture is not None:
        result.update(_digest(*capture, near_limit=SCENARIOS[args.child].near_limit))
    # Read only now, so no frame a scenario captured held what differs.
    import pydumpling
    from pydumpling.snapshot import values

    native = getattr(values, "_native", sys.modules.get("pydumpling._native"))
    statics = getattr(values, "_STATICS", None)
    result["fallbacks"] = None if statics is None else statics.fallbacks
    result["python"] = platform.python_version()
    result["package"] = str(Path(pydumpling.__file__).resolve().parent)
    result["native_loaded"] = native is not None
    result["native_file"] = getattr(native, "__file__", None)
    print(RESULT + json.dumps(result), flush=True)


# ----------------------------------------------------------- the parent side


def _error(message):
    print(f"never_slower: error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _modes(args):
    """The two kinds of child a run compares: (A, B), judged on A/B."""
    return ("tree", "base") if args.baseline else ("native", "pure")


def _spawn(args, name, mode, scratch):
    """Run one child; its result, or {"error": ...} if it failed."""
    prefixes = ("PYDUMPLING_", "COVERAGE_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    root = Path(args.baseline) if mode == "base" else REPO
    env.update(PYTHONPATH=str(root), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
    env.update(SCENARIOS[name].env)
    if mode != "native":
        env["PYDUMPLING_NATIVE"] = "0"
    command = [args.python, str(Path(__file__).resolve()), "--child", name]
    command += ["--scratch", scratch, "--min-runs", str(args.min_runs)]
    command += ["--min-time", str(args.min_time)]
    # Outside the checkout, so the working directory shadows nothing.
    done = subprocess.run(command, env=env, cwd=scratch, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith(RESULT)]
    if done.returncode or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-2:]
        return {"error": f"exit {done.returncode}: " + " | ".join(tail)}
    result = json.loads(lines[-1][len(RESULT) :])
    if result["package"] != str(root / "pydumpling"):
        _error(f"the {mode} child imported {result['package']}, not {root}")
    if result["native_loaded"] != (mode == "native"):
        state = "did not load" if mode == "native" else "loaded"
        _error(f"the {mode} child of {name} {state} pydumpling._native")
    return result


def _problems(children):
    return [c.get("error") or c.get("skipped") for c in children if "digest" not in c]


def _band(ratios, tolerance):
    """Whether a sweep's ratios pass, fail or leave it to extra rounds."""
    median = statistics.median(ratios)
    if median <= _CLEAR_PASS:
        return "pass"
    if median >= max(_CLEAR_FAIL, 1 + tolerance):
        return "fail"
    return "border"


def _shortfalls(native, pure):
    """How a near-limit native child captured less than the pure one, by the
    module docstring's definition; and whether the round shows the shift
    that definition allows: a root variable only the pure child captured,
    or more ``max_nodes`` cuts natively."""
    found = []
    for key, count in pure["roots"].items():
        mine = native["roots"].get(key, count)
        if mine > count:
            found.append(f"{key}: {mine} 'recursion' native, {count} pure")
    if (native["kept"] or 0) < (pure["kept"] or 0):
        found.append(f"kept {native['kept']} native, {pure['kept']} pure")
    mine, theirs = (dict(child["truncation"] or ()) for child in (native, pure))
    missing = not pure["roots"].keys() <= native["roots"].keys()
    return found, missing or mine.get("max_nodes", 0) > theirs.get("max_nodes", 0)


def _judge(scenario, pairs, args, swept):
    """The verdict on a scenario's rounds, the first ``swept`` of the sweep."""
    a, b = _modes(args)
    tolerance = args.tolerance
    entry = {"description": scenario.description, "near_limit": scenario.near_limit}
    entry.update(modes=[a, b], rounds=pairs, swept=swept)
    problems = _problems(pair[mode] for pair in pairs for mode in pair)
    if problems:
        errors = any("error" in pair[mode] for pair in pairs for mode in pair)
        return dict(entry, verdict="FAIL" if errors else "skip", reasons=problems[:1])
    first, second = ([pair[mode] for pair in pairs] for mode in (a, b))
    ratios = [x["us_per_op"] / y["us_per_op"] for x, y in zip(first, second)]
    digests = [{child["digest"] for child in side} for side in (first, second)]
    if len(digests[0]) > 1 or len(digests[1]) > 1:
        content = "unstable"
    else:
        content = "same" if digests[0] == digests[1] else "differs"
    band = _band(ratios[:swept], tolerance)
    median = statistics.median(ratios)
    reasons = []
    if band == "fail":
        reasons.append(f"median {a}/{b} >= {max(_CLEAR_FAIL, 1 + tolerance):.2f}")
    elif band == "border" and median > 1 + tolerance:
        reasons.append(f"median {a}/{b} of {len(ratios)} rounds > {1 + tolerance:.2f}")
    shortfalls, shifted, differs = [], 0, 0
    if scenario.near_limit and not args.baseline:
        for native, pure in zip(first, second):
            if not pure["overflowed"]:
                # Below the limit the content must be the same.
                differs += native["digest"] != pure["digest"]
                continue
            found, cut = _shortfalls(native, pure)
            shortfalls += [item for item in found if item not in shortfalls]
            shifted += cut
        if shortfalls:
            reasons.append("captures less: " + "; ".join(shortfalls[:3]))
        if differs:
            reasons.append(f"content differs in {differs} rounds below the limit")
    elif content != "same":
        reasons.append(f"content {content}")
    return dict(
        entry,
        verdict="FAIL" if reasons else "ok",
        reasons=reasons,
        band=band,
        median_ratio=median,
        sweep_median=statistics.median(ratios[:swept]),
        ratios=ratios,
        content=content,
        shortfalls=shortfalls,
        shifted_rounds=shifted,
        fallbacks=[child.get("fallbacks") for child in first],
        **{
            f"{mode}_us": statistics.median(child["us_per_op"] for child in side)
            for mode, side in ((a, first), (b, second))
        },
    )


def _print_table(entries, modes):
    fallbacks = (n for e in entries.values() for n in e.get("fallbacks", ()) or ())
    show = any(count is not None for count in fallbacks)
    widths = [28, 10, 10, 15, 10] + [9] * show
    header = ["scenario", f"{modes[0]} us", f"{modes[1]} us", "median (rounds)"]
    rows = [header + ["content"] + ["fallbacks"] * show + ["verdict"]]
    for name, entry in entries.items():
        verdict = entry["verdict"]
        if entry["reasons"]:
            verdict += ": " + "; ".join(entry["reasons"])
        if entry.get("shifted_rounds"):
            shifted, rounds = entry["shifted_rounds"], len(entry["ratios"])
            verdict += f" (shifted in {shifted} of {rounds})"
        if "ratios" not in entry:
            rows.append([name] + ["-"] * (len(widths) - 1) + [verdict])
            continue
        cells = [name] + [f"{entry[f'{mode}_us']:.2f}" for mode in modes]
        cells.append(f"{entry['median_ratio']:.3f} ({len(entry['ratios'])})")
        cells.append(entry["content"] + "~" * entry["near_limit"])
        cells += ["/".join(sorted(set(map(str, entry["fallbacks"]))))] * show
        rows.append(cells + [verdict])
    print()
    for row in rows:
        # Times and fallbacks align right, the rest left; the verdict last.
        cells = enumerate(zip(row, widths))
        print(
            *(c.rjust(w) if i in (1, 2, 5) else c.ljust(w) for i, (c, w) in cells),
            row[-1],
        )


def _selected(patterns, args):
    chosen = []
    for pattern in patterns or ["*"]:
        matches = [name for name in SCENARIOS if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            _error(f"no scenario matches {pattern!r}; see --list")
        chosen += [name for name in matches if name not in chosen]
    if args.baseline:
        left = [name for name in chosen if not SCENARIOS[name].baseline]
        if left:
            print(f"never_slower: not on a baseline tree: {' '.join(left)}", flush=True)
        chosen = [name for name in chosen if SCENARIOS[name].baseline]
    return [name for name in SCENARIOS if name in chosen]


def _note(pair, modes):
    """One progress line's worth about a pair of children."""
    if _problems(pair.values()):
        return "; ".join(problem for problem in _problems(pair.values()) if problem)
    a, b = (pair[mode]["us_per_op"] for mode in modes)
    return f"{modes[0]} {a:10.2f} us  {modes[1]} {b:10.2f} us  ratio {a / b:.3f}"


def _tree_digest(root):
    """A digest of what a run imports from ``root``: the package, the built
    extension included, and the benchmark scripts."""
    digest = hashlib.sha256()
    paths = sorted(
        path
        for pattern in ("pydumpling/**/*", "rust/bench/*.py")
        for path in Path(root).glob(pattern)
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:16]


def _run_round(args, name, order, scratch, label):
    """One pair of children, in ``order``; prints its progress line."""
    pair = {mode: _spawn(args, name, mode, scratch) for mode in order}
    pair = {mode: pair[mode] for mode in _modes(args)}
    print(f"{label} {name:<28} {_note(pair, _modes(args))}", flush=True)
    return pair


def run_parent(args):
    if args.baseline:
        args.baseline = os.path.abspath(args.baseline)
        if not (Path(args.baseline) / "pydumpling" / "__init__.py").is_file():
            _error(f"--baseline {args.baseline} holds no pydumpling package")
    names = _selected(args.scenarios, args)
    if os.sep in args.python or (os.altsep and os.altsep in args.python):
        # Children run in the scratch directory, where a relative path to the
        # interpreter no longer resolves.  abspath keeps a venv's symlink.
        args.python = os.path.abspath(args.python)
    probe = "import sys; print('%d.%d' % sys.version_info[:2])"
    try:
        version = subprocess.run(
            [args.python, "-c", probe], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        _error(f"--python {args.python} does not run: {error}")
    stem = "never_slower-baseline" if args.baseline else "never_slower"
    out = Path(args.out or HERE / "results" / f"{stem}-{version}.json")
    settings = ("rounds", "extra_rounds", "tolerance", "min_runs", "min_time")
    meta = {key: getattr(args, key) for key in settings + ("python", "baseline")}
    meta.update(argv=sys.argv, python_version=version, platform=platform.platform())
    meta.update(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    meta["tree"] = {
        "root": str(REPO),
        "digest": _tree_digest(REPO),
        "isolated": not (REPO / ".git").exists(),
    }
    if args.baseline:
        meta["baseline_digest"] = _tree_digest(args.baseline)
    meta["loadavg_start"] = os.getloadavg()
    modes = _modes(args)
    pairs = {name: [] for name in names}
    with tempfile.TemporaryDirectory(prefix="never_slower-") as scratch:
        for index in range(args.rounds):
            for position, name in enumerate(names):
                # Alternate which mode runs first, across rounds and scenarios.
                order = modes[:: -1 if (index + position) % 2 else 1]
                label = f"[{index + 1}/{args.rounds}]"
                pairs[name].append(_run_round(args, name, order, scratch, label))
        for name in names:
            # Scenarios near the line run alone for more rounds, ABBA.
            ratios = [
                pair[modes[0]]["us_per_op"] / pair[modes[1]]["us_per_op"]
                for pair in pairs[name]
                if not _problems(pair.values())
            ]
            if len(ratios) < args.rounds or _band(ratios, args.tolerance) != "border":
                continue
            for extra in range(args.extra_rounds):
                order = modes[:: -1 if extra % 2 else 1]
                label = f"[extra {extra + 1}/{args.extra_rounds}]"
                pairs[name].append(_run_round(args, name, order, scratch, label))
            ratios = [
                pair[modes[0]]["us_per_op"] / pair[modes[1]]["us_per_op"]
                for pair in pairs[name]
                if not _problems(pair.values())
            ]
            sweep = statistics.median(ratios[: args.rounds])
            print(
                f"{name}: median {modes[0]}/{modes[1]} {statistics.median(ratios):.3f}"
                f" over {len(ratios)} rounds (sweep {sweep:.3f})",
                flush=True,
            )
    entries = {
        name: _judge(SCENARIOS[name], pairs[name], args, args.rounds) for name in names
    }
    meta["loadavg_end"] = os.getloadavg()
    natives = (pair.get("native", {}) for runs in pairs.values() for pair in runs)
    meta["native_file"] = next(
        (c["native_file"] for c in natives if "native_file" in c), None
    )
    failed = [name for name, entry in entries.items() if entry["verdict"] == "FAIL"]
    skipped = sum(entry["verdict"] == "skip" for entry in entries.values())
    _print_table(entries, modes)
    print(
        f"\nnever_slower: {len(entries)} scenarios, {len(failed)} failed, "
        f"{skipped} skipped; {modes[0]}/{modes[1]} median of {args.rounds} rounds: "
        f"<= {_CLEAR_PASS} ok, >= {max(_CLEAR_FAIL, 1 + args.tolerance):.2f} FAIL, "
        f"between: {args.extra_rounds} more rounds, FAIL if the median of all "
        f"> {1 + args.tolerance:.2f}; FAIL if content differs"
        + ("" if args.baseline else " (near the limit, ~: if native captures less)")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {"meta": meta, "failed": failed, "scenarios": entries}
    out.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {out}")
    return 1 if failed else 0


def _not_copied(directory, names):
    """What --isolate leaves out of ``directory``'s ``names``."""
    where = Path(directory).relative_to(REPO).as_posix()
    left = _NOT_COPIED.get(where, set()) | _NEVER_COPIED
    return [name for name in names if name in left or name.endswith(".dump")]


def run_isolated(args, argv):
    """Copy the checkout into ``args.isolate`` and run the copy's script with
    the other arguments, so nothing the checkout's edits or builds do
    meanwhile reaches the run."""
    target = Path(args.isolate).resolve() / "tree"
    if target.exists():
        _error(f"{target} exists; --isolate takes a new or empty directory")
    shutil.copytree(REPO, target, ignore=_not_copied, symlinks=True)
    rest, skip = [], False
    for argument in argv:
        if skip:
            skip = False
        elif argument == "--isolate":
            skip = True
        elif not argument.startswith("--isolate="):
            rest.append(argument)
    script = target / "rust" / "bench" / Path(__file__).name
    print(
        f"never_slower: copied {REPO} to {target} (tree {_tree_digest(target)})",
        flush=True,
    )
    return subprocess.run([sys.executable, str(script), *rest]).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check that a native capture is never slower than pure Python."
    )
    add = parser.add_argument
    add("--scenarios", nargs="+", metavar="NAME", help="names or shell patterns")
    add("--list", action="store_true", help="list the scenarios and exit")
    add("--rounds", type=int, default=3, help="child pairs per scenario (3)")
    add("--extra-rounds", type=int, default=12, help="more for a border scenario (12)")
    add("--tolerance", type=float, default=0.02, help="FAIL above 1 + this (0.02)")
    add("--min-runs", type=int, default=20, help="timed runs per child, at least")
    add("--min-time", type=float, default=1.0, help="timed seconds per child, at least")
    add("--python", default=sys.executable, help="interpreter for the children")
    add("--out", help="JSON report; default results/never_slower-<python>.json")
    add("--baseline", metavar="DIR", help="compare the pure walk with DIR's")
    add("--isolate", metavar="DIR", help="run from a copy of the checkout in DIR")
    add("--child", choices=sorted(SCENARIOS), help=argparse.SUPPRESS)
    add("--scratch", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.min_runs < 4 or args.tolerance < 0:
        parser.error("--rounds must be >= 1, --min-runs >= 4 and --tolerance >= 0")
    if args.extra_rounds < 0:
        parser.error("--extra-rounds must be >= 0")
    if args.child:
        run_child(args)
    elif args.list:
        for scenario in SCENARIOS.values():
            mark = "~" if scenario.near_limit else " "
            print(f"{scenario.name:<28} {mark} {scenario.description}")
    elif args.isolate:
        return run_isolated(args, sys.argv[1:] if argv is None else argv)
    else:
        return run_parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
