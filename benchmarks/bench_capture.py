"""Measure snapshot capture, write, and load overhead on a fixed workload.

Run from the repository root:

    uv run --locked python benchmarks/bench_capture.py

The workload builds a caller chain of ``DEPTH`` frames whose locals hold
representative values (nested containers, strings, custom objects). Each case
reports the best and median wall time per operation so regressions in capture
or serialization show up as stable numbers. Compare output between revisions;
there is no pass/fail threshold.
"""

import os
import shutil
import statistics
import sys
import tempfile
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydumpling import (  # noqa: E402
    DumpOptions,
    load_dumpling,
    save_dumping,
)
from pydumpling.report import print_traceback_and_except  # noqa: E402

DEPTH = 24
RUNS = 20


class Record:
    """A plain object with attributes and slots, as applications define."""

    __slots__ = ("identifier", "payload")

    def __init__(self, identifier, payload):
        self.identifier = identifier
        self.payload = payload


def build_locals(index):
    return {
        "index": index,
        "ratio": 0.5 + index / DEPTH,
        "label": f"frame-{index}-" + "x" * (64 + index * 8),
        "matrix": [[index * 10 + column for column in range(24)] for _ in range(6)],
        "mapping": {
            f"key-{column}": Record(column, [column] * 8) for column in range(12)
        },
        "packed": ("alpha", index, 3.25, None, b"bytes-value" * 4),
        "lookup": {frozenset({index, index + 1}): "cluster", "plain": "entry"},
        "records": [
            Record(column, {"value": column, "tags": ["a", "b"]}) for column in range(8)
        ],
    }


def recurse(remaining, index):
    local = build_locals(index)
    if remaining <= 0:
        raise ValueError(f"workload failure at {index} with {len(local)} locals")
    return recurse(remaining - 1, index + 1)


def workload_exc_info():
    """Raise at the recursion floor and catch at the top.

    The exception must propagate through every frame to appear in the
    traceback; catching inside the innermost function yields one frame.
    """
    try:
        recurse(DEPTH, 0)
    except ValueError:
        return sys.exc_info()


def run_case(name, options, directory, *, load=False, report=False):
    exc_info = workload_exc_info()
    dump = directory / f"{name.replace('+', '-')}.dump"
    sink = open(os.devnull, "w")
    samples = []
    try:
        # The report's truncation notice is a stderr diagnostic by design;
        # keep it out of the measurement output.
        with redirect_stdout(sink), redirect_stderr(sink):
            for _ in range(RUNS):
                start = time.perf_counter()
                save_dumping(dump, exc_info=exc_info, options=options)
                if load:
                    dumpling = load_dumpling(dump)
                    if report:
                        print_traceback_and_except(dumpling, file=sink)
                samples.append((time.perf_counter() - start) * 1000)
    finally:
        sink.close()
    print(
        f"{name:<28} best {min(samples):8.2f} ms   "
        f"median {statistics.median(samples):8.2f} ms   "
        f"dump {dump.stat().st_size / 1024:8.1f} KiB"
    )


def memory_case(name, options, directory, exc_info):
    """Peak traced allocations during one save; tracemalloc skews time."""
    import tracemalloc

    dump = directory / f"{name.replace(' ', '-')}-memory.dump"
    sink = open(os.devnull, "w")
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            tracemalloc.start()
            try:
                save_dumping(dump, exc_info=exc_info, options=options)
            finally:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
    finally:
        sink.close()
    print(
        f"{name:<28} peak {peak / 1024:9.1f} KiB   "
        f"dump {dump.stat().st_size / 1024:8.1f} KiB"
    )
    return peak


def big_locals(index):
    return {
        "blocks": [f"{'y' * 4000}block-{index}-{position}" for position in range(96)],
        "table": {f"row-{row}": {"cells": [row] * 12} for row in range(150)},
    }


def recurse_big(remaining, index):
    local = big_locals(index)
    if remaining <= 0:
        raise ValueError(f"large workload at {index} with {len(local)} locals")
    return recurse_big(remaining - 1, index + 1)


def big_exc_info():
    try:
        recurse_big(DEPTH, 0)
    except ValueError:
        return sys.exc_info()


def main():
    # Unlimited capture keeps original objects; dill notes the by-reference
    # class inside this script, which is expected and not a measurement.
    warnings.filterwarnings("ignore", category=Warning)
    print(f"workload: {DEPTH} frames, python {sys.version.split()[0]}, {RUNS} runs")
    directory = Path(tempfile.mkdtemp(prefix="pydumpling-bench-"))
    try:
        run_case("bounded save", DumpOptions(), directory)
        run_case("bounded save+load", DumpOptions(), directory, load=True)
        run_case(
            "bounded save+load+report",
            DumpOptions(),
            directory,
            load=True,
            report=True,
        )
        run_case("unlimited save", DumpOptions.unlimited(), directory)
        run_case(
            "small-budget save", DumpOptions(max_frames=8, max_items=25), directory
        )
        print()
        # Peak traced memory while writing; tracemalloc overhead skews time,
        # so timing cases above do not trace.
        memory_case("bounded memory", DumpOptions(), directory, workload_exc_info())
        memory_case(
            "large bounded memory",
            DumpOptions(
                max_items=10000,
                max_nodes=500000,
                max_string_bytes=65536,
                max_dump_bytes=64 * 1024 * 1024,
            ),
            directory,
            big_exc_info(),
        )
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
