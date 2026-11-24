"""Measure snapshot capture, write, and load overhead on a fixed workload.

Run from the repository root:

    uv run --locked python benchmarks/bench_capture.py

For paired performance and content comparisons of installed wheels, run
the same script with each version installed in its own Python environment::

    python benchmarks/bench_capture.py --suite --compare-python /path/to/reference/python --json results.json

The legacy workload builds a caller chain of ``DEPTH`` frames whose locals hold
representative values (nested containers, strings, custom objects). Each case
reports the best and median wall time per operation so regressions in capture
or serialization show up as stable numbers. Compare output between revisions;
there is no pass/fail threshold.
"""

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

if not any(flag in sys.argv for flag in ("--case", "--suite")):
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


CASES = ("small", "shared", "mixed", "budget")


def snapshot_digest(name, payload):
    """Compare frame data, aliases, exception details and bounded capture results."""
    frames = []
    entry = payload["traceback"]
    while entry is not None:
        frame = entry.tb_frame
        frames.append(
            (frame.f_code.co_name, entry.tb_lineno, frame.f_locals, frame.f_globals)
        )
        entry = entry.tb_next
    locals_ = frames[-1][2]
    meta = payload["snapshot_meta"]
    if name == "shared":
        batch = locals_["batch"]
        assert len(frames) == 12 and len(batch) == 500
        assert all(frame[2]["batch"] is batch for frame in frames)
    elif name == "mixed":
        nested = locals_["order"].lines
        for level in reversed(range(8)):
            assert nested["level"] == level
            nested = nested["children"][0]
        assert nested["lines"][-1].lines[-1] == (60, "60")
    else:
        assert locals_["request_id"] == "r-42"
        if name == "budget":
            assert meta["truncation"].get("max_nodes", 0) > 0
            assert len(locals_["rows"]) < 1000
    record = (
        frames,
        payload["exc_extra"],
        payload.get("source"),
        meta["limits"],
        meta["original_frames"],
        meta["captured_frames"],
        meta["truncation"],
    )
    return hashlib.sha256(pickle.dumps(record, protocol=5)).hexdigest()


def suite_worker(name, runs):
    from capture_workloads import exception_case

    import pydumpling
    from pydumpling import pydumpling as writer

    exc_info, kwargs = exception_case(name)
    options = DumpOptions(**kwargs)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "example.dump"

        def save():
            save_dumping(path, exc_info=exc_info, options=options, strict=True)

        def samples(operation):
            operation()  # warm up outside the timed samples
            elapsed = []
            for _ in range(runs):
                start = time.perf_counter_ns()
                operation()
                elapsed.append((time.perf_counter_ns() - start) / 1e6)
            return elapsed

        # Obtain the public API's factory once so capture-only measures its
        # actual backend without gzip, fsync or setting up a different graph.
        build = None

        def keep_factory(filename, factory, options):
            nonlocal build
            build = factory

        original = writer._write_snapshot
        try:
            writer._write_snapshot = keep_factory
            save()
        finally:
            writer._write_snapshot = original
        assert build is not None
        capture_ms = samples(lambda: build(False))
        save_ms = samples(save)
        peak = peak_rss_bytes()
        capture_digest = snapshot_digest(name, build(False))
        with gzip.open(path, "rb") as stream:
            serialized_bytes = len(stream.read())
        public = load_dumpling(path)
        assert not public["snapshot_meta"].get("attempts"), "serialization retried"
        public_digest = snapshot_digest(name, public)
        return {
            "package": pydumpling.__file__,
            "python": sys.version.split()[0],
            "capture_ms": capture_ms,
            "save_ms": save_ms,
            "peak_rss_bytes": peak,
            "dump_bytes": path.stat().st_size,
            "serialized_bytes": serialized_bytes,
            "capture_digest": capture_digest,
            "save_digest": public_digest,
            "truncation": public["snapshot_meta"]["truncation"],
        }


def peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * (1 if sys.platform == "darwin" else 1024)


def run_suite(args):
    interpreters = {"current": sys.executable}
    if args.compare_python:
        interpreters["reference"] = args.compare_python
    samples = {name: {mode: [] for mode in interpreters} for name in CASES}
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as scratch:
        for name in CASES:
            for round_ in range(args.rounds):
                modes = list(interpreters)
                if round_ % 2:
                    modes.reverse()
                for mode in modes:
                    command = [
                        interpreters[mode],
                        str(Path(__file__).resolve()),
                        "--case",
                        name,
                        "--runs",
                        str(args.runs),
                    ]
                    raw = subprocess.check_output(
                        command, env=env, cwd=scratch, text=True
                    )
                    samples[name][mode].append(json.loads(raw))
    print(
        "case     mode        capture p50/p95 ms    save p50/p95 ms   peak RSS MiB   dump bytes"
    )
    summary = {name: {} for name in CASES}
    mismatch = []
    for name, results in samples.items():
        for key in ("capture_digest", "save_digest"):
            if len({r[key] for rounds in results.values() for r in rounds}) != 1:
                mismatch.append(f"{name}: {key}")
        for mode, rounds in results.items():
            row = summary[name][mode] = {}
            cells = []
            for metric in ("capture_ms", "save_ms"):
                times = sorted(value for r in rounds for value in r[metric])
                p95 = statistics.quantiles(times, n=100, method="inclusive")[94]
                p50 = statistics.median(times)
                row[metric] = {"p50": p50, "p95": p95}
                cells.append(f"{p50:7.2f}/{p95:<7.2f}")
            peaks = [
                r["peak_rss_bytes"] for r in rounds if r["peak_rss_bytes"] is not None
            ]
            row["peak_rss_bytes"] = max(peaks, default=None)
            row["median_dump_bytes"] = statistics.median(
                r["dump_bytes"] for r in rounds
            )
            row["median_serialized_bytes"] = statistics.median(
                r["serialized_bytes"] for r in rounds
            )
            rss = f"{max(peaks) / 2**20:.2f}" if peaks else "n/a"
            print(
                f"{name:<8} {mode:<10} {cells[0]}   {cells[1]}   {rss:>10} {row['median_dump_bytes']:>12g}"
            )
    if args.json:
        report = {"summary": summary, "rounds": samples, "mismatches": mismatch}
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    if mismatch:
        raise SystemExit("captured contents differ: " + ", ".join(mismatch))


def entrypoint():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", action="store_true", help="run the four exception workloads"
    )
    parser.add_argument(
        "--compare-python",
        help="alternate with this interpreter's installed pydumpling",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="fresh processes per workload and interpreter",
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="samples per operation in each child"
    )
    parser.add_argument(
        "--json", help="write raw timings, byte counts, RSS and content digests"
    )
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 2 or args.rounds < 1:
        parser.error("use at least two samples and one round")
    if args.case:
        print(json.dumps(suite_worker(args.case, args.runs)))
    elif args.suite:
        run_suite(args)
    else:
        main()


if __name__ == "__main__":
    entrypoint()
