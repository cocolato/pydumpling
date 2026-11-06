"""Benchmark pydumpling's capture walk: legacy copy, pure Python, native walker.

Each workload in ``workloads.py`` runs in its own fresh subprocess.  Inside it
the workload's exception is built once, then two measurements are taken after
a warm-up, each sampled until it has at least ``--min-runs`` runs and
``--min-time`` seconds of samples:

* ``save``: the full ``save_dumping(strict=True)`` into a temporary directory;
* ``capture``: ``SnapshotContext(options)`` + ``from_traceback`` +
  ``exception`` + ``embedded_sources``, without metadata or serialization.

Implementations (``--impl``):

* ``legacy``: the frozen pre-change package under ``--legacy-root``;
* ``pure``: the repository's package with ``PYDUMPLING_NATIVE=0``;
* ``native``: the repository's package with the compiled ``pydumpling._native``
  walker; fails if the extension does not load or a context has no walker.

Every subprocess gets ``PYTHONHASHSEED`` fixed (``--hash-seed``) so all
implementations walk identically laid out dicts and sets.  Peak RSS is
``ru_maxrss`` of the subprocess at its end, in MB of 2**20 bytes.  The SUITE
line sums the per-workload median times (and also gives their geometric
mean, which the largest workload does not dominate) and takes the largest
peak RSS.  Each result also records the capture's pickle digest, node count
and truncation counter, so ``--compare`` shows whether two implementations
captured identical snapshots as well as how fast.

Example::

    .venv/bin/python rust/bench/bench_walk.py --impl legacy \\
        --legacy-root /path/to/legacy_pkg --json rust/bench/results/legacy.json
    .venv/bin/python rust/bench/bench_walk.py --impl native --json native.json
    .venv/bin/python rust/bench/bench_walk.py --compare \\
        rust/bench/results/legacy.json native.json
"""

import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from time import perf_counter

# Keep __pycache__ out of rust/bench when this file imports workloads.py.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
IMPLS = ("legacy", "pure", "native")


# ------------------------------------------------------------------- the child


def _rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux KiB.
    return peak / (1 << 20) if sys.platform == "darwin" else peak / 1024


def _package_digest(package_dir):
    """A short hash of the imported package's sources, to tie results to code."""
    digest = hashlib.sha256()
    for path in sorted(Path(package_dir).rglob("*.py")):
        digest.update(path.relative_to(package_dir).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _fail(message):
    print(f"bench_walk: error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _import_impl(impl, legacy_root):
    """Import pydumpling, check it is the implementation asked for, and say so."""
    import pydumpling
    import pydumpling.snapshot.values as values

    package_dir = Path(pydumpling.__file__).resolve().parent
    expected = (
        Path(legacy_root).resolve() if impl == "legacy" else REPO
    ) / "pydumpling"
    if package_dir != expected:
        _fail(f"--impl {impl} imported {pydumpling.__file__}, expected {expected}")
    hook = getattr(values, "_native", None)
    native_file = getattr(hook, "__file__", None) if hook is not None else None
    if impl == "native" and hook is None:
        try:
            import pydumpling._native  # noqa: F401
        except ImportError as error:
            _fail(f"native extension pydumpling._native does not import: {error}")
        _fail(
            "pydumpling.snapshot.values._native is None or missing although "
            "pydumpling._native imports; the walker hook is not wired"
        )
    if impl != "native" and hook is not None:
        _fail(f"--impl {impl} loaded the native extension {native_file}")
    return pydumpling, package_dir, hasattr(values, "_native"), native_file


def _describe_walker(context):
    if not hasattr(context, "_walker"):
        return "absent (no _walker attribute)"
    walker = context._walker
    return "None" if walker is None else f"set ({type(walker).__qualname__})"


def _summary(samples):
    ordered = sorted(samples)
    q1, median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "runs": len(ordered),
        "sampled_s": round(sum(ordered), 4),
        "median_ms": statistics.median(ordered) * 1000,
        "min_ms": ordered[0] * 1000,
        "mean_ms": statistics.fmean(ordered) * 1000,
        "p25_ms": q1 * 1000,
        "p75_ms": q3 * 1000,
        "iqr_pct": 100 * (q3 - q1) / median if median else 0.0,
        "stdev_ms": statistics.stdev(ordered) * 1000,
    }


def _sample(function, *, min_runs, min_time, warmup_runs, warmup_time):
    """Warm up, then time ``function`` until both sample floors are reached."""
    started = perf_counter()
    warm = 0
    while warm < warmup_runs or perf_counter() - started < warmup_time:
        function()
        warm += 1
    gc.collect()
    samples = []
    total = 0.0
    while len(samples) < min_runs or total < min_time:
        begin = perf_counter()
        function()
        elapsed = perf_counter() - begin
        samples.append(elapsed)
        total += elapsed
    result = _summary(samples)
    result["warmup_runs"] = warm
    return result


def run_child(args):
    """Measure one workload with one implementation; write JSON to a file."""
    pydumpling, package_dir, hooked, native_file = _import_impl(
        args.impl, args.legacy_root
    )
    import workloads

    from pydumpling import DumpOptions, save_dumping
    from pydumpling.snapshot import SnapshotContext

    entries = {entry[0]: entry for entry in workloads.registry()}
    name, description, factory, option_kwargs = entries[args.child]
    options = DumpOptions(**option_kwargs)
    exc_type, exc_value, exc_tb = factory()
    exc_info = (exc_type, exc_value, exc_tb)

    probe = SnapshotContext(options)
    walker = _describe_walker(probe)
    if args.impl == "native" and getattr(probe, "_walker", None) is None:
        _fail(f"native run of {name}: SnapshotContext._walker is {walker}")
    if args.impl != "native" and getattr(probe, "_walker", None) is not None:
        _fail(f"--impl {args.impl} run of {name}: SnapshotContext._walker is set")
    del probe
    hook = (
        f"_native={native_file}"
        if native_file
        else ("_native=None" if hooked else "_native hook absent")
    )
    print(
        f"[{args.impl}] {name}: pydumpling={pydumpling.__file__} {hook} "
        f"walker={walker} python={platform.python_version()}",
        flush=True,
    )

    def capture():
        context = SnapshotContext(options)
        payload = {
            "traceback": context.from_traceback(exc_tb),
            "exc_extra": context.exception(exc_value),
        }
        source = context.embedded_sources()
        if source:
            payload["source"] = source
        return context, payload

    rss_setup = _rss_mb()
    workdir = Path(tempfile.mkdtemp(prefix=f"bench_walk-{name}-", dir=args.tmpdir))
    try:
        target = workdir / f"{name}.dump"

        def save():
            save_dumping(target, exc_info=exc_info, options=options, strict=True)

        floors = {
            "min_runs": args.min_runs,
            "min_time": args.min_time,
            "warmup_runs": args.warmup_runs,
            "warmup_time": args.warmup_time,
        }
        save_stats = _sample(save, **floors)
        dump_bytes = target.stat().st_size
        capture_stats = _sample(capture, **floors)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    context, payload = capture()
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    result = {
        "workload": name,
        "description": description,
        "impl": args.impl,
        "options": option_kwargs,
        "python": platform.python_version(),
        "executable": sys.executable,
        "pydumpling_file": pydumpling.__file__,
        "package_digest": _package_digest(package_dir),
        "native_hook": hooked,
        "native_file": native_file,
        "walker": walker,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "save": save_stats,
        "capture": capture_stats,
        "dump_bytes": dump_bytes,
        "capture_pickle_bytes": len(raw),
        # Equal digests across implementations mean byte-identical captures.
        "capture_pickle_sha256": hashlib.sha256(raw).hexdigest()[:16],
        "nodes": context._nodes,
        "frames": len(context.frames),
        "truncation": [[key, count] for key, count in context._truncation.items()],
        "rss_after_setup_mb": rss_setup,
        "peak_rss_mb": _rss_mb(),
    }
    Path(args.child_out).write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ the parent


def _child_env(impl, root, hash_seed):
    env = dict(os.environ)
    for name in (
        "PYDUMPLING_NATIVE",
        "COVERAGE_PROCESS_START",
        "COVERAGE_PROCESS_CONFIG",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(root)
    env["PYTHONHASHSEED"] = str(hash_seed)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if impl == "pure":
        env["PYDUMPLING_NATIVE"] = "0"
    return env


def _cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or None


def _selected(names, wanted):
    if not wanted:
        return names
    chosen = []
    for token in wanted.split(","):
        token = token.strip()
        matches = [name for name in names if token in (name, name.split("_")[0])]
        if not matches:
            raise SystemExit(f"bench_walk: unknown workload {token!r}; known: {names}")
        chosen.extend(match for match in matches if match not in chosen)
    return [name for name in names if name in chosen]


def _spread_pct(values):
    return 100 * (max(values) - min(values)) / statistics.median(values)


def _aggregate(rounds):
    """Per-workload medians over rounds, plus round-to-round spread."""
    out = {}
    for key in ("save", "capture"):
        medians = [result[key]["median_ms"] for result in rounds]
        out[f"{key}_median_ms"] = statistics.median(medians)
        out[f"{key}_round_medians_ms"] = medians
        out[f"{key}_round_spread_pct"] = (
            _spread_pct(medians) if len(medians) > 1 else None
        )
        out[f"{key}_iqr_pct"] = statistics.median(r[key]["iqr_pct"] for r in rounds)
        out[f"{key}_runs"] = [result[key]["runs"] for result in rounds]
    out["peak_rss_mb"] = max(result["peak_rss_mb"] for result in rounds)
    out["rss_after_setup_mb"] = max(result["rss_after_setup_mb"] for result in rounds)
    first = rounds[0]
    for key in (
        "description",
        "options",
        "dump_bytes",
        "capture_pickle_bytes",
        "capture_pickle_sha256",
        "nodes",
        "frames",
        "truncation",
        "walker",
    ):
        out[key] = first[key]
    digests = {result["capture_pickle_sha256"] for result in rounds}
    out["capture_pickle_stable"] = len(digests) == 1
    out["rounds"] = rounds
    return out


def _print_table(impl, results, suite):
    header = (
        f"{'workload':<20} {'save ms':>10} {'iqr%':>5} {'capture ms':>11} "
        f"{'iqr%':>5} {'rounds%':>8} {'rss MB':>8} {'nodes':>7}  truncation"
    )
    print(f"\n== {impl}")
    print(header)
    for name, entry in results.items():
        spread = entry["save_round_spread_pct"]
        truncation = ",".join(f"{key}={count}" for key, count in entry["truncation"])
        print(
            f"{name:<20} {entry['save_median_ms']:10.3f} {entry['save_iqr_pct']:5.1f} "
            f"{entry['capture_median_ms']:11.3f} {entry['capture_iqr_pct']:5.1f} "
            f"{'-' if spread is None else f'{spread:.1f}':>8} "
            f"{entry['peak_rss_mb']:8.1f} {entry['nodes']:7d}  {truncation}"
        )
    rounds = ""
    if len(suite["save_ms_per_round"]) > 1:
        per_round = "/".join(f"{value:.3f}" for value in suite["save_ms_per_round"])
        rounds = (
            f" save_ms_per_round={per_round} spread={suite['save_spread_pct']:.1f}%"
        )
    print(
        f"SUITE impl={impl} save_ms_sum={suite['save_ms']:.3f} "
        f"capture_ms_sum={suite['capture_ms']:.3f} "
        f"max_peak_rss_mb={suite['max_peak_rss_mb']:.1f} "
        f"save_geomean_ms={suite['save_geomean_ms']:.3f} "
        f"capture_geomean_ms={suite['capture_geomean_ms']:.3f}{rounds}",
        flush=True,
    )


def run_parent(args):
    import workloads

    names = _selected([entry[0] for entry in workloads.registry()], args.workloads)
    if args.impl == "legacy":
        if args.legacy_root is None:
            raise SystemExit(
                "bench_walk: --impl legacy needs --legacy-root (the directory "
                "holding the frozen pydumpling/ package) or PYDUMPLING_LEGACY_ROOT"
            )
        root = Path(args.legacy_root).resolve()
        if not (root / "pydumpling" / "__init__.py").is_file():
            raise SystemExit(f"bench_walk: no pydumpling/__init__.py under {root}")
        if root == REPO:
            raise SystemExit("bench_walk: --legacy-root is the repository itself")
    else:
        root = REPO
    env = _child_env(args.impl, root, args.hash_seed)
    meta = {
        "impl": args.impl,
        "argv": sys.argv,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "executable": args.python,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "loadavg_start": os.getloadavg(),
        "root": str(root),
        "hash_seed": args.hash_seed,
        "min_runs": args.min_runs,
        "min_time_s": args.min_time,
        "warmup_runs": args.warmup_runs,
        "warmup_time_s": args.warmup_time,
        "rounds": args.rounds,
        "rss_unit": "MB = 2**20 bytes, from ru_maxrss",
    }
    rounds = {name: [] for name in names}
    with tempfile.TemporaryDirectory(prefix="bench_walk-", dir=args.tmpdir) as scratch:
        for index in range(args.rounds):
            for name in names:
                out = Path(scratch) / f"{name}-{index}.json"
                command = [
                    args.python,
                    str(Path(__file__).resolve()),
                    "--impl",
                    args.impl,
                    "--child",
                    name,
                    "--child-out",
                    str(out),
                    "--min-runs",
                    str(args.min_runs),
                    "--min-time",
                    str(args.min_time),
                    "--warmup-runs",
                    str(args.warmup_runs),
                    "--warmup-time",
                    str(args.warmup_time),
                    "--tmpdir",
                    scratch,
                ]
                if args.impl == "legacy":
                    command += ["--legacy-root", str(root)]
                # The child runs outside the repository so its working
                # directory cannot shadow the package under test.
                completed = subprocess.run(command, env=env, cwd=scratch)
                if completed.returncode:
                    raise SystemExit(
                        f"bench_walk: {args.impl} child for {name} failed "
                        f"with exit code {completed.returncode}"
                    )
                rounds[name].append(json.loads(out.read_text()))
    results = {name: _aggregate(rounds[name]) for name in names}
    per_round = [
        sum(rounds[name][index]["save"]["median_ms"] for name in names)
        for index in range(args.rounds)
    ]
    suite = {
        "save_ms": sum(entry["save_median_ms"] for entry in results.values()),
        "capture_ms": sum(entry["capture_median_ms"] for entry in results.values()),
        "max_peak_rss_mb": max(entry["peak_rss_mb"] for entry in results.values()),
        # The sums are dominated by the largest workload; the geometric means
        # weigh every workload's speed-up equally.
        "save_geomean_ms": statistics.geometric_mean(
            entry["save_median_ms"] for entry in results.values()
        ),
        "capture_geomean_ms": statistics.geometric_mean(
            entry["capture_median_ms"] for entry in results.values()
        ),
        "save_ms_per_round": per_round,
        "save_spread_pct": _spread_pct(per_round) if len(per_round) > 1 else None,
    }
    meta["loadavg_end"] = os.getloadavg()
    meta["python"] = results[names[0]]["rounds"][0]["python"]
    meta["pydumpling_file"] = results[names[0]]["rounds"][0]["pydumpling_file"]
    meta["package_digest"] = results[names[0]]["rounds"][0]["package_digest"]
    _print_table(args.impl, results, suite)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {"meta": meta, "suite": suite, "workloads": results}
        path.write_text(json.dumps(document, indent=1) + "\n")
        print(f"wrote {path}")


def compare(base_path, other_path):
    """Print OTHER/BASE time ratios and whether the captures are identical."""
    documents = [json.loads(Path(path).read_text()) for path in (base_path, other_path)]
    base, other = (document["workloads"] for document in documents)
    labels = [document["meta"]["impl"] for document in documents]
    print(f"{other_path} ({labels[1]}) vs {base_path} ({labels[0]})")
    print(
        f"{'workload':<20} {'save x':>7} {'capture x':>10} {'rss x':>6}  "
        "same pickle/nodes/truncation"
    )
    for name in base:
        if name not in other:
            continue
        old, new = base[name], other[name]
        same = [
            old[key] == new[key]
            for key in ("capture_pickle_sha256", "nodes", "truncation")
        ]
        print(
            f"{name:<20} {old['save_median_ms'] / new['save_median_ms']:7.2f} "
            f"{old['capture_median_ms'] / new['capture_median_ms']:10.2f} "
            f"{new['peak_rss_mb'] / old['peak_rss_mb']:6.2f}  "
            + "/".join("yes" if flag else "NO" for flag in same)
        )
    # Over the workloads both files measured, so a partial run still compares.
    common = [name for name in base if name in other]
    ratios = []
    for key in ("save_median_ms", "capture_median_ms"):
        old = [base[name][key] for name in common]
        new = [other[name][key] for name in common]
        ratios.append(sum(old) / sum(new))
        ratios.append(statistics.geometric_mean(old) / statistics.geometric_mean(new))
    print(
        f"SUITE speed-up over {len(common)} workloads: save sum {ratios[0]:.2f}x, "
        f"save geomean {ratios[1]:.2f}x, capture sum {ratios[2]:.2f}x, "
        f"capture geomean {ratios[3]:.2f}x"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark pydumpling capture: legacy, pure Python or native.",
    )
    parser.add_argument("--impl", choices=IMPLS)
    parser.add_argument("--json", metavar="OUT", help="write the results here")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASE", "OTHER"),
        help="compare two result files (speed-up = BASE time / OTHER time) and exit",
    )
    parser.add_argument(
        "--legacy-root",
        default=os.environ.get("PYDUMPLING_LEGACY_ROOT"),
        help="directory holding the frozen pydumpling/ package (--impl legacy); "
        "default $PYDUMPLING_LEGACY_ROOT",
    )
    parser.add_argument(
        "--workloads",
        help="comma-separated names or prefixes (a,b,b2,c,d,e); default all",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="fresh subprocesses per workload; >1 reports round-to-round spread",
    )
    parser.add_argument("--min-runs", type=int, default=15)
    parser.add_argument("--min-time", type=float, default=0.5, metavar="SECONDS")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--warmup-time", type=float, default=0.1, metavar="SECONDS")
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter for the subprocesses (default: this one)",
    )
    parser.add_argument(
        "--tmpdir", help="parent of the temporary dump directories (default: system)"
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.compare:
        compare(*args.compare)
        return
    if args.impl is None:
        parser.error("--impl is required unless --compare is given")
    if args.min_runs < 2 or args.rounds < 1:
        parser.error("--min-runs must be at least 2 and --rounds at least 1")
    if not math.isfinite(args.min_time) or args.min_time < 0:
        parser.error("--min-time must be a nonnegative number")
    if args.child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
