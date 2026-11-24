# Optional Rust capture on Linux x86-64

The reference is the pure Python installation from `ed7ae815`. The candidate
adds a separate PyO3 wheel to the branch's pure Python wheel. Both public
writers use one native call per bounded capture. Python supplies the frame
roots, source positions, variable priorities and exception diagnostics. Rust
drives the child traversal with a heap stack. Python describes custom
objects one level at a time. The two sides share the context's identity memo
and node budget. Builtin exception diagnostics retain their existing small
reserved budget. Compression, serialization and file replacement use the
existing writer.

## Build and run

Run these commands in the branch checkout with a Rust toolchain and a
GIL-enabled CPython. Choose an installed interpreter for `python3.13`.

```sh
root=$PWD
work=$(mktemp -d)
git worktree add --detach "$work/reference" ed7ae815
python3.13 -m venv "$work/reader"
"$work/reader/bin/python" -m pip install "$work/reference"
python3.13 -m venv "$work/base"
"$work/base/bin/python" -m pip wheel --no-deps --wheel-dir "$work/wheels" . ./native
"$work/base/bin/python" -m pip install "$work"/wheels/pydumpling-*.whl pytest
python3.13 -m venv "$work/native"
"$work/native/bin/python" -m pip install "$work"/wheels/*.whl pytest

cp -R tests "$work/tests"
cd "$work"
for engine in base native; do
  PYDUMPLING_BASE_READER="$work/reader/bin/python" \
    "$work/$engine/bin/python" -I -m pytest -q "$work/tests"
  "$work/$engine/bin/python" -I "$root/benchmarks/bench_stack.py" "$engine"
done
"$work/native/bin/python" "$root/benchmarks/bench_capture.py" --suite \
  --compare-python "$work/reader/bin/python" --rounds 6 --runs 25 --json paired.json
```

The cloud build used one `cp312-abi3-manylinux_2_34_x86_64` extension wheel
on all three interpreters. The base wheel is `py3-none-any`. Uninstalling
`pydumpling-native` selects the Python engine in the next process.

## Compatibility and failure handling

* Both `save_dumping()` and `dump_current_traceback()` wrote format 2. In a
  separate environment the reference reader ran `--json` and pdb's `bt`,
  `up`, `p` and `pp` on the resulting files. Checks covered the fault line,
  frame order, cause, locals, custom attributes, cross-frame aliases and
  circular references.
* With `max_depth=None` and sufficient remaining budget, native capture
  preserved all 3,000 nested lists in the test and the local next to them.
  The reference reader recovered both. The pure engine retains its
  `recursion` placeholder on this input. Serialization still has its own
  limits.
* Exceeding the write budget or denying write permission preserves the
  previous file and the application's active exception. The strict call
  raises and the default call issues a warning. Unlimited capture uses
  Python from the outset.
* The released tag `v0.2.0` (`09820c7`) only reads format 1 and predates
  `--json`. It rejects format 2 from both `ed7ae815` and this branch with
  `Unsupported dumpling snapshot format version`. Both branch installations
  read files produced by that release.

## Stack measurement

The probe calls the public live-stack writer on a default Linux main thread.
It measures the distance to the same C stack marker inside the captured
leaf's attribute reader. It also reloads the file, checks the leaf, counts
native calls and checks for retries.

| CPython | Mixed graph, 4 / 12 / 20 levels (bytes) | Object-only graph, 4 / 12 / 20 levels (bytes) |
| --- | ---: | ---: |
| 3.12.14 | 2,432 / 2,368 / 2,368 | 2,368 / 2,368 / 2,368 |
| 3.13.5 | 2,288 / 2,208 / 2,208 | 2,208 / 2,208 / 2,208 |
| 3.14.10 | 2,144 / 2,128 / 2,128 | 2,128 / 2,128 / 2,128 |

The mixed graph alternates lists, dictionaries, tuples and objects. All 18
native samples entered the extension once. All 18 pure samples used the
same marker address at the origin and leaf. Native stack distance did not
increase as the graphs grew over these depths. The measurements cover the
capture traversal on Linux. The original macOS crash has no crash report
available for comparison.

## Installed test suite

| CPython | Base wheel (passed / skipped / failed) | Base and native wheels (passed / skipped / failed) |
| --- | ---: | ---: |
| 3.12.14 | 276 / 6 / 0 | 280 / 2 / 0 |
| 3.13.5 | 276 / 6 / 0 | 280 / 2 / 0 |
| 3.14.10 | 276 / 5 / 1 | 280 / 1 / 1 |

The single failing case, `test_native_pid_capture_preserves_business_frame`,
uses CPython's `sys.remote_exec` to attach to a second process by PID. The
reference also fails with `PyRuntime address lookup failed during debug
offsets initialization` on this host, with both the uv and GitHub Actions
interpreter builds. This leaves live PID attachment on 3.14 unverified here.
Both in-process capture APIs passed on all three versions.

The tested base wheel has SHA-256
`741ce18f41decb4fac7f9f4fb097d6f481b2711bba91e3d87c478025db6ca2b4`.
The native wheel has SHA-256
`28cff55236687e6ae44609765583f729a10bb391de6b9513fb5d58f3a6a3de41`.

## Performance

The benchmark alternated the original main wheel and the branch's installed
release wheels on one CPU of an AMD EPYC 9V74, using CPython 3.13.5. Each
workload and engine ran in six fresh processes with 25 timed samples per
operation per process. Capture measures the public writer's payload factory.
Save measures the entire `save_dumping()` call through file replacement.
Peak RSS is the largest process high-water mark across rounds. File sizes
are medians of the compressed dumps. Volatile process metadata can change
the compressed size slightly.

| Workload | Engine | Capture p50 / p95 (ms) | Save p50 / p95 (ms) | Peak RSS (MiB) | Dump (bytes) |
| --- | --- | ---: | ---: | ---: | ---: |
| Small exception | Main | 0.18 / 0.40 | 1.11 / 1.68 | 21.62 | 1,299 |
| | Native | 0.20 / 0.39 | 1.07 / 1.70 | 21.63 | 1,298.5 |
| Shared across frames | Main | 4.96 / 6.24 | 6.70 / 8.17 | 21.63 | 6,024 |
| | Native | 3.73 / 5.12 | 5.24 / 6.89 | 22.41 | 6,023 |
| Deep mixed containers | Main | 1.27 / 1.52 | 2.29 / 3.15 | 21.62 | 1,967 |
| | Native | 1.01 / 1.25 | 2.09 / 3.76 | 21.77 | 1,967 |
| Tight node budget | Main | 0.15 / 0.25 | 0.98 / 1.27 | 21.74 | 986 |
| | Native | 0.15 / 0.28 | 0.94 / 1.31 | 21.75 | 985.5 |

Every round matched the reference digest for both the captured payload and
the reloaded diagnostic content, including frame locals, aliases, exception
details and truncation counters. The four uncompressed sizes matched at
2,593, 20,176, 7,291 and 1,801 bytes respectively.

The clearest gain was the shared graph, with capture p50 down 24.9% and
save p50 down 21.8%. Mixed graphs also saved faster at the median, but
their save p95 rose 19.4%. Small-exception capture p50 rose 12.3%.
The extension remains opt-in.
