# Changelog

## Unreleased

### Runtime and capture

- Add `DumpOptions(embed_source_lines=N)`: store N lines of source context on
  each side of every captured frame's current line (`None` embeds whole
  files). `--debug` and `--rdebug` sessions show the embedded source through
  `list` for files missing on the viewing machine; local files keep
  precedence. When the byte budget is exceeded, source is omitted before
  reducing captured variables, and reported as `source_omitted`.
- Add `pydumpling --html FILE dump`: a standalone, script-free HTML report
  with the exception, a source window per frame, and locals tables.
- Stream every serialization attempt through gzip directly into its temporary
  dump file instead of building the full pickled snapshot in memory first. On
  a string-heavy workload the writer's peak drops by about 40 percent, and a
  failed attempt still leaves any previous dump untouched.
- Look up the value memo once per captured value instead of twice.

### Development and documentation

- Add `pydumpling --json` for an
  indented JSON report of a dump: capture metadata, exception details, frames
  with guarded `repr` locals (plus globals behind `--globals`), and truncation
  reasons. `pydumpling --info` prints the stored metadata as a human-readable
  summary. Both keep the quiet 141 pipe semantics of `--print`.
- Add `benchmarks/bench_capture.py`, a fixed-workload script comparing capture,
  write, load, and report timings between revisions.
- Split the internal `pydumpling.snapshot` module into a package
  (`snapshot.context`, `snapshot.values`, `snapshot.frames`,
  `snapshot.exceptions`, `snapshot.reduce`, `snapshot.identity`) and moved
  hooks to `pydumpling.hooks` and report formatting to `pydumpling.report`.
  `pydumpling.snapshot` and `pydumpling.helpers` keep re-exporting the same
  names; behavior is unchanged.

### Runtime and capture

- Record the creation time, capture duration, pid, host name, working
  directory, command line, and capturing thread in `snapshot_meta`; every
  field degrades to `None` when the environment cannot provide it.
- Give automatically generated dump names the source line, a UTC timestamp,
  the pid, and a per-process sequence, so repeated failures never overwrite an
  earlier snapshot.
- Add `DumpManager(directory, *, retention=50, options=None)` with
  `save_exception()`, `save_current()`, `target()`, and `prune()`: unique
  names in one directory, oldest dumps by file modification time removed beyond `retention`
  (`retention=None` keeps everything).
- `catch_any_exception(*, options=None, include_threads=False, manager=None)`
  can also hook `threading.excepthook` and save through a `DumpManager`;
  add `install_asyncio_handler(loop=None, *, options=None)` for event loops
  and `restore_exception_hooks()` to remove every installed hook while
  keeping later third-party hooks.
- Keep the value, type name, and attributes of `str`, `bytes`, `int`, and
  `float` subclass instances, such as `IntEnum` and `StrEnum` members, under
  the budgets of plain values instead of an empty placeholder.
- Save bounded snapshots faster: they are written with pickle before dill and
  compressed at gzip level 6. Unlimited capture keeps dill first.
- A captured attribute named like a pickling hook, such as `__reduce_ex__` or
  `__getstate__`, no longer makes a snapshot unsavable.
- Write snapshot format 2, which adds the scalar subclass copies. Format 1
  dumps still load; Pydumpling 0.2.0 cannot read format 2.

## 0.2.0 — 2026-10-18

### Runtime and capture

- Require Python 3.11+, dill 0.4.1+, and packaging 26.3+.
- Reuse captured frames and objects instead of recursively copying every caller
  chain; materialize modern line/column tables for Python 3.15 compatibility.
- Restart failed serialization in a fresh buffer and commit complete gzip dumps
  atomically, preserving existing files on failure.
- Capture once for serializer retries; reduce the captured graph before falling
  back to a minimal snapshot on byte-budget failures. Record failed attempts,
  cumulative omissions, and effective budgets in snapshot metadata.
- Preserve fault-frame locals ahead of globals and outer-frame locals during
  reduction, dropping code constants and bytecode before fault-frame values.
  Bound fallback diagnostic summaries so long exception reports do
  not prevent saving an otherwise usable traceback.
- Version the snapshot schema independently with `format_version=1`, including
  internal code tables; retain unversioned legacy reads and reject unknown formats.
  Apply historical package-version restrictions only to legacy format 0.
- Restore inspect, tracing, standard streams, and SIGINT state after debugging;
  close remote sockets on disconnect, timeout, and partial initialization.
- Preserve bounded built-in exception filenames, import details, and syntax
  locations. Redirect all remote debugger commands, including ``display``,
  through the same exception-safe standard-stream scope.
- Preserve snapshots when exception rendering fails, recording a diagnostic
  and truncation metadata in both normal and minimal captures. Retry mappings
  omitted at greater depth when capturing frame locals/globals within budget.
- Add `DumpOptions` budgets: 64 frames, 5 nested levels, 100 items per container,
  4,096 bytes per string, 10,000 visited values, and 16 MiB of uncompressed data.
  Use `DumpOptions.unlimited()` to explicitly request unrestricted capture.
- Keep warning-based writer failures by default; add `strict=True` for callers
  that need errors to propagate. `catch_any_exception(*, options=None)` accepts
  capture budgets and preserves ordinary exception reporting.
- Load at most 64 MiB of decompressed data by default. Pass `max_bytes=None`, or
  use `--max-load-mib 0`, for larger trusted snapshots.
- Add `capture_pid(pid, filename, *, options=None, timeout=30)` and the
  `--capture-pid` CLI action for local CPython 3.14+ processes with matching
  versions, same-user access, shared namespaces, and updated Pydumpling.
  A timeout stops waiting; an already sent request may still execute later.
- Provide an installed `pydumpling` console command, `--version`, and remote
  `--host` / `--port` options alongside `python -m pydumpling`.
- Write the `--print` report to standard output; notices stay on standard
  error, and a reader that exits early (`| head`) ends the command quietly.
- Capture `Counter`, `OrderedDict`, `defaultdict`, `namedtuple`, Python 3.15's
  `frozendict`, and other container subclasses with their entries, type name,
  and attributes instead of an empty placeholder.
- Keep file, function, variable, and attribute names out of `max_string_bytes`,
  so a small string budget no longer renames frames, locals, or attributes.
- A value nested too deeply for the interpreter no longer discards the other
  variables of its frame; it is reported as `recursion`. A removed working
  directory no longer fails the capture.
- Export `DumpFormatError`, `CaptureTimeoutError`, and `CaptureInterruptedError`
  from the `pydumpling` package.

### Development and documentation

- Consolidate development onto uv with a committed lockfile. Remove old PDM,
  exported requirements, tox, Flake8, and Hatch environment entry points.
- Use Ruff for lint/formatting and a direct pytest command with an 85% coverage
  floor. Expand CI to Linux Python 3.11–3.15 and Windows/macOS Python 3.14.
- Build and check both distributions, then validate installed-wheel entry points
  outside the checkout. CI does not publish releases automatically.
- Share package and documentation versions from `pydumpling/_version.py`.
  Add runnable examples, API reference, migration notes, and strict Sphinx builds.

### Migration notes

Existing API names remain available. Default snapshots can now summarize large
values; explicitly opt into unlimited capture when that tradeoff is appropriate.
Direct callers of `pydumpling.debug_dumpling.mock_inspect()` must now use
`with mock_inspect():` around their debugger session. Calling it alone no longer
patches `inspect`; leaving the context restores host state. `debug_dumpling` and
`r_post_mortem` handle this automatically.
Automatic filenames now use `source-name-line.dump` in the current working
directory, rather than writing alongside the source file. Explicit paths are
unchanged. Removed maintenance files remain recoverable from Git history.
Old 0.1 dumps are supported when their serialized objects match the interpreter;
cross-Python-version compatibility is not guaranteed. Keep corresponding source
files available, load only trusted dumps, and protect sensitive captured values.
A saved snapshot supports inspection and cannot resume the original program.
