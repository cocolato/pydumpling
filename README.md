# Pydumpling

English | [简体中文](README_zh.md)

Pydumpling saves Python exception or live stack snapshots to compressed `.dump`
files. Print a traceback, inspect saved variables with pdb, or open a remote
debugger. Derived from [elifiner/pydump](https://github.com/elifiner/pydump).

Requires Python 3.11+. Capturing another process requires CPython 3.14+.

## Install

~~~sh
python -m pip install pydumpling
pydumpling --version
~~~

These examples describe 0.2.0.

## Save an exception

Save this complete example as `example.py` and run `python example.py`:

~~~python
from pydumpling import save_dumping


def calculate():
    quantity = 3
    price = "12.50"
    return quantity + price


try:
    calculate()
except TypeError:
    save_dumping("error.dump", strict=True)
~~~

~~~sh
pydumpling --print error.dump
pydumpling --debug error.dump
# The module entry point also works:
python -m pydumpling --print error.dump
~~~

`--print` writes the traceback and exception text to standard output, so it
survives a pipe or a redirect; the omission notice stays on standard error.

Useful pdb commands include `bt`, `u`, `d`, `list`, `ll`, and `pp price`.
A snapshot lets you inspect saved state; it cannot resume the original program.

## Save the current stack

~~~python
from pydumpling import dump_current_traceback


def calculate():
    quantity = 3
    price = 12.5
    dump_current_traceback("current.dump", strict=True)
    return quantity * price


calculate()
~~~

Call `catch_any_exception()` once at application startup to capture unhandled
main-thread exceptions. It installs `sys.excepthook` and retains the previous
hook for normal exception reporting. It does not install thread or asyncio
exception handlers.

## Limit capture overhead

`DumpOptions()` applies these defaults:

| Setting | Default |
| --- | --- |
| `max_frames` | 64 frames |
| `max_depth` | 5 nested levels |
| `max_items` | 100 items per container |
| `max_string_bytes` | 4,096 bytes per captured string value |
| `max_nodes` | 10,000 visited values |
| `max_dump_bytes` | 16 MiB of uncompressed serialized data |

Values may be summarized when a budget is reached. Customize the budget or
explicitly request unrestricted capture:

~~~python
from pydumpling import DumpOptions, dump_current_traceback

dump_current_traceback(
    "small.dump",
    options=DumpOptions(max_frames=16, max_depth=3, max_items=25),
    strict=True,
)
dump_current_traceback("unlimited.dump", options=DumpOptions.unlimited(), strict=True)
~~~

Unrestricted capture can consume substantial resources and permits custom
object serialization hooks to execute. Omission reasons and applied budgets are
recorded in the loaded snapshot's `snapshot_meta` dictionary.
Unrestricted original objects can depend on their original environment or
custom loading hooks; successful round-trips are not guaranteed for every object.
The default readable snapshot mode is unaffected.

`save_dumping(filename=None, exc_info=None, *, options=None, strict=False)` and
`dump_current_traceback(filename=None, *, options=None, strict=False)` report
capture/write failures as warnings by default. Use `strict=True` to raise them.
`catch_any_exception(*, options=None)` accepts the same budgets.

Loading and debugging default to at most 64 MiB of decompressed dump data.
`load_dumpling(path, max_bytes=None)` or `debug_dumpling(path, max_bytes=None)`
disables this limit; the CLI equivalent is `--max-load-mib 0`.

## Capture a running process by PID

Run this target program with CPython 3.14+ and updated Pydumpling installed:

~~~python
import os
import time

import pydumpling

print(f"PID: {os.getpid()}", flush=True)
counter = 0
while True:
    counter += 1
    time.sleep(0.1)
~~~

In another terminal, replace `12345` with the printed PID:

~~~sh
pydumpling --capture-pid 12345 --output live.dump --timeout 30
pydumpling --print live.dump
~~~

The API is `capture_pid(pid, filename, *, options=None, timeout=30) -> Path`.
Controller and target must use the same CPython version on the same machine,
under the same user, with shared process and filesystem namespaces. The target
must have updated Pydumpling installed and permit CPython remote execution.
Operating system permissions may also prevent attachment.

A timeout stops waiting; an already dispatched request can still execute later.
The CLI prints the retained request directory on timeout. Avoid submitting
repeated requests while one is pending. Use `--unlimited` to remove capture
budgets explicitly.
Remove retained request files only after `status.json` confirms completion or
the target process has exited.

## Remote inspection

The debugger defaults to `127.0.0.1:4444`:

~~~sh
# Server:
pydumpling --rdebug error.dump --host 127.0.0.1 --port 4444

# Workstation:
ssh -L 4444:127.0.0.1:4444 user@server
# A second local terminal:
nc 127.0.0.1 4444
~~~

Remote pdb allows Python expression execution and has no built-in authentication
or encryption. Keep it on loopback and use an SSH tunnel.

## Trust and privacy

Only load trusted dumps: dill and pickle deserialization can execute code.
Size limits do not make an untrusted file safe. Snapshots can contain local
variables, credentials, personal data, and other application state; control file
access, retention, and sharing.

Inspect dumps with matching Python and source versions. Cross-version loading
and third-party debugger compatibility are not guaranteed. Old 0.1 dumps remain
readable when their objects are compatible with the interpreter.
See [0.2 migration notes](CHANGELOG.md).

## Development

~~~sh
uv sync --locked --dev --extra doc
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest --cov=pydumpling --cov-report=term-missing
uv run --locked --extra doc sphinx-build -W --keep-going -b html docs/source docs/build/html
uv build --sdist
uv build --wheel dist/pydumpling-0.2.0.tar.gz
uv run --locked twine check --strict dist/*
~~~

`uv.lock` is committed. Run `uv lock` after deliberately changing dependencies
and include the updated lockfile. CI checks Linux on Python 3.11–3.15 and Python
3.14 on Windows/macOS, with an 85% coverage floor, strict documentation builds,
and installed-wheel checks outside the source checkout. Hatchling builds
distributions; uv manages development environments.
