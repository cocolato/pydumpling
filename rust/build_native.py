"""Build pydumpling's optional native walker into the source tree.

    python rust/build_native.py [--debug] [cargo build arguments...]

The extension is built against the running interpreter with the stable ABI
(Python 3.12+) and copied to ``pydumpling/_native<suffix>``, where
``pydumpling.snapshot.values`` imports it.  ``RUSTFLAGS`` and any extra cargo
arguments pass through, so the same script builds instrumented,
profile-guided, and plain binaries; the build is a release build unless
``--debug``, ``--release``, or ``--profile`` says otherwise.  The previous
build stays in place until the new one is complete, and other builds of the
extension are removed so that none shadows it.
"""

import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CRATE = Path(__file__).resolve().parent
PACKAGE = CRATE.parent / "pydumpling"


def _suffix():
    """The stable-ABI extension suffix of this interpreter.

    Windows has no ``.abi3`` suffix: its plain ``.pyd``, listed last, is the
    one every version loads.
    """
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    return next((suffix for suffix in suffixes if ".abi3" in suffix), suffixes[-1])


def _build(argv):
    """Run cargo and return the path of the extension library it built."""
    debug = "--debug" in argv
    extra = [argument for argument in argv if argument != "--debug"]
    if any(argument.startswith("--message-format") for argument in extra):
        raise SystemExit("--message-format is the script's own: it reads cargo's JSON")
    profile = any(
        argument in ("--release", "-r", "--profile")
        or argument.startswith("--profile=")
        for argument in extra
    )
    command = [
        "cargo",
        "build",
        "--manifest-path",
        str(CRATE / "Cargo.toml"),
        "--lib",
        "--message-format=json-render-diagnostics",
    ]
    if not debug and not profile:
        command.append("--release")
    environment = dict(
        os.environ, PYO3_PYTHON=sys.executable, PYO3_BUILD_EXTENSION_MODULE="1"
    )
    process = subprocess.run(
        command + extra, check=True, env=environment, stdout=subprocess.PIPE, text=True
    )
    libraries = []
    for line in process.stdout.splitlines():
        message = json.loads(line)
        if (
            message.get("reason") == "compiler-artifact"
            and message["target"]["name"] == "_native"
            and "cdylib" in message["target"]["kind"]
        ):
            libraries = [
                name
                for name in message["filenames"]
                if name.endswith((".so", ".dylib", ".dll"))
            ]
    if len(libraries) != 1:
        raise SystemExit(f"cargo reported no single extension library: {libraries}")
    return Path(libraries[0])


def main(argv):
    library = _build(argv)
    destination = PACKAGE / ("_native" + _suffix())
    # Copy beside the destination and rename over it, so an interrupted
    # build never leaves a partial extension where Python imports it.
    handle, staged = tempfile.mkstemp(dir=PACKAGE, prefix=".native-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as target, open(library, "rb") as source:
            target.write(source.read())
        # mkstemp makes the file private; keep the mode cargo gave it.
        shutil.copymode(library, staged)
        os.replace(staged, destination)
    except BaseException:
        os.unlink(staged)
        raise
    # Another build of the extension, such as a version-specific one, would
    # be imported first; other files that share the prefix are left alone.
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for stale in PACKAGE.glob("_native.*"):
        if stale != destination and stale.name[len("_native") :] in suffixes:
            stale.unlink()
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1:])
