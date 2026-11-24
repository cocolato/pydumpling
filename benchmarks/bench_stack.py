"""Measure the native stack at a captured leaf on Linux, using installed wheels.

Run once for each interpreter and installation, on the default main thread::

    /path/to/base/bin/python -I benchmarks/bench_stack.py base > base.json
    /path/to/native/bin/python -I benchmarks/bench_stack.py native > native.json

A C compiler (cc with GCC/Clang builtins) is required. The byte count is the
distance from just before the public dump call to the same attribute-reading
hook in each capture. Compare depths within one interpreter and engine.
"""

import argparse
import ctypes
import json
import platform
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from pydumpling import DumpOptions, dump_current_traceback, load_dumpling
from pydumpling.snapshot import SnapshotContext

DEPTHS = (4, 12, 20)
TOKEN = "stack measurement reached the leaf"


class Leaf:
    def __init__(self):
        self.token = TOKEN


class Box:
    def __init__(self, child):
        self.child = child


def graph(depth, shape):
    leaf = Leaf()
    child = leaf
    for level in range(depth):
        if shape == "objects" or level % 4 == 3:
            child = Box(child)
        elif level % 4 == 0:
            child = [child]
        elif level % 4 == 1:
            child = {"child": child}
        else:
            child = (child,)
    return leaf, child


def stack_sampler(directory):
    # Return a marker from the same non-inlined C function at both call sites.
    # Only the difference is recorded, so ASLR and stack direction cancel out.
    library = directory / "stack_marker.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-O2", "-x", "c", "-", "-o", str(library)],
        input=(
            "#include <stdint.h>\n"
            "uintptr_t stack_marker(void) {\n"
            "    return (uintptr_t)__builtin_frame_address(0);\n"
            "}\n"
        ),
        text=True,
        check=True,
    )
    marker = ctypes.CDLL(str(library)).stack_marker
    marker.argtypes = ()
    marker.restype = ctypes.c_size_t
    return marker


def write(payload, path, options):
    # Capture exactly this frame. Its first argument is the graph to read back.
    dump_current_traceback(path, options=options, strict=True)


def measure(marker, engine, depth, shape, directory):
    leaf, payload = graph(depth, shape)
    path = directory / "capture.dump"
    options = DumpOptions(max_frames=1, max_depth=100, max_items=8, max_nodes=1000)
    stack_bytes = []
    native_entries = 0
    attributes = SnapshotContext._attributes

    def observed_attributes(value):
        if value is leaf:
            stack_bytes.append(abs(origin - marker()))
        return attributes(value)

    with ExitStack() as hooks:
        hooks.enter_context(
            patch.object(
                SnapshotContext, "_attributes", staticmethod(observed_attributes)
            )
        )
        if engine is not None:
            capture = engine.capture

            def observed_capture(context, program):
                nonlocal native_entries
                native_entries += 1
                return capture(context, program)

            hooks.enter_context(patch.object(engine, "capture", observed_capture))
        origin = marker()
        write(payload, path, options)

    result = load_dumpling(path)
    recovered = result["traceback"].tb_frame.f_locals["payload"]
    for _ in range(depth):
        if isinstance(recovered, (list, tuple)):
            recovered = recovered[0]
        elif isinstance(recovered, dict):
            recovered = recovered["child"]
        else:
            recovered = recovered.child
    if recovered.token != TOKEN or result["snapshot_meta"]["attempts"]:
        raise AssertionError("leaf was lost or capture retried")
    if len(stack_bytes) != 1 or native_entries != int(engine is not None):
        raise AssertionError(
            f"unexpected leaf samples {stack_bytes} or native entries {native_entries}"
        )
    return {
        "stack_bytes": stack_bytes[0],
        "native_entries": native_entries,
        "leaf_read": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installation", choices=["base", "native"])
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this benchmark builds a Linux shared library")
    try:
        import pydumpling_native as engine
    except ModuleNotFoundError as exc:
        if exc.name != "pydumpling_native":
            raise
        engine = None
    if (engine is not None) != (args.installation == "native"):
        parser.error("installed extension does not match the requested installation")
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        marker = stack_sampler(directory)
        rows = [
            {
                "shape": shape,
                "depth": depth,
                **measure(marker, engine, depth, shape, directory),
            }
            for shape in ("mixed", "objects")
            for depth in DEPTHS
        ]
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": sys.version,
                "installation": args.installation,
                "pydumpling": metadata.version("pydumpling"),
                "pydumpling_native": metadata.version("pydumpling-native")
                if engine
                else None,
                "samples": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
