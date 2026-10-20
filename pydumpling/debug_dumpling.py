"""Load trusted snapshots and inspect them with the standard debugger."""

import gzip
import inspect
import linecache
import pdb
import pickle
import signal
import sys
import threading
import types
import zlib
from contextlib import contextmanager
from os import PathLike
from typing import Any

import dill
from packaging.version import InvalidVersion, Version

from .fake_types import FakeCode, FakeFrame, FakeTraceback
from .snapshot import SNAPSHOT_FORMAT_VERSION

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_COMPAT_LOCK = threading.RLock()


class DumpFormatError(ValueError):
    """A snapshot cannot be decoded or has an unsupported structure."""


def _normalize_positions(code, lasti):
    if isinstance(code, FakeCode):
        # Old snapshots can lack a position table; modern inspect/traceback
        # still index that iterator by the saved bytecode instruction offset.
        count = getattr(code, "_co_position_count", 0)
        if not isinstance(count, int):
            count = 0
        code._co_position_count = max(count, lasti // 2 + 1, 1)


def _validate_frames(frame, validated):
    active = set()
    while frame is not None:
        if not isinstance(frame, (FakeFrame, types.FrameType)):
            raise DumpFormatError("Invalid frame in dumpling traceback")
        identity = id(frame)
        if identity in active:
            raise DumpFormatError("Cycle in dumpling frame chain")
        if identity in validated:
            break
        active.add(identity)
        code = getattr(frame, "f_code", None)
        if not isinstance(code, (FakeCode, types.CodeType)):
            raise DumpFormatError("Invalid code in dumpling frame")
        if not all(
            isinstance(getattr(code, field, None), str)
            for field in ("co_filename", "co_name")
        ):
            raise DumpFormatError("Invalid code metadata in dumpling frame")
        if not isinstance(getattr(frame, "f_locals", None), dict):
            raise DumpFormatError("Invalid locals in dumpling frame")
        if not isinstance(getattr(frame, "f_globals", None), dict):
            raise DumpFormatError("Invalid globals in dumpling frame")
        if not isinstance(getattr(frame, "f_lineno", None), int):
            raise DumpFormatError("Invalid line number in dumpling frame")
        if not isinstance(getattr(frame, "f_lasti", None), int):
            raise DumpFormatError("Invalid instruction offset in dumpling frame")
        _normalize_positions(code, frame.f_lasti)
        if not hasattr(frame, "f_back"):
            raise DumpFormatError("Missing frame chain in dumpling traceback")
        frame = frame.f_back
    validated.update(active)


def _validate_dumpling(dumpling):
    if not isinstance(dumpling, dict):
        raise DumpFormatError("Dumpling data must be a dictionary")
    format_version = dumpling.get("format_version", 0)
    if (
        type(format_version) is not int
        or not 0 <= format_version <= SNAPSHOT_FORMAT_VERSION
    ):
        raise DumpFormatError("Unsupported dumpling snapshot format version")
    version = dumpling.get("version")
    if not isinstance(version, str):
        raise DumpFormatError("Missing or invalid dumpling version")
    try:
        package_version = Version(version)
    except InvalidVersion as error:
        raise DumpFormatError("Invalid dumpling version: %s" % version) from error
    if format_version == 0 and not Version("0.0.1") <= package_version < Version(
        "1.0.0"
    ):
        raise DumpFormatError("Unsupported dumpling version: %s" % version)
    if dumpling.get("dump_type", "DILL") not in ("DILL", "PICKLE"):
        raise DumpFormatError("Unsupported dumpling serialization format")
    if dumpling.get("exc_extra") is not None and not isinstance(
        dumpling["exc_extra"], dict
    ):
        raise DumpFormatError("Invalid exception metadata in dumpling")
    metadata = dumpling.get("snapshot_meta")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise DumpFormatError("Invalid snapshot metadata in dumpling")
        if metadata.get("truncation") is not None and not isinstance(
            metadata["truncation"], dict
        ):
            raise DumpFormatError("Invalid snapshot truncation metadata")

    traceback = dumpling.get("traceback")
    if traceback is None:
        raise DumpFormatError("Dumpling has no traceback")
    visited = set()
    frames = set()
    while traceback is not None:
        if not isinstance(traceback, (FakeTraceback, types.TracebackType)):
            raise DumpFormatError("Invalid dumpling traceback")
        if id(traceback) in visited:
            raise DumpFormatError("Cycle in dumpling traceback chain")
        visited.add(id(traceback))
        if not isinstance(getattr(traceback, "tb_lineno", None), int):
            raise DumpFormatError("Invalid line number in dumpling traceback")
        if not isinstance(getattr(traceback, "tb_lasti", None), int):
            raise DumpFormatError("Invalid instruction offset in dumpling traceback")
        frame = getattr(traceback, "tb_frame", None)
        if frame is None:
            raise DumpFormatError("Missing frame in dumpling traceback")
        _validate_frames(frame, frames)
        _normalize_positions(frame.f_code, traceback.tb_lasti)
        if not hasattr(traceback, "tb_next"):
            raise DumpFormatError("Missing traceback chain in dumpling")
        traceback = traceback.tb_next
    return dumpling


def load_dumpling(
    filename: str | PathLike[str], *, max_bytes: int | None = DEFAULT_MAX_BYTES
) -> dict[str, Any]:
    """Load a trusted gzip/dill snapshot, limiting decompressed input bytes.

    Unpickling executes Python code. Only load snapshots from trusted sources.
    Pass ``max_bytes=None`` to read a trusted snapshot without a size limit.
    Invalid gzip or snapshot data raises ``DumpFormatError``.
    """
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
    ):
        raise ValueError("max_bytes must be a positive integer or None")
    try:
        with gzip.open(filename, "rb") as stream:
            payload = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
    except (gzip.BadGzipFile, EOFError, zlib.error) as error:
        raise DumpFormatError("Cannot decode dumpling gzip data") from error
    if max_bytes is not None and len(payload) > max_bytes:
        raise DumpFormatError("Dumpling exceeds the %d-byte read limit" % max_bytes)
    try:
        dumpling = dill.loads(payload)
    except Exception as error:
        # A separate immutable buffer keeps the pickle fallback at byte zero.
        try:
            dumpling = pickle.loads(payload)
        except Exception:
            raise DumpFormatError("Cannot decode dumpling data") from error
    return _validate_dumpling(dumpling)


def _embedded_lines(entry):
    """The validated ``{lineno: text}`` mapping of one embedded file, or None."""
    lines = entry.get("lines") if isinstance(entry, dict) else None
    if not isinstance(lines, dict) or not lines:
        return None
    for number, text in lines.items():
        if type(number) is not int or number < 1 or type(text) is not str:
            return None
    return lines


def _matches_local(filename, lines, total) -> bool:
    """Whether the file on this machine still holds every embedded line.

    linecache ends every full line with a newline, so an embedded line
    without one was cut to its bound and only has to start the local line.
    """
    linecache.checkcache(filename)
    local = linecache.getlines(filename)
    if not local or (type(total) is int and total != len(local)):
        return False
    for number, text in lines.items():
        if number > len(local):
            return False
        line = local[number - 1]
        if line != text and (text.endswith("\n") or not line.startswith(text)):
            return False
    return True


@contextmanager
def embedded_source(sources):
    """Serve a dump's embedded source through ``linecache`` while active.

    Embedded lines are what ran, so they stand in for a local file that is
    missing or no longer matches them; an unchanged local file keeps serving
    its whole text. Lines the dump left out read as empty strings. Injected
    entries use a ``None`` mtime, so ``linecache.checkcache`` keeps them, and
    the entries they replaced return on exit. Malformed entries from a
    hand-edited dump are skipped, not fatal.
    """
    with _COMPAT_LOCK:
        replaced = {}
        try:
            for filename, entry in sources.items() if isinstance(sources, dict) else ():
                lines = _embedded_lines(entry)
                if (
                    not isinstance(filename, str)
                    or lines is None
                    or _matches_local(filename, lines, entry.get("total"))
                ):
                    continue
                replaced[filename] = linecache.cache.get(filename)
                padded = [""] * max(lines)
                for number, text in lines.items():
                    padded[number - 1] = text
                linecache.cache[filename] = (
                    sum(map(len, padded)),
                    None,
                    padded,
                    filename,
                )
            yield
        finally:
            for filename, previous in replaced.items():
                if previous is None:
                    linecache.cache.pop(filename, None)
                else:
                    linecache.cache[filename] = previous


@contextmanager
def mock_inspect(sources=None):
    """Use with ``with mock_inspect():``; restore inspect/tracing state on exit.

    ``sources`` is a dump's embedded source mapping; while the context is
    active, pdb's ``list`` shows it, as :func:`embedded_source` serves it.
    """
    with _COMPAT_LOCK, embedded_source(sources):
        original = inspect.isframe, inspect.istraceback, inspect.iscode
        original_trace = sys.gettrace()
        main_thread = threading.current_thread() is threading.main_thread()
        original_sigint = signal.getsignal(signal.SIGINT) if main_thread else None
        previous_sigint = pdb.Pdb._previous_sigint_handler
        frame_traces = []
        frame = sys._getframe().f_back
        while frame is not None:
            frame_traces.append(
                (frame, frame.f_trace, frame.f_trace_lines, frame.f_trace_opcodes)
            )
            frame = frame.f_back
        inspect.isframe = lambda obj: isinstance(obj, (types.FrameType, FakeFrame))
        inspect.istraceback = lambda obj: isinstance(
            obj, (types.TracebackType, FakeTraceback)
        )
        inspect.iscode = lambda obj: isinstance(obj, (types.CodeType, FakeCode))
        try:
            yield
        finally:
            inspect.isframe, inspect.istraceback, inspect.iscode = original
            # Older Bdb quit/continue stops the process-wide trace; continue
            # also deletes local trace hooks along the live caller stack.
            for frame, trace, lines, opcodes in frame_traces:
                frame.f_trace = trace
                frame.f_trace_lines = lines
                frame.f_trace_opcodes = opcodes
            frame_traces.clear()
            if sys.gettrace() is not original_trace:
                sys.settrace(original_trace)
            if main_thread:
                if signal.getsignal(signal.SIGINT) is not original_sigint:
                    signal.signal(signal.SIGINT, original_sigint)
                pdb.Pdb._previous_sigint_handler = previous_sigint


def debug_dumpling(
    dump_file: str | PathLike[str],
    pdb: Any = pdb,
    *,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
) -> None:
    dumpling = load_dumpling(dump_file, max_bytes=max_bytes)
    from .report import print_snapshot_notice

    print_snapshot_notice(dumpling)
    with mock_inspect(dumpling.get("source")):
        pdb.post_mortem(dumpling["traceback"])
