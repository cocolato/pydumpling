"""Capture bounded snapshots and commit complete dumps atomically."""

import gzip
import inspect
import itertools
import os
import pickle
import re
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from traceback import clear_frames
from types import FrameType, TracebackType
from typing import Callable

import dill

from ._version import __version__
from .fake_types import FakeTraceback
from .options import DumpOptions
from .snapshot import SNAPSHOT_FORMAT_VERSION, SnapshotContext, reduce_snapshot
from .snapshot.context import _utf8_prefix

Pathish = str | os.PathLike[str]

DUMP_FILE_EXTENSION = ".dump"
# The sequence keeps names unique even within the same millisecond.
_DUMP_SEQUENCE = itertools.count(1)
# The writer's temporary file is mkstemp's ".{name}.XXXXXXXX" next to the
# target; a generated name leaves room for it within the 255-byte file name
# limit that common filesystems share.
_DUMP_NAME_LIMIT = 255 - len("..") - 8
# The timestamp and sequence that end every generated name.
_DUMP_NAME_TAIL = re.compile(
    r"-(\d{8}T\d{6}\.\d{3})-\d+-(\d+)" + re.escape(DUMP_FILE_EXTENSION) + "$"
)


class DumpTooLargeError(ValueError):
    """Even a reduced snapshot cannot fit the requested byte budget."""


class _LimitedWriter:
    """A binary stream wrapper that aborts once serialized bytes exceed a budget.

    The counter watches the uncompressed bytes the serializer produces, exactly
    as the whole-buffer writer it replaced did; compressed output can be far
    smaller.
    """

    def __init__(self, stream, limit: int | None):
        self._stream = stream
        self.limit = limit
        self.written = 0

    def write(self, data: bytes) -> int:
        self.written += len(data)
        if self.limit is not None and self.written > self.limit:
            raise DumpTooLargeError(f"Snapshot exceeds {self.limit} serialized bytes")
        return self._stream.write(data)


def _stream_dump(payload: dict, output, serializer: str, limit: int | None) -> None:
    """Serialize ``payload`` through gzip into ``output``, replacing its content.

    The pickled form is compressed and written as it is produced, so the full
    serialized snapshot never sits in memory.
    """
    output.seek(0)
    output.truncate()
    # Level 6 compresses faster than the default 9 for a slightly larger
    # file; the gain depends on the data.
    with gzip.GzipFile(
        filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6
    ) as compressed:
        try:
            writer = _LimitedWriter(compressed, limit)
            if serializer == "dill":
                dill.dump(payload, writer, protocol=pickle.HIGHEST_PROTOCOL, byref=True)
            else:
                pickle.dump(payload, writer, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as error:
            # dill's pure-Python framer keeps a view of its frame buffer in the
            # failed frames.  Clear their locals so the kept error does not
            # leave that buffer exported until garbage collection.  Every
            # exception, KeyboardInterrupt and SystemExit included, is then
            # re-raised unchanged.
            clear_frames(error.__traceback__)
            raise


def _write_snapshot(
    filename: Pathish,
    factory: Callable[[bool], dict],
    options: DumpOptions,
) -> Path:
    """Capture a snapshot and commit the first attempt that fits, atomically.

    Every attempt rewrites one temporary file, so a path that cannot be
    created fails before any capture work and a commit error is never retried
    as a serialization failure. Any failure leaves a previous dump at
    ``filename`` intact.
    """
    target = Path(filename).absolute()
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            _write_attempts(output, factory, options)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target


def _write_attempts(output, factory: Callable[[bool], dict], options: DumpOptions):
    attempts = []
    stage = "full"
    try:
        payload = factory(False)
    except Exception as error:
        attempts.append({"stage": stage, "error": type(error).__name__})
        payload = reduce_snapshot(factory(True), options, stage="minimal")
        stage = "minimal"
    payload["format_version"] = SNAPSHOT_FORMAT_VERSION
    # A bounded snapshot holds only builtins and pydumpling's own types, which
    # C pickle writes many times faster than dill.  Original objects kept by an
    # unlimited capture, such as lambdas or local classes, may need dill.
    serializers = ("dill", "pickle") if options.is_unlimited else ("pickle", "dill")
    while True:
        metadata = payload["snapshot_meta"]
        metadata["attempts"] = attempts
        if stage != "full":
            metadata["truncation"]["serialization_fallback"] = 1
        too_large = False
        for serializer in serializers:
            payload["dump_type"] = serializer.upper()
            try:
                _stream_dump(payload, output, serializer, options.max_dump_bytes)
                return
            except Exception as error:
                last_error = error
                too_large |= isinstance(error, DumpTooLargeError)
                attempts.append(
                    {
                        "stage": stage,
                        "serializer": serializer,
                        "error": type(error).__name__,
                    }
                )
        if stage == "minimal":
            raise last_error
        while True:
            stage = (
                {
                    "full": "source",
                    "source": "globals",
                    "globals": "outer",
                    "outer": "reduced",
                    "reduced": "minimal",
                }[stage]
                if too_large
                else "minimal"
            )
            try:
                reduced = reduce_snapshot(payload, options, stage=stage)
            except Exception as error:
                if stage == "minimal":
                    raise
                attempts.append({"stage": stage, "error": type(error).__name__})
                stage = "minimal"
                payload = reduce_snapshot(payload, options, stage="minimal")
                break
            unchanged = (
                reduced["snapshot_meta"]["truncation"]
                == payload["snapshot_meta"]["truncation"]
            )
            payload = reduced
            if stage not in ("source", "globals", "outer") or not unchanged:
                break


def _options(options: DumpOptions | None) -> DumpOptions:
    if options is None:
        return DumpOptions()
    if not isinstance(options, DumpOptions):
        raise TypeError("options must be a DumpOptions instance or None")
    return options


def _default_filename(frame: FrameType, lineno: int) -> str:
    """A unique name for ``frame``, so repeated failures never overwrite."""
    return str(_dump_target(Path(frame.f_code.co_filename).name, lineno))


def _dump_target(
    stem: str, lineno: int | None = None, directory: Pathish | None = None
) -> Path:
    """A timestamped dump path that never reuses a name in this process.

    The path is ``stem-lineno-timestamp-pid-sequence.dump`` in UTC
    milliseconds. The per-process sequence keeps repeated captures unique;
    retention uses file modification times rather than name order. A long
    stem is cut, by its UTF-8 size, so this name and the writer's temporary
    name both fit file name limits; the snapshot keeps the full source path.
    """
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem) or "snapshot"
    parts = [] if lineno is None else [str(lineno)]
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3])
    parts.append(str(os.getpid()))
    parts.append(f"{next(_DUMP_SEQUENCE):06d}")
    tail = "-" + "-".join(parts) + DUMP_FILE_EXTENSION
    budget = _DUMP_NAME_LIMIT - len(tail)
    if len(stem.encode("utf-8", "surrogatepass")) > budget:
        stem = _utf8_prefix(stem, budget) or "snapshot"
    parent = Path() if directory is None else Path(directory)
    return parent / (stem + tail)


def _dump_order(name: str) -> tuple[str, int]:
    """The UTC timestamp and sequence a generated name carries, for ordering.

    Names this package did not generate sort before every generated name.
    """
    match = _DUMP_NAME_TAIL.search(name)
    return (match[1], int(match[2])) if match else ("", 0)


def _exc_info(
    exc_info: tuple | None,
) -> tuple[type[BaseException], BaseException, TracebackType]:
    """The validated exception triple; ``None`` means the active exception."""
    exc_type, exc_value, exc_tb = sys.exc_info() if exc_info is None else exc_info
    if not isinstance(exc_value, BaseException) or not isinstance(
        exc_tb, TracebackType
    ):
        raise ValueError("save_dumping requires an active exception with a traceback")
    if not isinstance(exc_type, type) or not issubclass(exc_type, BaseException):
        raise ValueError("exc_info must contain an exception type, value and traceback")
    return exc_type, exc_value, exc_tb


def _innermost_traceback(exc_tb: TracebackType) -> TracebackType:
    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next
    return exc_tb


def _complete_payload(context: SnapshotContext, payload: dict) -> dict:
    """Add embedded source, then the metadata that also counts its truncation."""
    source = context.embedded_sources()
    if source:
        payload["source"] = source
    payload["snapshot_meta"] = context.metadata()
    return payload


def _capture_frame(
    frame: FrameType,
    filename: Pathish | None = None,
    *,
    options: DumpOptions | None = None,
) -> Path:
    """Capture the supplied business frame; used by local and PID capture."""
    if not isinstance(frame, FrameType):
        raise ValueError("No Python frame is available to capture")
    settings = _options(options)
    filename = (
        filename if filename is not None else _default_filename(frame, frame.f_lineno)
    )

    def factory(minimal: bool) -> dict:
        context = SnapshotContext(settings, minimal=minimal)
        payload = {"traceback": context.from_frame(frame), "version": __version__}
        return _complete_payload(context, payload)

    return _write_snapshot(filename, factory, settings)


def _warn_or_raise(error: Exception, strict: bool) -> None:
    if strict:
        raise error
    warnings.warn(
        f"Unexpected error: {error} when dumping traceback",
        RuntimeWarning,
        stacklevel=3,
    )


def save_dumping(
    filename: Pathish | None = None,
    exc_info: tuple[type[BaseException], BaseException, TracebackType] | None = None,
    *,
    options: DumpOptions | None = None,
    strict: bool = False,
) -> None:
    """Save an exception snapshot. By default errors warn instead of propagating."""
    try:
        settings = _options(options)
        exc_type, exc_value, exc_tb = _exc_info(exc_info)
        if filename is None:
            innermost = _innermost_traceback(exc_tb)
            filename = _default_filename(innermost.tb_frame, innermost.tb_lineno)

        def factory(minimal: bool) -> dict:
            context = SnapshotContext(settings, minimal=minimal)
            payload = {
                "traceback": context.from_traceback(exc_tb),
                "version": __version__,
                "exc_extra": context.exception(exc_value),
            }
            return _complete_payload(context, payload)

        _write_snapshot(filename, factory, settings)
    except Exception as error:
        _warn_or_raise(error, strict)


def dump_current_traceback(
    filename: Pathish | None = None,
    *,
    options: DumpOptions | None = None,
    strict: bool = False,
) -> None:
    """Save the caller's stack without requiring an exception."""
    frame = inspect.currentframe()
    try:
        _capture_frame(
            frame.f_back if frame is not None else None, filename, options=options
        )
    except Exception as error:
        _warn_or_raise(error, strict)
    finally:
        del frame


def gen_tb_from_frame(
    frame: FrameType, *, options: DumpOptions | None = None
) -> FakeTraceback:
    """Build a snapshot traceback including the supplied frame."""
    return SnapshotContext(_options(options)).from_frame(frame)
