"""Capture bounded snapshots and commit complete dumps atomically."""

import gzip
import inspect
import itertools
import linecache
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
from .snapshot.context import _CODE_TEXT_LIMIT

Pathish = str | os.PathLike[str]

DUMP_FILE_EXTENSION = ".dump"
# The sequence keeps names unique even within the same millisecond.
_DUMP_SEQUENCE = itertools.count(1)


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


def _stream_dump(
    payload: dict, filename: Pathish, serializer: str, limit: int | None
) -> Path:
    """Serialize through gzip straight into an atomically committed dump.

    The pickled form is compressed and written as it is produced, so the full
    serialized snapshot never sits in memory. Any failure after the temporary
    file exists still leaves a previous dump at ``filename`` intact.
    """
    target = Path(filename).absolute()
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            # Level 6 compresses faster than the default 9 for a slightly
            # larger file; the gain depends on the data.
            with gzip.GzipFile(
                filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6
            ) as compressed:
                try:
                    writer = _LimitedWriter(compressed, limit)
                    if serializer == "dill":
                        dill.dump(
                            payload,
                            writer,
                            protocol=pickle.HIGHEST_PROTOCOL,
                            byref=True,
                        )
                    else:
                        pickle.dump(payload, writer, protocol=pickle.HIGHEST_PROTOCOL)
                except BaseException as error:
                    # dill's pure-Python framer keeps a view of its frame buffer
                    # in the failed frames.  Clear their locals so the kept
                    # error does not leave that buffer exported until garbage
                    # collection.  Every exception, KeyboardInterrupt and
                    # SystemExit included, is then re-raised unchanged.
                    clear_frames(error.__traceback__)
                    raise
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return target


def _write_snapshot(
    filename: Pathish,
    factory: Callable[[bool], dict],
    options: DumpOptions,
) -> Path:
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
                return _stream_dump(
                    payload, filename, serializer, options.max_dump_bytes
                )
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
                continue
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
    retention uses file modification times rather than name order.
    """
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem) or "snapshot"
    parts = [stem]
    if lineno is not None:
        parts.append(str(lineno))
    parts.append(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3])
    parts.append(str(os.getpid()))
    parts.append(f"{next(_DUMP_SEQUENCE):06d}")
    parent = Path() if directory is None else Path(directory)
    return parent / ("-".join(parts) + DUMP_FILE_EXTENSION)


def _innermost_traceback(exc_tb: TracebackType) -> TracebackType:
    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next
    return exc_tb


def _embedded_sources(context: SnapshotContext, settings: DumpOptions) -> dict:
    """Read a window of source around each captured line, best effort.

    ``embed_source_lines`` is zero (embed nothing), a window half-width, or
    ``None`` for whole files. Reading runs through linecache and tolerates
    files that moved or vanished; each line is bounded like code metadata.
    """
    span = settings.embed_source_lines
    if span == 0:
        return {}
    wanted = {}
    for frame in context.frames.values():
        filename = frame.f_code.co_filename
        if filename.startswith("<") and filename.endswith(">"):
            continue
        wanted.setdefault(filename, set()).add(frame.f_lineno)
    sources = {}
    for filename, linenos in wanted.items():
        lines = linecache.getlines(filename)
        if not lines:
            continue
        if span is None:
            selected = {
                number: lines[number - 1] for number in range(1, len(lines) + 1)
            }
        else:
            selected = {}
            for lineno in linenos:
                for number in range(
                    max(1, lineno - span), min(len(lines), lineno + span) + 1
                ):
                    selected[number] = lines[number - 1]
        sources[filename] = {
            "lines": {
                number: text
                if len(text) <= _CODE_TEXT_LIMIT
                else text[:_CODE_TEXT_LIMIT]
                for number, text in selected.items()
            },
            "total": len(lines),
        }
    return sources


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
        traceback = context.from_frame(frame)
        payload = {
            "traceback": traceback,
            "version": __version__,
            "snapshot_meta": context.metadata(),
        }
        if not minimal:
            source = _embedded_sources(context, settings)
            if source:
                payload["source"] = source
        return payload

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
        exc_type, exc_value, exc_tb = sys.exc_info() if exc_info is None else exc_info
        if not isinstance(exc_value, BaseException) or not isinstance(
            exc_tb, TracebackType
        ):
            raise ValueError(
                "save_dumping requires an active exception with a traceback"
            )
        if not isinstance(exc_type, type) or not issubclass(exc_type, BaseException):
            raise ValueError(
                "exc_info must contain an exception type, value and traceback"
            )
        innermost = _innermost_traceback(exc_tb)
        filename = (
            filename
            if filename is not None
            else _default_filename(innermost.tb_frame, innermost.tb_lineno)
        )

        def factory(minimal: bool) -> dict:
            context = SnapshotContext(settings, minimal=minimal)
            traceback = context.from_traceback(exc_tb)
            extra = context.exception(exc_value)
            payload = {
                "traceback": traceback,
                "version": __version__,
                "exc_extra": extra,
                "snapshot_meta": context.metadata(),
            }
            if not minimal:
                source = _embedded_sources(context, settings)
                if source:
                    payload["source"] = source
            return payload

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
