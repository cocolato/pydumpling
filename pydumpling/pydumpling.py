"""Capture bounded snapshots and commit complete dumps atomically."""

import gzip
import inspect
import io
import os
import pickle
import re
import sys
import tempfile
import warnings
from pathlib import Path
from traceback import clear_frames
from types import FrameType, TracebackType
from typing import Callable

import dill

from ._version import __version__
from .fake_types import FakeTraceback
from .options import DumpOptions
from .snapshot import SNAPSHOT_FORMAT_VERSION, SnapshotContext, reduce_snapshot

Pathish = str | os.PathLike[str]


class DumpTooLargeError(ValueError):
    """Even a reduced snapshot cannot fit the requested byte budget."""


class _LimitedBuffer(io.BytesIO):
    def __init__(self, limit: int | None):
        super().__init__()
        self.limit = limit

    def write(self, data: bytes) -> int:
        if self.limit is not None and self.tell() + len(data) > self.limit:
            raise DumpTooLargeError(f"Snapshot exceeds {self.limit} serialized bytes")
        return super().write(data)


def _serialize(payload: dict, serializer: str, limit: int | None) -> bytes:
    with _LimitedBuffer(limit) as buffer:
        try:
            if serializer == "dill":
                dill.dump(payload, buffer, protocol=pickle.HIGHEST_PROTOCOL, byref=True)
            else:
                pickle.dump(payload, buffer, protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as error:
            # dill's pure-Python framer keeps a view of its frame buffer in the
            # failed frames.  Clear their locals so the kept error does not
            # leave that buffer exported until garbage collection.  Every
            # exception, KeyboardInterrupt and SystemExit included, is then
            # re-raised unchanged.
            clear_frames(error.__traceback__)
            raise
        return buffer.getvalue()


def _atomic_write(filename: Pathish, data: bytes) -> Path:
    target = Path(filename).absolute()
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            # Level 6 compresses faster than the default 9 for a slightly
            # larger file; the gain depends on the data.
            with gzip.GzipFile(
                filename="", fileobj=output, mode="wb", mtime=0, compresslevel=6
            ) as compressed:
                compressed.write(data)
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
                data = _serialize(payload, serializer, options.max_dump_bytes)
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
            return _atomic_write(filename, data)
        if stage == "minimal":
            raise last_error
        while True:
            stage = (
                {
                    "full": "globals",
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
            if stage not in ("globals", "outer") or not unchanged:
                break


def _options(options: DumpOptions | None) -> DumpOptions:
    if options is None:
        return DumpOptions()
    if not isinstance(options, DumpOptions):
        raise TypeError("options must be a DumpOptions instance or None")
    return options


def _default_filename(frame: FrameType, lineno: int) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(frame.f_code.co_filename).name)
    return f"{name or 'snapshot'}-{lineno}.dump"


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
        return {
            "traceback": traceback,
            "version": __version__,
            "snapshot_meta": context.metadata(),
        }

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
        innermost = exc_tb
        while innermost.tb_next is not None:
            innermost = innermost.tb_next
        filename = (
            filename
            if filename is not None
            else _default_filename(innermost.tb_frame, innermost.tb_lineno)
        )

        def factory(minimal: bool) -> dict:
            context = SnapshotContext(settings, minimal=minimal)
            traceback = context.from_traceback(exc_tb)
            extra = context.exception(exc_value)
            return {
                "traceback": traceback,
                "version": __version__,
                "exc_extra": extra,
                "snapshot_meta": context.metadata(),
            }

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
