"""A managed dump directory: unique names in, oldest snapshots out."""

import inspect
import stat
from os import PathLike
from pathlib import Path
from types import FrameType, TracebackType

from .options import DumpOptions
from .pydumpling import (
    DUMP_FILE_EXTENSION,
    _capture_frame,
    _dump_order,
    _dump_target,
    _exc_info,
    _innermost_traceback,
    _warn_or_raise,
    save_dumping,
)

ExcInfo = tuple[type[BaseException], BaseException, TracebackType]


class DumpManager:
    """Save snapshots into one directory, keeping only the newest dumps.

    Every save writes a uniquely named file (see :func:`target`) and then
    deletes the oldest dumps beyond ``retention``. Pass ``options`` once to
    apply capture budgets to every save; a per-call ``options`` overrides it.
    """

    def __init__(
        self,
        directory: PathLike[str] | str,
        *,
        retention: int | None = 50,
        options: DumpOptions | None = None,
    ):
        if retention is not None and (type(retention) is not int or retention <= 0):
            raise ValueError("retention must be a positive integer or None")
        self._check(options)
        self.directory = Path(directory)
        self.retention = retention
        self.options = options

    def target(self, *, stem: str = "snapshot", lineno: int | None = None) -> Path:
        """A fresh dump path in the managed directory, which is created first.

        An unwritable directory raises here, before any capture work.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        return _dump_target(stem, lineno, directory=self.directory)

    def save_exception(
        self,
        exc_info: ExcInfo | None = None,
        *,
        options: DumpOptions | None = None,
        strict: bool = False,
    ) -> Path | None:
        """Save an exception snapshot like :func:`save_dumping`, then prune.

        Returns the written path, or ``None`` when a non-strict save failed.
        """
        self._check(options)
        try:
            exc_info = _exc_info(exc_info)
            innermost = _innermost_traceback(exc_info[2])
            filename = self.target(
                stem=Path(innermost.tb_frame.f_code.co_filename).name,
                lineno=innermost.tb_lineno,
            )
            save_dumping(
                filename, exc_info, options=self._settings(options), strict=True
            )
        except Exception as error:
            _warn_or_raise(error, strict)
            return None
        self.prune()
        return filename

    def save_current(
        self, *, options: DumpOptions | None = None, strict: bool = False
    ) -> Path | None:
        """Save the caller's stack like :func:`dump_current_traceback`, then prune.

        The manager's own frames stay out of the captured stack.
        """
        self._check(options)
        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            if not isinstance(caller, FrameType):
                raise ValueError("No Python frame is available to capture")
            filename = self.target(
                stem=Path(caller.f_code.co_filename).name, lineno=caller.f_lineno
            )
            _capture_frame(caller, filename, options=self._settings(options))
        except Exception as error:
            _warn_or_raise(error, strict)
            return None
        finally:
            del frame
        self.prune()
        return filename

    def prune(self) -> list[Path]:
        """Delete the oldest dumps beyond ``retention``; return removed paths.

        Order by file modification time. Coarse filesystem clocks give quick
        successive saves equal times; the UTC timestamp and sequence in
        generated names then decide, never the source file or line.
        """
        if self.retention is None:
            return []
        dumps = []
        for path in self.directory.glob("*" + DUMP_FILE_EXTENSION):
            try:
                info = path.stat()
            except OSError:
                continue  # Another saver may have pruned this file already.
            if stat.S_ISREG(info.st_mode):
                dumps.append((info.st_mtime_ns, *_dump_order(path.name), path))
        dumps.sort()
        removed = []
        for *_, path in dumps[: max(0, len(dumps) - self.retention)]:
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                continue  # A concurrently removed file is already pruned.
        return removed

    def _check(self, options: DumpOptions | None) -> None:
        if options is not None and not isinstance(options, DumpOptions):
            raise TypeError("options must be a DumpOptions instance or None")

    def _settings(self, options: DumpOptions | None) -> DumpOptions | None:
        return options if options is not None else self.options
