"""Compatibility re-exports; the implementations moved to focused modules.

Hooks live in :mod:`pydumpling.hooks` and report formatting in
:mod:`pydumpling.report`. Import from those modules in new code.
"""

from .hooks import (
    catch_any_exception,
    install_asyncio_handler,
    restore_exception_hooks,
)
from .pydumpling import DUMP_FILE_EXTENSION, save_dumping
from .report import (
    discard_stream,
    print_snapshot_notice,
    print_traceback_and_except,
    validate_file_name,
)

__all__ = [
    "DUMP_FILE_EXTENSION",
    "catch_any_exception",
    "discard_stream",
    "install_asyncio_handler",
    "print_snapshot_notice",
    "print_traceback_and_except",
    "restore_exception_hooks",
    "save_dumping",
    "validate_file_name",
]
