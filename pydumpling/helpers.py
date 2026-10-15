"""CLI formatting and installation of the optional exception hook."""

import argparse
import os.path
import sys
from traceback import print_exception, print_tb
from typing import TextIO

from .options import DumpOptions
from .pydumpling import save_dumping

DUMP_FILE_EXTENSION = ".dump"


def print_snapshot_notice(dumpling_result: dict, *, file: TextIO | None = None) -> None:
    metadata = dumpling_result.get("snapshot_meta") or {}
    truncation = metadata.get("truncation") or {}
    if truncation:
        reasons = ", ".join(
            f"{name}={count}" for name, count in sorted(truncation.items()) if count
        )
        if reasons:
            print(
                f"Snapshot is incomplete ({reasons}).",
                file=file if file is not None else sys.stderr,
            )


def print_traceback_and_except(dumpling_result: dict) -> None:
    print_snapshot_notice(dumpling_result)
    exc_tb = dumpling_result["traceback"]
    extra = dumpling_result.get("exc_extra") or {}
    formatted = extra.get("formatted")
    if formatted:
        print_tb(exc_tb)
        text = "".join(formatted) if isinstance(formatted, list) else str(formatted)
        print(text, file=sys.stderr, end="" if text.endswith("\n") else "\n")
    elif extra.get("exc_type") is not None and extra.get("exc_value") is not None:
        print_exception(extra["exc_type"], extra["exc_value"], exc_tb)
    else:
        print_tb(exc_tb)


def catch_any_exception(*, options: DumpOptions | None = None) -> None:
    """Install one snapshot hook and always preserve the previous exception hook."""
    if options is not None and not isinstance(options, DumpOptions):
        raise TypeError("options must be a DumpOptions instance or None")
    if getattr(sys.excepthook, "_pydumpling_hook", False):
        sys.excepthook._pydumpling_options = options
        return
    original_hook = sys.excepthook

    def hook(exc_type, exc_value, exc_tb):
        try:
            save_dumping(
                exc_info=(exc_type, exc_value, exc_tb), options=hook._pydumpling_options
            )
        finally:
            original_hook(exc_type, exc_value, exc_tb)

    hook._pydumpling_hook = True
    hook._pydumpling_options = options
    sys.excepthook = hook


def validate_file_name(file_name: str) -> str:
    """Validate file arguments while leaving reads and format errors to the loader."""
    if not file_name.endswith(DUMP_FILE_EXTENSION):
        raise argparse.ArgumentTypeError("File must be a .dump file")
    if not os.path.isfile(file_name):
        raise argparse.ArgumentTypeError(
            f"File {file_name} not found or not a regular file"
        )
    return file_name
