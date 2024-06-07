import sys
import os.path
import argparse
from traceback import print_exception, print_tb

from .pydumpling import save_dumping


DUMP_FILE_EXTENSION: str = ".dump"


def print_traceback_and_except(dumpling_result):
    exc_tb = dumpling_result["traceback"]
    except_extra = dumpling_result.get("exc_extra")
    exc_type = except_extra["exc_type"] if except_extra else None
    exc_value = except_extra["exc_value"] if except_extra else None
    if exc_type and exc_value:
        print_exception(exc_type, exc_value, exc_tb)
    else:
        print_tb(exc_tb)


def catch_any_exception():
    original_hook = sys.excepthook

    def _hook(exc_type, exc_value, exc_tb):
        save_dumping(exc_info=(exc_type, exc_value, exc_tb))
        original_hook(exc_type, exc_value, exc_tb)  # call sys original hook

    sys.excepthook = _hook


def validate_file_name(file_name: str) -> str:
    """check file extension name and exists"""
    if not file_name.endswith(DUMP_FILE_EXTENSION):
        raise argparse.ArgumentTypeError("File must be .dump file")
    if not os.path.exists(file_name):
        raise argparse.ArgumentTypeError(f"File {file_name} not found")
    return file_name
