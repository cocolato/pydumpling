import argparse
import os.path
from .debug_dumpling import debug_dumpling, load_dumpling
from .helpers import print_traceback_and_except

DUMP_FILE_EXTENSION: str = ".dump"


def validate_file_name(file_name: str) -> str:
    """check file extension name and exists"""
    if not file_name.endswith(DUMP_FILE_EXTENSION):
        raise argparse.ArgumentTypeError("File must be .dump file")
    if not os.path.exists(file_name):
        raise argparse.ArgumentTypeError(f"File {file_name}  not found")
    return file_name


parser = argparse.ArgumentParser(
    description="pydumpling cli tools",
    prog="pydumpling",
    usage="%(prog)s [options] filename"
)

# print or debug
pydumpling_cli_action_group = parser.add_mutually_exclusive_group(required=True)

pydumpling_cli_action_group.add_argument(
    "--print",
    action="store_true",
    help="print traceback information"
)

pydumpling_cli_action_group.add_argument(
    "--debug",
    action="store_true",
    help="enter pdb debugging interface"
)

parser.add_argument(
    "filename",
    type=validate_file_name,
    help="the .dump file"
)


def main() -> None:
    args = parser.parse_args()
    file_name = args.filename
    if args.print:
        dumpling_ = load_dumpling(file_name)
        print_traceback_and_except(dumpling_)
    elif args.debug:
        debug_dumpling(file_name)
