import argparse

from .debug_dumpling import debug_dumpling, load_dumpling
from .helpers import print_traceback_and_except, validate_file_name
from .rpdb import r_post_mortem
from .vis import generate_vis

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

pydumpling_cli_action_group.add_argument(
    "--rdebug",
    action="store_true",
    help="enter rpdb debugging interface"
)

pydumpling_cli_action_group.add_argument(
    "--vis", action="store_true", help="visualize the traceback"
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
    elif args.rdebug:
        r_post_mortem(file_name)
    elif args.vis:
        generate_vis(file_name)
