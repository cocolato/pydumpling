"""Command-line entry points for stored snapshots and native PID capture."""

import argparse
import math
import sys

from ._version import __version__
from .capture import capture_pid
from .debug_dumpling import DEFAULT_MAX_BYTES, debug_dumpling, load_dumpling
from .helpers import discard_stream, print_traceback_and_except, validate_file_name
from .options import DumpOptions
from .rpdb import DEFAULT_ADDR, DEFAULT_PORT, r_post_mortem


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _port(value: str) -> int:
    result = int(value)
    if not 1 <= result <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return result


def _timeout(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return result


def _load_limit(value: str) -> int | None:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError(
            "load limit must be nonnegative; 0 disables it"
        )
    return result * 1024 * 1024 if result else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pydumpling",
        description="Save Python snapshots and inspect trusted .dump files.",
        epilog="Only load trusted dumps: dill/pickle can execute code. PID capture requires CPython 3.14+.",
    )
    parser.add_argument(
        "--version", action="version", version=f"pydumpling {__version__}"
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--print", action="store_true", help="print a stored traceback"
    )
    actions.add_argument(
        "--debug", action="store_true", help="debug a snapshot with pdb"
    )
    actions.add_argument(
        "--rdebug", action="store_true", help="serve snapshot pdb over TCP"
    )
    actions.add_argument(
        "--capture-pid",
        type=_positive_int,
        metavar="PID",
        help="save another process's main-thread stack",
    )
    parser.add_argument(
        "filename", nargs="?", type=validate_file_name, help="trusted .dump file"
    )
    parser.add_argument(
        "--output", metavar="FILE", help="required output path for PID capture"
    )
    parser.add_argument("--host", help=f"remote bind address (default: {DEFAULT_ADDR})")
    parser.add_argument(
        "--port", type=_port, help=f"remote bind port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--timeout",
        type=_timeout,
        help="seconds to wait for a remote connection or PID capture (PID default: 30)",
    )
    parser.add_argument(
        "--unlimited", action="store_true", help="disable PID snapshot budgets"
    )
    parser.add_argument(
        "--max-load-mib",
        type=_load_limit,
        default=argparse.SUPPRESS,
        metavar="MIB",
        help="decompressed input limit (default: 64; 0 disables)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return zero on success; usage errors exit 2 and operation errors return 1.

    An interrupt returns 130, and ``--print`` into a closed pipe returns 141.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.capture_pid is not None:
        if args.filename is not None or not args.output:
            parser.error(
                "--capture-pid requires --output and does not accept a dump filename"
            )
        if args.host is not None or args.port is not None:
            parser.error("--host and --port are only valid with --rdebug")
        if hasattr(args, "max_load_mib"):
            parser.error("--max-load-mib is only valid when reading a dump")
    else:
        if args.filename is None:
            parser.error("a .dump filename is required")
        if args.output is not None or args.unlimited:
            parser.error("--output and --unlimited are only valid with --capture-pid")
        if not args.rdebug and (
            args.host is not None or args.port is not None or args.timeout is not None
        ):
            parser.error(
                "--host, --port and --timeout require --rdebug or --capture-pid"
            )
    args.max_load_mib = getattr(args, "max_load_mib", DEFAULT_MAX_BYTES)
    try:
        if args.capture_pid is not None:
            output = capture_pid(
                args.capture_pid,
                args.output,
                options=DumpOptions.unlimited() if args.unlimited else None,
                timeout=args.timeout if args.timeout is not None else 30.0,
            )
            print(output)
        elif args.print:
            dumpling = load_dumpling(args.filename, max_bytes=args.max_load_mib)
            try:
                print_traceback_and_except(dumpling)
                sys.stdout.flush()
            except BrokenPipeError:
                # The report's reader closed the pipe early, as ``| head``
                # does; the notice handles its own stream.  Stop quietly like
                # tools ended by SIGPIPE, so the final flush cannot fail again.
                discard_stream(sys.stdout)
                return 141  # 128 + SIGPIPE
        elif args.debug:
            debug_dumpling(args.filename, max_bytes=args.max_load_mib)
        else:
            r_post_mortem(
                args.filename,
                addr=args.host if args.host is not None else DEFAULT_ADDR,
                port=args.port if args.port is not None else DEFAULT_PORT,
                timeout=args.timeout,
                max_bytes=args.max_load_mib,
            )
    except KeyboardInterrupt as error:
        detail = f": {error}" if str(error) else ""
        print(f"pydumpling: interrupted{detail}", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"pydumpling: {error}", file=sys.stderr)
        return 1
    return 0
