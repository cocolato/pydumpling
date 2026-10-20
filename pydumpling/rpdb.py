"""A small TCP transport for inspecting trusted offline snapshots."""

import math
import pdb
import socket
import sys
import threading
from contextlib import contextmanager
from os import PathLike

from .debug_dumpling import (
    _COMPAT_LOCK,
    DEFAULT_MAX_BYTES,
    load_dumpling,
    mock_inspect,
)

DEFAULT_ADDR = "127.0.0.1"
DEFAULT_PORT = 4444


class FileObjectWrapper:
    """Retained for users of the internal module."""

    def __init__(self, fileobject, stdio):
        self._obj = fileobject
        self._io = stdio

    def __getattr__(self, attr):
        try:
            return getattr(self._obj, attr)
        except AttributeError:
            return getattr(self._io, attr)


class Rpdb(pdb.Pdb):
    def __init__(
        self,
        addr: str = DEFAULT_ADDR,
        port: int = DEFAULT_PORT,
        *,
        timeout: float | None = None,
    ) -> None:
        """Wait for one client; timeout limits connection waiting only."""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number or None")
        self.port = port
        self.skt = None
        self.clientsocket = None
        self.handle = None
        self._closed = False
        try:
            self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.skt.bind((addr, port))
            self.port = self.skt.getsockname()[1]
            self.skt.listen(1)
            self.skt.settimeout(timeout)
            try:
                sys.stderr.write("pdb is running on %s:%d\n" % self.skt.getsockname())
            except (OSError, ValueError):
                pass
            self.clientsocket, _ = self.skt.accept()
            self.clientsocket.settimeout(None)
            self.handle = self.clientsocket.makefile(
                "rw", encoding="utf-8", newline="\n"
            )
            super().__init__(completekey="tab", stdin=self.handle, stdout=self.handle)
        except BaseException:
            self.shutdown()
            raise

    @contextmanager
    def _redirect_stdio(self):
        # Every command can execute Python or invoke user-defined repr hooks.
        # Keep all command IO remote without replacing stdio between commands.
        with _COMPAT_LOCK:
            original = sys.stdin, sys.stdout, sys.stderr
            sys.stdin, sys.stdout, sys.stderr = self.stdin, self.stdout, self.stdout
            try:
                yield
            finally:
                sys.stdin, sys.stdout, sys.stderr = original

    def onecmd(self, line):
        with self._redirect_stdio():
            return super().onecmd(line)

    def preloop(self):
        # Python 3.11 reevaluates registered display expressions here, before
        # onecmd runs. Newer versions dispatch this through an internal command.
        with self._redirect_stdio():
            return super().preloop()

    def shutdown(self):
        """Close acquired resources, also after partial setup or prior cleanup."""
        if self._closed:
            return
        self._closed = True
        for resource in (self.handle, self.clientsocket, self.skt):
            if resource is not None:
                try:
                    resource.close()
                except (OSError, ValueError):
                    pass

    def do_continue(self, arg):
        try:
            return super().do_continue(arg)
        finally:
            self.shutdown()

    do_c = do_cont = do_continue

    def do_quit(self, arg):
        try:
            return super().do_quit(arg)
        finally:
            self.shutdown()

    do_q = do_exit = do_quit

    def do_EOF(self, arg):
        try:
            return super().do_EOF(arg)
        finally:
            self.shutdown()


class OccupiedPorts:
    """Legacy dump compatibility; new sessions do not register ports here."""

    def __init__(self):
        self.lock = threading.RLock()
        self.claims = {}

    def claim(self, port, handle):
        with self.lock:
            self.claims[port] = id(handle)

    def is_claimed(self, port, handle):
        with self.lock:
            return self.claims.get(port) == id(handle)

    def unclaim(self, port):
        with self.lock:
            self.claims.pop(port, None)


def r_post_mortem(
    dump_file: str | PathLike[str],
    addr: str = DEFAULT_ADDR,
    port: int = DEFAULT_PORT,
    *,
    timeout: float | None = None,
    max_bytes: int | None = DEFAULT_MAX_BYTES,
) -> None:
    dumpling = load_dumpling(dump_file, max_bytes=max_bytes)
    from .report import print_snapshot_notice

    # Accept neither holds the compatibility lock nor changes process stdio.
    debugger = Rpdb(addr=addr, port=port, timeout=timeout)
    try:
        with mock_inspect(dumpling.get("source")):
            print_snapshot_notice(dumpling, file=debugger.stdout)
            debugger.reset()
            debugger.interaction(None, dumpling["traceback"])
    finally:
        debugger.shutdown()
