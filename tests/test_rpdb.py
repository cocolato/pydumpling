import importlib
import inspect
import io
import signal
import socket
import sys

import pytest

from pydumpling.rpdb import Rpdb, r_post_mortem
from tests.test_debug import make_dumpling, write_dumpling


def loopback_client(monkeypatch, commands):
    """Connect before accept, making real network checks bounded and deterministic."""
    module = importlib.import_module("pydumpling.rpdb")
    original_socket = socket.socket
    resources = {}
    original_streams = sys.stdin, sys.stdout, sys.stderr
    original_inspect = inspect.isframe, inspect.istraceback, inspect.iscode

    class Listener(original_socket):
        def accept(self):
            assert (sys.stdin, sys.stdout, sys.stderr) == original_streams
            assert (
                inspect.isframe,
                inspect.istraceback,
                inspect.iscode,
            ) == original_inspect
            peer = original_socket(socket.AF_INET, socket.SOCK_STREAM)
            peer.settimeout(2)
            resources["peer"] = peer
            resources["listener"] = self
            self.bound_address = self.getsockname()
            peer.connect(self.getsockname())
            if commands:
                peer.sendall(commands)
            else:
                peer.shutdown(socket.SHUT_WR)
            connection, address = super().accept()
            resources["connection"] = connection
            return connection, address

    monkeypatch.setattr(module.socket, "socket", Listener)
    return resources


@pytest.mark.parametrize("finish", [b"q\n", b"c\n", b""])
def test_remote_pdb_commands_and_cleanup(tmp_path, monkeypatch, capsys, finish):
    filename, _ = write_dumpling(tmp_path)
    commands = (
        b"p value\n!print('statement-output')\np print('expression-output')\n"
        b"p input('input-prompt')\nremote-input\n"
        b"display print('display-output')\ndisplay input('display-prompt')\n"
        b"display-value\ndisplay 1 / 0\n!print('after-display-error')\n"
    )
    resources = loopback_client(monkeypatch, commands + finish)
    # An explicit half-close makes EOF exercise the same command loop.
    if not finish:
        original_accept = socket.socket.accept

        def accept_and_eof(listener):
            result = original_accept(listener)
            resources["peer"].shutdown(socket.SHUT_WR)
            return result

        monkeypatch.setattr(socket.socket, "accept", accept_and_eof)
    streams = sys.stdin, sys.stdout, sys.stderr
    original_trace = sys.gettrace()
    original_sigint = signal.getsignal(signal.SIGINT)
    inspect_functions = inspect.isframe, inspect.istraceback, inspect.iscode
    original_precmd = Rpdb.precmd

    def check_between_commands(debugger, line):
        assert (sys.stdin, sys.stdout, sys.stderr) == streams
        return original_precmd(debugger, line)

    monkeypatch.setattr(Rpdb, "precmd", check_between_commands)
    try:
        r_post_mortem(filename, port=0, timeout=1)
        chunks = []
        while data := resources["peer"].recv(4096):
            chunks.append(data)
        output = b"".join(chunks).decode()
        assert "42" in output
        assert "statement-output" in output
        assert "expression-output" in output
        assert "remote-input" in output
        assert "(Pdb) display-output\n" in output
        assert "display input('display-prompt'): 'display-value'" in output
        assert "** raised ZeroDivisionError:" in output
        assert "after-display-error" in output
        process_output = capsys.readouterr()
        assert "display-output" not in process_output.out
        assert "display-prompt" not in process_output.out
        assert resources["connection"].gettimeout() is None
        assert resources["connection"].fileno() == -1
        assert resources["listener"].fileno() == -1
        with socket.socket() as rebound:
            rebound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            rebound.bind(resources["listener"].bound_address)
        assert (sys.stdin, sys.stdout, sys.stderr) == streams
        assert sys.gettrace() is original_trace
        assert signal.getsignal(signal.SIGINT) is original_sigint
        assert (
            inspect.isframe,
            inspect.istraceback,
            inspect.iscode,
        ) == inspect_functions
    finally:
        for resource in resources.values():
            resource.close()


def test_accept_timeout_releases_listener_and_preserves_stdio(monkeypatch):
    module = importlib.import_module("pydumpling.rpdb")
    original_socket = socket.socket
    sockets = []

    def make_socket(*args, **kwargs):
        listener = original_socket(*args, **kwargs)
        sockets.append(listener)
        return listener

    monkeypatch.setattr(module.socket, "socket", make_socket)
    streams = sys.stdin, sys.stdout, sys.stderr
    with pytest.raises(TimeoutError):
        Rpdb(port=0, timeout=0.02)
    assert sockets[0].fileno() == -1
    assert (sys.stdin, sys.stdout, sys.stderr) == streams


def test_remote_exception_and_repeated_shutdown(tmp_path, monkeypatch):
    filename, _ = write_dumpling(tmp_path)
    resources = loopback_client(monkeypatch, b"q\n")
    debuggers = []
    streams = sys.stdin, sys.stdout, sys.stderr

    def fail_interaction(debugger, frame, traceback):
        debuggers.append(debugger)
        with debugger._redirect_stdio():
            assert sys.stdout is debugger.stdout
            raise KeyboardInterrupt()

    monkeypatch.setattr(Rpdb, "interaction", fail_interaction)
    try:
        with pytest.raises(KeyboardInterrupt):
            r_post_mortem(filename, port=0, timeout=1)
        debuggers[0].shutdown()
        assert resources["connection"].fileno() == -1
        assert resources["listener"].fileno() == -1
        assert (sys.stdin, sys.stdout, sys.stderr) == streams
    finally:
        for resource in resources.values():
            resource.close()


def test_client_disconnect_releases_remote_resources(tmp_path, monkeypatch):
    filename, _ = write_dumpling(tmp_path)
    resources = loopback_client(monkeypatch, b"")
    original_accept = socket.socket.accept

    def accept_then_disconnect(listener):
        result = original_accept(listener)
        resources["peer"].close()
        return result

    monkeypatch.setattr(socket.socket, "accept", accept_then_disconnect)
    try:
        try:
            r_post_mortem(filename, port=0, timeout=1)
        except OSError:
            # A TCP close can be observed as EOF or as a reset while writing.
            pass
        assert resources["connection"].fileno() == -1
        assert resources["listener"].fileno() == -1
    finally:
        for resource in resources.values():
            resource.close()


def test_pdb_initialization_failure_releases_accepted_socket(monkeypatch):
    resources = loopback_client(monkeypatch, b"q\n")

    def fail_initialization(self, **kwargs):
        raise RuntimeError("Pdb initialization failed")

    monkeypatch.setattr(Rpdb.__bases__[0], "__init__", fail_initialization)
    try:
        with pytest.raises(RuntimeError, match="initialization failed"):
            Rpdb(port=0, timeout=1)
        assert resources["connection"].fileno() == -1
        assert resources["listener"].fileno() == -1
    finally:
        for resource in resources.values():
            resource.close()


def test_real_interactive_console_uses_remote_streams():
    debugger = Rpdb.__new__(Rpdb)
    output = io.StringIO()
    Rpdb.__bases__[0].__init__(
        debugger,
        stdin=io.StringIO("print(value)\nraise ValueError('console-error')\nexit()\n"),
        stdout=output,
        readrc=False,
    )
    debugger.reset()
    debugger.setup(None, make_dumpling()["traceback"])
    streams = sys.stdin, sys.stdout, sys.stderr
    try:
        debugger.onecmd("interact")
    except SystemExit:
        # Before Python 3.13, exit() leaves code.interact via SystemExit.
        pass
    assert "42" in output.getvalue()
    assert "console-error" in output.getvalue()
    assert (sys.stdin, sys.stdout, sys.stderr) == streams


def test_interact_restores_all_streams_after_failure(monkeypatch):
    debugger = Rpdb.__new__(Rpdb)
    Rpdb.__bases__[0].__init__(
        debugger, stdin=io.StringIO(), stdout=io.StringIO(), readrc=False
    )
    streams = sys.stdin, sys.stdout, sys.stderr

    def fail_interact(self, arg):
        assert sys.stdin is debugger.stdin
        assert sys.stdout is debugger.stdout
        assert sys.stderr is debugger.stdout
        raise RuntimeError("interactive console failed")

    monkeypatch.setattr(Rpdb.__bases__[0], "do_interact", fail_interact)
    with pytest.raises(RuntimeError):
        debugger.onecmd("interact")
    assert (sys.stdin, sys.stdout, sys.stderr) == streams
