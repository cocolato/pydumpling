import socket
import sys
import threading
from unittest.mock import Mock, patch

import pytest

from pydumpling.rpdb import DEFAULT_ADDR, DEFAULT_PORT, FileObjectWrapper, OccupiedPorts, Rpdb, r_post_mortem


@pytest.fixture
def mock_socket():
    with patch('socket.socket') as mock:
        sock = Mock()
        sock.getsockname.return_value = ('127.0.0.1', 4444)
        mock.return_value = sock
        yield sock


@pytest.fixture
def mock_file_handle():
    handle = Mock()
    handle.makefile.return_value = Mock()
    return handle


@pytest.fixture
def occupied_ports():
    return OccupiedPorts()


def test_file_object_wrapper_gets_fileobject_attr():
    fileobj = Mock()
    fileobj.test_attr = "test_value"
    stdio = Mock()
    wrapper = FileObjectWrapper(fileobj, stdio)
    assert wrapper.test_attr == "test_value"


def test_file_object_wrapper_gets_stdio_attr():
    fileobj = Mock()
    stdio = Mock()
    stdio.test_attr = "test_value"
    wrapper = FileObjectWrapper(fileobj, stdio)
    wrapper.test_attr = getattr(stdio, 'test_attr')
    assert wrapper.test_attr == "test_value"


def test_file_object_wrapper_attr_error():
    fileobj = Mock(spec=[])
    stdio = Mock(spec=[])
    wrapper = FileObjectWrapper(fileobj, stdio)
    with pytest.raises(AttributeError, match="Attribute nonexistent_attr is not found"):
        wrapper.nonexistent_attr


def test_rpdb_init(mock_socket, mock_file_handle):
    mock_socket.accept.return_value = (mock_file_handle, '127.0.0.1')

    with patch('sys.stdout'), patch('sys.stdin'), patch('sys.stderr'):
        debugger = Rpdb()

        mock_socket.setsockopt.assert_called_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        mock_socket.bind.assert_called_with((DEFAULT_ADDR, DEFAULT_PORT))
        mock_socket.listen.assert_called_with(1)
        mock_socket.accept.assert_called_once()


def test_rpdb_init_stderr_error(mock_socket, mock_file_handle):
    mock_socket.accept.return_value = (mock_file_handle, '127.0.0.1')

    with patch('sys.stdout'), patch('sys.stdin'), \
         patch('sys.stderr.write', side_effect=IOError):
        debugger = Rpdb()
        mock_socket.accept.assert_called_once()


def test_rpdb_shutdown(mock_socket, mock_file_handle):
    mock_socket.accept.return_value = (mock_file_handle, '127.0.0.1')

    with patch('sys.stdout') as mock_stdout, patch('sys.stdin') as mock_stdin, patch('sys.stderr'):
        debugger = Rpdb()
        debugger.shutdown()

        assert sys.stdout == mock_stdout
        assert sys.stdin == mock_stdin
        mock_socket.shutdown.assert_called_with(socket.SHUT_RDWR)
        mock_socket.close.assert_called_once()


def test_rpdb_do_continue(mock_socket, mock_file_handle):
    mock_socket.accept.return_value = (mock_file_handle, '127.0.0.1')

    with patch('sys.stdout'), patch('sys.stdin'), patch('sys.stderr'), \
         patch('pdb.Pdb.do_continue') as mock_continue:
        debugger = Rpdb()
        mock_continue.return_value = None
        debugger.do_continue('arg')

        mock_continue.assert_called_once_with(debugger, 'arg')
        mock_socket.shutdown.assert_called_with(socket.SHUT_RDWR)
        mock_socket.close.assert_called_once()


def test_rpdb_do_quit(mock_socket, mock_file_handle):
    mock_socket.accept.return_value = (mock_file_handle, '127.0.0.1')

    with patch('sys.stdout'), patch('sys.stdin'), patch('sys.stderr'), \
         patch('pdb.Pdb.do_quit') as mock_quit:
        debugger = Rpdb()
        mock_quit.return_value = None
        debugger.do_quit('arg')

        mock_quit.assert_called_once_with(debugger, 'arg')
        mock_socket.shutdown.assert_called_with(socket.SHUT_RDWR)
        mock_socket.close.assert_called_once()


def test_rpdb_do_EOF(mock_socket, mock_file_handle):
    mock_socket.accept.return_value = (mock_file_handle, '127.0.0.1')

    with patch('sys.stdout'), patch('sys.stdin'), patch('sys.stderr'), \
         patch('pdb.Pdb.do_EOF') as mock_eof:
        debugger = Rpdb()
        mock_eof.return_value = None
        debugger.do_EOF('arg')

        mock_eof.assert_called_once_with(debugger, 'arg')
        mock_socket.shutdown.assert_called_with(socket.SHUT_RDWR)
        mock_socket.close.assert_called_once()


def test_occupied_ports_claim(occupied_ports):
    handle = Mock()
    occupied_ports.claim(4444, handle)
    assert occupied_ports.is_claimed(4444, handle)


def test_occupied_ports_unclaim(occupied_ports):
    handle = Mock()
    occupied_ports.claim(4444, handle)
    occupied_ports.unclaim(4444)
    assert not occupied_ports.is_claimed(4444, handle)


def test_occupied_ports_is_claimed_false(occupied_ports):
    handle1 = Mock()
    handle2 = Mock()
    occupied_ports.claim(4444, handle1)
    assert not occupied_ports.is_claimed(4444, handle2)


def test_occupied_ports_thread_safety(occupied_ports):
    def claim_port():
        handle = Mock()
        occupied_ports.claim(4444, handle)
        assert occupied_ports.is_claimed(4444, handle)
        occupied_ports.unclaim(4444)

    threads = [threading.Thread(target=claim_port) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


@patch('pydumpling.rpdb.mock_inspect')
@patch('pydumpling.rpdb.load_dumpling')
@patch('pydumpling.rpdb.Rpdb')
def test_r_post_mortem(mock_rpdb_class, mock_load, mock_inspect):
    mock_debugger = Mock()
    mock_rpdb_class.return_value = mock_debugger
    mock_load.return_value = {"traceback": Mock()}

    r_post_mortem("dump.file")

    mock_inspect.assert_called_once()
    mock_load.assert_called_with("dump.file")
    mock_debugger.interaction.assert_called_once()


@patch('pydumpling.rpdb.mock_inspect')
@patch('pydumpling.rpdb.load_dumpling')
@patch('pydumpling.rpdb.Rpdb')
def test_r_post_mortem_custom_addr_port(mock_rpdb_class, mock_load, mock_inspect):
    mock_debugger = Mock()
    mock_rpdb_class.return_value = mock_debugger
    mock_load.return_value = {"traceback": Mock()}

    r_post_mortem("dump.file", addr="0.0.0.0", port=5555)

    mock_rpdb_class.assert_called_once_with(addr="0.0.0.0", port=5555)
    mock_inspect.assert_called_once()
    mock_load.assert_called_with("dump.file")
    mock_debugger.interaction.assert_called_once()
