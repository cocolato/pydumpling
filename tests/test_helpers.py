import pytest
import sys
import os.path
from argparse import ArgumentTypeError
from unittest.mock import Mock, patch, MagicMock
from pydumpling.fake_types import FakeCode, FakeFrame, FakeTraceback

from pydumpling.helpers import validate_file_name, print_traceback_and_except, catch_any_exception


def test_validate_file_name(tmp_path):
    # Create a temp dump file
    dump_file = tmp_path / "test.dump"
    dump_file.write_text("")

    assert validate_file_name(str(dump_file)) == str(dump_file)

    with pytest.raises(ArgumentTypeError, match="File must be .dump file"):
        validate_file_name("test.txt")

    with pytest.raises(ArgumentTypeError, match="File missing.dump not found"):
        validate_file_name("missing.dump")


def test_print_traceback_and_except(capsys):
    # Create a mock traceback
    mock_code = MagicMock(spec=FakeCode)
    mock_code.co_filename = "test.py"
    mock_code.co_name = "test_func"
    mock_code.co_positions = lambda: [(1, None, None, None)]

    mock_frame = MagicMock(spec=FakeFrame)
    mock_frame.f_code = mock_code
    mock_frame.f_lineno = 1
    mock_frame.f_globals = {}
    mock_frame.f_locals = {}

    mock_tb = MagicMock(spec=FakeTraceback)
    mock_tb.tb_frame = mock_frame
    mock_tb.tb_lineno = 1
    mock_tb.tb_next = None
    mock_tb.tb_lasti = 0

    # Test with exc_extra
    dumpling_result = {
        "traceback": mock_tb,
        "exc_extra": {
            "exc_type": ValueError,
            "exc_value": ValueError("test error")
        }
    }

    print_traceback_and_except(dumpling_result)
    captured = capsys.readouterr()
    assert "test.py" in captured.err
    assert "test error" in captured.err

    # Test without exc_extra
    dumpling_result = {"traceback": mock_tb}
    print_traceback_and_except(dumpling_result)
    captured = capsys.readouterr()
    assert "test.py" in captured.err

    # Test with None values
    dumpling_result = {
        "traceback": mock_tb,
        "exc_extra": {
            "exc_type": None,
            "exc_value": None
        }
    }
    print_traceback_and_except(dumpling_result)
    captured = capsys.readouterr()
    assert "test.py" in captured.err


def test_catch_any_exception():
    original_hook = sys.excepthook
    mock_save = Mock()

    with patch("pydumpling.helpers.save_dumping", mock_save):
        catch_any_exception()

        # Create a mock traceback
        mock_code = MagicMock(spec=FakeCode)
        mock_code.co_filename = "test.py"
        mock_code.co_name = "test_func"
        mock_code.co_positions = lambda: [(1, None, None, None)]

        mock_frame = MagicMock(spec=FakeFrame)
        mock_frame.f_code = mock_code
        mock_frame.f_lineno = 1
        mock_frame.f_globals = {}
        mock_frame.f_locals = {}

        mock_tb = MagicMock(spec=FakeTraceback)
        mock_tb.tb_frame = mock_frame
        mock_tb.tb_lineno = 1
        mock_tb.tb_next = None
        mock_tb.tb_lasti = 0

        # Simulate exception
        exc_type = ValueError
        exc_value = ValueError("test error")

        sys.excepthook(exc_type, exc_value, mock_tb)

        mock_save.assert_called_once_with(exc_info=(exc_type, exc_value, mock_tb))

    # Clean up
    sys.excepthook = original_hook
