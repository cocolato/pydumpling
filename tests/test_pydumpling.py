import inspect
import os
import pickle
import sys
import warnings
from unittest.mock import MagicMock, mock_open, patch

import dill
import pytest

from pydumpling.fake_types import FakeFrame, FakeTraceback
from pydumpling.pydumpling import (dump_current_traceback, gen_tb_from_frame,
                                 save_dumping)


@pytest.fixture
def mock_frame():
    frame = MagicMock()
    frame.f_back = None
    frame.f_lasti = 123
    frame.f_lineno = 456
    frame.f_code.co_filename = os.path.abspath("test.py")
    frame.f_locals = {}
    frame.f_globals = {}
    frame.f_builtins = {}
    return frame


@pytest.fixture
def mock_traceback(mock_frame):
    tb = MagicMock()
    tb.tb_frame = mock_frame
    tb.tb_lasti = mock_frame.f_lasti
    tb.tb_lineno = mock_frame.f_lineno
    tb.tb_next = None
    return tb


def test_save_dumping_with_exc_info(tmp_path, mock_traceback):
    exc_type = ValueError
    exc_value = ValueError("test error")
    exc_info = (exc_type, exc_value, mock_traceback)

    filename = str(tmp_path / "test.dump")

    with patch("gzip.open", mock_open()) as mock_file:
        save_dumping(filename, exc_info)
        mock_file.assert_called_once_with(filename, "wb")


def test_save_dumping_without_filename(mock_traceback):
    exc_type = ValueError
    exc_value = ValueError("test error")
    exc_info = (exc_type, exc_value, mock_traceback)

    expected_filename = f"{os.path.abspath('test.py')}-456.dump"

    with patch("gzip.open", mock_open()) as mock_file:
        save_dumping(exc_info=exc_info)
        mock_file.assert_called_once_with(expected_filename, "wb")


def test_save_dumping_dill_fallback(mock_traceback):
    exc_type = ValueError
    exc_value = ValueError("test error")
    exc_info = (exc_type, exc_value, mock_traceback)

    with patch("gzip.open", mock_open()) as mock_file:
        with patch("dill.dump", side_effect=Exception("dill error")):
            with patch("pickle.dump") as mock_pickle:
                save_dumping("test.dump", exc_info)
                assert mock_pickle.called


def test_dump_current_traceback(tmp_path, mock_frame):
    filename = str(tmp_path / "test.dump")

    with patch("inspect.currentframe", return_value=mock_frame):
        with patch("gzip.open", mock_open()) as mock_file:
            dump_current_traceback(filename)
            mock_file.assert_called_once_with(filename, "wb")


def test_dump_current_traceback_without_filename(mock_frame):
    expected_filename = f"{os.path.abspath('test.py')}:456.dump"

    with patch("inspect.currentframe", return_value=mock_frame):
        with patch("gzip.open", mock_open()) as mock_file:
            dump_current_traceback()
            mock_file.assert_called_once_with(expected_filename, "wb")


def test_dump_current_traceback_dill_fallback(mock_frame):
    with patch("inspect.currentframe", return_value=mock_frame):
        with patch("gzip.open", mock_open()) as mock_file:
            with patch("dill.dump", side_effect=Exception("dill error")):
                with patch("pickle.dump") as mock_pickle:
                    dump_current_traceback("test.dump")
                    assert mock_pickle.called


def test_gen_tb_from_frame_no_back(mock_frame):
    tb = gen_tb_from_frame(mock_frame)
    assert isinstance(tb, FakeTraceback)
    assert tb.tb_frame.f_code.co_filename == os.path.abspath("test.py")
    assert tb.tb_lasti == mock_frame.f_lasti
    assert tb.tb_lineno == mock_frame.f_lineno
    assert tb.tb_next is None


def test_gen_tb_from_frame_with_back():
    back_frame = MagicMock()
    back_frame.f_back = None
    back_frame.f_lasti = 789
    back_frame.f_lineno = 101112
    back_frame.f_code.co_filename = os.path.abspath("back.py")
    back_frame.f_locals = {}
    back_frame.f_globals = {}
    back_frame.f_builtins = {}

    current_frame = MagicMock()
    current_frame.f_back = back_frame
    current_frame.f_lasti = 123
    current_frame.f_lineno = 456
    current_frame.f_code.co_filename = os.path.abspath("test.py")
    current_frame.f_locals = {}
    current_frame.f_globals = {}
    current_frame.f_builtins = {}

    tb = gen_tb_from_frame(current_frame)
    assert isinstance(tb, FakeTraceback)
    assert tb.tb_frame.f_code.co_filename == os.path.abspath("back.py")
    assert tb.tb_next is None


def test_save_dumping_error_warning():
    with patch("gzip.open", side_effect=Exception("test error")):
        with pytest.warns(RuntimeWarning, match="Unexpected error: test error"):
            save_dumping("test.dump")


def test_dump_current_traceback_error_warning():
    with patch("inspect.currentframe", side_effect=Exception("test error")):
        with pytest.warns(RuntimeWarning, match="Unexpected error: test error"):
            dump_current_traceback("test.dump")


def test_gen_tb_from_frame_multiple_back():
    frame3 = MagicMock()
    frame3.f_back = None
    frame3.f_lasti = 111
    frame3.f_lineno = 333
    frame3.f_code.co_filename = os.path.abspath("frame3.py")
    frame3.f_locals = {}
    frame3.f_globals = {}
    frame3.f_builtins = {}

    frame2 = MagicMock()
    frame2.f_back = frame3
    frame2.f_lasti = 222
    frame2.f_lineno = 222
    frame2.f_code.co_filename = os.path.abspath("frame2.py")
    frame2.f_locals = {}
    frame2.f_globals = {}
    frame2.f_builtins = {}

    frame1 = MagicMock()
    frame1.f_back = frame2
    frame1.f_lasti = 333
    frame1.f_lineno = 111
    frame1.f_code.co_filename = os.path.abspath("frame1.py")
    frame1.f_locals = {}
    frame1.f_globals = {}
    frame1.f_builtins = {}

    tb = gen_tb_from_frame(frame1)
    assert isinstance(tb, FakeTraceback)
    assert tb.tb_frame.f_code.co_filename == os.path.abspath("frame3.py")
    assert tb.tb_next.tb_frame.f_code.co_filename == os.path.abspath("frame2.py")
    assert tb.tb_next.tb_next is None
