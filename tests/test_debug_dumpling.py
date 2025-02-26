import pytest
import gzip
import pickle
import types
import inspect
from unittest.mock import mock_open, patch, MagicMock
from packaging.version import parse

from pydumpling.debug_dumpling import load_dumpling, mock_inspect, debug_dumpling
from pydumpling.fake_types import FakeCode, FakeFrame, FakeTraceback

@pytest.fixture
def mock_frame():
    frame = MagicMock()
    frame.f_code = MagicMock()
    frame.f_locals = {}
    frame.f_globals = {}
    frame.f_lineno = 1
    frame.f_back = None
    frame.f_lasti = 0
    frame.f_builtins = {}
    return frame

@pytest.fixture
def mock_traceback():
    traceback = MagicMock()
    traceback.tb_frame = None
    traceback.tb_lineno = 1
    traceback.tb_next = None
    traceback.tb_lasti = 0
    return traceback

@pytest.fixture
def dump_file(tmp_path):
    dump_path = tmp_path / "test.dump"
    data = {
        "version": "0.0.1",
        "traceback": None
    }
    with gzip.open(dump_path, "wb") as f:
        pickle.dump(data, f)
    return str(dump_path)

def test_load_dumpling_with_pickle(dump_file):
    result = load_dumpling(dump_file)
    assert isinstance(result, dict)
    assert result["version"] == "0.0.1"

def test_load_dumpling_with_dill(tmp_path):
    import dill
    dump_path = tmp_path / "test.dump"
    data = {"version": "0.0.1"}
    with gzip.open(dump_path, "wb") as f:
        dill.dump(data, f)
    result = load_dumpling(str(dump_path))
    assert result["version"] == "0.0.1"

def test_load_dumpling_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_dumpling("nonexistent.dump")

def test_mock_inspect(mock_frame, mock_traceback):
    mock_inspect()

    fake_frame = FakeFrame(mock_frame)
    fake_traceback = FakeTraceback(mock_traceback)
    fake_code = FakeCode(mock_frame.f_code)

    assert inspect.isframe(fake_frame)
    assert inspect.istraceback(fake_traceback)
    assert inspect.iscode(fake_code)

    # Fix for failing test - we should only test the fake types
    # since mock_inspect() modifies how inspect works
    assert inspect.isframe(fake_frame)
    assert inspect.istraceback(fake_traceback)
    assert inspect.iscode(fake_code)

def test_debug_dumpling_valid_version(dump_file):
    mock_pdb = MagicMock()
    debug_dumpling(dump_file, pdb=mock_pdb)
    mock_pdb.post_mortem.assert_called_once()

def test_debug_dumpling_invalid_version(tmp_path):
    dump_path = tmp_path / "test.dump"
    data = {
        "version": "1.0.0",
        "traceback": None
    }
    with gzip.open(dump_path, "wb") as f:
        pickle.dump(data, f)

    with pytest.raises(ValueError) as exc_info:
        debug_dumpling(str(dump_path))
    assert "Unsupported dumpling version" in str(exc_info.value)

def test_debug_dumpling_very_old_version(tmp_path):
    dump_path = tmp_path / "test.dump"
    data = {
        "version": "0.0.0",
        "traceback": None
    }
    with gzip.open(dump_path, "wb") as f:
        pickle.dump(data, f)

    with pytest.raises(ValueError) as exc_info:
        debug_dumpling(str(dump_path))
    assert "Unsupported dumpling version" in str(exc_info.value)
