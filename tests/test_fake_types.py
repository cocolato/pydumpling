import pytest
from unittest.mock import Mock, patch
from pydumpling.fake_types import FakeType, FakeTraceback, FakeFrame, FakeClass, FakeCode


# Fixture for mock traceback
@pytest.fixture
def mock_traceback():
    tb = Mock()
    tb.tb_frame = Mock()
    tb.tb_frame.f_code = Mock()
    tb.tb_frame.f_code.co_filename = "test.py"
    tb.tb_frame.f_code.co_name = "test_func"
    tb.tb_frame.f_code.co_consts = (None,)
    tb.tb_frame.f_code.co_firstlineno = 1
    tb.tb_frame.f_code.co_lnotab = b""
    tb.tb_frame.f_code.co_varnames = ("arg1",)
    tb.tb_frame.f_code.co_flags = 0
    tb.tb_frame.f_code.co_code = b""
    tb.tb_frame.f_code.co_lines = lambda: []
    tb.tb_frame.f_locals = {"var1": "val1"}
    tb.tb_frame.f_globals = {"global1": "globalval1"}
    tb.tb_frame.f_lineno = 42
    tb.tb_frame.f_back = None
    tb.tb_frame.f_lasti = 0
    tb.tb_frame.f_builtins = {}
    tb.tb_lineno = 42
    tb.tb_next = None
    tb.tb_lasti = 0
    return tb


# Fixture for mock frame
@pytest.fixture
def mock_frame():
    frame = Mock()
    frame.f_code = Mock()
    frame.f_code.co_filename = "test.py"
    frame.f_code.co_name = "test_func"
    frame.f_code.co_consts = (None,)
    frame.f_code.co_firstlineno = 1
    frame.f_code.co_lnotab = b""
    frame.f_code.co_varnames = ("arg1",)
    frame.f_code.co_flags = 0
    frame.f_code.co_code = b""
    frame.f_code.co_lines = lambda: []
    frame.f_locals = {"var1": "val1"}
    frame.f_globals = {"global1": "globalval1"}
    frame.f_lineno = 42
    frame.f_back = None
    frame.f_lasti = 0
    frame.f_builtins = {}
    return frame


@pytest.fixture
def mock_code():
    code = Mock()
    code.co_filename = "test.py"
    code.co_name = "test_func"
    code.co_argcount = 1
    code.co_consts = (None,)
    code.co_firstlineno = 1
    code.co_lnotab = b""
    code.co_varnames = ("arg1",)
    code.co_flags = 0
    code.co_code = b""
    code.co_lines = lambda: []
    return code


def test_fake_traceback_init_none():
    ft = FakeTraceback()
    assert ft.tb_frame is None
    assert ft.tb_lineno is None
    assert ft.tb_next is None
    assert ft.tb_lasti == 0


def test_fake_traceback_init(mock_traceback):
    ft = FakeTraceback(mock_traceback)
    assert isinstance(ft.tb_frame, FakeFrame)
    assert ft.tb_lineno == 42
    assert ft.tb_next is None
    assert ft.tb_lasti == 0


def test_fake_frame_init(mock_frame):
    ff = FakeFrame(mock_frame)
    assert isinstance(ff.f_code, FakeCode)
    assert ff.f_globals == {"global1": "globalval1"}
    assert ff.f_lineno == 42
    assert ff.f_back is None
    assert ff.f_lasti == 0
    assert ff.f_builtins == {}


def test_fake_code_init(mock_code):
    fc = FakeCode(mock_code)
    assert fc.co_filename.endswith("test.py")
    assert fc.co_name == "test_func"
    assert fc.co_argcount == 1
    assert fc.co_consts == (None,)
    assert fc.co_firstlineno == 1
    assert fc.co_lnotab == b""
    assert fc.co_varnames == ("arg1",)
    assert fc.co_flags == 0
    assert fc.co_code == b""
    assert fc._co_lines == []


def test_fake_code_co_lines(mock_code):
    mock_code.co_lines = lambda: [(1, 0, "line1")]
    fc = FakeCode(mock_code)
    assert list(fc.co_lines()) == [(1, 0, "line1")]


def test_fake_code_with_kwonlyargcount(mock_code):
    mock_code.co_kwonlyargcount = 1
    fc = FakeCode(mock_code)
    assert fc.co_kwonlyargcount == 1


def test_fake_code_with_positions(mock_code):
    mock_code.co_positions = [(1, 1, 2, 2)]
    fc = FakeCode(mock_code)
    assert fc.co_positions == [(1, 1, 2, 2)]


def test_safe_repr():
    class TestObject:
        def __repr__(self):
            raise ValueError("test error")

    obj = TestObject()
    result = FakeType._safe_repr(obj)
    assert result.startswith("repr error")


def test_convert():
    test_val = "test string"
    result = FakeType._convert(test_val)
    assert result == test_val


def test_convert_none():
    assert FakeType._convert(None) is None


def test_convert_sequence():
    test_list = [1, 2, 3]
    result = FakeType._convert(test_list)
    assert list(result) == test_list


def test_convert_dict():
    test_dict = {"key": "value"}
    result = FakeType._convert_dict(test_dict)
    assert result == {"key": "value"}


def test_convert_dict_with_complex_keys():
    test_dict = {1: "value1", "key2": "value2"}
    result = FakeType._convert_dict(test_dict)
    assert result == {1: "value1", "key2": "value2"}
