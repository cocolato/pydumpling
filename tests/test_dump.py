import os
from pydumpling import save_dumping, dump_current_traceback
import pytest


def inner():
    a = 1
    b = "2"
    dump_current_traceback("current_tb.dump")
    c = a + b  # noqa: F841


def outer():
    inner()


@pytest.mark.order(1)
def test_dumpling():
    try:
        outer()
    except Exception as e:
        print(e)
        save_dumping(filename="test.dump")

    assert os.path.exists("test.dump")
