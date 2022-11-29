import os
from pydumpling import save_dumping
import pytest


def inner():
    a = 1
    b = "2"
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
