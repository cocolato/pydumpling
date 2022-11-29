from pydumpling import load_dumpling, __version__
from pydumpling.fake_types import FakeTraceback
import pytest


@pytest.mark.order(2)
def test_debug():
    dumpling = load_dumpling("test.dump")
    assert isinstance(dumpling["traceback"], FakeTraceback)
    assert dumpling["version"] == __version__
