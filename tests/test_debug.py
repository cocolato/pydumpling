from pydumpling import load_dumpling, __version__
from pydumpling.fake_types import FakeTraceback
import pytest


@pytest.mark.order(2)
def test_debug():
    dumpling = load_dumpling("test.dump")
    assert isinstance(dumpling["traceback"], FakeTraceback)
    assert dumpling["version"] == __version__

    dumpling2 = load_dumpling("current_tb.dump")
    assert isinstance(dumpling2["traceback"], FakeTraceback)
    assert dumpling2["version"] == __version__
