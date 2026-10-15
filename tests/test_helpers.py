import argparse
import sys

import pytest

from pydumpling import DumpOptions
from pydumpling.helpers import (
    catch_any_exception,
    print_snapshot_notice,
    print_traceback_and_except,
    validate_file_name,
)


def test_file_validation_requires_regular_dump(tmp_path):
    path = tmp_path / "valid.dump"
    path.touch()
    assert validate_file_name(str(path)) == str(path)
    with pytest.raises(argparse.ArgumentTypeError, match="\\.dump"):
        validate_file_name("invalid.txt")
    with pytest.raises(argparse.ArgumentTypeError, match="not found"):
        validate_file_name(str(tmp_path / "missing.dump"))
    directory = tmp_path / "directory.dump"
    directory.mkdir()
    with pytest.raises(argparse.ArgumentTypeError, match="regular file"):
        validate_file_name(str(directory))


def test_exception_hook_is_idempotent_and_preserves_previous_hook(monkeypatch):
    import pydumpling.helpers as helpers

    called = []
    monkeypatch.setattr(
        sys, "excepthook", lambda *info: called.append(("original", info))
    )
    monkeypatch.setattr(
        helpers, "save_dumping", lambda **kwargs: called.append(("dump", kwargs))
    )
    catch_any_exception()
    hook = sys.excepthook
    settings = DumpOptions(max_frames=1)
    catch_any_exception(options=settings)
    assert sys.excepthook is hook
    error = ValueError("example")
    hook(ValueError, error, None)
    assert [name for name, _ in called] == ["dump", "original"]
    assert called[0][1]["options"] is settings

    def fail(**kwargs):
        raise OSError("save failed")

    monkeypatch.setattr(helpers, "save_dumping", fail)
    with pytest.raises(OSError):
        hook(ValueError, error, None)
    assert called[-1][0] == "original"
    with pytest.raises(TypeError):
        catch_any_exception(options={})


def test_falsey_exception_prints_and_notices_are_visible(capsys, monkeypatch):
    import pydumpling.helpers as helpers

    class FalseyError(Exception):
        def __bool__(self):
            return False

    error = FalseyError("example")
    calls = []
    monkeypatch.setattr(helpers, "print_exception", lambda *args: calls.append(args))
    print_traceback_and_except(
        {
            "traceback": None,
            "exc_extra": {"exc_type": FalseyError, "exc_value": error},
            "snapshot_meta": {"truncation": {"frames": 2}},
        }
    )
    assert calls[0][1] is error
    assert "frames=2" in capsys.readouterr().err
    print_snapshot_notice({})
    assert capsys.readouterr().err == ""
