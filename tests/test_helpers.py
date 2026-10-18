import argparse
import io
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
    monkeypatch.setattr(
        helpers,
        "format_exception",
        lambda *args: calls.append(args) or ["FalseyError: example\n"],
    )
    print_traceback_and_except(
        {
            "traceback": None,
            "exc_extra": {"exc_type": FalseyError, "exc_value": error},
            "snapshot_meta": {"truncation": {"frames": 2}},
        }
    )
    assert calls[0][1] is error
    captured = capsys.readouterr()
    # The report is the command's product; the notice is a diagnostic.
    assert captured.out == "FalseyError: example\n"
    assert "frames=2" in captured.err
    print_snapshot_notice({})
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    ("encoding", "errors", "text", "expected"),
    [
        ("ascii", "strict", "/tmp/\udcff caf\xe9", b"/tmp/\\udcff caf\\xe9"),
        # Redirected stdout on Windows: a legacy code page with surrogateescape.
        (
            "cp1252",
            "surrogateescape",
            "\u914d\u7f6e caf\xe9",
            b"\\u914d\\u7f6e caf\xe9",
        ),
        # UTF-8 mode: surrogateescape cannot encode a lone high surrogate.
        ("utf-8", "surrogateescape", "split \ud83d", b"split \\ud83d"),
    ],
)
def test_report_escapes_text_the_stream_cannot_encode(encoding, errors, text, expected):
    stream = io.TextIOWrapper(
        io.BytesIO(), encoding=encoding, errors=errors, newline="\n"
    )
    print_traceback_and_except(
        {"traceback": None, "exc_extra": {"formatted": [f"OSError: {text}\n"]}},
        file=stream,
    )
    stream.flush()
    # The report is escaped as standard error would escape it, not aborted.
    assert stream.buffer.getvalue() == b"OSError: " + expected + b"\n"


def test_discarding_a_stream_without_a_descriptor_is_a_no_op():
    from pydumpling.helpers import discard_stream

    stream = io.StringIO()
    discard_stream(stream)
    stream.write("still usable")
    assert stream.getvalue() == "still usable"


def test_report_keeps_what_the_stream_handler_can_encode():
    stream = io.TextIOWrapper(
        io.BytesIO(), encoding="utf-8", errors="surrogateescape", newline="\n"
    )
    print_traceback_and_except(
        {"traceback": None, "exc_extra": {"formatted": ["OSError: /tmp/\udcff\n"]}},
        file=stream,
    )
    stream.flush()
    # surrogateescape restores the undecodable byte of the original path.
    assert stream.buffer.getvalue() == b"OSError: /tmp/\xff\n"
