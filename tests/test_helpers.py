import argparse
import asyncio
import io
import sys
import threading
import warnings
from types import SimpleNamespace

import pytest

from pydumpling import DumpOptions, load_dumpling
from pydumpling.dump_manager import DumpManager
from pydumpling.helpers import (
    catch_any_exception,
    install_asyncio_handler,
    print_snapshot_notice,
    print_traceback_and_except,
    restore_exception_hooks,
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
    import pydumpling.hooks as hooks

    called = []
    monkeypatch.setattr(
        sys, "excepthook", lambda *info: called.append(("original", info))
    )
    monkeypatch.setattr(
        hooks, "save_dumping", lambda **kwargs: called.append(("dump", kwargs))
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

    monkeypatch.setattr(hooks, "save_dumping", fail)
    with pytest.raises(OSError):
        hook(ValueError, error, None)
    assert called[-1][0] == "original"
    with pytest.raises(TypeError):
        catch_any_exception(options={})


def test_falsey_exception_prints_and_notices_are_visible(capsys, monkeypatch):
    import pydumpling.report as report

    class FalseyError(Exception):
        def __bool__(self):
            return False

    error = FalseyError("example")
    calls = []
    monkeypatch.setattr(
        report,
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


@pytest.fixture
def clean_hooks(monkeypatch):
    """Record the current hooks so every installed hook is undone on exit."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    yield
    restore_exception_hooks()


def test_thread_hook_saves_and_preserves_previous_hook(monkeypatch, clean_hooks):
    import pydumpling.hooks as hooks

    called = []
    monkeypatch.setattr(
        threading, "excepthook", lambda args: called.append(("original", args))
    )
    monkeypatch.setattr(
        hooks, "save_dumping", lambda **kwargs: called.append(("dump", kwargs))
    )
    catch_any_exception(include_threads=True)
    hook = threading.excepthook

    def raise_value_error():
        raise ValueError("from a thread")

    try:
        raise_value_error()
    except ValueError as error:
        args = SimpleNamespace(
            exc_type=ValueError,
            exc_value=error,
            exc_traceback=error.__traceback__,
        )
    hook(args)
    assert [name for name, _ in called] == ["dump", "original"]
    # Re-running updates budgets without replacing the hook.
    settings = DumpOptions(max_frames=1)
    catch_any_exception(include_threads=True, options=settings)
    assert threading.excepthook is hook
    hook(args)
    assert called[-2][1]["options"] is settings
    # A manual call without a traceback saves nothing but still reports.
    hook(SimpleNamespace(exc_type=ValueError, exc_value=None, exc_traceback=None))
    assert [name for name, _ in called][-2:] == ["original", "original"]
    with pytest.raises(TypeError):
        catch_any_exception(include_threads=True, options={})


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_thread_hook_saves_a_real_thread_dump(tmp_path, monkeypatch, clean_hooks):
    monkeypatch.chdir(tmp_path)
    catch_any_exception(include_threads=True)
    started = threading.Event()

    def worker():
        started.set()
        raise ValueError("worker failure")

    thread = threading.Thread(target=worker, name="pydumpling-worker")
    thread.start()
    started.wait()
    thread.join()
    dumps = list(tmp_path.glob("*.dump"))
    assert len(dumps) == 1
    restored = load_dumpling(dumps[0])
    assert restored["snapshot_meta"]["thread_name"] == "pydumpling-worker"
    assert restored["exc_extra"]["type_name"].endswith("ValueError")


def test_restore_exception_hooks_returns_the_previous_hooks(clean_hooks):
    original = sys.excepthook
    original_thread = threading.excepthook
    catch_any_exception(include_threads=True)
    assert sys.excepthook is not original
    assert threading.excepthook is not original_thread
    restore_exception_hooks()
    assert sys.excepthook is original
    assert threading.excepthook is original_thread
    # Restore is idempotent.
    restore_exception_hooks()
    assert sys.excepthook is original


def test_restore_keeps_a_later_third_party_hook(clean_hooks):
    def third_party(*info):
        pass

    catch_any_exception(include_threads=True)
    sys.excepthook = third_party
    restore_exception_hooks()
    assert sys.excepthook is third_party


def test_hook_with_manager_saves_into_the_managed_directory(tmp_path, clean_hooks):
    manager = DumpManager(tmp_path / "dumps", retention=1)
    catch_any_exception(manager=manager)
    try:
        raise ValueError("example")
    except ValueError as error:
        sys.excepthook(ValueError, error, error.__traceback__)
    assert len(list((tmp_path / "dumps").glob("*.dump"))) == 1
    with pytest.raises(ValueError):
        catch_any_exception(options=DumpOptions(), manager=manager)
    with pytest.raises(TypeError):
        catch_any_exception(manager="not a manager")


def test_asyncio_handler_saves_and_still_reports(monkeypatch, clean_hooks):
    import pydumpling.hooks as hooks

    saved = []
    monkeypatch.setattr(hooks, "save_dumping", lambda **kwargs: saved.append(kwargs))
    loops = []
    settings = DumpOptions(max_frames=1)

    async def main():
        loop = asyncio.get_running_loop()
        loops.append(loop)
        install_asyncio_handler()
        handler = loop.get_exception_handler()
        # Re-running updates budgets without replacing the handler.
        install_asyncio_handler(options=settings)
        assert loop.get_exception_handler() is handler
        try:
            raise ValueError("from the loop")
        except ValueError as error:
            loop.call_exception_handler({"message": "example", "exception": error})

    asyncio.run(main())
    assert len(saved) == 1
    assert saved[0]["options"] is settings
    restore_exception_hooks()
    # The previous default handler is back in place.
    assert loops[0].get_exception_handler() is None


def test_asyncio_handler_requires_a_loop():
    with pytest.raises(RuntimeError, match="running event loop"):
        install_asyncio_handler()


@pytest.mark.parametrize("custom_handler", [False, True])
def test_asyncio_reports_original_exception_when_saving_fails(
    monkeypatch, clean_hooks, custom_handler
):
    import pydumpling.pydumpling as writer

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_stream_dump", fail)
    reported = []
    loop = asyncio.new_event_loop()
    try:
        if custom_handler:
            loop.set_exception_handler(lambda loop, context: reported.append(context))
        else:
            monkeypatch.setattr(loop, "default_exception_handler", reported.append)
        install_asyncio_handler(loop)
        try:
            raise ValueError("original failure")
        except ValueError as error:
            context = {"message": "task failed", "exception": error}
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="disk full"):
                loop.get_exception_handler()(loop, context)
        assert reported == [context]
    finally:
        loop.close()
