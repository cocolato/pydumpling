import json
import subprocess
import sys

import pytest

from pydumpling import DumpOptions, dump_current_traceback, save_dumping
from pydumpling.cli import main


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "example.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    return path


def test_module_help_version_and_print(snapshot, capsys):
    for argument in ("--help", "--version"):
        result = subprocess.run(
            [sys.executable, "-m", "pydumpling", argument],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "pydumpling" in result.stdout
    assert main(["--print", str(snapshot)]) == 0
    captured = capsys.readouterr()
    # The traceback is the command's product and must survive a pipe.
    assert "in snapshot" in captured.out
    assert "Snapshot is incomplete" not in captured.out


def test_print_sends_the_report_to_stdout_and_the_notice_to_stderr(tmp_path, capsys):
    path = tmp_path / "notice.dump"
    dump_current_traceback(
        path, options=DumpOptions(max_frames=1, max_items=1), strict=True
    )
    assert main(["--print", str(path)]) == 0
    captured = capsys.readouterr()
    # A pipe or a redirect must receive the whole traceback...
    assert 'File "' in captured.out
    # ...while the diagnostic notice stays on stderr.
    assert "Snapshot is incomplete" in captured.err
    assert "Snapshot is incomplete" not in captured.out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe semantics")
def test_print_into_a_closed_pipe_stops_quietly(snapshot):
    import os

    reader, writer = os.pipe()
    os.close(reader)  # The reader is gone before the report is written.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pydumpling", "--print", str(snapshot)],
            stdout=writer,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        os.close(writer)
    # Like a tool ended by SIGPIPE: status 128 + 13 and no error report.
    assert result.returncode == 141
    assert "Broken pipe" not in result.stderr
    assert "Exception ignored" not in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe semantics")
def test_print_with_a_closed_stderr_still_writes_the_report(snapshot):
    import os

    reader, writer = os.pipe()
    os.close(reader)  # Only the notice's reader is gone.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pydumpling", "--print", str(snapshot)],
            stdout=subprocess.PIPE,
            stderr=writer,
            text=True,
            check=False,
        )
    finally:
        os.close(writer)
    assert result.returncode == 0
    assert 'File "' in result.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--print"],
        ["--capture-pid", "12"],
        ["--capture-pid", "0"],
        ["--capture-pid", "12", "--output", "test.dump", "--port", "4444"],
        ["--capture-pid", "12", "--output", "test.dump", "--max-load-mib", "1"],
        ["--capture-pid", "12", "--output", "test.dump", "--max-load-mib", "64"],
        ["--capture-pid", "12", "--output", "test.dump", "--max-load-mib", "0"],
        ["--capture-pid", "12", "--output", "test.dump", "--format", "json"],
        ["--capture-pid", "12", "--output", "test.dump", "--globals"],
        ["--rdebug", "--port", "0"],
        ["--rdebug", "--timeout", "nan"],
        ["--print", "--max-load-mib", "-1"],
    ],
)
def test_invalid_usage(arguments, monkeypatch):
    def unexpected_capture(*args, **kwargs):
        pytest.fail("invalid usage must not attempt PID capture")

    monkeypatch.setattr("pydumpling.cli.capture_pid", unexpected_capture)
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_file_action_option_validation_and_corrupt_dump(snapshot, tmp_path, capsys):
    for extra in (["--host", "localhost"], ["--output", "out.dump"], ["--unlimited"]):
        with pytest.raises(SystemExit) as error:
            main(["--print", str(snapshot), *extra])
        assert error.value.code == 2
    corrupt = tmp_path / "corrupt.dump"
    corrupt.touch()
    assert main(["--print", str(corrupt)]) == 1
    stderr = capsys.readouterr().err
    assert "pydumpling:" in stderr
    assert "Traceback (most recent call last)" not in stderr


def test_cli_dispatches_remote_local_and_pid_options(snapshot, monkeypatch, capsys):
    import pydumpling.cli as cli

    calls = []
    monkeypatch.setattr(
        cli, "debug_dumpling", lambda *args, **kwargs: calls.append(("debug", kwargs))
    )
    monkeypatch.setattr(
        cli, "r_post_mortem", lambda *args, **kwargs: calls.append(("remote", kwargs))
    )
    monkeypatch.setattr(
        cli,
        "capture_pid",
        lambda *args, **kwargs: calls.append(("pid", kwargs)) or snapshot,
    )
    assert main(["--debug", str(snapshot), "--max-load-mib", "0"]) == 0
    assert calls[-1][1]["max_bytes"] is None
    for extra in ([], ["--max-load-mib", "64"]):
        assert main(["--debug", str(snapshot), *extra]) == 0
        assert calls[-1][1]["max_bytes"] == 64 * 1024 * 1024
    assert (
        main(
            [
                "--rdebug",
                str(snapshot),
                "--host",
                "127.0.0.1",
                "--port",
                "5555",
                "--timeout",
                "2",
            ]
        )
        == 0
    )
    assert calls[-1][1]["port"] == 5555 and calls[-1][1]["timeout"] == 2
    assert main(["--capture-pid", "123", "--output", str(snapshot), "--unlimited"]) == 0
    assert calls[-1][1]["options"] == DumpOptions.unlimited()
    assert str(snapshot) in capsys.readouterr().out

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "capture_pid", interrupt)
    assert main(["--capture-pid", "123", "--output", str(snapshot)]) == 130


def test_json_and_info_usage_validation(snapshot):
    # Actions are mutually exclusive; --globals requires --json.
    for arguments in (
        ["--print", "--json", str(snapshot)],
        ["--print", "--format", "json", str(snapshot)],
        ["--globals", "--print", str(snapshot)],
        ["--globals", "--debug", str(snapshot)],
        ["--globals", "--info", str(snapshot)],
        ["--json", "--globals", "--host", "localhost", str(snapshot)],
    ):
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 2


def test_json_reports_stack_snapshot(snapshot, capsys):
    assert main(["--json", str(snapshot)]) == 0
    report = json.loads(capsys.readouterr().out)
    # A stack snapshot has no exception; frames run outermost to innermost.
    assert report["exception"] is None
    assert report["created_at"].endswith("Z")
    assert report["thread_name"] == "MainThread"
    frame = report["frames"][0]
    assert frame["filename"].endswith("test_cli.py")
    assert frame["qualname"] == "snapshot"
    assert type(frame["locals"]["path"]) is str
    assert "globals" not in frame
    # The fixture keeps one frame of a long pytest caller chain.
    assert report["truncation"]["max_frames"] >= 1


def test_json_reports_exception_and_truncation(tmp_path, capsys):
    path = tmp_path / "failure.dump"

    try:
        quantity = 3
        payload = list(range(50))
        raise TypeError("bad combination")
    except TypeError:
        save_dumping(path, options=DumpOptions(max_items=8), strict=True)
    assert quantity == 3 and len(payload) == 50

    assert main(["--json", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["exception"]["type"].endswith("TypeError")
    assert report["exception"]["message"] == "bad combination"
    assert "TypeError" in report["exception"]["formatted"]
    assert report["truncation"]["max_items"] >= 1
    frame = report["frames"][-1]
    assert frame["locals"]["quantity"] == "3"
    assert frame["locals"]["payload"] == "[0, 1, 2, 3, 4, 5, 6, 7]"


def test_json_globals_are_opt_in(snapshot, capsys):
    assert main(["--json", str(snapshot)]) == 0
    assert "globals" not in json.loads(capsys.readouterr().out)["frames"][0]
    assert main(["--json", "--globals", str(snapshot)]) == 0
    with_globals = json.loads(capsys.readouterr().out)
    globals_entry = with_globals["frames"][0]["globals"]
    assert globals_entry["__name__"] == "'tests.test_cli'"


def test_info_prints_metadata(snapshot, capsys):
    assert main(["--info", str(snapshot)]) == 0
    out = capsys.readouterr().out
    assert "created_at:" in out
    assert f"python: {sys.version.split()[0]}" in out
    assert "frames: 1 captured of" in out
    assert "limits:" in out
    assert "max_frames=1" in out
    assert "captured by: MainThread" in out
    assert "attempts: none" in out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe semantics")
def test_json_into_a_closed_pipe_stops_quietly(snapshot):
    import os

    reader, writer = os.pipe()
    os.close(reader)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pydumpling", "--json", str(snapshot)],
            stdout=writer,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    finally:
        os.close(writer)
    assert result.returncode == 141
    assert "Broken pipe" not in result.stderr


def test_html_writes_a_report_and_validates_usage(snapshot, tmp_path, capsys):
    target = tmp_path / "report.html"
    assert main(["--html", str(target), str(snapshot)]) == 0
    assert capsys.readouterr().out.strip() == str(target.resolve())
    page = target.read_text(encoding="utf-8")
    assert page.startswith("<!DOCTYPE html>")
    assert "snapshot" in page
    # --html is an action: no mixing with the JSON modifiers or remote flags.
    for extra in (
        ["--globals"],
        ["--host", "localhost"],
        ["--print"],
    ):
        with pytest.raises(SystemExit) as error:
            main(["--html", str(target), str(snapshot), *extra])
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--html", str(target)])
    assert error.value.code == 2
