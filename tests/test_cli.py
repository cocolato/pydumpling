import subprocess
import sys

import pytest

from pydumpling import DumpOptions, dump_current_traceback
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
