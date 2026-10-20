"""The managed dump directory: unique names in, oldest snapshots out."""

import os
import re
from datetime import datetime

import pytest

from pydumpling import DumpManager, DumpOptions, load_dumpling, save_dumping


def make_error(message):
    try:
        raise ValueError(message)
    except ValueError as error:
        return error


def exc_info(message="example"):
    error = make_error(message)
    return (ValueError, error, error.__traceback__)


def test_retention_and_options_are_validated(tmp_path):
    with pytest.raises(ValueError, match="retention"):
        DumpManager(tmp_path, retention=0)
    with pytest.raises(ValueError, match="retention"):
        DumpManager(tmp_path, retention=True)
    with pytest.raises(TypeError, match="options"):
        DumpManager(tmp_path, options={})
    # Unlimited retention is an explicit choice, not an error.
    assert DumpManager(tmp_path, retention=None).retention is None


def test_save_exception_names_and_prunes(tmp_path):
    manager = DumpManager(tmp_path / "dumps", retention=2)
    paths = []
    for index in range(4):
        path = manager.save_exception(exc_info(str(index)))
        assert path is not None and path.is_file()
        paths.append(path)
    # The two oldest dumps were pruned as the retention filled up.
    assert not paths[0].exists()
    assert not paths[1].exists()
    remaining = sorted(tmp_path.glob("dumps/*.dump"))
    assert remaining == sorted(paths[2:])
    assert manager.prune() == []


def test_save_exception_reports_failure_only_in_strict_mode(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("occupied")
    manager = DumpManager(blocker)
    with pytest.warns(RuntimeWarning, match="Unexpected error"):
        assert manager.save_exception(exc_info()) is None
    with pytest.raises(OSError):
        manager.save_exception(exc_info(), strict=True)


def test_save_exception_uses_the_active_exception(tmp_path):
    manager = DumpManager(tmp_path)
    try:
        raise ValueError("active")
    except ValueError:
        path = manager.save_exception(strict=True)
    assert path is not None
    restored = load_dumpling(path)
    assert restored["exc_extra"]["message"] == "active"


def test_save_current_captures_the_caller_without_manager_frames(tmp_path):
    manager = DumpManager(tmp_path)
    path = manager.save_current(strict=True)
    assert path is not None and path.is_file()
    names = []
    node = load_dumpling(path)["traceback"]
    while node is not None:
        names.append(node.tb_frame.f_code.co_name)
        node = node.tb_next
    # The innermost frame is the business caller; manager frames stay out.
    assert names[-1] == "test_save_current_captures_the_caller_without_manager_frames"
    assert "save_current" not in names
    with pytest.raises(TypeError, match="options"):
        manager.save_current(options={})


def test_per_call_options_override_the_manager_options(tmp_path):
    manager = DumpManager(tmp_path, options=DumpOptions(max_frames=2))
    path = manager.save_current(options=DumpOptions(max_frames=1), strict=True)
    assert load_dumpling(path)["snapshot_meta"]["limits"]["max_frames"] == 1


def test_target_names_carry_line_timestamp_and_pid(tmp_path):
    manager = DumpManager(tmp_path)
    first = manager.target(stem="example app", lineno=3)
    assert first.parent == tmp_path
    assert re.fullmatch(
        rf"example app-3-\d{{8}}T\d{{6}}\.\d{{3}}-{re.escape(str(os.getpid()))}"
        r"-\d{6}\.dump",
        first.name,
    )
    # A name already on disk is never handed out again.
    first.write_text("occupied")
    second = manager.target(stem="example app", lineno=3)
    assert second != first
    # The default stem keeps a valid, path-free name.
    assert manager.target().name.startswith("snapshot-")


def test_unlimited_retention_keeps_everything(tmp_path):
    manager = DumpManager(tmp_path, retention=None)
    for index in range(3):
        assert manager.save_exception(exc_info(str(index)), strict=True)
    assert len(list(tmp_path.glob("*.dump"))) == 3
    assert manager.prune() == []


@pytest.mark.parametrize(
    "locations", [(("z.py", 1), ("a.py", 1)), (("same.py", 9), ("same.py", 10))]
)
def test_retention_uses_write_time_across_source_locations(tmp_path, locations):
    manager = DumpManager(tmp_path, retention=1, options=DumpOptions(max_frames=1))
    paths = []
    for filename, lineno in locations:
        try:
            exec(
                compile(
                    "\n" * (lineno - 1) + "raise ValueError('boom')", filename, "exec"
                )
            )
        except ValueError:
            paths.append(manager.save_exception(strict=True))
        assert paths[-1].is_file()
        if len(paths) == 1:
            os.utime(paths[0], ns=(1, 1))
    assert not paths[0].exists()
    assert paths[1].is_file()


def test_prune_skips_files_that_disappear_during_scan(tmp_path, monkeypatch):
    from pathlib import Path

    paths = [tmp_path / name for name in ("gone.dump", "old.dump", "new.dump")]
    for index, path in enumerate(paths, start=1):
        path.touch()
        # Whole seconds: NTFS keeps 100 ns units, so 1, 2 and 3 ns would tie.
        os.utime(path, (index, index))
    (tmp_path / "directory.dump").mkdir()
    original_stat = Path.stat

    def disappearing_stat(path, *args, **kwargs):
        if path == paths[0]:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", disappearing_stat)
    assert DumpManager(tmp_path, retention=1).prune() == [paths[1]]
    assert paths[2].is_file()


def test_same_millisecond_names_still_prune_oldest_first(tmp_path, monkeypatch):
    import pydumpling.pydumpling as writer

    class FrozenClock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 19, 8, 12, 44, 123000, tzinfo=tz)

    # Equal name timestamps must neither collide nor override file write times.
    monkeypatch.setattr(writer, "datetime", FrozenClock)
    manager = DumpManager(tmp_path, retention=2)
    paths = [manager.save_exception(exc_info(str(index))) for index in range(4)]
    remaining = sorted(tmp_path.glob("*.dump"))
    assert remaining == sorted(paths[2:])
    assert [path.name for path in remaining] == [
        paths[2].name,
        paths[3].name,
    ]


@pytest.mark.parametrize(
    "locations", [(("z.py", 1), ("a.py", 1)), (("same.py", 99), ("same.py", 100))]
)
def test_equal_write_times_keep_the_newest_save(tmp_path, locations):
    # Coarse filesystem clocks give quick successive saves the same mtime.
    writer = DumpManager(tmp_path, retention=None, options=DumpOptions(max_frames=1))
    paths = []
    for filename, lineno in locations:
        try:
            exec(
                compile(
                    "\n" * (lineno - 1) + "raise ValueError('boom')", filename, "exec"
                )
            )
        except ValueError:
            paths.append(writer.save_exception(strict=True))
    foreign = tmp_path / "manual.dump"
    foreign.write_bytes(paths[0].read_bytes())
    for path in (*paths, foreign):
        os.utime(path, ns=(5, 5))
    assert DumpManager(tmp_path, retention=1).prune() == [foreign, paths[0]]
    assert paths[1].is_file()


def test_save_exception_requires_an_exception_before_touching_disk(tmp_path):
    manager = DumpManager(tmp_path / "dumps")
    with pytest.raises(ValueError, match="active exception"):
        manager.save_exception(strict=True)
    with pytest.warns(RuntimeWarning, match="active exception"):
        assert manager.save_exception() is None
    assert not (tmp_path / "dumps").exists()


def test_target_creates_the_directory_for_a_direct_save(tmp_path):
    target = DumpManager(tmp_path / "new" / "dumps").target(stem="direct")
    assert target.parent.is_dir()
    try:
        raise ValueError("direct")
    except ValueError:
        save_dumping(target, strict=True)
    assert load_dumpling(target)["exc_extra"]["message"] == "direct"
