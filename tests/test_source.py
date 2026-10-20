"""Embedded source windows and their injection into linecache."""

import io
import linecache
import sys
from dataclasses import replace

import pytest

from pydumpling import DumpOptions, load_dumpling, save_dumping
from pydumpling.debug_dumpling import mock_inspect
from pydumpling.pydumpling import _capture_frame
from pydumpling.snapshot import reduce_snapshot
from pydumpling.snapshot.context import _CODE_TEXT_LIMIT

THIS_FILE = __file__


def failing_function(marker=None):
    try:
        raise ValueError("source check")
    except ValueError:
        return sys.exc_info()


def frame_for(filename, source):
    namespace = {}
    exec(compile(source, filename, "exec"), namespace)
    return namespace["business"]()


def test_embedding_captures_a_window_around_each_frame(tmp_path):
    path = tmp_path / "window.dump"
    exc_type, exc_value, exc_tb = failing_function()
    save_dumping(
        path,
        (exc_type, exc_value, exc_tb),
        options=DumpOptions(embed_source_lines=2),
        strict=True,
    )
    entry = load_dumpling(path)["source"][THIS_FILE]
    total = len(linecache.getlines(THIS_FILE))
    assert entry["total"] == total
    linenos = sorted(entry["lines"])
    raise_line = exc_tb.tb_lineno
    assert linenos == list(range(raise_line - 2, raise_line + 3))
    assert "ValueError" in entry["lines"][raise_line]
    # A window does not reach a function defined far away at the top.
    assert 1 not in linenos


def test_embedding_is_off_by_default_and_whole_file_on_request(tmp_path):
    exc_type, exc_value, exc_tb = failing_function()

    plain = tmp_path / "plain.dump"
    save_dumping(plain, (exc_type, exc_value, exc_tb), strict=True)
    assert "source" not in load_dumpling(plain)

    whole = tmp_path / "whole.dump"
    save_dumping(
        whole,
        (exc_type, exc_value, exc_tb),
        options=DumpOptions(embed_source_lines=None),
        strict=True,
    )
    entry = load_dumpling(whole)["source"][THIS_FILE]
    assert sorted(entry["lines"]) == list(range(1, entry["total"] + 1))


def test_embedding_skips_pseudo_and_missing_files(tmp_path):
    source = "import sys\ndef business():\n    return sys._getframe()\n"
    # A compiled string frame's pseudo name is skipped, whatever the caller
    # chain contributes; a real-looking name with no file on disk embeds
    # nothing for that name either.
    frame = frame_for("<string>", source)
    path = tmp_path / "string.dump"
    _capture_frame(frame, path, options=DumpOptions(embed_source_lines=3))
    assert "<string>" not in (load_dumpling(path).get("source") or {})

    missing = str(tmp_path / "gone.py")
    frame = frame_for(missing, source)
    path = tmp_path / "missing.dump"
    _capture_frame(frame, path, options=DumpOptions(embed_source_lines=3))
    assert missing not in (load_dumpling(path).get("source") or {})


def test_long_source_lines_are_bounded_like_code_metadata(tmp_path):
    module = tmp_path / "longline.py"
    module.write_text(
        "import sys\n"
        "def business():\n"
        "    x = '" + "u" * (_CODE_TEXT_LIMIT + 900) + "'\n"
        "    return sys._getframe()\n"
    )
    frame = frame_for(str(module), module.read_text())
    path = tmp_path / "long.dump"
    _capture_frame(frame, path, options=DumpOptions(embed_source_lines=1))
    entry = load_dumpling(path)["source"][str(module)]
    longest = max(len(text) for text in entry["lines"].values())
    assert longest <= _CODE_TEXT_LIMIT


@pytest.mark.parametrize("already_minimal", [False, True])
def test_minimal_reduction_drops_embedded_source(already_minimal):
    exc_type, exc_value, exc_tb = failing_function()
    context_payload = {
        "traceback": None,
        "snapshot_meta": {"truncation": {"max_items": 1}},
        "source": {THIS_FILE: {"lines": {1: "x"}, "total": 1}},
    }
    # Hand the reduction a real traceback through a context capture.
    from pydumpling.snapshot import SnapshotContext

    context = SnapshotContext(DumpOptions(max_frames=1), minimal=already_minimal)
    context_payload["traceback"] = context.from_traceback(exc_tb)
    context_payload["snapshot_meta"] = context.metadata()
    reduced = reduce_snapshot(
        context_payload, DumpOptions(max_frames=1), stage="minimal"
    )
    assert "source" not in reduced
    assert reduced["snapshot_meta"]["truncation"]["source_omitted"] == 1


def test_byte_budget_drops_source_before_captured_values(tmp_path):
    module = tmp_path / "large.py"
    module.write_text(
        "def business():\n"
        "    marker = {'answer': 42}\n"
        "    alias = marker\n"
        "    raise ValueError('source budget')\n" + "# source padding\n" * 2000
    )
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    try:
        namespace["business"]()
    except ValueError:
        exc_info = sys.exc_info()
    path = tmp_path / "source-budget.dump"
    save_dumping(
        path,
        exc_info,
        options=DumpOptions(max_frames=1, max_dump_bytes=8192, embed_source_lines=None),
        strict=True,
    )
    result = load_dumpling(path)
    variables = result["traceback"].tb_frame.f_locals
    assert variables["marker"] == {"answer": 42}
    assert variables["alias"] is variables["marker"]
    assert "source" not in result
    assert result["exc_extra"]["message"] == "source budget"
    metadata = result["snapshot_meta"]
    assert metadata["truncation"] == {
        "max_frames": 1,
        "source_omitted": 1,
        "serialization_fallback": 1,
    }
    assert {attempt["stage"] for attempt in metadata["attempts"]} == {"full"}


def test_injection_fills_linecache_only_without_a_local_copy(tmp_path):
    module = tmp_path / "vanishing.py"
    module.write_text("def business():\n    raise ValueError('gone file')\n")
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    try:
        namespace["business"]()
    except ValueError:
        exc_info = sys.exc_info()
    path = tmp_path / "vanish.dump"
    save_dumping(path, exc_info, options=DumpOptions(embed_source_lines=1), strict=True)
    module.unlink()
    entry = load_dumpling(path)["source"][str(module)]
    assert entry["lines"]

    with mock_inspect({str(module): entry}):
        lines = linecache.getlines(str(module))
        assert "gone file" in "".join(lines)
        # Entries with no mtime survive a cache refresh.
        linecache.checkcache(str(module))
        assert "gone file" in "".join(linecache.getlines(str(module)))
    assert str(module) not in linecache.cache

    # An unchanged local file keeps its whole text; malformed entries are
    # skipped.
    first = linecache.getlines(THIS_FILE)[0]
    with mock_inspect({THIS_FILE: {"lines": {1: first}}, "bad": {"lines": 0}}):
        assert len(linecache.getlines(THIS_FILE)) > 5
        assert "bad" not in linecache.cache
    # A changed one yields to the lines that ran, and comes back afterwards.
    with mock_inspect({THIS_FILE: {"lines": {1: "fabricated\n"}}}):
        assert linecache.getlines(THIS_FILE) == ["fabricated\n"]
    assert len(linecache.getlines(THIS_FILE)) > 5


def test_a_frame_listed_twice_embeds_both_windows(tmp_path):
    module = tmp_path / "reraise.py"
    module.write_text(
        "def inner():\n"
        "    raise KeyError('k')\n"
        "def business():\n"
        "    try:\n"
        "        inner()\n"
        "    except KeyError as error:\n"
        + "        pass\n" * 20
        + "        raise error\n"
    )
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    try:
        namespace["business"]()
    except KeyError:
        exc_info = sys.exc_info()
    path = tmp_path / "reraise.dump"
    save_dumping(path, exc_info, options=DumpOptions(embed_source_lines=1), strict=True)
    lines = load_dumpling(path)["source"][str(module)]["lines"]
    # The re-raise at line 27 and the original call at line 5 share a frame.
    assert {4, 5, 6, 26, 27} <= set(lines)


def test_cut_source_lines_are_counted_and_unlimited_keeps_them(tmp_path):
    module = tmp_path / "longline.py"
    long_line = "        raise ValueError('long')  # " + "u" * (_CODE_TEXT_LIMIT + 900)
    module.write_text(
        "import sys\n"
        "def business():\n"
        "    try:\n" + long_line + "\n"
        "    except ValueError:\n"
        "        return sys.exc_info()\n"
    )
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    exc_info = namespace["business"]()

    bounded = tmp_path / "bounded.dump"
    save_dumping(
        bounded, exc_info, options=DumpOptions(embed_source_lines=1), strict=True
    )
    result = load_dumpling(bounded)
    assert len(result["source"][str(module)]["lines"][4]) < len(long_line)
    assert result["snapshot_meta"]["truncation"]["source_lines"] == 1

    whole = tmp_path / "whole.dump"
    options = replace(DumpOptions.unlimited(), embed_source_lines=1)
    save_dumping(whole, exc_info, options=options, strict=True)
    result = load_dumpling(whole)
    assert result["source"][str(module)]["lines"][4] == long_line + "\n"
    assert "source_lines" not in result["snapshot_meta"]["truncation"]


def test_print_reads_embedded_source_for_a_missing_file(tmp_path):
    from pydumpling.report import print_traceback_and_except

    module = tmp_path / "printed.py"
    module.write_text("def business():\n    raise ValueError('printed source')\n")
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    try:
        namespace["business"]()
    except ValueError:
        exc_info = sys.exc_info()
    path = tmp_path / "printed.dump"
    save_dumping(path, exc_info, options=DumpOptions(embed_source_lines=1), strict=True)
    module.unlink()
    linecache.checkcache(str(module))
    output = io.StringIO()
    print_traceback_and_except(load_dumpling(path), file=output)
    assert "raise ValueError('printed source')" in output.getvalue()
    assert str(module) not in linecache.cache


def test_embedding_refreshes_a_stale_linecache_entry(tmp_path):
    module = tmp_path / "reloaded.py"
    template = (
        "import sys\ndef business():\n    marker = {!r}\n    return sys._getframe()\n"
    )
    module.write_text(template.format("old"))
    # Cached as a traceback or inspect call would, then edited and reloaded.
    assert "'old'" in "".join(linecache.getlines(str(module)))
    module.write_text(template.format("newer"))
    frame = frame_for(str(module), module.read_text())
    assert frame.f_locals["marker"] == "newer"
    path = tmp_path / "reloaded.dump"
    _capture_frame(frame, path, options=DumpOptions(embed_source_lines=1))
    lines = load_dumpling(path)["source"][str(module)]["lines"]
    assert lines[3] == "    marker = 'newer'\n"


def test_embedding_skips_a_file_cut_short_before_its_captured_line(tmp_path):
    module = tmp_path / "shrunk.py"
    module.write_text(
        "import sys\ndef business():\n    value = 1\n    return sys._getframe()\n"
    )
    frame = frame_for(str(module), module.read_text())
    module.write_text("# emptied\n")
    path = tmp_path / "shrunk.dump"
    _capture_frame(frame, path, options=DumpOptions(embed_source_lines=1))
    assert str(module) not in (load_dumpling(path).get("source") or {})
