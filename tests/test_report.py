"""The plain-data report view shared by the JSON and info reporters."""

import io
import json
import sys

import pytest

from pydumpling import DumpOptions, dump_current_traceback, load_dumpling, save_dumping
from pydumpling.report import (
    _json_mapping,
    _safe_repr,
    _source_window,
    dump_to_dict,
    print_info_report,
    print_json_report,
)


def test_safe_repr_guards_hostile_values():
    class Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    assert _safe_repr(Hostile()) == "<repr error>"
    assert _safe_repr(3) == "3"


def test_json_mapping_stringifies_nonstring_keys():
    assert _json_mapping({(1, 2): "v", "plain": 3}) == {
        "(1, 2)": "'v'",
        "plain": "3",
    }


def test_dump_to_dict_reports_frames_globals_and_exception(tmp_path):
    path = tmp_path / "plain.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    result = dump_to_dict(load_dumpling(path), include_globals=True)
    assert result["exception"] is None
    assert result["pydumpling_version"]
    assert result["python_version"] == result["python_version"].strip()
    frame = result["frames"][0]
    assert frame["filename"].endswith("test_report.py")
    assert frame["qualname"] == "test_dump_to_dict_reports_frames_globals_and_exception"
    assert type(frame["locals"]["path"]) is str
    assert frame["globals"]["__name__"] == "'tests.test_report'"


def test_dump_to_dict_exc_reports_bounded_messages(tmp_path):
    path = tmp_path / "exc.dump"
    try:
        holder = object()
        raise ValueError("structured message")
    except ValueError:
        save_dumping(path, strict=True)
    assert holder is not None
    result = dump_to_dict(load_dumpling(path))
    assert result["exception"]["type"].endswith("ValueError")
    assert result["exception"]["message"] == "structured message"
    assert "ValueError" in result["exception"]["formatted"]
    assert result["frames"][-1]["locals"]["holder"].startswith("<")


def test_print_json_report_and_info_write_to_the_given_stream(tmp_path):
    path = tmp_path / "both.dump"
    try:
        raise KeyError("missing")
    except KeyError:
        save_dumping(path, strict=True)
    dumpling = load_dumpling(path)
    json_sink = io.StringIO()
    print_json_report(dumpling, file=json_sink)
    report = json.loads(json_sink.getvalue())
    assert report["exception"]["type"].endswith("KeyError")
    info_sink = io.StringIO()
    print_info_report(dumpling, file=info_sink)
    info = info_sink.getvalue()
    assert "pydumpling:" in info
    assert "truncation:" in info
    assert "attempts: none" in info


def test_info_report_degrades_without_metadata():
    sink = io.StringIO()
    print_info_report({"snapshot_meta": {}}, file=sink)
    text = sink.getvalue()
    assert "created_at: unknown" in text
    assert "python: unknown" in text
    assert "truncation: none" in text
    assert "frames: ? captured of ? original" in text


def test_render_html_escapes_and_shows_frames_and_locals(tmp_path):
    from pydumpling.report import render_html

    path = tmp_path / "html.dump"
    try:
        markup = "<script>&</script>"
        number = 7
        raise ValueError("broken <thing> & stuff")
    except ValueError:
        save_dumping(path, options=DumpOptions(embed_source_lines=2), strict=True)
    assert markup or number  # both locals must exist at capture time
    page = render_html(load_dumpling(path))
    assert page.startswith("<!DOCTYPE html>")
    assert "ValueError" in page and "broken &lt;thing&gt; &amp; stuff" in page
    # Locals are escaped, not interpreted.
    assert "<script>" not in page
    assert "&lt;script&gt;&amp;&lt;/script&gt;" in page
    assert "test_render_html_escapes_and_shows_frames_and_locals" in page
    assert "raise ValueError" in page  # the embedded source window


def test_render_html_prefers_embedded_source_over_a_changed_local(tmp_path):
    from pydumpling.report import render_html

    module = tmp_path / "report_src.py"
    module.write_text("def business():\n    raise ValueError('original line')\n")
    namespace = {}
    exec(compile(module.read_text(), str(module), "exec"), namespace)
    try:
        namespace["business"]()
    except ValueError:
        exc_info = sys.exc_info()
    path = tmp_path / "src.dump"
    save_dumping(path, exc_info, options=DumpOptions(embed_source_lines=1), strict=True)
    module.write_text("def business():\n    raise ValueError('rewritten line')\n")
    page = render_html(load_dumpling(path))
    assert "original line" in page
    assert "rewritten line" not in page


def test_render_html_without_exception_or_source(tmp_path):
    from pydumpling.report import render_html

    path = tmp_path / "plain.dump"
    dump_current_traceback(path, options=DumpOptions(max_frames=1), strict=True)
    page = render_html(load_dumpling(path))
    assert "Stack snapshot" in page
    assert "<pre class=source>" in page  # local source still serves the frame


def test_write_html_report_returns_an_absolute_path(tmp_path):
    from pydumpling.report import write_html_report

    path = tmp_path / "written.dump"
    try:
        raise KeyError("written")
    except KeyError:
        save_dumping(path, strict=True)
    target = tmp_path / "out" / "report.html"
    target.parent.mkdir()
    returned = write_html_report(load_dumpling(path), str(target))
    assert returned == str(target.resolve())
    assert target.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


@pytest.mark.parametrize(
    "lineno, expected", [(1, [1, 2]), (3, [2, 3, 4]), (5, [4, 5]), (8, [])]
)
def test_source_window_reads_only_requested_local_lines(monkeypatch, lineno, expected):
    class SourceLines(list):
        def __iter__(self):
            pytest.fail("source windows must not scan the whole file")

    lines = SourceLines([f"line {number}\n" for number in range(1, 6)])
    monkeypatch.setattr("pydumpling.report.linecache.getlines", lambda filename: lines)
    assert _source_window({}, "source.py", lineno, 1) == [
        (number, lines[number - 1]) for number in expected
    ]


def test_source_window_looks_up_embedded_lines_without_scanning(monkeypatch):
    class SourceLines(dict):
        def items(self):
            pytest.fail("source windows must not scan the whole mapping")

    def no_local_source(filename):
        pytest.fail("embedded source must take precedence")

    monkeypatch.setattr("pydumpling.report.linecache.getlines", no_local_source)
    dumpling = {
        "source": {"source.py": {"lines": SourceLines({2: "saved\n", 3: None})}}
    }
    assert _source_window(dumpling, "source.py", 3, 1) == [(2, "saved\n")]
    assert _source_window(dumpling, "source.py", 20, 1) == []
    assert _source_window({}, "<string>", 1, 1) == []
