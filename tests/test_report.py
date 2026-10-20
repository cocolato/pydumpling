"""The plain-data report view shared by the JSON and info reporters."""

import io
import json
import re
import sys

import pytest

from pydumpling import DumpOptions, dump_current_traceback, load_dumpling, save_dumping
from pydumpling.debug_dumpling import embedded_source
from pydumpling.report import (
    _json_mapping,
    _safe_repr,
    _source_window,
    dump_to_dict,
    print_info_report,
    print_json_report,
    render_html,
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
    assert _source_window("source.py", lineno, 1) == [
        (number, lines[number - 1]) for number in expected
    ]


def test_source_window_shows_embedded_lines_for_a_changed_file(tmp_path):
    module = tmp_path / "source.py"
    module.write_text("".join(f"local {number}\n" for number in range(1, 6)))
    filename = str(module)
    local = [(number, f"local {number}\n") for number in (2, 3, 4)]
    changed = {filename: {"lines": {2: "saved 2\n", 3: "saved 3\n"}, "total": 5}}
    with embedded_source(changed):
        # Lines the dump left out are skipped instead of mixing in local ones.
        assert _source_window(filename, 3, 1) == [(2, "saved 2\n"), (3, "saved 3\n")]
    assert _source_window(filename, 3, 1) == local
    unchanged = {filename: {"lines": {3: "local 3\n"}, "total": 5}}
    with embedded_source(unchanged):
        assert _source_window(filename, 3, 1) == local
    assert _source_window("<string>", 1, 1) == []


def test_html_marks_the_current_line(tmp_path):
    path = tmp_path / "mark.dump"
    try:
        raise KeyError("marked")
    except KeyError:
        save_dumping(path, options=DumpOptions(embed_source_lines=1), strict=True)
    page = render_html(load_dumpling(path))
    marked = re.findall(r'<span class="line mark">([^<]*)</span>', page)
    assert len(marked) == 1
    assert "raise KeyError(&quot;marked&quot;)" in marked[0]
    assert "pre.source .mark{" in page


def test_reports_derive_text_from_exception_objects(tmp_path):
    namespace = {}
    exec(
        compile(
            "import sys\n"
            "def business():\n"
            "    try:\n"
            "        raise ValueError('actual failure')\n"
            "    except ValueError:\n"
            "        return sys.exc_info()\n",
            str(tmp_path / "unlimited.py"),
            "exec",
        ),
        namespace,
    )
    path = tmp_path / "unlimited.dump"
    save_dumping(path, namespace["business"](), options=DumpOptions.unlimited())
    dumpling = load_dumpling(path)
    # Unlimited capture keeps the exception object itself, not its text.
    assert "message" not in dumpling["exc_extra"]
    exception = dump_to_dict(dumpling)["exception"]
    assert exception["type"] == "ValueError"
    assert exception["message"] == "actual failure"
    assert exception["formatted"] == "ValueError: actual failure\n"
    assert "<h1>ValueError: actual failure</h1>" in render_html(dumpling)

    # Older snapshots can carry only the exception type and value.
    legacy = dict(
        dumpling, exc_extra={"exc_type": KeyError, "exc_value": KeyError("k")}
    )
    exception = dump_to_dict(legacy)["exception"]
    assert exception == {
        "type": "KeyError",
        "message": "'k'",
        "formatted": "KeyError: 'k'\n",
    }
    assert "Stack snapshot" not in render_html(legacy)


def test_html_export_escapes_lone_surrogates_and_keeps_old_report(
    tmp_path, monkeypatch
):
    from pydumpling.cli import main

    path = tmp_path / "surrogate.dump"
    try:
        # What os.fsdecode(b"\xff") yields for an undecodable POSIX file name.
        raise FileNotFoundError("missing \udcff")
    except FileNotFoundError:
        save_dumping(path, strict=True)
    target = tmp_path / "report.html"
    target.write_text("previous report", encoding="utf-8")

    def broken(dumpling, *, span):
        raise RuntimeError("render failed")

    with monkeypatch.context() as patch:
        patch.setattr("pydumpling.report.render_html", broken)
        assert main(["--html", str(target), str(path)]) == 1
    assert target.read_text(encoding="utf-8") == "previous report"

    assert main(["--html", str(target), str(path)]) == 0
    assert "missing \\udcff" in target.read_text(encoding="utf-8")


def test_exception_view_falls_back_when_the_class_hides_its_name():
    class Meta(type):
        @property
        def __module__(cls):
            raise RuntimeError("no module")

    class Hidden(Exception, metaclass=Meta):
        pass

    dumpling = {"exc_extra": {"exc_type": Hidden, "exc_value": Hidden("boom")}}
    exception = dump_to_dict(dumpling)["exception"]
    assert exception["type"] == "?"
    assert exception["message"] == "boom"
    assert "<h1>?: boom</h1>" in render_html(dumpling)
