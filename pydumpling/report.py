"""CLI formatting and printable dump reports."""

import argparse
import html
import json
import linecache
import os
import sys
from traceback import format_exception, format_tb
from typing import TextIO

from .pydumpling import DUMP_FILE_EXTENSION


def print_snapshot_notice(dumpling_result: dict, *, file: TextIO | None = None) -> None:
    metadata = dumpling_result.get("snapshot_meta") or {}
    truncation = metadata.get("truncation") or {}
    if truncation:
        reasons = ", ".join(
            f"{name}={count}" for name, count in sorted(truncation.items()) if count
        )
        if reasons:
            print(
                f"Snapshot is incomplete ({reasons}).",
                file=file if file is not None else sys.stderr,
            )


def print_traceback_and_except(
    dumpling_result: dict, *, file: TextIO | None = None
) -> None:
    """Print the stored traceback and exception text to standard output.

    The requested report is the command's product, so it goes to ``file``
    (standard output by default) and stays usable in a pipe or a redirect.
    The truncation notice remains a diagnostic and stays on standard error.
    """
    try:
        print_snapshot_notice(dumpling_result)
    except BrokenPipeError:
        # Only the diagnostic's reader is gone; the report is still written.
        discard_stream(sys.stderr)
    exc_tb = dumpling_result["traceback"]
    extra = dumpling_result.get("exc_extra") or {}
    formatted = extra.get("formatted")
    if formatted:
        text = "".join(formatted) if isinstance(formatted, list) else str(formatted)
        lines = [*format_tb(exc_tb), text if text.endswith("\n") else text + "\n"]
    elif extra.get("exc_type") is not None and extra.get("exc_value") is not None:
        lines = format_exception(extra["exc_type"], extra["exc_value"], exc_tb)
    else:
        lines = format_tb(exc_tb)
    _write_report("".join(lines), file if file is not None else sys.stdout)


def discard_stream(stream: TextIO) -> None:
    """Point a stream whose reader has gone at devnull so later flushes succeed."""
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return  # No descriptor, as for an in-process StringIO.
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, descriptor)
    finally:
        os.close(devnull)


def _write_report(text: str, file: TextIO) -> None:
    encoding = getattr(file, "encoding", None)
    if encoding:
        try:
            text.encode(encoding, getattr(file, "errors", None) or "strict")
        except UnicodeEncodeError:
            # Escape what the stream cannot encode, as standard error does.
            # Even surrogateescape, the Windows and UTF-8 mode default, fails
            # on CJK text in a legacy code page or on a lone high surrogate.
            text = text.encode(encoding, "backslashreplace").decode(encoding)
    file.write(text)


def validate_file_name(file_name: str) -> str:
    """Validate file arguments while leaving reads and format errors to the loader."""
    if not file_name.endswith(DUMP_FILE_EXTENSION):
        raise argparse.ArgumentTypeError("File must be a .dump file")
    if not os.path.isfile(file_name):
        raise argparse.ArgumentTypeError(
            f"File {file_name} not found or not a regular file"
        )
    return file_name


def _safe_repr(value):
    try:
        return repr(value)
    except Exception:
        return "<repr error>"


def _json_mapping(values):
    """A mapping of guarded reprs; JSON needs plain-string keys."""
    return {
        name if type(name) is str else _safe_repr(name): _safe_repr(value)
        for name, value in values.items()
    }


def dump_to_dict(dumpling: dict, *, include_globals: bool = False) -> dict:
    """Extract a plain-data report view of a loaded dump.

    Values become guarded ``repr`` strings, frames run outermost to innermost,
    and an absent exception yields ``None``. Shared by the JSON and HTML
    reporters.
    """
    metadata = dumpling.get("snapshot_meta") or {}
    extra = dumpling.get("exc_extra") or {}
    frames = []
    node = dumpling.get("traceback")
    index = 0
    while node is not None:
        frame = node.tb_frame
        code = frame.f_code
        entry = {
            "index": index,
            "filename": code.co_filename,
            "qualname": getattr(code, "co_qualname", code.co_name),
            "lineno": node.tb_lineno,
            "locals": _json_mapping(frame.f_locals),
        }
        if include_globals:
            entry["globals"] = _json_mapping(frame.f_globals)
        frames.append(entry)
        index += 1
        node = node.tb_next
    exception = None
    formatted = extra.get("formatted")
    if extra.get("type_name") is not None or extra.get("message") or formatted:
        exception = {
            "type": extra.get("type_name"),
            "message": extra.get("message"),
            "formatted": (
                "".join(formatted) if isinstance(formatted, list) else str(formatted)
            ),
        }
    return {
        "format_version": dumpling.get("format_version"),
        "pydumpling_version": dumpling.get("version"),
        "python_version": metadata.get("python_version"),
        "created_at": metadata.get("created_at"),
        "pid": metadata.get("pid"),
        "thread_name": metadata.get("thread_name"),
        "duration_ms": metadata.get("duration_ms"),
        "exception": exception,
        "frames": frames,
        "truncation": metadata.get("truncation") or {},
    }


def print_json_report(
    dumpling: dict,
    *,
    include_globals: bool = False,
    file: TextIO | None = None,
) -> None:
    """Print a dump as an indented JSON report; unknown values become strings."""
    payload = json.dumps(
        dump_to_dict(dumpling, include_globals=include_globals),
        indent=2,
        ensure_ascii=True,
        default=str,
    )
    _write_report(payload + "\n", file if file is not None else sys.stdout)


def print_info_report(dumpling: dict, *, file: TextIO | None = None) -> None:
    """Print stored snapshot metadata as a human-readable summary."""
    metadata = dumpling.get("snapshot_meta") or {}
    lines = [
        f"created_at: {metadata.get('created_at') or 'unknown'}",
        "pydumpling: %s (format %s, %s)"
        % (
            dumpling.get("version") or "unknown",
            dumpling.get("format_version", "?"),
            dumpling.get("dump_type", "?"),
        ),
        f"python: {metadata.get('python_version') or 'unknown'}",
        "captured by: %s (pid %s) in %s ms"
        % (
            metadata.get("thread_name") or "unknown",
            metadata.get("pid", "?"),
            metadata.get("duration_ms", "?"),
        ),
        "frames: %s captured of %s original"
        % (metadata.get("captured_frames", "?"), metadata.get("original_frames", "?")),
    ]
    limits = metadata.get("limits") or {}
    if limits:
        lines.append(
            "limits: " + ", ".join(f"{name}={value}" for name, value in limits.items())
        )
    effective = metadata.get("effective_limits")
    if effective:
        lines.append(
            "effective limits: "
            + ", ".join(f"{name}={value}" for name, value in effective.items())
        )
    truncation = {
        name: count
        for name, count in (metadata.get("truncation") or {}).items()
        if count
    }
    lines.append(
        "truncation: "
        + (
            ", ".join(f"{name}={count}" for name, count in sorted(truncation.items()))
            or "none"
        )
    )
    attempts = metadata.get("attempts")
    if attempts:
        lines.extend(f"attempt: {attempt}" for attempt in attempts)
    else:
        lines.append("attempts: none")
    _write_report("\n".join(lines) + "\n", file if file is not None else sys.stdout)


def _source_window(dumpling: dict, filename: str, lineno: int, span: int):
    """Numbered lines around ``lineno``: embedded source first, then local.

    Embedded source is the ground truth of what ran; the local file only
    serves frames whose snapshot carried none.
    """
    first = max(1, lineno - span)
    last = lineno + span
    entry = (dumpling.get("source") or {}).get(filename)
    embedded = entry.get("lines") if isinstance(entry, dict) else None
    if isinstance(embedded, dict) and embedded:
        return [
            (number, text)
            for number in range(first, last + 1)
            if type(text := embedded.get(number)) is str
        ]
    if filename.startswith("<") and filename.endswith(">"):
        return []
    lines = linecache.getlines(filename)
    return [
        (number, lines[number - 1])
        for number in range(first, min(len(lines), last) + 1)
    ]


_HTML_STYLE = """
body{font-family:sans-serif;margin:2rem;color:#1a1a1a}
h1{font-size:1.4rem;margin:0 0 .25rem}
h2{font-size:1.05rem;border-bottom:2px solid #ddd;padding-bottom:.2rem}
.meta{color:#666;font-size:.85rem;margin-bottom:1rem}
.exception{background:#fdecea;border-left:4px solid #d93025;
padding:.6rem .9rem;margin:1rem 0;white-space:pre-wrap;font-family:monospace}
section.frame{border:1px solid #ddd;border-radius:6px;margin:1rem 0;overflow:hidden}
.frame-head{background:#f5f5f5;padding:.45rem .9rem;font-family:monospace;
font-size:.9rem;white-space:pre-wrap}
.frame-head b{color:#b06000}
pre.source{margin:0;padding:.6rem .9rem;background:#fafafa;overflow-x:auto;
font-size:.85rem;line-height:1.45}
pre.source mark{background:#fff3cd;display:inline-block;width:100%}
table.locals{border-collapse:collapse;width:100%;font-size:.85rem}
table.locals td{border-top:1px solid #eee;padding:.25rem .9rem;
vertical-align:top;font-family:monospace}
table.locals td:first-child{color:#0b5394;white-space:nowrap}
.notice{color:#8a6d3b;font-size:.85rem}
"""


def render_html(dumpling: dict, *, span: int = 3) -> str:
    """Render a loaded dump as one standalone HTML page, no scripts."""
    data = dump_to_dict(dumpling)
    exception = data["exception"]
    meta = [
        data["created_at"] or "unknown time",
        f"python {data['python_version'] or '?'}",
        f"pydumpling {data['pydumpling_version'] or '?'}",
        f"captured by {data['thread_name'] or '?'}"
        + (f" (pid {data['pid']})" if data["pid"] is not None else ""),
    ]
    truncation = data["truncation"]
    parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<title>pydumpling snapshot</title>"
        f"<style>{_HTML_STYLE}</style></head><body>",
    ]
    if exception:
        title = html.escape(f"{exception['type']}: {exception['message']}")
    else:
        title = "Stack snapshot"
    parts.append(f"<h1>{title}</h1>")
    parts.append(f"<div class=meta>{html.escape(' | '.join(meta))}</div>")
    if exception and exception["formatted"]:
        parts.append(
            "<div class=exception>"
            + html.escape(exception["formatted"].rstrip("\n"))
            + "</div>"
        )
    if truncation:
        reasons = ", ".join(
            f"{name}={count}" for name, count in sorted(truncation.items()) if count
        )
        if reasons:
            parts.append(
                f"<div class=notice>Snapshot is incomplete "
                f"({html.escape(reasons)}).</div>"
            )
    parts.append(f"<h2>Frames ({len(data['frames'])})</h2>")
    for frame in data["frames"]:
        parts.append("<section class=frame>")
        location = html.escape(
            f"{frame['filename']}:{frame['lineno']} in {frame['qualname']}"
        )
        parts.append(f"<div class=frame-head><b>#{frame['index']}</b> {location}</div>")
        window = _source_window(dumpling, frame["filename"], frame["lineno"], span)
        if window:
            rendered = []
            for number, text in window:
                line = html.escape(text.rstrip("\n")) or "&nbsp;"
                marker = " mark" if number == frame["lineno"] else ""
                rendered.append(f"<span class=line{marker}>{number:4d} {line}</span>")
            parts.append("<pre class=source>" + "\n".join(rendered) + "</pre>")
        if frame["locals"]:
            rows = "".join(
                f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>"
                for name, value in frame["locals"].items()
            )
            parts.append(f"<table class=locals>{rows}</table>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "".join(parts)


def write_html_report(dumpling: dict, filename: str, *, span: int = 3):
    """Write the standalone HTML report and return its path."""
    target = os.path.abspath(filename)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_html(dumpling, span=span))
    return target
