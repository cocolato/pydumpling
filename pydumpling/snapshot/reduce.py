"""Reduction of a captured snapshot without rereading live frames."""

import copy
from collections import Counter
from dataclasses import asdict, replace

from ..options import DumpOptions
from .context import SnapshotContext, _utf8_prefix


def _omit_frame_variables(traceback, truncation, *, outer):
    nodes = []
    while traceback is not None:
        nodes.append(traceback)
        traceback = traceback.tb_next
    originals = {id(node.tb_frame): node.tb_frame for node in nodes}
    frames = {identity: copy.copy(frame) for identity, frame in originals.items()}
    codes = {}
    innermost = nodes[-1].tb_frame
    for identity, original in originals.items():
        frame = frames[identity]
        frame.f_back = frames.get(id(original.f_back))
        if frame.f_globals:
            truncation["globals_omitted"] += 1
        frame.f_globals = {}
        if outer:
            code_id = id(frame.f_code)
            if code_id not in codes:
                code = copy.copy(frame.f_code)
                if code.co_consts or code.co_code:
                    truncation["code_payload_omitted"] += 1
                code.co_consts = ()
                code.co_code = b""
                codes[code_id] = code
            frame.f_code = codes[code_id]
        if outer and original is not innermost:
            if frame.f_locals:
                truncation["outer_locals_omitted"] += 1
            frame.f_locals = {}
    head = None
    for original in reversed(nodes):
        node = copy.copy(original)
        node.tb_frame = frames[id(original.tb_frame)]
        node.tb_next = head
        head = node
    return head


def _summarize_exception(extra, budget):
    lines = extra.get("formatted", [])
    fields = [extra.get("type_name", ""), extra.get("message", ""), *lines]
    if (
        sum(len(text[: budget + 1].encode("utf-8", "surrogatepass")) for text in fields)
        <= budget
    ):
        return extra, False
    type_name = _utf8_prefix(extra.get("type_name", ""), min(128, budget // 8))
    message = _utf8_prefix(extra.get("message", ""), min(256, budget // 8))
    available = budget - len((type_name + message).encode("utf-8"))
    remaining = max(0, available - 80)
    summary = []
    kept = 0
    for line in lines:
        if remaining <= 1:
            break
        prefix = _utf8_prefix(line, min(256, remaining - 1))
        kept += len(prefix)
        if not prefix.endswith("\n"):
            prefix += "\n"
        summary.append(prefix)
        remaining -= len(prefix.encode("utf-8"))
    omitted = sum(map(len, lines)) - kept
    if omitted:
        marker = f"... {omitted} diagnostic characters omitted\n"
        summary.append(_utf8_prefix(marker, min(80, available)))
    return {
        "exc_type": None,
        "exc_value": None,
        "type_name": type_name,
        "message": message,
        "formatted": summary,
    }, True


def reduce_snapshot(payload: dict, options: DumpOptions, *, stage: str) -> dict:
    """Reduce captured values without rereading the application's live frames."""
    metadata = dict(payload["snapshot_meta"])
    truncation = Counter(metadata["truncation"])
    if stage in ("globals", "outer"):
        reduced = dict(
            payload,
            traceback=_omit_frame_variables(
                payload["traceback"], truncation, outer=stage == "outer"
            ),
        )
    elif stage == "source" or (
        stage == "minimal" and truncation.get("minimal_snapshot")
    ):
        reduced = dict(payload)
    else:
        reduced = _reduce_values(payload, options, minimal=stage == "minimal")
        metadata = reduced["snapshot_meta"]
        truncation = Counter(metadata["truncation"])
    if stage in ("source", "minimal") and "source" in reduced:
        del reduced["source"]
        truncation["source_omitted"] = 1
    extra = reduced.get("exc_extra")
    if stage != "source" and extra and extra.get("formatted"):
        budget = max(1, min(4096, (options.max_dump_bytes or 16384) // 4))
        reduced["exc_extra"], summarized = _summarize_exception(extra, budget)
        if summarized:
            truncation["exception_summary"] += 1
    metadata["truncation"] = dict(truncation)
    reduced["snapshot_meta"] = metadata
    return reduced


def _reduce_values(payload: dict, options: DumpOptions, *, minimal: bool) -> dict:
    defaults = DumpOptions()
    limits = replace(
        options,
        **{
            name: max(1, (getattr(options, name) or getattr(defaults, name)) // 2)
            for name in ("max_depth", "max_items", "max_string_bytes", "max_nodes")
        },
    )
    context = SnapshotContext(limits, minimal=minimal)
    reduced = dict(payload, traceback=context.from_traceback(payload["traceback"]))
    extra = payload.get("exc_extra")
    if minimal and extra:
        if extra.get("formatted"):
            reduced["exc_extra"] = dict(extra, exc_type=None, exc_value=None)
        elif extra.get("exc_value") is not None:
            reduced["exc_extra"] = context.exception(extra["exc_value"])
    metadata = dict(payload["snapshot_meta"])
    truncation = Counter(metadata["truncation"])
    # Only what the reduction itself cut: the process context in the original
    # metadata, and anything its argv or cwd cut there, is already counted.
    truncation.update(context._truncation)
    metadata.update(truncation=dict(truncation), effective_limits=asdict(limits))
    reduced["snapshot_meta"] = metadata
    return reduced
