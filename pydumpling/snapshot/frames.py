"""Capture of frames, their code objects, and the traceback chain."""

import builtins
import itertools
from collections import deque

from ..fake_types import FakeCode, FakeFrame, FakeTraceback


class FramesMixin:
    """Frame and code capture shared by the composed :class:`SnapshotContext`."""

    def _native_items(self, values):
        limit = self.options.max_items
        if limit is None:
            return tuple(values)
        items = tuple(itertools.islice(values, limit + 1))
        if len(items) > limit:
            self._truncation["max_items"] += 1
        return items[:limit]

    def code(self, code):
        if id(code) in self.codes:
            return self.codes[id(code)]
        captured = object.__new__(FakeCode)
        self.codes[id(code)] = captured
        self._code_sources[id(code)] = code
        filename = self._code_text(code.co_filename)
        captured.co_filename = self._absolute_filename(filename)
        captured.co_name = self._code_text(code.co_name)
        captured.co_qualname = (
            self._code_text(getattr(code, "co_qualname", code.co_name))
            if type(getattr(code, "co_qualname", None)) is str
            else captured.co_name
        )
        captured.co_firstlineno = code.co_firstlineno
        captured.co_flags = code.co_flags
        captured.co_varnames = tuple(
            self._code_text(name) for name in self._native_items(code.co_varnames)
        )
        argcount = getattr(code, "co_argcount", 0)
        kwcount = getattr(code, "co_kwonlyargcount", 0)
        captured.co_argcount = (
            min(argcount, len(captured.co_varnames)) if type(argcount) is int else 0
        )
        captured.co_kwonlyargcount = (
            min(kwcount, max(0, len(captured.co_varnames) - captured.co_argcount))
            if type(kwcount) is int
            else 0
        )
        captured.co_posonlyargcount = (
            min(getattr(code, "co_posonlyargcount", 0), captured.co_argcount)
            if type(getattr(code, "co_posonlyargcount", 0)) is int
            else 0
        )
        named_args = captured.co_argcount + captured.co_kwonlyargcount
        if captured.co_flags & 4 and named_args >= len(captured.co_varnames):
            captured.co_flags &= ~4
        if captured.co_flags & 8 and named_args + bool(captured.co_flags & 4) >= len(
            captured.co_varnames
        ):
            captured.co_flags &= ~8
        captured.co_names = (
            tuple(
                self._code_text(name)
                for name in self._native_items(getattr(code, "co_names", ()))
            )
            if type(getattr(code, "co_names", ())) is tuple
            else ()
        )
        captured.co_lnotab = b""  # Compatibility field; never read native co_lnotab.
        captured.co_consts = ()
        captured.co_code = b""
        captured._co_lines = []
        captured._co_positions = ()
        captured._co_positions_sparse = {}
        native_bytes = code.co_code
        captured._co_position_count = (
            len(native_bytes) // 2 if type(native_bytes) is bytes else 0
        )
        if not self.minimal:
            lines = getattr(code, "co_lines", None)
            positions = getattr(code, "co_positions", None)
            try:
                captured._co_lines = (
                    list(self._native_items(lines())) if callable(lines) else []
                )
            except TypeError:
                captured._co_lines = []
            try:
                captured._co_positions = self._native_items(
                    positions() if callable(positions) else positions or ()
                )
            except TypeError:
                captured._co_positions = ()
            captured._co_position_count = max(
                captured._co_position_count, len(captured._co_positions)
            )
        if not self._capturing_frames:
            self._code_payload(code, captured)
        return captured

    def _code_payload(self, original, captured):
        if self.minimal or id(original) in self._payload_codes:
            return
        self._payload_codes.add(id(original))
        constants = self.value(original.co_consts)
        captured.co_consts = constants if type(constants) is tuple else ()
        bytecode = original.co_code
        captured.co_code = (
            self._bounded_bytes(bytecode, self._code_bytes_limit, "code_metadata")
            if type(bytecode) is bytes
            else b""
        )

    def _location(self, original, captured, lasti, lineno):
        if not any(start <= lasti < end for start, end, _ in captured._co_lines):
            captured._co_lines.append((lasti, lasti + 2, lineno))
            captured._co_lines.sort(key=lambda item: item[0])
        index = max(0, lasti // 2)
        if (
            index < len(captured._co_positions)
            or index in captured._co_positions_sparse
        ):
            return
        position = (lineno, lineno, None, None)
        if not self.minimal:
            positions = getattr(original, "co_positions", None)
            if callable(positions):
                try:
                    position = next(
                        itertools.islice(positions(), index, index + 1), position
                    )
                except TypeError:
                    pass
        captured._co_positions_sparse[index] = position
        captured._co_position_count = max(captured._co_position_count, index + 1)

    def from_frame(self, frame):
        if frame is None:
            raise ValueError("No frame is available to capture")
        entries = []
        seen = set()
        current = frame
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            self._original_frames += 1
            limit = self.options.max_frames
            if limit is None or len(entries) < limit:
                entries.append((current, current.f_lineno, current.f_lasti))
            current = current.f_back
        if current is not None:
            self._truncation["frame_cycle"] += 1
        if self._original_frames > len(entries):
            self._truncation["max_frames"] += self._original_frames - len(entries)
        return self._build(list(reversed(entries)))

    def from_traceback(self, traceback):
        if traceback is None:
            raise ValueError("No traceback is available to capture")
        entries = deque(maxlen=self.options.max_frames)
        seen = set()
        current = traceback
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            self._original_frames += 1
            entries.append((current.tb_frame, current.tb_lineno, current.tb_lasti))
            current = current.tb_next
        if current is not None:
            self._truncation["traceback_cycle"] += 1
        if self._original_frames > len(entries):
            self._truncation["max_frames"] += self._original_frames - len(entries)
        return self._build(list(entries))

    def _build(self, entries):
        self._capturing_frames = True
        for original, lineno, lasti in entries:
            if id(original) not in self.frames:
                frame = object.__new__(FakeFrame)
                self.frames[id(original)] = frame
                frame.f_code = self.code(original.f_code)
                frame.f_lineno = lineno
                frame.f_lasti = lasti
                frame.f_back = None
                frame.f_locals = {}
                frame.f_globals = {}
                frame.f_builtins = builtins.__dict__
            self._location(
                original.f_code, self.frames[id(original)].f_code, lasti, lineno
            )
        # Capture innermost locals first, then outer locals, then globals.
        if not self.minimal:
            # Register every namespace before capturing any value, so a local
            # holding globals() cannot capture that namespace first with its
            # variable names cut to the string budget.
            namespaces = [
                (original, original.f_locals, original.f_globals)
                for original, _, _ in entries
            ]
            for _, local_names, global_names in namespaces:
                self._namespaces.update((id(local_names), id(global_names)))
            for original, local_names, _ in reversed(namespaces):
                frame = self.frames[id(original)]
                code = frame.f_code
                argument_count = (
                    code.co_argcount
                    + code.co_kwonlyargcount
                    + bool(code.co_flags & 4)
                    + bool(code.co_flags & 8)
                )
                frame.f_locals = self._mapping(
                    local_names,
                    priority=("self",) + code.co_varnames[:argument_count],
                )
            for original, _, global_names in reversed(namespaces):
                frame = self.frames[id(original)]
                frame.f_globals = self._mapping(
                    global_names, priority=("__name__",) + frame.f_code.co_names
                )
        elif entries:
            self._truncation["minimal_snapshot"] += 1
        for original, _, _ in entries:
            self.frames[id(original)].f_back = self.frames.get(id(original.f_back))
        self._capturing_frames = False
        for identifier, captured in tuple(self.codes.items()):
            self._code_payload(self._code_sources[identifier], captured)
        head = None
        tail = None
        for original, lineno, lasti in entries:
            node = FakeTraceback()
            node.tb_frame = self.frames[id(original)]
            node.tb_lineno = lineno
            node.tb_lasti = lasti
            if tail is None:
                head = node
            else:
                tail.tb_next = node
            tail = node
        return head
