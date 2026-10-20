"""Capture of exception instances, messages, and chained diagnostics."""

import builtins
import itertools
import reprlib
import traceback
from dataclasses import replace

from ..fake_types import FakeStr, TruncatedValue
from .identity import _derives

_BUILTIN_EXCEPTIONS = {
    id(value): value
    for value in builtins.__dict__.values()
    if isinstance(value, type) and issubclass(value, BaseException)
}


class ExceptionsMixin:
    """Exception capture shared by the composed :class:`SnapshotContext`."""

    def _builtin_exception(self, exc):
        if _BUILTIN_EXCEPTIONS.get(id(type(exc))) is not type(exc) or _derives(
            exc, BaseExceptionGroup
        ):
            return None
        if id(exc) in self._exception_copies:
            return self._exception_copies[id(exc)]
        self._sources[id(exc)] = exc
        self._exception_copies[id(exc)] = None
        # Reserve a small diagnostic budget even when locals exhaust theirs.
        limits = replace(self.options, max_depth=3, max_items=8, max_nodes=32)
        # A fresh context of the same composed class; importing the context
        # module here would be circular.
        metadata = type(self)(limits, serializer=self.serializer, minimal=True)
        try:
            # Builtin reduction includes filenames/state without instance hooks.
            reduced = type(exc).__reduce__(exc)
            arguments = metadata.value(reduced[1])
            saved = type(exc)(*arguments)
            if len(reduced) > 2 and type(reduced[2]) is dict:
                state = metadata._attribute_mapping(
                    reduced[2], 0, priority=("name", "path")
                )
                BaseException.__setstate__(saved, state)
            if _derives(exc, OSError) and getattr(exc, "winerror", None) is not None:
                # OSError reduction omits winerror; retain it in pickle state too.
                saved.winerror = metadata.value(exc.winerror)
                saved.__dict__["winerror"] = saved.winerror
            self._exception_copies[id(exc)] = saved
            return saved
        except Exception:
            self._truncation["exception_state"] += 1
            return None
        finally:
            self._truncation.update(metadata._truncation)

    def _exception_message(self, exc):
        try:
            return self._render_exception_message(exc)
        except Exception:
            self._truncation["exception_format"] += 1
            return self._text("<exception message unavailable>")

    def _render_exception_message(self, exc):
        saved = self._builtin_exception(exc)
        if saved is not None:
            return self._text(str(saved))
        args = BaseException.args.__get__(exc, type(exc))
        # Basic exception information has a small reserved budget: exhausting
        # the application's value graph must not erase the reason it failed.
        converted = []
        for arg in itertools.islice(args, self.options.max_items):
            if type(arg) is str:
                converted.append(self._text(arg))
            elif type(arg) is bool or type(arg) is float or arg is None:
                converted.append(arg)
            elif type(arg) is int and (
                self.options.max_string_bytes is None
                or arg.bit_length() <= self.options.max_string_bytes * 8
            ):
                converted.append(arg)
            elif self.minimal:
                converted.append(
                    TruncatedValue("minimal_snapshot", self._type_name(arg))
                )
            else:
                converted.append(self.value(arg, 1))
        if self.options.max_items is not None and len(args) > self.options.max_items:
            self._truncation["exception_items"] += 1
        converted = tuple(converted)
        if not converted:
            return ""
        if len(converted) == 1 and (
            type(converted[0]) is str or type(converted[0]) is FakeStr
        ):
            # As str(exc) would, without the subclass's own __str__.
            return str.__str__(converted[0])
        formatter = reprlib.Repr()
        formatter.maxstring = self.options.max_string_bytes or 4096
        formatter.maxother = formatter.maxstring
        return self._text(formatter.repr(converted))

    def exception(self, exc):
        if not isinstance(exc, BaseException):
            raise TypeError("exc must be an exception instance")
        type_name = self._type_name(exc)
        extra = {
            "exc_type": None,
            "exc_value": None,
            "type_name": type_name,
        }
        if self._unlimited:
            if self._serializable((type(exc), exc)):
                extra.update(exc_type=type(exc), exc_value=exc)
                return extra
            self._truncation["unserializable_exception"] += 1
        extra["message"] = self._exception_message(exc)
        saved = self._builtin_exception(exc)
        if not self.minimal and saved is not None:
            extra.update(exc_type=type(saved), exc_value=saved)
        formatted = []
        seen = set()
        pending = [(exc, 0)]
        count = 0
        limit = self.options.max_items or 100
        while pending and count < limit:
            current, depth = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            text = self._exception_message(current)
            saved = self._builtin_exception(current)
            indent = "  " * min(depth, self.options.max_depth or 5)
            lines = [self._type_name(current) + ": " + text + "\n"]
            if saved is not None:
                try:
                    lines = traceback.format_exception_only(type(saved), saved)
                except Exception:
                    self._truncation["exception_format"] += 1
            formatted.extend(indent + line for line in lines)
            count += 1
            attributes = self._attributes(current)
            current_notes = attributes.get("__notes__") if attributes else None
            if saved is None and (
                type(current_notes) is list or type(current_notes) is tuple
            ):
                for note in current_notes[: max(0, limit - count)]:
                    if type(note) is str:
                        formatted.append(self._text(note + "\n"))
                        count += 1
            # MRO identity, not isinstance: for a non-group exception the
            # type check would fall back to the application's ``__class__``.
            if _derives(current, BaseExceptionGroup):
                children = BaseExceptionGroup.exceptions.__get__(current, type(current))
                for child in reversed(children[:limit]):
                    pending.append((child, depth + 1))
            cause = BaseException.__cause__.__get__(current, type(current))
            context = BaseException.__context__.__get__(current, type(current))
            suppressed = BaseException.__suppress_context__.__get__(
                current, type(current)
            )
            chained = cause if cause is not None else (None if suppressed else context)
            if chained is not None:
                pending.append((chained, depth))
        if pending:
            self._truncation["exception_items"] += 1
        extra["formatted"] = formatted
        return extra
