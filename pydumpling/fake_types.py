"""Pickleable frame adapters used by the standard post-mortem debugger."""

import builtins


class FakeType:
    @classmethod
    def _safe_repr(cls, value):
        try:
            return repr(value)
        except Exception:
            return "repr error"

    @classmethod
    def _convert(cls, value):
        from .options import DumpOptions
        from .snapshot import SnapshotContext

        return SnapshotContext(DumpOptions()).value(value)

    @classmethod
    def _convert_dict(cls, value):
        return cls._convert(value)

    @classmethod
    def _convert_obj(cls, value):
        return cls._convert(value)

    @classmethod
    def _convert_seq(cls, value):
        return map(cls._convert, value)


class TruncatedValue:
    """A visible placeholder for data omitted from a bounded snapshot."""

    def __init__(self, reason, type_name="object"):
        self.reason = reason
        self.type_name = type_name

    def __repr__(self):
        return "<%s omitted: %s>" % (self.type_name, self.reason)


class FakeTraceback(FakeType):
    def __init__(self, traceback=None):
        self.tb_frame = None
        self.tb_lineno = None
        self.tb_next = None
        self.tb_lasti = 0
        if traceback is not None:
            from .options import DumpOptions
            from .snapshot import SnapshotContext

            captured = SnapshotContext(DumpOptions()).from_traceback(traceback)
            if captured is not None:
                self.__dict__.update(captured.__dict__)


class FakeFrame(FakeType):
    def __init__(self, frame):
        from .options import DumpOptions
        from .snapshot import SnapshotContext

        context = SnapshotContext(DumpOptions())
        context.from_frame(frame)
        self.__dict__.update(context.frames[id(frame)].__dict__)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("f_builtins", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.f_builtins = builtins.__dict__
        self.__dict__.setdefault("f_locals", {})
        self.__dict__.setdefault("f_globals", {})
        self.__dict__.setdefault("f_back", None)
        self.__dict__.setdefault("f_lasti", 0)


class FakeClass(FakeType):
    def __init__(self, repr, vals):
        self.__dict__.update(vals)
        self.__repr = repr

    def __repr__(self):
        return self.__dict__.get("_FakeClass__repr", "<object snapshot>")


class FakeCode(FakeType):
    def __init__(self, code):
        from .options import DumpOptions
        from .snapshot import SnapshotContext

        self.__dict__.update(SnapshotContext(DumpOptions()).code(code).__dict__)

    def co_lines(self):
        lines = self.__dict__.get("_co_lines")
        if lines:
            return iter(lines)
        # Old Python < 3.10 dumps only stored the legacy line table.
        line = self.__dict__.get("co_firstlineno", 1)
        offset = 0
        result = []
        table = self.__dict__.get("co_lnotab", b"")
        for address_delta, line_delta in zip(table[::2], table[1::2]):
            next_offset = offset + address_delta
            if next_offset > offset:
                result.append((offset, next_offset, line))
            offset = next_offset
            line += line_delta if line_delta < 128 else line_delta - 256
        result.append(
            (offset, max(offset + 2, len(self.__dict__.get("co_code", b""))), line)
        )
        return iter(result)

    def co_positions(self):
        positions = self.__dict__.get("_co_positions", ())
        sparse = self.__dict__.get("_co_positions_sparse", {})
        count = max(
            self.__dict__.get("_co_position_count", 0),
            len(positions),
            len(self.__dict__.get("co_code", b"")) // 2,
            max(sparse, default=-1) + 1,
            1,
        )
        for index in range(count):
            if index in sparse:
                yield sparse[index]
            elif index < len(positions):
                yield positions[index]
            else:
                yield (None, None, None, None)

    def __setstate__(self, state):
        state = state.copy()
        legacy_positions = state.pop("co_positions", None)
        if "_co_positions" not in state and legacy_positions is not None:
            try:
                values = (
                    legacy_positions()
                    if callable(legacy_positions)
                    else legacy_positions
                )
                state["_co_positions"] = tuple(values)
            except Exception:
                state["_co_positions"] = ()
        state.setdefault("_co_positions", ())
        state.setdefault("_co_positions_sparse", {})
        self.__dict__.update(state)
