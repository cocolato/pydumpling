"""Pickleable frame adapters used by the standard post-mortem debugger."""

import builtins
import copyreg
import pprint
import reprlib


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


class FakeContainer:
    """Mixin for a captured instance of a dict, list, tuple, or set subclass.

    Entries live in the builtin base, ``_base``.  The original type name,
    instance attributes, slots, and named fields live in ``__dict__``.
    """

    __slots__ = ()
    _type_key = "_FakeContainer__type"

    def __hash__(self):
        # Immutable copies hash like the tuple or frozenset they equal, so an
        # equal plain value finds them in a captured dict or set.  Mutable
        # copies, and immutable ones holding unhashable entries, hash by
        # identity so every copy still works as a set member or mapping key.
        base = type(self)._base
        if base is tuple or base is frozenset:
            try:
                return base.__hash__(self)
            except TypeError:
                pass
        return object.__hash__(self)

    @reprlib.recursive_repr()
    def __repr__(self):
        return "%s(%r)" % (_original_name(self), type(self)._base(self))

    def __setstate__(self, state):
        attributes, entries = state
        if entries:
            type(self)._extend(self, entries)
        if attributes:
            self.__dict__.update(attributes)


class FakeDict(FakeContainer, dict):
    _base, _extend = dict, dict.update


class FakeList(FakeContainer, list):
    _base, _extend = list, list.extend


class FakeTuple(FakeContainer, tuple):
    _base, _extend = tuple, None


class FakeSet(FakeContainer, set):
    _base, _extend = set, set.update


class FakeFrozenSet(FakeContainer, frozenset):
    _base, _extend = frozenset, None


class FakeScalar:
    """Mixin for a captured instance of a str, bytes, int, or float subclass.

    The value lives in the builtin base, ``_base``, so the copy hashes and
    compares like the plain value it equals.  The original type name, instance
    attributes, and slots live in ``__dict__``.
    """

    __slots__ = ()
    _type_key = "_FakeScalar__type"

    def __repr__(self):
        return "%s(%r)" % (_original_name(self), type(self)._base(self))


class FakeStr(FakeScalar, str):
    _base = str


class FakeBytes(FakeScalar, bytes):
    _base = bytes


class FakeInt(FakeScalar, int):
    _base = int


class FakeFloat(FakeScalar, float):
    _base = float


def _original_name(copy):
    cls = type(copy)
    return copy.__dict__.get(cls._type_key, cls._base.__name__)


def _pprint_copy(printer, value, stream, indent, allowance, context, level):
    name = _original_name(value)
    stream.write(name + "(")
    indent += len(name) + 1
    if type(value)._base is dict and dict.__len__(value):
        # A plain dict would have its keys sorted; keep the captured order.
        stream.write("{")
        printer._format_dict_items(
            list(dict.items(value)), stream, indent, allowance + 2, context, level
        )
        stream.write("}")
    else:
        printer._format(
            type(value)._base(value), stream, indent, allowance + 1, context, level
        )
    stream.write(")")


# pprint, and so pdb's ``pp``, lays out only types whose ``__repr__`` it knows;
# without this a long copy would print on a single line.
if hasattr(pprint.PrettyPrinter, "_dispatch") and hasattr(
    pprint.PrettyPrinter, "_format_dict_items"
):
    pprint.PrettyPrinter._dispatch[FakeContainer.__repr__] = _pprint_copy
    pprint.PrettyPrinter._dispatch[FakeScalar.__repr__] = _pprint_copy


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


# Snapshot copies carry captured attributes in ``__dict__``, where a name such
# as ``__reduce_ex__``, ``__getstate__`` or ``items`` would hide the method the
# default reduction looks up on the instance.  pickle, dill, and copy consult
# copyreg's dispatch table first, so these reducers look nothing up on the copy.


def _reduce_object(copy):
    return copyreg.__newobj__, (type(copy),), copy.__dict__


def _reduce_container(copy):
    # Entries are read through the builtin base.  Mutable entries travel in the
    # state, restored after the empty copy is memoized, so a copy containing
    # itself still pickles; immutable entries cannot refer back to their copy.
    cls = type(copy)
    base = cls._base
    entries = list(dict.items(copy) if base is dict else base.__iter__(copy))
    if cls._extend is None:
        return cls, (entries,), (copy.__dict__, ())
    return cls, (), (copy.__dict__, entries)


def _reduce_scalar(copy):
    # The value travels as a plain instance of the base, which any Python loads.
    cls = type(copy)
    return cls, (cls._base(copy),), copy.__dict__


copyreg.pickle(FakeClass, _reduce_object)
for _copy_type in (FakeDict, FakeList, FakeTuple, FakeSet, FakeFrozenSet):
    copyreg.pickle(_copy_type, _reduce_container)
for _copy_type in (FakeStr, FakeBytes, FakeInt, FakeFloat):
    copyreg.pickle(_copy_type, _reduce_scalar)
del _copy_type
