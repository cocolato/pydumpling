"""Seeded A/B parity between the native walker and the pure-Python walk.

Every seed draws random ``DumpOptions``, builds a random object graph and call
stack, and runs one random sequence of capture operations twice on the same
live objects: on a context served by ``pydumpling._native`` and on a pure
Python one.  The pickled results, the truncation counters in key order, the
node count, and the kinds of what ``objects`` and ``_sources`` end up holding,
in their order, must match byte for byte, and the native walk must run exactly
the application hooks the pure one runs: none, outside contrived calls.  A
first, discarded pure run materializes lazily created state, such as an
exception's instance dict, that changes every later capture.  Some cases run
a few frames below the recursion limit, where the result depends on the stack
depth, so both walks start from the same depth.

A captured set iterates in hash order, and ``TruncatedValue``, ``FakeClass``
and NaN ``FakeFloat`` copies hash by identity, so while these tests run they
hash by content instead.  Copies that still hash by identity, such as those of
hashable mutable container subclasses, never share a set with other members.

``PYDUMPLING_PARITY_SEEDS`` sets the number of seeds (300 by default).  A seed
reproduces under the same ``PYTHONHASHSEED``.
"""

import asyncio
import builtins
import contextlib
import copy
import functools
import gc
import gzip
import inspect
import os
import pickle
import random
import reprlib
import signal
import subprocess
import sys
import types
import warnings
import weakref
from collections import Counter, OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import IntEnum, StrEnum

import dill
import pytest

from pydumpling import dump_current_traceback, save_dumping
from pydumpling.fake_types import (
    FakeClass,
    FakeContainer,
    FakeDict,
    FakeFloat,
    FakeList,
    FakeStr,
    FakeType,
    TruncatedValue,
)
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot import context as context_module
from pydumpling.snapshot import values as values_module

if (
    getattr(values_module, "_native", None) is None
    or os.environ.get("PYDUMPLING_NATIVE") == "0"
):
    pytest.skip(
        "the native walker is unavailable: pydumpling._native is not built, "
        "or PYDUMPLING_NATIVE=0",
        allow_module_level=True,
    )

SEEDS = int(os.environ.get("PYDUMPLING_PARITY_SEEDS", "300"))
DUMP_SEEDS = 20


# --- Walk mode -------------------------------------------------------------


@contextlib.contextmanager
def _mode(native):
    """Build contexts on the native walker, or switch it off so they run pure.

    ``PYDUMPLING_NATIVE=0`` also keeps contexts created inside the walk, such
    as an exception's metadata context, in pure Python.
    """
    saved = os.environ.get("PYDUMPLING_NATIVE")
    if native:
        os.environ.pop("PYDUMPLING_NATIVE", None)
    else:
        os.environ["PYDUMPLING_NATIVE"] = "0"
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("PYDUMPLING_NATIVE", None)
        else:
            os.environ["PYDUMPLING_NATIVE"] = saved


def _check_walker(context, native):
    """The context must be served by the walker its mode asks for.

    The native walker lives on the context, as ``_walker`` or as the
    ``value`` entry point it installs.
    """
    state = vars(context)
    served = state.get("_walker")
    if served is None:
        served = state.get("value")
    assert (served is not None) is native, f"native={native} but walker={served!r}"


def _traced():
    """Whether a profiler, tracer or monitoring tool, such as coverage's, is
    set: the walker then hands back each call before it runs Python code."""
    all_events = getattr(sys.monitoring, "_all_events", dict)
    return (
        sys.getprofile() is not None
        or sys.gettrace() is not None
        or bool(all_events())
        or any(sys.monitoring.get_tool(tool) is not None for tool in range(6))
    )


# --- Application types whose hooks must never run during a capture ---------

_HOOK_CALLS = []
_ARMED = [False]


def _called(name):
    """Record an application hook that ran while a capture was in progress."""
    if _ARMED[0]:
        _HOOK_CALLS.append(name)


@contextlib.contextmanager
def _armed():
    """Collect the application hooks run inside the block."""
    _HOOK_CALLS.clear()
    _ARMED[0] = True
    try:
        yield _HOOK_CALLS
    finally:
        _ARMED[0] = False


class _Meta(type):
    """A metaclass whose lookup, comparison and hash are application code."""

    def __getattribute__(cls, name):
        _called("metaclass __getattribute__")
        return type.__getattribute__(cls, name)

    def __eq__(cls, other):
        _called("metaclass __eq__")
        return type.__eq__(cls, other)

    def __hash__(cls):
        _called("metaclass __hash__")
        return type.__hash__(cls)

    def __instancecheck__(cls, instance):
        _called("metaclass __instancecheck__")
        return type.__instancecheck__(cls, instance)


class _MetaObject(metaclass=_Meta):
    """An instance of a class with a hooked metaclass."""


class _Plain:
    """An object with an ordinary instance dict."""


class _Slotted:
    """An object with slots and no instance dict."""

    __slots__ = ("first", "second")


class _SlotsAndDict:
    """An object with both a slot and an instance dict."""

    __slots__ = ("__dict__", "first")


class _Planted:
    """A class whose ``__dict__`` is a property that must never run."""

    @property
    def __dict__(self):
        _called("__dict__ property")
        return {}


class _Hostile:
    """An object whose customizable hooks are application code."""

    def __getattribute__(self, name):
        _called("__getattribute__")
        return object.__getattribute__(self, name)

    def __repr__(self):
        _called("__repr__")
        return "<hostile>"

    def __reduce_ex__(self, protocol):
        _called("__reduce_ex__")
        return object.__reduce_ex__(self, protocol)

    def __eq__(self, other):
        _called("__eq__")
        return self is other

    def __hash__(self):
        _called("__hash__")
        return 7

    @property
    def __class__(self):
        _called("__class__")
        return _Hostile


class _Key:
    """A dict key whose hash and equality are application code."""

    def __init__(self, label):
        self.label = label

    def __hash__(self):
        _called("key __hash__")
        return hash(self.label)

    def __eq__(self, other):
        _called("key __eq__")
        return self is other


class _List(list):
    """A list subclass whose protocol methods are application code."""

    def __iter__(self):
        _called("list __iter__")
        return list.__iter__(self)

    def __len__(self):
        _called("list __len__")
        return list.__len__(self)

    def __getitem__(self, index):
        _called("list __getitem__")
        return list.__getitem__(self, index)


class _SlottedList(list):
    __slots__ = ("owner", "unset")


class _HashList(list):
    """A hashable list subclass; its copy hashes by identity."""

    __hash__ = object.__hash__


class _Labelled(list):
    __slots__ = ("label",)


class _Relabelled(_Labelled):
    """A property hides the base slot, as it would in attribute lookup."""

    @property
    def label(self):
        _called("label property")
        return "property"


class _Dict(dict):
    """A dict subclass whose protocol methods are application code."""

    def items(self):
        _called("dict items")
        return dict.items(self)

    def keys(self):
        _called("dict keys")
        return dict.keys(self)

    def __iter__(self):
        _called("dict __iter__")
        return dict.__iter__(self)

    def __getitem__(self, key):
        _called("dict __getitem__")
        return dict.__getitem__(self, key)

    def __len__(self):
        _called("dict __len__")
        return dict.__len__(self)


class _SlottedDict(dict):
    __slots__ = ("owner",)


class _HashDict(dict):
    """A hashable dict subclass; its copy hashes by identity."""

    __hash__ = object.__hash__


class _Tuple(tuple):
    """A tuple subclass with an instance dict."""


class _OddTuple(tuple):
    """Hashable even with unhashable entries; its copy then hashes by identity."""

    def __hash__(self):
        return 1


class _Set(set):
    """A hashable set subclass, which can contain itself."""

    __hash__ = object.__hash__


class _FrozenSet(frozenset):
    """A frozenset subclass with an instance dict."""


class _Ordered(OrderedDict):
    """An OrderedDict subclass with an instance dict."""


_Record = namedtuple("_Record", "x y")


class _Text(str):
    """A str subclass whose protocol methods are application code."""

    def __getitem__(self, index):
        _called("str __getitem__")
        return str.__getitem__(self, index)

    def __len__(self):
        _called("str __len__")
        return str.__len__(self)

    def __iter__(self):
        _called("str __iter__")
        return str.__iter__(self)

    def __str__(self):
        _called("str __str__")
        return str.__str__(self)

    def __repr__(self):
        _called("str __repr__")
        return str.__repr__(self)

    def __eq__(self, other):
        _called("str __eq__")
        return str.__eq__(self, other)

    def __ne__(self, other):
        _called("str __ne__")
        return str.__ne__(self, other)

    def __hash__(self):
        _called("str __hash__")
        return str.__hash__(self)


class _SlotText(str):
    __slots__ = ("child",)


class _Blob(bytes):
    """A bytes subclass whose protocol methods are application code."""

    def __getitem__(self, index):
        _called("bytes __getitem__")
        return bytes.__getitem__(self, index)

    def __len__(self):
        _called("bytes __len__")
        return bytes.__len__(self)

    def __bytes__(self):
        _called("bytes __bytes__")
        return bytes(self)


class _Count(int):
    """An int subclass whose conversions are application code."""

    def __int__(self):
        _called("int __int__")
        return int.__int__(self)

    def __index__(self):
        _called("int __index__")
        return int.__index__(self)

    def __repr__(self):
        _called("int __repr__")
        return int.__repr__(self)

    def bit_length(self):
        _called("int bit_length")
        return int.bit_length(self)


class _Ratio(float):
    """A float subclass with slots whose conversions are application code."""

    __slots__ = ("unit", "unset")

    def __float__(self):
        _called("float __float__")
        return float.__float__(self)

    def __repr__(self):
        _called("float __repr__")
        return float.__repr__(self)


class _Color(IntEnum):
    RED = 1
    HUGE = 1 << 70


class _Mode(StrEnum):
    FAST = "fast"
    WIDE = "饺子😀" * 3


class _Error(Exception):
    """A custom exception: its message comes from its args, never ``__str__``."""

    def __str__(self):
        _called("exception __str__")
        return "custom"

    def __repr__(self):
        _called("exception __repr__")
        return "custom"


class _ShadyError(Exception):
    """An exception that lies about its class."""

    @property
    def __class__(self):
        _called("exception __class__")
        return _ShadyError


class _Group(ExceptionGroup):
    """A custom exception group."""


class _Namespace:
    """A mapping that is not a dict, as a metaclass ``__prepare__`` may return.

    Capture can read it only through its own ``items``, so both walks must
    call that equally often.  A ``broken`` one fails like a concurrent change.
    """

    def __init__(self, entries=(), *, broken=False):
        self.data = dict(entries)
        self.broken = broken

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def items(self):
        _called("mapping items")
        if self.broken:
            raise RuntimeError("mapping changed size during iteration")
        return list(self.data.items())


class _NamespaceMeta(type):
    """A metaclass whose class bodies run in a :class:`_Namespace`."""

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        return _Namespace()

    def __new__(mcls, name, bases, namespace, **kwargs):
        return super().__new__(mcls, name, bases, dict(namespace.data))


class _Stolen:
    """Its ``__dict__`` descriptor belongs to ``type`` and refuses instances."""

    __dict__ = type.__dict__["__dict__"]


class _Fielded(tuple):
    """A tuple subclass whose ``_fields`` are not all names."""

    _fields = ("x", 1)


# --- Content hashes for copies that hash by identity -----------------------


# --- Random options and values ---------------------------------------------


def _truncated_hash(self):
    return hash(("TruncatedValue", self.reason, self.type_name))


def _fake_class_hash(self):
    label = self.__dict__.get("_FakeClass__repr")
    return hash(("FakeClass", label if type(label) is str else None))


def _fake_float_hash(self):
    return 0x7FF8 if float.__ne__(self, self) else float.__hash__(self)


@pytest.fixture(autouse=True)
def _content_hashes(monkeypatch):
    """Make a captured set's order independent of where copies were allocated."""
    monkeypatch.setattr(TruncatedValue, "__hash__", _truncated_hash)
    monkeypatch.setattr(FakeClass, "__hash__", _fake_class_hash)
    monkeypatch.setattr(FakeFloat, "__hash__", _fake_float_hash)


@pytest.fixture(autouse=True)
def _whole_int_reprs():
    """From 3.13.6 and 3.14 ``reprlib`` writes an int too long to convert as a
    placeholder with its address, so an exception message would differ
    between any two captures; lifting the limit from 3.13 covers them."""
    if sys.version_info < (3, 13):
        yield
        return
    saved = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(saved)


@pytest.fixture(autouse=True)
def _addressless_repr_fallbacks(monkeypatch):
    """``reprlib`` writes a value whose ``repr`` fails, as one may a few frames
    below the recursion limit, as ``<type instance at address>``, the copy's
    address, so an exception message would differ between any two captures.
    A tracer, coverage's included, uses some of the limit itself, so a case
    that fits without one may fail there."""
    original = reprlib.Repr.repr_instance

    def repr_instance(self, x, level):
        text = original(self, x, level)
        name = x.__class__.__name__
        return (
            f"<{name} instance>" if text == f"<{name} instance at {id(x):#x}>" else text
        )

    monkeypatch.setattr(reprlib.Repr, "repr_instance", repr_instance)


_NAMES = (
    "self",
    "__name__",
    "__builtins__",
    "_FakeClass__repr",
    "_FakeContainer__type",
    "_FakeScalar__type",
    "items",
    "__reduce_ex__",
    "名前",
)
_ATTRIBUTE_NAMES = (
    "owner",
    "label",
    "items",
    "__reduce_ex__",
    "_FakeClass__repr",
    "_FakeContainer__type",
    "_FakeScalar__type",
    "_value_",
    "名前",
    "a_rather_long_attribute_name_well_past_small_budgets",
)
_LOCAL_NAMES = (
    "value",
    "items",
    "payload",
    "counter",
    "名前",
    "a_rather_long_local_variable_name_past_small_budgets",
)
_PATTERNS = ("a", "ab", "abc ", "é", "饺", "😀", "a饺", "\ud800", "x\ud800é", "aé饺😀")
_WIDE = ("é", "饺", "😀", "\ud800")
_FLOATS = (0.0, -0.0, 1.5, -2.25, 0.1, 1e308, 5e-324, float("inf"), float("-inf"))
_INTS = (0, 1, -1, 7, 255, 256, -5, -6, 2**31 - 1, 2**63 - 1, -(2**63), 2**64 - 1)
_INTS += (2**64, -(2**64))
_LEAVES = (
    "none",
    "int",
    "float",
    "complex",
    "text",
    "data",
    "scalar",
    "special",
    "snapshot",
)
_LEAF_WEIGHTS = (6, 10, 5, 2, 18, 5, 6, 6, 2)
_BUILDERS = (
    "list_value",
    "tuple_value",
    "dict_value",
    "set_value",
    "ordered_value",
    "subclass_value",
    "object_value",
    "cycle_value",
    "exception",
)
_BUILDER_WEIGHTS = (14, 8, 14, 6, 3, 12, 9, 3, 3)
_KEYS = (
    "text",
    "shared",
    "int",
    "float",
    "plain",
    "tuple",
    "frozenset",
    "subclass",
    "object",
    "snapshot",
    "unstable",
)
_KEY_WEIGHTS = (30, 6, 12, 5, 5, 8, 4, 6, 5, 2, 3)
_FROZENDICT = getattr(builtins, "frozendict", None)


def _pick(rng, choices, weights):
    return rng.choices(choices, weights)[0]


def _options(rng, deep):
    """Random budgets; never ``DumpOptions.unlimited()``."""
    nodes = (None, 1000, 10000) if deep else (None, 1, 5, 50, 1000, 10000)
    depth = _pick(rng, (None, 1, 2, 3, 5, 8), (3, 1, 1, 2, 2, 2))
    options = DumpOptions(
        max_frames=rng.choice((1, 2, 3, 64, None)),
        max_depth=None if deep else depth,
        max_items=_pick(rng, (None, 1, 2, 3, 10, 100), (2, 1, 1, 2, 3, 2)),
        max_string_bytes=_pick(rng, (None, 1, 3, 8, 64, 4096), (2, 1, 2, 2, 2, 2)),
        max_nodes=_pick(rng, nodes, (2,) + (1,) * (len(nodes) - 2) + (2,)),
    )
    assert not options.is_unlimited
    return options


def _captured_copies(rng):
    """Copies an earlier capture produced, to be captured again."""
    holder = _Plain()
    holder.answer = 42
    holder.text = "captured text"
    holder.self = holder
    label = _Text("label")
    label.owner = "team"
    source = [
        holder,
        Counter("aab"),
        _Record(1, [2]),
        label,
        {"a": {"b": {"c": [1]}}},
        _FrozenSet({1, 2}),
        1 << 80,
        "x" * 40,
        [0] * 5,
    ]
    options = DumpOptions(
        max_depth=rng.choice((2, 5)),
        max_items=rng.choice((3, 100)),
        max_string_bytes=rng.choice((4, 4096)),
    )
    with _mode(native=False):
        copies = list(SnapshotContext(options).value(source))
    omitted = TruncatedValue("max_depth", "original.Type")
    shadowed = FakeClass(
        "<original.Type snapshot>", {"omitted": omitted, "_FakeClass__repr": "x"}
    )
    shadowed.self = shadowed
    return copies + [omitted, shadowed, FakeStr("plain copy")]


class _Graph:
    """Random application values for one seed, built deterministically.

    Choices never depend on hashes or ids, so a seed builds the same graph in
    every process; set iteration order is shared by both walks.
    """

    def __init__(self, rng, options):
        self.rng = rng
        self.text_limit = options.max_string_bytes or rng.choice((8, 64))
        self.item_limit = options.max_items or rng.choice((3, 10))
        self.budget = rng.randint(10, 120)
        self.pool = []  # Shared references and cycles are drawn from here.
        self.texts = []  # Shared str and bytes objects, usable as keys.
        self.mappings = []  # Sources for ``_mapping`` operations.
        self.attribute_dicts = []  # Sources for ``_attribute_mapping``.
        self.exceptions = []
        self.snapshots = None
        self.keep = []

    # Sizes and leaves.

    def size(self):
        """A container length, often at the item budget's boundary."""
        rng = self.rng
        roll = rng.random()
        if roll < 0.6:
            return rng.randint(0, 3)
        if roll < 0.9:
            return max(0, self.item_limit + rng.randint(-1, 1))
        return rng.randint(4, 12)

    def items(self, depth):
        count = self.size()
        if count > 12:
            return [self.leaf() for _ in range(count)]
        return [self.value(depth) for _ in range(count)]

    def value(self, depth=3):
        """Any application value nested at most ``depth`` levels."""
        rng = self.rng
        if self.pool and rng.random() < 0.12:
            return rng.choice(self.pool)
        if depth <= 0 or self.budget <= 0 or rng.random() < 0.35:
            return self.leaf()
        self.budget -= 1
        return getattr(self, _pick(rng, _BUILDERS, _BUILDER_WEIGHTS))(depth - 1)

    def leaf(self):
        rng = self.rng
        kind = _pick(rng, _LEAVES, _LEAF_WEIGHTS)
        if kind == "none":
            return rng.choice((None, True, False))
        if kind == "int":
            return self.integer()
        if kind == "float":
            return self.real()
        if kind == "complex":
            return rng.choice((1 + 2j, complex(float("nan"), 1), complex(-0.0, 1e999)))
        if kind == "text":
            return self.shared(self.text())
        if kind == "data":
            return self.shared(self.data())
        if kind == "scalar":
            return self.scalar()
        if kind == "special":
            return self.special()
        return self.snapshot()

    def shared(self, value):
        """Make ``value`` available for later shared references."""
        if self.rng.random() < 0.3:
            if type(value) is not bytearray:
                self.texts.append(value)
            self.pool.append(value)
        return value

    def text(self):
        """A str around the byte budget, multi-byte and surrogate ones included."""
        rng = self.rng
        limit = self.text_limit
        roll = rng.random()
        if roll < 0.15:
            return rng.choice(_NAMES)
        pattern = rng.choice(_PATTERNS)
        if roll < 0.35:
            # A multi-byte character straddles the byte budget.
            head = "a" * max(0, limit - rng.randint(0, 3))
            return head + rng.choice(_WIDE) + pattern[: rng.randint(0, 2)]
        length = rng.choice(
            (0, 1, 2, limit // 3, limit // 3 + 1, limit - 1, limit, limit + 1)
            + (2 * limit + 1,)
        )
        length = max(0, length)
        return (pattern * (length // len(pattern) + 1))[:length]

    def data(self):
        rng = self.rng
        limit = self.text_limit
        length = max(0, rng.choice((0, 1, limit - 1, limit, limit + 1, 2 * limit)))
        raw = (b"\x00\xffab\x80" * (length // 5 + 1))[:length]
        return bytearray(raw) if rng.random() < 0.3 else raw

    def integer(self):
        """Small, 64-bit edge and huge ints, around the bit budget too."""
        rng = self.rng
        roll = rng.random()
        if roll < 0.5:
            return rng.choice(_INTS)
        if roll < 0.85:
            number = (1 << (8 * self.text_limit + rng.randint(-2, 1))) - rng.randint(
                0, 1
            )
            return -number if rng.random() < 0.3 else number
        if roll < 0.95:
            return rng.randint(-(10**30), 10**30)
        return rng.choice((True, False))

    def real(self):
        rng = self.rng
        return float("nan") if rng.random() < 0.15 else rng.choice(_FLOATS)

    def scalar(self):
        """A str, bytes, int or float subclass instance, maybe with attributes."""
        rng = self.rng
        kind = rng.randrange(7)
        if kind == 0:
            value = _Text(self.text())
            if rng.random() < 0.5:
                self.attributes(value, 1)
        elif kind == 1:
            value = _SlotText(self.text())
            if rng.random() < 0.6:
                value.child = rng.choice((_SlotText(self.text()), self.text()))
        elif kind == 2:
            value = _Blob(self.data())
            if rng.random() < 0.3:
                self.attributes(value, 1)
        elif kind == 3:
            value = _Count(self.integer())
            if rng.random() < 0.3:
                self.attributes(value, 1)
        elif kind == 4:
            value = _Ratio(self.real())
            if rng.random() < 0.6:
                value.unit = self.value(1)
        elif kind == 5:
            value = rng.choice((_Color.RED, _Color.HUGE))
        else:
            value = rng.choice((_Mode.FAST, _Mode.WIDE))
        return value

    def special(self):
        """Functions, classes, modules and other objects without a copy type."""
        rng = self.rng
        kind = rng.randrange(12)
        if kind == 0:

            def function(item):
                return item

            if rng.random() < 0.5:
                function.tag = self.text()
            return function
        if kind == 1:
            return rng.choice((_Plain, _Slotted, _MetaObject, int, _Error, _Dict))
        if kind == 2:
            module = types.ModuleType("parity_module")
            module.answer = self.leaf()
            return module
        if kind == 3:
            return (item for item in ())
        if kind == 4:
            return rng.choice((range(rng.randint(0, 5)), slice(1, None, 2)))
        if kind == 5:
            return memoryview(b"view")
        if kind == 6:
            return rng.choice((Ellipsis, NotImplemented, len, print, object()))
        if kind == 7:
            partial = functools.partial(len, "text")
            partial.note = self.leaf()
            return partial
        if kind == 8:
            return types.SimpleNamespace(first=self.leaf())
        if kind == 9:
            return (lambda: None).__code__
        if kind == 10:
            return rng.choice((sys.flags, sys.version_info))
        return self.exception(0)

    def snapshot(self):
        """A copy an earlier capture produced."""
        if self.snapshots is None:
            self.snapshots = _captured_copies(self.rng)
        return self.rng.choice(self.snapshots)

    # Keys and members.

    def key(self, unstable, depth=2):
        """A hashable value; only ``unstable`` ones may copy to an identity hash."""
        rng = self.rng
        kind = _pick(rng, _KEYS, _KEY_WEIGHTS)
        if kind == "shared" and self.texts:
            return rng.choice(self.texts)
        if kind == "int":
            return self.integer()
        if kind == "float":
            return self.real()
        if kind == "plain":
            return rng.choice((None, True, False, b"key", 2j))
        if kind == "tuple" and depth > 0:
            return tuple(
                self.key(unstable, depth - 1) for _ in range(rng.randint(0, 3))
            )
        if kind == "frozenset" and depth > 0:
            count = rng.randint(0, 3)
            return frozenset(
                self.key(unstable and count <= 1, depth - 1) for _ in range(count)
            )
        if kind == "subclass":
            return rng.choice(
                (
                    _Text(self.text()),
                    _Color.RED,
                    _Color.HUGE,
                    _Mode.WIDE,
                    _Count(self.integer()),
                    _FrozenSet((1, "a")),
                )
            )
        if kind == "object":
            return rng.choice((_Key(self.text()), _Hostile(), _Plain(), _Plain, len))
        if kind == "snapshot":
            copy = self.snapshot()
            if type(copy) in (TruncatedValue, FakeClass, FakeStr):
                return copy
        if kind == "unstable" and unstable:
            return self.unstable_key()
        return self.shared(self.text())

    def unstable_key(self):
        """A hashable value whose copy hashes by identity."""
        rng = self.rng
        kind = rng.randrange(5 if _FROZENDICT is None else 6)
        if kind == 0:
            return _HashList(self.items(1))
        if kind == 1:
            value = _HashDict({self.key(False): self.leaf()})
            self.mappings.append(value)
            return value
        if kind == 2:
            return _OddTuple(([self.leaf()], self.leaf()))
        if kind == 3:
            value = _Set()
            value.add(value)
            return value
        if kind == 4:
            return _Set({self.key(False)})
        # A frozendict hashes its values too, so they must be keys as well.
        return _FROZENDICT({self.key(False): self.key(False)})

    def plain_key(self):
        """A key OrderedDict can iterate in link order: builtin types only."""
        rng = self.rng
        kind = rng.randrange(4)
        if kind == 0:
            return self.text()
        if kind == 1:
            return self.integer()
        if kind == 2:
            return rng.choice((None, b"key", 2.5, 3j, False))
        return (self.text(), rng.randint(0, 3))

    # Containers.

    def list_value(self, depth):
        result = []
        self.pool.append(result)
        result.extend(self.items(depth))
        return result

    def tuple_value(self, depth):
        if self.rng.random() < 0.2:
            # A tuple reachable from its own list: a recursive immutable reference.
            inner = [self.value(depth)]
            result = (inner, *self.items(depth))
            inner.append(result)
        else:
            result = tuple(self.items(depth))
        self.pool.append(result)
        return result

    def dict_value(self, depth):
        rng = self.rng
        result = {}
        self.pool.append(result)
        count = self.size()
        for _ in range(count):
            result[self.key(True)] = self.leaf() if count > 12 else self.value(depth)
        roll = rng.random()
        if roll < 0.08:
            result["__builtins__"] = rng.choice((builtins, {"len": len}))
        elif roll < 0.15:
            result["self"] = result
        self.mappings.append(result)
        return result

    def set_value(self, depth):
        rng = self.rng
        count = min(self.size(), 12)
        members = [self.key(count <= 1) for _ in range(count)]
        kind = rng.randrange(4)
        if kind == 0:
            return set(members)
        if kind == 1:
            return frozenset(members)
        value = (_Set if kind == 2 else _FrozenSet)(members)
        if rng.random() < 0.5:
            self.attributes(value, depth)
        return value

    def ordered_value(self, depth):
        """OrderedDicts in link order, or in insertion order for hooked keys."""
        rng = self.rng
        kind = rng.randrange(4)
        result = (_Ordered if kind == 3 else OrderedDict)()
        self.pool.append(result)
        for _ in range(min(self.size(), 12)):
            key = self.key(True) if kind == 1 else self.plain_key()
            result[key] = self.value(depth)
        keys = list(dict.keys(result))
        if keys and rng.random() < 0.7:
            result.move_to_end(rng.choice(keys), last=rng.random() < 0.5)
        if kind == 3:
            self.attributes(result, depth)
        self.mappings.append(result)
        return result

    def subclass_value(self, depth):
        """Container subclasses, with and without attributes and slots."""
        rng = self.rng
        kind = rng.randrange(17 if _FROZENDICT is None else 18)
        if kind == 16:
            value = _Fielded(self.items(depth))
        elif kind == 0:
            value = _List()
            self.pool.append(value)
            value.extend(self.items(depth))
            self.attributes(value, depth)
        elif kind == 1:
            value = _SlottedList(self.items(depth))
            if rng.random() < 0.6:
                value.owner = self.value(depth)
        elif kind == 2:
            value = _HashList(self.items(depth))
        elif kind == 3:
            value = _Relabelled(self.items(depth))
            _Labelled.label.__set__(value, self.leaf())
        elif kind == 4:
            value = _Labelled(self.items(depth))
            if rng.random() < 0.5:
                value.label = self.value(depth)
        elif kind in (5, 6, 7):
            value = (_Dict, _SlottedDict, _HashDict)[kind - 5]()
            self.pool.append(value)
            for _ in range(min(self.size(), 12)):
                value[self.key(True)] = self.value(depth)
            if kind == 5:
                self.attributes(value, depth)
            elif kind == 6 and rng.random() < 0.6:
                value.owner = self.value(depth)
            self.mappings.append(value)
        elif kind == 8:
            value = _Tuple(self.items(depth))
            self.attributes(value, depth)
        elif kind == 9:
            value = _OddTuple(([self.leaf()], *self.items(depth)))
        elif kind == 10:
            value = _Record(self.value(depth), self.value(depth))
        elif kind == 11:
            value = os.stat_result(tuple(range(10)))
        elif kind == 12:
            if rng.random() < 0.3:
                value = _Set()
                value.add(value)  # A copy containing itself must be alone.
            else:
                count = min(self.size(), 12)
                value = _Set(self.key(count <= 1) for _ in range(count))
            value.owner = self.leaf()
        elif kind == 13:
            value = defaultdict(rng.choice((list, None)))
            for _ in range(min(self.size(), 12)):
                value[self.key(True)] = self.value(depth)
            self.mappings.append(value)
        elif kind == 14:
            value = Counter(self.text() for _ in range(self.size()))
            self.mappings.append(value)
        elif kind == 15:
            value = _FrozenSet(self.key(False) for _ in range(min(self.size(), 12)))
            self.attributes(value, depth)
        else:
            value = _FROZENDICT({self.key(True): self.value(depth) for _ in range(3)})
            self.mappings.append(value)
        return value

    def object_value(self, depth):
        """Objects with instance dicts, slots only, or hooked lookups."""
        rng = self.rng
        kind = rng.randrange(10)
        if kind == 0:
            value = _Plain()
            self.pool.append(value)
            self.attributes(value, depth)
        elif kind == 8:
            value = _Stolen()
            value.hidden = self.leaf()
        elif kind == 9:
            value = _Namespace({"entry": self.value(depth)})
            self.mappings.append(value)
        elif kind == 1:
            value = _Slotted()
            if rng.random() < 0.6:
                value.first = self.value(depth)
        elif kind == 2:
            value = _SlotsAndDict()
            value.first = self.value(depth)
            self.attributes(value, depth)
        elif kind == 3:
            value = _Planted()
            value.hidden = self.leaf()
        elif kind == 4:
            value = _MetaObject()
            self.attributes(value, depth)
        elif kind == 5:
            value = _Hostile()
            self.attributes(value, depth)
        elif kind == 6:

            def value():
                return None

            self.attributes(value, depth)
        else:
            value = self.exception(depth)
        return value

    def cycle_value(self, depth):
        rng = self.rng
        kind = rng.randrange(6)
        if kind == 0:
            value = [self.leaf()]
            value.append(value)
        elif kind == 1:
            value = {"kept": self.leaf()}
            value["self"] = value
            self.mappings.append(value)
        elif kind == 2:
            inner = [self.leaf()]
            value = (inner, self.leaf())
            inner.append(value)
        elif kind == 3:
            value = _Plain()
            value.me = value
            value.other = self.value(depth)
            self.attribute_dicts.append(value.__dict__)
        elif kind == 4:
            value = _Set()
            value.add(value)
            value.owner = self.leaf()
        else:
            value = _Text("loop")
            value.self = value
        self.pool.append(value)
        return value

    def attributes(self, target, depth):
        """Give ``target`` random instance attributes, names of any type too."""
        rng = self.rng
        namespace = target.__dict__
        for index in range(rng.choice((0, 1, 1, 2, 3, self.item_limit + 1))):
            name = rng.choice(_ATTRIBUTE_NAMES)
            namespace[name + str(index) if rng.random() < 0.5 else name] = self.value(
                depth
            )
        roll = rng.random()
        if roll < 0.12:
            namespace[rng.choice((1, None, (1, "a"), 2.5))] = self.value(depth)
        elif roll < 0.2:
            namespace[_Text("named")] = self.value(depth)
        elif roll < 0.28:
            namespace["me"] = target
        self.attribute_dicts.append(namespace)

    def exception(self, depth):
        """Builtin, custom and group exceptions with args, notes and chains."""
        rng = self.rng
        count = rng.choice((0, 1, 1, 2, self.item_limit + 1))
        args = [self.value(depth) for _ in range(count)]
        kind = rng.randrange(15 if depth > 0 else 13)
        if kind == 0:
            error = ValueError(*args)
        elif kind == 1:
            error = KeyError(self.key(True))
        elif kind == 2:
            error = OSError(2, "missing", self.text(), None, "destination")
            if rng.random() < 0.5:
                error.winerror = rng.choice((2, self.integer()))
        elif kind == 3:
            error = FileNotFoundError(2, self.text())
        elif kind == 4:
            error = UnicodeDecodeError("utf-8", b"\xff" + self.data(), 0, 1, "bad")
        elif kind == 5:
            error = SyntaxError(self.text(), ("<parity>", 1, 2, self.text(), 1, 3))
        elif kind == 6:
            error = ImportError(*args, name=self.text(), path=self.text())
        elif kind == 7:
            error = rng.choice((StopIteration, SystemExit))(*args)
        elif kind == 8:
            error = rng.choice((RecursionError, KeyboardInterrupt, Exception))(*args)
        elif kind == 9:
            error = _Error(*args)
        elif kind == 10:
            error = _ShadyError(*args)
        elif kind == 11:
            error = ZeroDivisionError("division by zero")
        elif kind == 12:
            error = BaseException(*args)
        else:
            children = [self.exception(depth - 1) for _ in range(rng.randint(1, 3))]
            if all(isinstance(child, Exception) for child in children):
                group = rng.choice((ExceptionGroup, _Group, BaseExceptionGroup))
            else:
                group = BaseExceptionGroup
            error = group(self.text(), children)
        roll = rng.random()
        if roll < 0.3:
            error.add_note(self.text())
        elif roll < 0.4:
            error.__notes__ = rng.choice((["note", 1], ("tuple note",), "not a list"))
        if rng.random() < 0.25:
            self.attributes(error, depth)
        if self.exceptions and rng.random() < 0.3:
            error.__cause__ = rng.choice(self.exceptions)
        if self.exceptions and rng.random() < 0.3:
            error.__context__ = rng.choice(self.exceptions)
            error.__suppress_context__ = rng.random() < 0.3
        if self.exceptions and rng.random() < 0.05:
            rng.choice(self.exceptions).__context__ = error  # A chain cycle.
        self.exceptions.append(error)
        return error


def _chain(rng, graph, length):
    """A value nested ``length`` levels deep, built without recursion."""
    kind = rng.choice(
        ("list", "dict", "tuple", "object", "slots", "subclass", "frozenset", "mixed")
    )
    pad = graph.text() if rng.random() < 0.5 else None
    node = None
    for _ in range(length):
        step = kind
        if kind == "mixed":
            step = rng.choice(("list", "dict", "tuple", "object"))
        if step == "list":
            node = [node] if pad is None else [pad, node]
        elif step == "dict":
            node = {"next": node} if pad is None else {"pad": pad, "next": node}
        elif step == "tuple":
            node = (pad, node)
        elif step == "object":
            holder = _Plain()
            holder.pad = pad
            holder.next = node
            node = holder
        elif step == "slots":
            holder = _SlotText(pad or "level")
            holder.child = node
            node = holder
        elif step == "subclass":
            node = _Dict(next=node)
            node.pad = pad
        else:
            node = frozenset((pad, node))
    return node


# --- Generated call stacks -------------------------------------------------


def _signature(rng, index):
    """A parameter list for level ``index``, and how to build its call."""
    kind = rng.randrange(6)
    if kind == 0:
        return "()", lambda ref: "()"
    if kind == 1:
        return f"(a{index}, b{index}=None)", lambda ref: f"({ref()})"
    if kind == 2:
        return "(self, *args, **kwargs)", lambda ref: f"({ref()}, {ref()}, k={ref()})"
    if kind == 3:
        return (
            f"(a{index}, *, flag{index})",
            lambda ref: f"({ref()}, flag{index}={ref()})",
        )
    if kind == 4:
        return (
            f"(a{index}, /, b{index}, **extra{index})",
            lambda ref: f"({ref()}, {ref()}, more={ref()})",
        )
    return f"(*rest{index})", lambda ref: f"({ref()}, {ref()})"


@dataclass
class _Stack:
    """A generated call stack whose frames hold random locals.

    A raising stack runs inside a generator that stays suspended, so the
    frames' ``f_back`` chain ends inside generated code instead of reaching
    pytest's frames.  A capturing stack is called afresh for every dump.
    """

    namespace: dict
    depth: int
    entry: object = None
    generator: object = None
    caught: object = None
    traceback: object = None
    frames: list = field(default_factory=list)

    def close(self):
        if self.generator is not None:
            self.generator.close()


def _stack(graph, rng, name, *, capture=False, extra=()):
    """Generate, compile and run a call stack for ``graph``'s values.

    Its locals and globals come from ``graph``; ``extra`` values become
    locals too.  Unless ``capture`` is set the innermost frame raises one of
    the graph's exceptions, which the bottom frame catches.
    """
    payload = []
    extra = list(extra)

    def ref(value=None):
        payload.append(graph.value(rng.randint(0, 3)) if value is None else value)
        return f"_P[{len(payload) - 1}]"

    levels = rng.randint(1, 4)
    signatures = [_signature(rng, index) for index in range(levels)]
    lines = []
    for index, (parameters, _) in enumerate(signatures):
        lines.append(f"def _level{index}{parameters}:")
        for position in range(rng.randint(0, 4)):
            lines.append(f"    {rng.choice(_LOCAL_NAMES)}{index}_{position} = {ref()}")
        if extra and (index == levels - 1 or rng.random() < 0.5):
            lines.append(f"    deep{index} = {ref(extra.pop())}")
        if rng.random() < 0.25:
            lines.append(f"    held{index} = {ref()}")
            lines.append(f"    def _closure{index}():")
            lines.append(f"        return held{index}")
        if index + 1 < levels:
            lines.append(f"    return _level{index + 1}{signatures[index + 1][1](ref)}")
        elif capture:
            lines.append("    _CAPTURE()")
        else:
            lines.append(f"    raise {ref(graph.exception(1))}")
    call = f"_level0{signatures[0][1](ref)}"
    bottom = rng.randrange(2 if capture else 3)
    indent = "    " if capture else "        "
    body = [f"{indent}{call}"]
    depth = levels + 1
    module = None
    locals_ = None
    if bottom == 1:
        meta = "(metaclass=_NAMESPACE)" if rng.random() < 0.4 else ""
        body = [
            f"{indent}class _Body{meta}:",
            f"{indent}    attribute = {ref()}",
            f"{indent}    {call}",
        ]
        depth += 1
    elif bottom == 2:
        source = f"module_value = {ref()}\n{call}\n"
        module = compile(source, name + "-module", "exec")
        body = [f"{indent}exec(_MODULE, globals(), _LOCALS)"]
        locals_ = rng.choice((None, {}, OrderedDict(), _Dict(), _Namespace()))
        if locals_ is not None:
            graph.mappings.append(locals_)
        depth += 1
    lines.append("def _entry():")
    for position in range(rng.randint(0, 2)):
        lines.append(f"    entry_{position} = {ref()}")
    if capture:
        lines.extend(body)
    else:
        lines.append("    try:")
        lines.extend(body)
        lines.append("    except BaseException as error:")
        lines.append("        caught = error")
        lines.append("    yield caught")
    kind = rng.randrange(2 if capture else 3)
    if kind == 0:
        namespace = {"__name__": name}
    elif kind == 1:
        holder = types.ModuleType(name)
        graph.keep.append(holder)
        namespace = holder.__dict__
    else:
        # Name lookups call its hooks while the stack is built, before any
        # capture is armed.
        namespace = _Dict(__name__=name)
    for position in range(rng.randint(0, 4)):
        namespace[f"{rng.choice(_LOCAL_NAMES)}_{position}"] = graph.value(2)
    namespace.update(_P=payload, _MODULE=module, _LOCALS=locals_, _CAPTURE=None)
    namespace["_NAMESPACE"] = _NamespaceMeta
    graph.mappings.append(namespace)
    exec(compile("\n".join(lines) + "\n", name, "exec"), namespace)
    stack = _Stack(namespace, depth)
    if capture:
        stack.entry = namespace["_entry"]
        return stack
    stack.generator = namespace["_entry"]()
    stack.caught = next(stack.generator)
    stack.traceback = stack.caught.__traceback__
    node = stack.traceback
    while node is not None:
        stack.frames.append(node.tb_frame)
        node = node.tb_next
    return stack


def _fake_namespace(graph, rng):
    """A frame namespace stand-in, sometimes shared with the value graph."""
    if graph.mappings and rng.random() < 0.3:
        return rng.choice(graph.mappings)
    namespace = rng.choice((dict, dict, _Dict, OrderedDict, FakeDict, _Namespace))()
    if type(namespace) is _Namespace and rng.random() < 0.2:
        namespace.broken = True
    for position in range(rng.randint(0, 5)):
        namespace[f"{rng.choice(_LOCAL_NAMES)}{position}"] = graph.value(2)
    if rng.random() < 0.2:
        namespace[graph.key(True)] = graph.leaf()
    graph.mappings.append(namespace)
    return namespace


def _fake_stack(graph, rng, codes):
    """Frame and traceback stand-ins with odd line numbers and cycles."""
    frames = []
    for _ in range(rng.randint(1, 4)):
        local_names = _fake_namespace(graph, rng)
        frames.append(
            types.SimpleNamespace(
                f_code=rng.choice(codes),
                f_locals=local_names,
                f_globals=(
                    local_names if rng.random() < 0.2 else _fake_namespace(graph, rng)
                ),
                f_builtins={},
                f_lineno=rng.choice((1, 3, 0, -1, None)),
                f_lasti=rng.choice((0, 2, 7, -2, 200)),
                f_back=frames[-1] if frames else None,
            )
        )
    if rng.random() < 0.2:
        frames[0].f_back = frames[-1]
    traceback = None
    for frame in reversed(frames):
        traceback = types.SimpleNamespace(
            tb_frame=frame,
            tb_lineno=frame.f_lineno,
            tb_lasti=frame.f_lasti,
            tb_next=traceback,
        )
    if rng.random() < 0.2:
        tail = traceback
        while tail.tb_next is not None:
            tail = tail.tb_next
        tail.tb_next = traceback
    return traceback, frames[-1]


# --- Cases -----------------------------------------------------------------


@dataclass
class _Case:
    """One seed: options, the operations to replay, and what keeps them alive."""

    seed: int
    options: DumpOptions
    ops: list
    margin: object  # Frames left below the recursion limit, or None.
    stacks: list

    def close(self):
        for stack in self.stacks:
            stack.close()


def _keys(source):
    return list(dict.keys(source) if isinstance(source, dict) else source.keys())


def _priority(rng, source):
    names = [key for key in _keys(source) if type(key) is str]
    chosen = rng.sample(names, rng.randint(0, min(3, len(names))))
    if rng.random() < 0.2:
        chosen.append(rng.choice(("absent", "__builtins__", "self")))
    return tuple(chosen)


def _case(seed):
    """Build the options, values, stacks and operations of one seed."""
    rng = random.Random(seed)
    deep = seed % 10 == 9 or rng.random() < 0.02
    options = _options(rng, deep)
    graph = _Graph(rng, options)
    roots = [graph.value(rng.randint(2, 6)) for _ in range(rng.randint(1, 4))]
    chain = None
    if deep:
        length = _pick(rng, (50, 200, 600, 990, 1500, 3000), (1, 1, 1, 2, 2, 3))
        chain = _chain(rng, graph, length)
    stack = _stack(
        graph,
        rng,
        f"<parity-stack-{seed}>",
        extra=[chain] if chain is not None and rng.random() < 0.4 else (),
    )
    codes = [frame.f_code for frame in stack.frames] + [(lambda: None).__code__]
    fake_traceback, fake_frame = _fake_stack(graph, rng, codes)
    if not graph.exceptions:
        graph.exception(2)
    ops = []
    for _ in range(rng.randint(1, 4)):
        if ops and rng.random() < 0.2:
            ops.append(rng.choice(ops))
            continue
        kind = _pick(
            rng,
            ("value", "mapping", "attributes", "traceback", "frame", "exception"),
            (40, 14, 10, 14, 10, 12),
        )
        if kind == "value":
            target = rng.choice(roots) if rng.random() < 0.6 else graph.value(3)
            ops.append(("value", target, rng.choice((0, 0, 1, 2))))
        elif kind == "mapping":
            if rng.random() < 0.15:
                source = rng.choice(stack.frames).f_locals
            else:
                source = rng.choice(graph.mappings)
            ops.append(
                (
                    "mapping",
                    source,
                    rng.choice((0, 0, 1, 2)),
                    _priority(rng, source),
                    rng.random() < 0.3,
                    rng.choice((dict, FakeDict)),
                )
            )
        elif kind == "attributes":
            source = rng.choice(graph.attribute_dicts or graph.mappings)
            if not isinstance(source, dict):
                source = dict(source)
            names = [key for key in _keys(source) if type(key) is str]
            private = None
            if rng.random() < 0.4:
                private = rng.choice(names + ["_FakeClass__repr", "absent"])
            ops.append(
                (
                    "attributes",
                    source,
                    rng.choice((0, 0, 1, 2)),
                    _priority(rng, source),
                    private,
                )
            )
        elif kind == "traceback":
            choices = [stack.traceback, fake_traceback]
            if stack.traceback.tb_next is not None:
                choices.append(stack.traceback.tb_next)
            ops.append(("traceback", rng.choice(choices)))
        elif kind == "frame":
            ops.append(("frame", rng.choice(stack.frames + [fake_frame])))
        else:
            ops.append(("exception", rng.choice(graph.exceptions)))
    hashable = [source for source in graph.mappings if type(source) is _HashDict]
    if hashable and rng.random() < 0.3:
        # Captured first as a plain dict, it no longer fits as a key or member.
        ops.insert(0, ("mapping", rng.choice(hashable), 0, (), False, dict))
    if chain is not None:
        target = rng.choice((chain, {"deep": chain, "kept": graph.leaf()}, [chain]))
        ops.insert(rng.randint(0, len(ops)), ("value", target, rng.choice((0, 0, 1))))
    margin = None
    if seed % 10 == 3:
        # Near enough that even a shallow walk overflows, or only a deep one.
        margin = rng.randint(4, 40) if rng.random() < 0.5 else rng.randint(40, 160)
    return _Case(seed, options, ops, margin, [stack])


# --- Running both walks ----------------------------------------------------


def _comparable(value, seen=None):
    """``value`` in a form that pickles: itself if it does, otherwise its type
    and parts, each object once, so an outcome that does not pickle, such as
    one holding a local class, still compares."""
    try:
        pickle.dumps(value, protocol=5)
        return value
    except Exception:
        pass
    # Each object seen stays held, so no later one reuses its id.
    seen = {} if seen is None else seen
    if id(value) in seen:
        return ("seen", seen[id(value)][0])
    seen[id(value)] = (len(seen), value)
    kind = type(value)
    if kind in (list, tuple):
        return (kind.__name__, [_comparable(item, seen) for item in value])
    if kind is dict:
        pairs = value.items()
        return (
            "dict",
            [(_comparable(k, seen), _comparable(v, seen)) for k, v in pairs],
        )
    return (
        str(kind.__module__),
        str(kind.__qualname__),
        _comparable_parts(value, seen),
    )


def _comparable_parts(value, seen):
    """What ``_comparable`` compares of an object that does not pickle: its
    instance dict, its slots and other members, an exception's ``args``,
    and a dict, list or tuple subclass's items, read past its class's own
    methods."""
    try:
        state = object.__getattribute__(value, "__dict__")
    except Exception:
        state = None
    parts = [_comparable(state, seen) if type(state) is dict else None]
    for cls in type(value).__mro__:
        for name, member in vars(cls).items():
            if isinstance(member, types.MemberDescriptorType):
                try:
                    parts.append((name, _comparable(member.__get__(value), seen)))
                except AttributeError:
                    parts.append((name, "unset"))
    if isinstance(value, BaseException):
        parts.append(_comparable(BaseException.args.__get__(value), seen))
    for base, items in (
        (dict, dict.items),
        (list, list.__iter__),
        (tuple, tuple.__iter__),
    ):
        if isinstance(value, base):
            parts.append(_comparable(list(items(value)), seen))
            break
    return parts


def test_comparable_reads_what_a_pickle_would_keep():
    """An outcome that does not pickle compares by an exception's ``args``,
    a subclass's items and slots too, not only its instance dict."""

    class Raised(Exception):
        pass

    class Items(list):
        pass

    class Pairs(dict):
        pass

    class Slotted:
        __slots__ = ("held", "unset")

    def outcome(arg, item, pair, held):
        slotted = Slotted()
        slotted.held = held
        return _comparable([Raised(arg), Items([item]), Pairs(k=pair), slotted])

    same = outcome("a", 1, 2, 3)
    assert same == outcome("a", 1, 2, 3)
    for other in [("b", 1, 2, 3), ("a", 0, 2, 3), ("a", 1, 0, 3), ("a", 1, 2, 0)]:
        assert outcome(*other) != same


@dataclass
class _Outcome:
    results: list
    truncation: list
    nodes: int
    hooks: list
    memo: object = None
    blob: bytes = b""

    def __post_init__(self):
        outcome = _comparable((self.results, self.truncation, self.nodes, self.memo))
        self.blob = pickle.dumps(outcome, protocol=5)


def _memo(context):
    """The kinds of what ``objects`` and ``_sources`` hold, in their order:
    the walker writes each entry where and when the Python walk writes it.
    Read without looking a key up, which could run one's ``__eq__``."""
    held = (vars(context).get(name) for name in ("objects", "_sources"))
    return [
        [type(item).__name__ for item in dict.values(memo)]
        if isinstance(memo, dict)
        else None
        for memo in held
    ]


def _apply(context, op):
    """Run one operation; an exception the walk raises is part of its result."""
    kind = op[0]
    try:
        if kind == "value":
            return context.value(op[1], op[2])
        if kind == "mapping":
            return context._mapping(
                op[1], op[2], priority=op[3], counted=op[4], container=op[5]
            )
        if kind == "attributes":
            return context._attribute_mapping(
                op[1], op[2], priority=op[3], private=op[4]
            )
        if kind == "traceback":
            return context.from_traceback(op[1])
        if kind == "frame":
            return context.from_frame(op[1])
        return context.exception(op[1])
    except Exception as error:
        return ("raised", type(error).__name__, str(error))


def _stack_depth():
    frame = sys._getframe(1)
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def _descend(depth, run):
    """Call ``run`` from ``depth`` more Python frames."""
    if depth <= 0:
        return run()
    return _descend(depth - 1, run)


def _below_limit(margin, run):
    """Call ``run`` ``margin`` frames below the recursion limit, or directly."""
    if margin is None:
        return run()
    return _descend(sys.getrecursionlimit() - _stack_depth() - margin, run)


def _run(case, native):
    """Replay the case's operations on a fresh context in the given mode."""
    with _mode(native):
        context = SnapshotContext(case.options)
        _check_walker(context, native)
        with _armed() as calls:
            results = _below_limit(
                case.margin, lambda: [_apply(context, op) for op in case.ops]
            )
        hooks = list(calls)
    return _Outcome(
        results,
        list(context._truncation.items()),
        context._nodes,
        hooks,
        _memo(context),
    )


# --- Diagnosis -------------------------------------------------------------


def _first_byte(left, right):
    """Where two byte strings first differ."""
    pairs = enumerate(zip(left, right))
    return next((at for at, (x, y) in pairs if x != y), min(len(left), len(right)))


def _assert_parity(case, native, pure):
    """Both walks must pickle to the same bytes and run the same hooks; a
    failure names the seed, the options and the first byte that differs."""
    if native.blob == pure.blob and native.hooks == pure.hooks:
        return
    pickles = "pickles are identical"
    if native.blob != pure.blob:
        pickles = (
            f"pickles first differ at byte {_first_byte(native.blob, pure.blob)} "
            f"(native {len(native.blob)} bytes, pure {len(pure.blob)})"
        )
    raise AssertionError(
        f"native and pure walks differ for seed {case.seed}\n"
        f"options: {case.options}\n"
        f"{pickles}\n"
        f"application hooks ran: native {native.hooks}, pure {pure.hooks}\n"
        f"reproduce: PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')} "
        f"pytest {__file__} -k 'test_walk_parity and [{case.seed}]'"
    )


# --- Tests -----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(SEEDS))
def test_walk_parity(seed):
    """The native and pure walks capture one random case identically."""
    case = _case(seed)
    try:
        # A first capture materializes lazily created state, such as an
        # exception's instance dict, that changes every later capture.
        _run(case, native=False)
        native = _run(case, native=True)
        pure = _run(case, native=False)
    finally:
        case.close()
    _assert_parity(case, native, pure)


def test_parity_check_detects_differences():
    """A broken comparison must not pass silently: a change of value,
    sharing, type, order, node count or hooks fails it, naming the seed."""
    for seed in (0, 9, 13):
        case = _case(seed)
        try:
            first = _run(case, native=False)
            second = _run(case, native=False)
        finally:
            case.close()
        _assert_parity(case, first, second)
        second.results.append("extra")
        with pytest.raises(AssertionError, match=f"differ for seed {seed}\n"):
            _assert_parity(case, first, _Outcome(**vars(second)))

    shared = "shared text"
    results = [[shared, shared, {"key": 1.5}], TruncatedValue("max_depth", "list")]
    truncation = [("max_depth", 1), ("max_items", 2)]
    original = _Outcome(results, truncation, 3, [])

    def copy():
        """An equal copy, sharing included, whose objects are all new."""
        return pickle.loads(pickle.dumps((results, truncation)))

    _assert_parity(case, original, _Outcome(*copy(), 3, []))
    changed, same = copy()
    changed[0][2]["key"] = 2.5
    unshared, same_too = copy()
    unshared[0][1] = "".join(("shared ", "text"))
    retyped, also_same = copy()
    retyped[1] = None
    copied, reordered = copy()
    perturbations = [
        _Outcome(changed, same, 3, []),
        _Outcome(unshared, same_too, 3, []),
        _Outcome(retyped, also_same, 3, []),
        _Outcome(copied, reordered[::-1], 3, []),
        _Outcome(*copy(), 4, []),
        # The reason a placeholder shares with the counter's key.
        _Outcome(results, [("".join(("max_", "depth")), 1), truncation[1]], 3, []),
    ]
    for outcome in perturbations:
        with pytest.raises(AssertionError, match="pickles first differ"):
            _assert_parity(case, original, outcome)
    with pytest.raises(
        AssertionError,
        match=r"pickles are identical\napplication hooks ran: native \[\], "
        r"pure \['__repr__'\]",
    ):
        _assert_parity(case, original, _Outcome(*copy(), 3, ["__repr__"]))
    assert [_first_byte(b"abc", b"abd"), _first_byte(b"ab", b"abc")] == [2, 2]


class _FrozenClock:
    """Stands in for the context module's ``datetime``: a fixed creation time."""

    @staticmethod
    def now(tz=None):
        return datetime(2026, 1, 2, 3, 4, 5, 678000, tzinfo=UTC)


def _dump(path, write, native):
    """Write one dump in the given mode; return its error, payload and hooks."""
    if path.exists():
        path.unlink()
    with _mode(native), _armed() as calls:
        try:
            write()
            error = None
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
    payload = gzip.decompress(path.read_bytes()) if path.exists() else None
    return error, payload, list(calls)


def _assert_same_dump(seed, name, options, native, pure):
    if native == pure:
        return
    lines = [
        f"{name} dumps differ for dump seed {seed}",
        f"options: {options}",
        f"errors: native {native[0]!r}, pure {pure[0]!r}",
        f"application hooks ran: native {native[2]}, pure {pure[2]}",
    ]
    if native[1] is not None and pure[1] is not None and native[1] != pure[1]:
        lines.append(f"payloads first differ at byte {_first_byte(native[1], pure[1])}")
        for label, payload in (("native", native[1]), ("pure", pure[1])):
            metadata = dill.loads(payload).get("snapshot_meta", {})
            lines.append(f"{label} truncation: {metadata.get('truncation')}")
            lines.append(f"{label} attempts: {metadata.get('attempts')}")
    raise AssertionError("\n".join(lines))


@pytest.mark.parametrize("seed", range(DUMP_SEEDS))
def test_dump_parity(seed, tmp_path, monkeypatch):
    """Whole dumps, metadata included, are byte-identical in both modes."""
    monkeypatch.setattr(context_module.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(context_module, "datetime", _FrozenClock)
    rng = random.Random(1_000_003 + seed)
    options = replace(
        _options(rng, deep=False),
        max_dump_bytes=rng.choice((16 << 20, 1 << 16, 8192, 2048)),
    )
    graph = _Graph(rng, options)
    path = tmp_path / "parity.dump"
    raising = _stack(graph, rng, f"<parity-dump-{seed}>")
    capturing = _stack(graph, rng, f"<parity-capture-{seed}>", capture=True)
    # Only generated frames may be captured: pytest's change between calls.
    capturing.namespace["_CAPTURE"] = functools.partial(
        dump_current_traceback,
        path,
        options=replace(options, max_frames=rng.randint(1, capturing.depth)),
        strict=True,
    )
    error = raising.caught
    writers = {
        "save_dumping": lambda: save_dumping(
            path,
            exc_info=(type(error), error, error.__traceback__),
            options=options,
            strict=True,
        ),
        "dump_current_traceback": capturing.entry,
    }
    try:
        for name, write in writers.items():
            _dump(path, write, native=False)
            native = _dump(path, write, native=True)
            pure = _dump(path, write, native=False)
            _assert_same_dump(seed, name, options, native, pure)
    finally:
        raising.close()


# --- Targeted cases ----------------------------------------------------------
#
# Paths random graphs rarely reach: mappings that are not dicts (their own
# ``items()`` is application code, which only the Python walk runs), errors
# raised part way through a walk, odd type names, a collection during a walk,
# and calls handed back to the Python walk.


class _ItemsSource:
    """A mapping that is not a dict, read through its own ``items()``."""

    def __init__(self, entries, shape, fail_on=None, collect=False):
        self.entries = entries
        self.shape = shape
        self.fail_on = fail_on
        self.collect = collect
        self.passes = 0

    def _rows(self):
        self.passes += 1
        _called(f"iter {self.passes}")
        if self.collect:
            gc.collect()
        for index, (key, value) in enumerate(self.entries):
            if self.fail_on == (self.passes, index):
                raise _Error("items failed")
            if self.shape == "lists":
                yield [key, value]
            elif self.shape == "triples":
                yield (key, value, None)
            elif self.shape == "singles":
                yield (key,)
            else:
                yield (key, value)

    def items(self):
        _called("items")
        if self.shape == "generator":
            return self._rows()
        if self.shape == "list":
            return list(self._rows())
        return _ItemsView(self)


class _ItemsView:
    """A re-iterable items view whose iteration is application code."""

    def __init__(self, source):
        self.source = source

    def __iter__(self):
        return self.source._rows()


_ITEMS_SHAPES = ("view", "list", "generator", "lists", "triples", "singles")


def _items_case(shape, priority, fail_on=None, collect=False, max_items=None):
    shared = [1, "two", (3,)]
    entries = [("alpha", shared), ("beta", [shared]), ("gamma", "text"), ("delta", 4)]
    source = _ItemsSource(entries, shape, fail_on=fail_on, collect=collect)
    options = DumpOptions(max_items=max_items, max_depth=None, max_nodes=None)
    ops = [
        ("mapping", source, 1, priority, False, dict),
        # A second call sees what the first left behind, error or not.
        ("value", [shared, source, entries], 0),
        ("mapping", source, 1, priority, False, dict),
    ]
    return _Case(0, options, ops, None, [])


@pytest.mark.parametrize("shape", _ITEMS_SHAPES)
@pytest.mark.parametrize("priority", [(), ("gamma",), ("delta", "absent")])
@pytest.mark.parametrize("max_items", [None, 1, 2])
def test_items_mapping_parity(shape, priority, max_items):
    """A non-dict mapping's own ``items()`` is called and read the same way."""
    case = _items_case(shape, priority, max_items=max_items)
    native = _run(case, native=True)
    pure = _run(_items_case(shape, priority, max_items=max_items), native=False)
    _assert_parity(case, native, pure)


@pytest.mark.parametrize("fail_on", [(1, 0), (1, 2), (2, 1), (3, 0), (3, 3)])
@pytest.mark.parametrize("priority", [(), ("beta",)])
def test_error_part_way_through_a_walk(fail_on, priority):
    """An error inside the walk keeps exactly what the Python walk keeps."""
    case = _items_case("view", priority, fail_on=fail_on)
    native = _run(case, native=True)
    pure = _run(_items_case("view", priority, fail_on=fail_on), native=False)
    _assert_parity(case, native, pure)


def test_collection_during_a_walk():
    """A garbage collection while the walker holds the context is safe."""
    case = _items_case("view", ("beta",), collect=True)
    native = _run(case, native=True)
    pure = _run(_items_case("view", ("beta",), collect=True), native=False)
    _assert_parity(case, native, pure)


class _QualName(str):
    """A ``__qualname__`` that is a str subclass."""


def _odd_types():
    odd_name = type("OddName", (), {})
    odd_name.__qualname__ = _QualName("Outer.OddName")
    odd_module = type("OddModule", (), {"__slots__": ("value",)})
    odd_module.__module__ = 7
    no_module = type("NoModule", (), {"__module__": None})
    return odd_name, odd_module, no_module


def test_odd_type_names():
    """Type names read through the descriptors match for odd classes."""

    def case():
        odd_name, odd_module, no_module = _odd_types()
        slotted = odd_module()
        slotted.value = [1]
        values = [odd_name(), slotted, no_module(), odd_name, [odd_module]]
        options = DumpOptions(max_depth=None, max_string_bytes=8)
        return _Case(0, options, [("value", values, 0)], None, [])

    first = case()
    _assert_parity(first, _run(first, native=True), _run(case(), native=False))


def test_dispatch_descriptor_and_handbacks():
    """The class attribute is the native descriptor around the Python method,
    and a walk deeper than the native stack budget is handed back intact."""
    descriptor = vars(values_module.ValuesMixin)["value"]
    assert SnapshotContext.value is descriptor
    assert descriptor.__wrapped__.__qualname__ == "ValuesMixin.value"
    assert descriptor.__doc__ == descriptor.__wrapped__.__doc__
    nested = []
    for _ in range(600):
        nested = [nested, {"key": nested}]
    options = DumpOptions(max_depth=None, max_nodes=None, max_items=None)
    results = {}
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(options)
            _check_walker(context, native)
            captured = context.value(nested)
            if native:
                assert context._walker.handbacks >= 1
            results[native] = pickle.dumps(
                (captured, list(context._truncation.items()), context._nodes),
                protocol=5,
            )
    assert results[True] == results[False]


def _same_in_both_modes(build, options, calls):
    """Run ``calls(context, *built)`` in each mode on fresh objects from
    ``build()`` and require byte-identical results and the same hooks."""
    outcomes = {}
    for native in (False, True, False):
        built = build()
        with _mode(native):
            context = SnapshotContext(options)
            _check_walker(context, native)
            with _armed() as hooks:
                results = []
                for call in calls(context, *built):
                    try:
                        results.append(call())
                    except Exception as error:
                        results.append(("raised", type(error).__name__, str(error)))
            outcomes[native] = _Outcome(
                results,
                list(context._truncation.items()),
                context._nodes,
                list(hooks),
                _memo(context),
            )
    native, pure = outcomes[True], outcomes[False]
    assert native.hooks == pure.hooks
    assert native.blob == pure.blob, (native.results, pure.results)
    return native, pure


def _contents(context):
    """What ``objects`` and ``_sources`` hold, in their order: the walker
    writes each entry where and when the Python walk writes it."""
    return (
        list(map(repr, context.objects.values())),
        list(map(repr, context._sources.values())),
    )


def test_class_level_and_bound_calls():
    """``SnapshotContext.value(ctx, ...)`` routes like ``ctx.value(...)``, and
    the bound call behaves like the bound method it stands in for."""
    descriptor = vars(values_module.ValuesMixin)["value"]

    def build():
        shared = [1, "two", (3,)]
        return ({"a": shared, "b": [shared]}, shared)

    def calls(context, data, shared):
        return [
            lambda: SnapshotContext.value(context, data),
            lambda: SnapshotContext._mapping(context, data, 1, counted=True),
            lambda: context.value(shared, 1),
        ]

    _same_in_both_modes(build, DumpOptions(max_nodes=9), calls)
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
            bound = context.value
            assert bound.__self__ is context
            assert getattr(bound.__func__, "__wrapped__", bound.__func__) is (
                descriptor.__wrapped__
            )
            assert bound == context.value and hash(bound) == hash(context.value)
            assert bound != SnapshotContext(DumpOptions()).value
            assert bound.__name__ == "value"
            assert bound.__doc__ == descriptor.__wrapped__.__doc__
            assert bound.__qualname__ == "ValuesMixin.value"
            assert str(inspect.signature(bound)) == "(value, depth=0)"
            assert inspect.getsource(bound).lstrip().startswith("def value(")
            assert bound.__reduce__() == (getattr, (context, "value"))
            if native:
                assert context._walker.handbacks == 0


class _Redacting(SnapshotContext):
    """A context that overrides a helper the native walker carries out."""

    def _text(self, value):
        return "<redacted>"


class _Composed(SnapshotContext):
    """A context subclass that changes nothing in the walk."""


def test_overriding_a_walk_method_keeps_the_python_walk():
    """A subclass overriding a walk method captures in Python, which calls it:
    its walker hands the context there for good at the first call."""
    with _mode(True):
        redacting = _Redacting(DumpOptions())
        assert redacting.value({"key": ["secret"]}) == {"<redacted>": ["<redacted>"]}
        assert vars(redacting).get("_walker") is None
        composed = _Composed(DumpOptions())
        composed.value([1])
        assert isinstance(vars(composed).get("_walker"), values_module._native.Walker)


def _child_environment():
    """The environment of a child interpreter that imports this checkout."""
    package = os.path.dirname(os.path.dirname(os.path.abspath(values_module.__file__)))
    root = os.path.dirname(package)
    path = os.environ.get("PYTHONPATH")
    return dict(os.environ, PYTHONPATH=root if not path else root + os.pathsep + path)


_THREAD_SCRIPT = """
import os, pickle, sys, threading
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

node = None
for _ in range(3000):
    node = [node]
outcomes = {}

def work(native):
    os.environ["PYDUMPLING_NATIVE"] = "1" if native else "0"
    context = SnapshotContext(DumpOptions(max_depth=None, max_nodes=None))
    captured = context.value({"deep": node})
    outcomes[native] = pickle.dumps(
        (captured, dict(context._truncation), context._nodes), protocol=5
    )

threading.stack_size(int(sys.argv[1]) * 1024)
for native in (True, False):
    thread = threading.Thread(target=work, args=(native,))
    thread.start()
    thread.join()
print("same" if outcomes[True] == outcomes[False] else "different")
"""


@pytest.mark.parametrize("stack_kib", [64, 160, 512])
def test_small_thread_stack(stack_kib):
    """A walk on a thread with a small native stack hands back, not crashes."""
    process = subprocess.run(
        [sys.executable, "-c", _THREAD_SCRIPT, str(stack_kib)],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "same"


class _Nest:
    """Each level is an ``__init__`` call, which Python 3.13+ runs beneath a
    shim frame that counts toward the recursion limit but is not visible."""

    def __init__(self, levels, run):
        self.result = run() if levels <= 0 else _Nest(levels - 1, run).result


@pytest.mark.parametrize("margin", [40, 90, 140, 200, 320])
def test_hidden_init_frames_near_the_limit(margin):
    """Capture beneath a chain of ``__init__`` calls near the recursion limit."""

    def build():
        nested = []
        for _ in range(150):
            nested = [nested, {"key": nested}]
        return (nested,)

    def calls(context, nested):
        def deep():
            levels = (sys.getrecursionlimit() - _stack_depth() - margin) // 2
            return _Nest(levels, lambda: context.value(nested)).result

        return [deep, lambda: context._mapping({"nested": nested})]

    _same_in_both_modes(build, DumpOptions(max_depth=None, max_nodes=None), calls)


def test_empty_bytes_that_are_not_the_singleton():
    """An empty bytes object that is not ``b""`` is captured as Python slices it."""
    zero = len(sys.argv) * 0

    def build():
        return (b"x" * zero, b"y" * zero)

    for options in (DumpOptions(), DumpOptions(max_string_bytes=None)):
        _same_in_both_modes(build, options, lambda c, a, b: [lambda: c.value([a, b])])


class _CollidingName(str):
    """A class-dict key that hashes like the names the walk looks up."""

    def __hash__(self):
        _called("colliding __hash__")
        return hash("__dict__")

    def __eq__(self, other):
        _called("colliding __eq__")
        return str.__eq__(self, other)


@pytest.mark.filterwarnings("ignore:non-string key:RuntimeWarning")
def test_class_dict_keys_that_are_not_str():
    """Types whose class dicts hold other keys run their hooks only in Python."""

    def build():
        odd = type("Odd", (), {_CollidingName("tag"): 1, _Key("other"): 2})
        big = type("Big", (int,), {_CollidingName("tag"): 1})
        space = type("Space", (dict,), {_CollidingName("tag"): 1})
        child = type("Child", (odd,), {"__slots__": ()})
        instance = odd()
        instance.x = [1]
        return (instance, big(1 << 40), space(a=1), child(), odd)

    def calls(context, instance, big, space, child, odd):
        return [
            lambda: context.value([instance]),
            lambda: context.value(big),
            lambda: context._mapping(space),
            lambda: context.value((child, odd), 1),
        ]

    for options in (DumpOptions(), DumpOptions(max_depth=1, max_string_bytes=1)):
        _same_in_both_modes(build, options, calls)


def _keyed_source():
    return ({"a": [1], "b": "text", "c": (2,)},)


@pytest.mark.parametrize(
    "call",
    [
        lambda c, s: c._mapping(s, priority=(key for key in ("b", "a"))),
        lambda c, s: c._mapping(s, priority=[_Key("a"), "b"]),
        lambda c, s: c._mapping(s, priority=None),
        lambda c, s: c._mapping(s, priority={"c", "a"}),
        lambda c, s: c._mapping(s, counted=1),
        lambda c, s: c._mapping(s, container=OrderedDict),
        lambda c, s: c._mapping(s, 0, container=FakeDict, counted=True),
        lambda c, s: c._mapping(s, depth=1),
        lambda c, s: c._mapping(s, 0, 1),
        lambda c, s: c._mapping(s, extra=1),
        lambda c, s: c._mapping(source=s),
        lambda c, s: c.value(s, True),
        lambda c, s: c.value(s, 2**70),
        lambda c, s: c.value(s, depth=-1),
        lambda c, s: c.value(value=s),
        lambda c, s: c.value(s, 0, 1),
        lambda c, s: c.value(),
    ],
)
def test_call_arguments_of_any_shape(call):
    """Arguments the walker does not read itself go to Python as passed."""
    _same_in_both_modes(
        _keyed_source, DumpOptions(max_items=2), lambda c, s: [lambda: call(c, s)]
    )


@pytest.mark.parametrize(
    "depth, handed_back",
    [
        (-(2**63), True),
        (-(2**63) + 1, True),
        (-(2**62) - 1, True),
        (-(2**62), False),
        (2**62, False),
        (2**62 + 1, True),
    ],
)
def test_a_depth_beyond_two_to_the_62_is_handed_back(depth, handed_back):
    """A depth of more than 2**62 in magnitude, which adding to could
    overflow, hands the call back, down to -2**63, whose magnitude no i64
    holds; the walker takes one within that bound."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=2))
        _check_walker(context, native)
        found[native] = pickle.dumps(context.value([[[1]]], depth), 4)
    assert found[True] == found[False]
    assert context._walker.handbacks == (1 if handed_back else 0)


def test_copy_of_a_capturing_context():
    """A shallow copy of a native context shares its walker, as it shares
    ``objects``; a deep copy, which has its own, captures in Python."""

    def build():
        shared = [1, [2], {"three": 3}]
        return (shared, {"again": shared})

    def calls(context, shared, holder):
        def copied():
            twin = copy.copy(context)
            assert vars(twin).get("_walker") is vars(context).get("_walker")
            deep = copy.deepcopy(context)
            assert vars(deep).get("_walker") is None
            return (twin.value(holder), twin._nodes, deep.value([shared]), deep._nodes)

        return [lambda: context.value(shared), copied]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_frame_namespaces_stay_native():
    """A frame's own locals are read natively, without a hand-back."""
    with _mode(True):
        context = SnapshotContext(DumpOptions())
        local = {"nested": [1, 2, 3]}  # noqa: F841
        context.from_frame(sys._getframe())
        assert context._walker.handbacks == 0 or _traced()


def test_bases_changed_between_calls():
    """A class whose ``__bases__`` change between two calls is reread."""

    def build():
        class WithDict:
            pass

        class Planted:
            @property
            def __dict__(self):
                return {}

        class Child(Planted):
            pass

        first, second = Child(), Child()
        object.__setattr__(first, "x", 1)
        object.__setattr__(second, "y", [1, 2])
        return (Child, WithDict, first, second)

    def calls(context, child, with_dict, first, second):
        def rebased():
            child.__bases__ = (with_dict,)
            return context.value(second)

        return [lambda: context.value(first), rebased]

    _same_in_both_modes(build, DumpOptions(), calls)


_RELOAD_SCRIPT = """
import sys
from pydumpling.fake_types import FakeDict as OldFakeDict
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext as Old

for name in [name for name in sys.modules if name.split(".")[0] == "pydumpling"]:
    del sys.modules[name]
from pydumpling import _native
from pydumpling.fake_types import FakeDict
from pydumpling.options import DumpOptions as NewOptions
from pydumpling.snapshot import SnapshotContext as New

class Space(dict):
    pass

new = New(NewOptions(max_depth=None))
old = Old(DumpOptions(max_depth=None))
assert isinstance(vars(new)["_walker"], _native.Walker)
assert FakeDict is not OldFakeDict
assert type(new.value(Space(a=1))) is FakeDict
assert type(old.value(Space(a=1))) is OldFakeDict
print("ok")
"""


def test_reimported_package_has_its_own_classes():
    """A re-imported package's walker copies into that import's classes."""
    process = subprocess.run(
        [sys.executable, "-c", _RELOAD_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "ok"


def test_errors_and_hand_backs_inside_native_calls():
    """An error the native walk raises keeps what it did; odd memo entries and
    keys the private filter would hash go to Python."""

    def build():
        reason = TruncatedValue(_Text("max_depth"), "odd")
        planted = _Plain()
        vars(planted)[_Key("key")] = [1]
        vars(planted)["name"] = "text"
        return (reason, planted, _List([1, 2]))

    def calls(context, reason, planted, listed):
        copy = SnapshotContext(DumpOptions()).value(planted)
        return [
            lambda: context._mapping([1, 2]),
            lambda: context._mapping(listed, 1),
            lambda: context.value(reason),
            lambda: context._mapping(reason),
            lambda: context.value([copy, copy]),
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


class _ClassOnce:
    """A memo entry whose class answers ``__class__`` itself, naming a
    placeholder on the first read only, as a lazy proxy resolves."""

    reason = "max_depth"

    def __init__(self):
        self.reads = 0

    @property
    def __class__(self):
        _called("__class__")
        self.reads += 1
        return TruncatedValue if self.reads == 1 else dict


class _ClassLookup:
    """A memo entry whose ``__getattribute__`` answers ``__class__``."""

    reason = "max_depth"

    def __getattribute__(self, name):
        _called(f"__getattribute__ {name}")
        if name == "__class__":
            return TruncatedValue
        return object.__getattribute__(self, name)


@pytest.mark.parametrize("how", ["value", "mapping", "attribute mapping"])
@pytest.mark.parametrize("entry", [_ClassOnce, _ClassLookup, _Plain, int])
def test_a_memo_entry_that_answers_its_own_class(entry, how):
    """A memo entry Python code adds for a mapping, whose class answers the
    ``__class__`` the walk's ``isinstance`` reads, runs that hook once, as in
    the Python walk, whether the mapping is an instance's attributes or
    captured itself; an entry of a builtin type stays native."""

    def build():
        holder = _Plain()
        holder.payload = [1, 2]
        return (holder, entry())

    def calls(context, holder, planted):
        attributes = vars(holder)

        def plant():
            context.objects[id(attributes)] = planted
            context._sources[id(attributes)] = attributes

        return [
            plant,
            {
                "value": lambda: context.value(holder),
                "mapping": lambda: context._mapping(attributes),
                "attribute mapping": lambda: context._attribute_mapping(attributes, 0),
            }[how],
        ]

    _same_in_both_modes(build, DumpOptions(), calls)
    if entry is int:
        with _mode(True):
            context = SnapshotContext(DumpOptions())
            for call in calls(context, *build()):
                call()
            assert context._walker.handbacks == 0 or _traced()


def test_type_names_read_natively():
    """Odd but plain class names and modules are read without a hand-back."""

    def build():
        _, odd_module, no_module = _odd_types()
        surrogate = type("Lone", (), {})
        surrogate.__qualname__ = "Lone\ud800"
        slotted = odd_module()
        slotted.value = [1]
        return (slotted, no_module(), surrogate(), [surrogate])

    def calls(context, *values):
        return [lambda: context.value(list(values)), lambda: context.value(values, 9)]

    for options in (DumpOptions(), DumpOptions(max_depth=1)):
        native, _ = _same_in_both_modes(build, options, calls)


def test_descriptor_attributes_and_odd_receivers():
    """The descriptor reads like the function, and a class-level call on an
    object that is no context fails as the function would."""
    descriptor = vars(values_module.ValuesMixin)["_mapping"]
    assert SnapshotContext._mapping.__name__ == "_mapping"
    assert SnapshotContext._mapping.__qualname__ == "ValuesMixin._mapping"
    assert (
        descriptor.__wrapped__
        is vars(values_module.ValuesMixin)["_mapping"].__wrapped__
    )
    with _mode(True):
        context = SnapshotContext(DumpOptions())
        assert repr(context.value).startswith("<bound method ValuesMixin.value of ")
    _same_in_both_modes(
        lambda: (),
        DumpOptions(),
        lambda c: [
            lambda: SnapshotContext.value(object(), 1),
            lambda: SnapshotContext.value(c, 1, depth=True),
            lambda: SnapshotContext._mapping(c, {"a": 1}, priority=frozenset({"a"})),
        ],
    )


@pytest.mark.parametrize("then", ["call", "copy"])
@pytest.mark.parametrize("written", ["new", "replacing"])
def test_memo_entries_written_between_calls(written, then):
    """Copies Python memoizes in ``objects`` between calls, new or in place of
    one the walker made, are what both walks return from then on, and both
    dicts hold what the Python walk leaves, whether the context captures
    next or is copied first."""

    def build():
        withheld = {"secret": [1]}
        shared = ["s", ("t", 1)]
        return (withheld, shared, {"cfg": withheld, "other": [2], "s": shared})

    def calls(context, withheld, shared, holder):
        def memo():
            return _contents(context)

        def seed():
            original = withheld if written == "new" else shared
            context.objects[id(original)] = "<withheld>"
            context._sources[id(original)] = original
            if then == "call":
                return context.value(holder), memo()
            twin = copy.copy(context)
            return context.value(holder), twin.value(holder), memo()

        return [
            lambda: context.value([3, shared]),
            seed,
            lambda: context.value(withheld),
            lambda: context.value({"again": shared}),
            memo,
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_full_collections_during_a_native_walk():
    """Full collections that traverse the walker mid-call change nothing."""

    def build():
        shared = [_Record(1, [2]), _Fielded((1, 2)), _Slotted()]
        return ([{"k": shared, "n": list(range(40))} for _ in range(30)],)

    saved = gc.get_threshold()
    gc.set_threshold(1, 1, 1)
    try:
        _same_in_both_modes(
            build, DumpOptions(), lambda c, graph: [lambda: c.value(graph)]
        )
    finally:
        gc.set_threshold(*saved)


# --- Context state, lifecycle and API edges ---------------------------------


def _mutated_options(**changes):
    """Options whose fields hold values validation would refuse."""

    def mutate(context):
        options = replace(context.options)
        for name, value in changes.items():
            object.__setattr__(options, name, value)
        context.options = options

    return mutate


class _Layouts(dict):
    """A layout cache whose lookups are application code."""

    def get(self, key, default=None):
        _called("layouts get")
        return dict.get(self, key, default)


_STATE_CHANGES = {
    "truncation dict": lambda c: setattr(c, "_truncation", {}),
    "float tally": lambda c: c._truncation.__setitem__("max_items", 1.5),
    "int subclass tally": lambda c: c._truncation.__setitem__("max_items", _Count(2)),
    "sources OrderedDict": lambda c: setattr(c, "_sources", OrderedDict()),
    "layouts dict subclass": lambda c: setattr(c, "_layouts", _Layouts()),
    "node marker deleted": lambda c: delattr(c, "_node_marker"),
    "code text limit deleted": lambda c: delattr(c, "_code_text_limit"),
    "truncation Counter subclass": lambda c: setattr(
        c, "_truncation", type("Tally", (Counter,), {})()
    ),
    "namespaces frozenset": lambda c: setattr(c, "_namespaces", frozenset()),
    "negative nodes": lambda c: setattr(c, "_nodes", -1),
    "float nodes": lambda c: setattr(c, "_nodes", 1.5),
    "tiny code text limit": lambda c: setattr(c, "_code_text_limit", 3),
    "unlimited flag": lambda c: setattr(c, "_unlimited", 1),
    "objects OrderedDict": lambda c: setattr(c, "objects", OrderedDict()),
    "objects reset": lambda c: setattr(c, "objects", {}),
    "max_items -1": _mutated_options(max_items=-1),
    "max_items float": _mutated_options(max_items=2.5),
    "max_items True": _mutated_options(max_items=True),
    "max_depth float": _mutated_options(max_depth=2.0),
    "max_string_bytes 0": _mutated_options(max_string_bytes=0),
    "max_nodes maxsize": _mutated_options(max_nodes=sys.maxsize),
    "max_nodes beyond 64 bits": _mutated_options(max_nodes=2**70),
    "max_items maxsize": _mutated_options(max_items=sys.maxsize),
    "walk method on the instance": lambda c: setattr(
        c, "_text", lambda value: "<redacted>"
    ),
    "non-str instance key": lambda c: vars(c).__setitem__(1, 2),
}


@pytest.mark.parametrize("change", sorted(_STATE_CHANGES))
def test_context_state_changed_between_calls(change):
    """State the walker cannot reproduce, set between calls, gives what the
    Python walk gives, errors included."""

    def build():
        plain = _Plain()
        plain.a_long_attribute_name = ["x" * 40, {"inner": (1, 2.5)}]
        data = {"key": "y" * 40, "items": list(range(12)), "obj": plain}
        return (data, [[["deep"], plain]])

    def calls(context, data, nested):
        def changed():
            _STATE_CHANGES[change](context)
            return context.value(data)

        return [
            lambda: context.value(data),
            changed,
            # Fresh data that truncates and copies named tuples reads the
            # tallies and the layout cache.
            lambda: context.value((list(range(12)), _Record(1, 2), _Record(3, 4))),
            lambda: context.value(nested),
            lambda: context._mapping({"k": nested}, 0),
        ]

    _same_in_both_modes(build, DumpOptions(max_nodes=40, max_items=8), calls)


def test_options_subclass_keeps_the_python_walk():
    """Options of another class may compute their budgets in application
    code, so their contexts capture in Python."""

    class Options(DumpOptions):
        pass

    data = {"key": ["v" * 30, (1, 2)]}
    results = {}
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(Options(max_string_bytes=8))
            results[native] = pickle.dumps(
                (context.value(data), dict(context._truncation), context._nodes)
            )
            assert vars(context).get("_walker") is None
    assert results[True] == results[False]


def test_method_objects_pickle_copy_and_weakref():
    """The class attribute pickles and copies as itself, and the bound method
    reads like the Python walk's own, in both modes."""
    function = vars(values_module.ValuesMixin)["value"].__wrapped__
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
            _check_walker(context, native)
            for name in ("value", "_mapping"):
                attribute = getattr(SnapshotContext, name)
                assert pickle.loads(pickle.dumps(attribute)) is attribute
                assert copy.deepcopy(attribute) is attribute
                assert copy.copy(attribute) is attribute
                assert weakref.ref(attribute)() is attribute
            bound = context.value
            assert inspect.ismethod(bound)
            assert weakref.WeakMethod(bound)() == bound
            assert bound.__doc__ == function.__doc__
            assert SnapshotContext.value.__doc__ == function.__doc__
            assert pickle.loads(pickle.dumps(bound))([1]) == [1]


def test_keyword_names_that_are_not_utf8():
    """A keyword name with a lone surrogate fails as the function would."""
    _same_in_both_modes(
        lambda: ([1],),
        DumpOptions(),
        lambda c, v: [
            lambda: c.value(v, **{"\udc80": 1}),
            lambda: c._mapping({"a": 1}, **{"dep\ud800th": 1}),
            lambda: SnapshotContext.value(c, v, **{"\udc80": 1}),
        ],
    )


class _Evil:
    """A key whose hash and equality are application code."""

    def __init__(self, target):
        self.target = target

    def __hash__(self):
        _called("_Evil.__hash__")
        return hash(self.target)

    def __eq__(self, other):
        _called("_Evil.__eq__")
        return self is other


class _Reading:
    """A descriptor whose ``__get__`` is application code."""

    def __get__(self, instance, owner=None):
        _called("_Reading.__get__")
        return {"from": "descriptor"}


@pytest.mark.parametrize("planted", ["__doc__", "__annotations__"])
def test_application_code_behind_builtin_lookups(planted):
    """A key colliding with a name the walk looks up, and a foreign getset
    planted as a metaclass's ``__dict__``, run in both walks alike."""

    def build():
        meta = type.__new__(type, "Meta", (type,), {"__dict__": type.__dict__[planted]})
        cls = meta("Cls", (), {planted: _Reading()})
        fake = FakeClass("<snap>", {_Evil("_FakeClass__repr"): 1})
        copies = []
        for copy_type, original in ((FakeList, [1]), (FakeDict, {"a": 1})):
            copied = copy_type(original)
            vars(copied)[_Evil(copy_type._type_key)] = 1
            vars(copied)[copy_type._type_key] = "app.Thing"
            copies.append(copied)
        return (cls, _odd_types()[0](), fake, copies)

    def calls(context, cls, handed_back, fake, copies):
        return [
            lambda: context.value([cls]),
            lambda: context.value([cls, handed_back]),
            lambda: context.value([fake, fake]),
            lambda: context.value(copies),
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_deep_caller_stack_under_a_raised_limit():
    """A capture far below a raised recursion limit stays native however
    deep the caller's stack is, and matches the Python walk.  The graph is
    shallow enough for the native stack budget of any build."""
    nested = None
    for _ in range(12):
        nested = [nested, {"k": (nested,)}]
    saved = sys.getrecursionlimit()
    sys.setrecursionlimit(40000)
    try:
        outcomes = {}
        for native in (True, False):
            with _mode(native):
                context = SnapshotContext(DumpOptions(max_depth=None, max_nodes=None))
                captured = _descend(5000, lambda: context.value(nested))
                if native:
                    assert context._walker.handbacks == 0 or _traced()
                outcomes[native] = pickle.dumps(
                    (captured, dict(context._truncation), context._nodes), protocol=5
                )
        assert outcomes[True] == outcomes[False]
    finally:
        sys.setrecursionlimit(saved)


_AUDIT_SCRIPT = """
import os, pickle, sys
from pydumpling import _native
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

def deny(event, arguments):
    if event == "sys._getframe":
        raise RuntimeError("denied")

sys.addaudithook(deny)
nested = None
for _ in range(15):
    nested = [nested, {"k": nested}]
outcomes = {}
for native in (True, False):
    os.environ["PYDUMPLING_NATIVE"] = "1" if native else "0"
    context = SnapshotContext(DumpOptions(max_depth=None, max_nodes=None))
    captured = context.value({"deep": nested})
    walker = vars(context).get("_walker")
    outcomes[native] = pickle.dumps(
        (captured, dict(context._truncation), context._nodes), protocol=5
    )
    if native:
        print("walker", isinstance(walker, _native.Walker), walker.handbacks)
print("same" if outcomes[True] == outcomes[False] else "different")
"""


def test_audit_hook_refusing_frame_access():
    """An audit hook that refuses ``sys._getframe`` costs the native walk
    nothing but a measured start: the capture stays native and the same."""
    process = subprocess.run(
        [sys.executable, "-c", _AUDIT_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.split() == ["walker", "True", "0", "same"]


def test_placeholders_replaced_in_place():
    """A depth placeholder replaced in place, by a call the Python walk takes
    or by a native one after the walker wrote it, is the copy later calls
    reuse, natively or in Python."""

    def build():
        return ({"a": {"leaf": [1]}}, {"b": {"leaf": [2]}}, _odd_types()[0]())

    def calls(context, first, second, handed_back):
        return [
            lambda: context.value([first, second]),
            # A bool depth is not one the walker reads: Python takes the call.
            lambda: context._mapping(first, False),
            lambda: context.value({"again": first}),
            lambda: copy.copy(context) and None,
            lambda: context._mapping(second, 0),
            lambda: context.value([second, handed_back]),
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=1), calls)


@pytest.mark.parametrize("key", [1, 1.5, b"reason"], ids=["int", "float", "bytes"])
def test_a_placeholder_with_a_key_not_a_str_is_handed_back(key):
    """``_mapping`` meeting a depth placeholder in its memo reads its
    ``reason`` only from an instance dict of exact str keys: one that holds
    another, a bytes key too, which compared with a str warns under
    ``python -b``, hands the call back, and the Python walk replaces it."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=1))
        _check_walker(context, native)
        inner = {"k": 1}
        (placeholder,) = context.value([inner])
        vars(placeholder)[key] = 1
        found[native] = (context._mapping(inner), dict(context._truncation))
    assert found[True] == found[False] == ({"k": 1}, {"max_depth": 1})
    assert context._walker.handbacks == 1


@pytest.mark.parametrize("answer", ["a str subclass", "max_depth", "nothing"])
def test_a_placeholder_class_hook_runs_once_where_mapping_meets_it(monkeypatch, answer):
    """``_mapping`` meeting a depth placeholder reads its ``reason``, which
    an attribute hook application code gave ``TruncatedValue`` answers with
    a str subclass, the reason itself, or ``AttributeError``: the hook runs
    once, as in the Python walk, and the call goes on as there."""

    class Reason(str):
        pass

    def build():
        return ({"k": 1},)

    def calls(context, mapping):
        def hooked():
            def reading(placeholder, name):
                if name != "reason":
                    return object.__getattribute__(placeholder, name)
                _called("placeholder reason")
                if answer == "nothing":
                    raise AttributeError(name)
                reason = object.__getattribute__(placeholder, name)
                return Reason(reason) if answer == "a str subclass" else reason

            with monkeypatch.context() as patch:
                patch.setattr(TruncatedValue, "__getattribute__", reading)
                return context._mapping(mapping)

        return [
            lambda: context.value([mapping]) and None,  # too deep: a placeholder
            hooked,
            lambda: context.value(mapping),
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=1), calls)


@pytest.mark.parametrize("taken", ["natively", "by a hand-back", "natively, copied"])
def test_a_placeholder_captured_as_itself_and_replaced_in_place(taken):
    """A depth placeholder captured as its own copy is, to the Python walk's
    ``_mapping``, a depth placeholder like any other: it replaces it in
    place before it fails on the placeholder's missing ``items``. The
    walker, which handed that call back, takes the replacement up and
    returns it from then on, as the Python walk does."""

    def build():
        return (TruncatedValue("max_depth", "dict"),)

    def calls(context, placeholder):
        taking = {
            "natively": [lambda: context.value(placeholder) is placeholder],
            "by a hand-back": [lambda: context.value(value=placeholder) is placeholder],
            "natively, copied": [
                lambda: copy.copy(context) and None,
                lambda: context.value(placeholder) is placeholder,
            ],
        }
        return [
            *taking[taken],
            lambda: context._mapping(placeholder, 0),
            lambda: context.value(placeholder),
            lambda: context.value([placeholder])[0],
            lambda: copy.copy(context).objects[id(placeholder)],
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_a_placeholder_replaced_in_place_is_let_go():
    """A depth placeholder that a root capture replaces in place is held
    nowhere once that call returns, as in the Python walk, where the copy
    took its place in ``objects``."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=1))
        _check_walker(context, native)
        mapping = {"k": 1}
        placeholder = weakref.ref(context.value(mapping, 1))
        context._mapping(mapping)
        gc.collect()
        assert placeholder() is None


def _inside_the_walk(context):
    """Whether a native walk on ``context`` is running on this thread."""
    try:
        context._walker.handbacks
    except RuntimeError:
        return True
    except AttributeError:
        return False  # A callback-safe hand-back temporarily removes the walker.
    return False


@contextlib.contextmanager
def _collecting(callback):
    """Collect garbage as often as possible, calling ``callback`` each time."""
    saved = gc.get_threshold()
    gc.callbacks.append(callback)
    gc.set_threshold(1, 1, 1)
    try:
        yield
    finally:
        gc.set_threshold(*saved)
        gc.callbacks.remove(callback)


def test_capture_reentered_by_a_collection():
    """A collection callback that captures on the context while the native
    walk runs is refused with RuntimeError; outside the walk it captures as
    usual; and the outer capture is unchanged.

    A collection runs inside a native call only at the walker's signal
    checks, one per 4096 nodes, so the graph crosses many of them.
    """
    graph = [[i, str(i)] for i in range(20000)]
    results = {}
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
            _check_walker(context, native)
        calls = []

        def callback(phase, info, context=context, calls=calls, native=native):
            if phase != "start" or len(calls) >= 50:
                return
            inside = native and _inside_the_walk(context)
            try:
                calls.append((inside, context.value(["inner", 2.5])))
            except RuntimeError as error:
                calls.append((inside, error))

        with _collecting(callback):
            captured = context.value(graph)
        inside = [outcome for within, outcome in calls if within]
        outside = [outcome for within, outcome in calls if not within]
        assert all(copy == ["inner", 2.5] for copy in outside)
        if native:
            assert inside, "no collection ran during the native walk"
            assert all(
                isinstance(error, RuntimeError) and "already capturing" in str(error)
                for error in inside
            )
        else:
            assert outside and not inside
        results[native] = pickle.dumps(captured, protocol=5)
    assert results[True] == results[False]


_REENTRY_AUDIT_SCRIPT = """
import copy, sys, threading
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

mode = sys.argv[1]
context = SnapshotContext(DumpOptions())
assert context._walker is not None
entered, done = threading.Event(), threading.Event()
errors, independent = [], []
fired = False

def capture_same():
    try:
        if mode == 'copy':
            copy.copy(context)
        else:
            context.value([1])
    except RuntimeError as error:
        errors.append(str(error))
    finally:
        done.set()

def audit(event, args):
    global fired
    if event != 'sys._getframe' or fired:
        return
    fired = True
    if mode == 'other':
        entered.set()
        assert done.wait(5), 'second capture did not finish'
    elif mode == 'separate':
        independent.append(SnapshotContext(DumpOptions()).value([1]))
    else:
        capture_same()

if mode == 'other':
    def other():
        if entered.wait(5):
            capture_same()
    thread = threading.Thread(target=other)
    thread.start()

sys.addaudithook(audit)
def descend(depth):
    if depth:
        return descend(depth - 1)
    return context.value(['outer'])

assert descend(600) == ['outer']
if mode == 'other':
    thread.join(5)
    assert not thread.is_alive()
assert fired
if mode == 'separate':
    assert independent == [[1]]
elif mode == 'copy':
    assert errors == []
else:
    assert len(errors) == 1 and 'already capturing' in errors[0], errors
print('ok')
"""


@pytest.mark.parametrize("mode", ["same", "other", "copy", "separate"])
def test_unanticipated_capture_reentry(mode):
    """Unknown same-context reentry fails closed; independent contexts work,
    and so does copying the context, which captures nothing."""
    process = subprocess.run(
        [sys.executable, "-c", _REENTRY_AUDIT_SCRIPT, mode],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "ok"


def test_descriptor_without_an_owner_or_arguments():
    """A descriptor built without an owner cannot be pickled, and a call with
    no context fails as the function's own does."""
    function = vars(values_module.ValuesMixin)["value"].__wrapped__
    with pytest.raises(TypeError, match="without an owner"):
        pickle.dumps(values_module._native.Dispatch(function))
    with pytest.raises(TypeError) as native_error:
        vars(values_module.ValuesMixin)["value"]()
    with pytest.raises(TypeError) as pure_error:
        function()
    assert str(native_error.value) == str(pure_error.value)


@pytest.mark.parametrize("method", ["value", "_mapping"])
def test_keyword_self_preserves_memo(method):
    """A keyword receiver must use the same memo as a positional receiver."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=3))
        source = [1, 2] if method == "value" else {"key": 1}
        captured = getattr(context, method)(source)
        keyword = "value" if method == "value" else "source"
        again = getattr(SnapshotContext, method)(self=context, **{keyword: source})
        assert again is captured
        assert context.value(source) is captured
        assert context._nodes == 3
        assert not context._truncation


def test_same_size_instance_change_honors_walk_override():
    """Replacing an unrelated attribute with a helper must retire the walker."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        context.scratch = None
        context.value(1)
        size = len(vars(context))
        del context.scratch
        context._text = lambda value: "<redacted>"
        assert len(vars(context)) == size
        assert context.value("secret") == "<redacted>"


def test_class_change_after_construction_honors_walk_override():
    """A helper installed on the context class also invalidates native dispatch."""
    for native in (False, True):

        class Context(SnapshotContext):
            pass

        with _mode(native):
            context = Context(DumpOptions())
        context.value(1)
        Context._text = lambda self, value: "<redacted>"
        assert context.value("secret") == "<redacted>"


def test_deleted_unused_helper_retires_walker(monkeypatch):
    """A missing helper need not fail a Python capture that never calls it."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        context.value(1)
        with monkeypatch.context() as patch:
            patch.delattr(SnapshotContext, "_text")
            assert context.value(42) == 42


@contextlib.contextmanager
def _walk_callback(kind, callback):
    """Run a callback at a Python helper, restoring any test runner hook."""
    getter = sys.getprofile if kind == "profile" else sys.gettrace
    setter = sys.setprofile if kind == "profile" else sys.settrace
    previous = getter()

    def hook(frame, event, arg):
        if event == "call" and frame.f_code is SnapshotContext._bounded_text.__code__:
            callback()
        return hook

    setter(hook)
    try:
        yield
    finally:
        setter(previous)


def test_gc_callback_capture_shares_memo_and_budget():
    """GC callbacks use the same memo and budget as the enclosing capture."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=20, max_items=None))
        graph = [[i] for i in range(2000)]
        inner = []

        def callback(phase, info):
            if phase == "start" and not inner:
                inner.append(context.value(graph[0]))

        with _collecting(callback):
            captured = context.value(graph)
        assert inner and inner[0] is captured[0]
        assert context.value(graph[0]) is captured[0]
        copy.copy(context)
        assert context._nodes == len(context.objects) <= 20


def _handed_back():
    """A value the walker leaves to the Python walk: its type has an odd name."""
    return _odd_types()[0]()


@pytest.mark.parametrize("kind", ["profile", "trace"])
def test_tracing_hook_reentering_a_hand_back(kind):
    """A tracing hook runs only in the Python walk. One that captures there,
    in a call the walker handed back, shares that call's copies."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_string_bytes=8))
        shared, inner = [], []

        def capture():
            if not inner:
                inner.append(context.value(shared))

        with _walk_callback(kind, capture):
            captured = context.value([shared, "x" * 100, _handed_back()])
        assert len(inner) == 1
        assert inner[0] is captured[0]
        assert context.value(shared) is captured[0]
        if native:
            assert context._walker.handbacks == 1


@pytest.mark.parametrize("local", [False, True], ids=["global", "local"])
def test_monitoring_callback_reentering_a_hand_back(local):
    """A monitoring callback, global or set on one code object, likewise."""
    monitoring = sys.monitoring
    tool_id = next(
        (tool for tool in reversed(range(6)) if monitoring.get_tool(tool) is None),
        None,
    )
    if tool_id is None:
        pytest.skip("all monitoring tool ids are in use")
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_string_bytes=8))
        shared, inner = [], []
        code = SnapshotContext._bounded_text.__code__

        def callback(started, offset):
            if started is code and not inner:
                inner.append(context.value(shared))

        monitoring.use_tool_id(tool_id, "pydumpling-reentry-test")
        try:
            monitoring.register_callback(tool_id, monitoring.events.PY_START, callback)
            if local:
                monitoring.set_local_events(tool_id, code, monitoring.events.PY_START)
            else:
                monitoring.set_events(tool_id, monitoring.events.PY_START)
            captured = context.value([shared, "x" * 100, _handed_back()])
        finally:
            monitoring.set_events(tool_id, 0)
            monitoring.set_local_events(tool_id, code, 0)
            monitoring.register_callback(tool_id, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool_id)
        assert len(inner) == 1
        assert inner[0] is captured[0]
        assert context.value(shared) is captured[0]
        if native:
            assert context._walker.handbacks == 1


@pytest.mark.parametrize("kind", ["profile", "trace"])
def test_tracing_hook_shares_the_node_budget(kind):
    """The nested capture spends the remaining node budget of the call it
    runs in, natively handed back or in Python."""
    outcomes = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_string_bytes=8, max_nodes=4))
        inner = []

        def capture():
            if not inner:
                inner.append(context.value(["inner", 2.5]))

        with _walk_callback(kind, capture):
            captured = context.value([1, "x" * 100, _handed_back()])
        copy.copy(context)
        assert inner == [[]]
        assert context._nodes == len(context.objects) == 4
        outcomes[native] = pickle.dumps(
            (captured, inner, list(context._truncation.items()), context._nodes),
            protocol=5,
        )
    assert outcomes[True] == outcomes[False]


@contextlib.contextmanager
def _recording(kind, seen, callers=None):
    """Record each Python function a profiler, tracer or monitoring tool sees,
    and in ``callers`` the file of the frame that called it."""

    def record(frame):
        seen.append(frame.f_code)
        if callers is not None:
            caller = frame.f_back
            callers.append(caller.f_code.co_filename if caller else None)

    if kind == "monitoring":
        monitoring = sys.monitoring
        tool_id = next(
            (tool for tool in reversed(range(6)) if monitoring.get_tool(tool) is None),
            None,
        )
        if tool_id is None:
            pytest.skip("all monitoring tool ids are in use")
        monitoring.use_tool_id(tool_id, "pydumpling-probe-test")
        monitoring.register_callback(
            tool_id,
            monitoring.events.PY_START,
            lambda code, offset: record(sys._getframe(1)),
        )
        monitoring.set_events(tool_id, monitoring.events.PY_START)
        try:
            yield
        finally:
            monitoring.set_events(tool_id, 0)
            monitoring.register_callback(tool_id, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool_id)
        return
    getter = sys.getprofile if kind == "profile" else sys.gettrace
    setter = sys.setprofile if kind == "profile" else sys.settrace

    def hook(frame, event, arg):
        if event == "call":
            record(frame)
        return hook

    previous = getter()
    setter(hook)
    try:
        yield
    finally:
        setter(previous)


@pytest.mark.parametrize("kind", ["profile", "trace", "monitoring"])
def test_tracing_hooks_never_see_the_stack_probe(kind):
    """Under a profiler, tracer or monitoring tool a shallow capture stays
    native and runs no Python code; a deep one, which needs the Python stack
    probe, goes to the Python walk before the probe runs."""
    options = DumpOptions(max_depth=None, max_items=None, max_nodes=None)
    deep = [0]
    for level in range(200):
        deep = [level, deep]
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value(deep), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    shallow, nested = [], []
    with _recording(kind, shallow):
        assert context.value([[1, "a"], {"k": (2.5,)}]) == [[1, "a"], {"k": (2.5,)}]
    with _recording(kind, nested):
        captured = context.value(deep)
    assert context._walker.handbacks == 1
    assert pickle.dumps(captured, protocol=5) == expected
    walk_files = {values_module.__file__, context_module.__file__}
    assert not [code for code in shallow if code.co_filename in walk_files]
    assert values_module._headroom.__code__ not in nested
    assert SnapshotContext.value.__wrapped__.__code__ in nested


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is unavailable")
def test_signal_handlers_and_asyncio_keep_the_native_walk():
    """Installing a Python signal handler, as ``asyncio.run`` installs one for
    SIGINT, leaves the walker in use and the handler installed: a handler
    runs inside a native call only at the walker's signal checks."""
    with _mode(True):
        context = SnapshotContext(DumpOptions())

    def handler(number, frame):
        pass

    previous = signal.signal(signal.SIGUSR1, handler)
    try:
        assert context.value([1, 2]) == [1, 2]
        assert signal.getsignal(signal.SIGUSR1) is handler
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert context._walker.handbacks == 0

    async def main():
        with _mode(True):
            inner = SnapshotContext(DumpOptions())
        assert inner.value({"a": [1]}) == {"a": [1]}
        return inner._walker.handbacks

    assert asyncio.run(main()) == 0


def test_native_calls_run_no_import_or_class_hooks():
    """A native call runs neither ``builtins.__import__`` nor a hook of the
    context's metaclass, such as a property in place of ``__mro__`` or an
    attribute hook, or of a descriptor set on its class."""
    lookups = []

    def logged(name):
        raw = type.__dict__[name]

        def get(cls):
            lookups.append(name)
            return raw.__get__(cls, type(cls))

        return property(get)

    names = ("__mro__", "__bases__", "__dict__", "__name__", "__flags__")
    Logged = type("Logged", (type,), {name: logged(name) for name in names})

    class Meta(type):
        def __getattribute__(cls, name):
            lookups.append(name)
            return super().__getattribute__(name)

    class Descriptor:
        def __get__(self, instance, owner=None):
            lookups.append(("__get__", instance is None))
            return types.MethodType(SnapshotContext._text, instance)

    class Hooked(SnapshotContext, metaclass=Meta):
        pass

    with _mode(True):
        hooked = Hooked(DumpOptions())
    assert hooked.value([1, "x"]) == [1, "x"]
    assert hooked._walker is not None
    assert hooked._walker.handbacks == 0 or _traced()
    assert lookups == []

    class Context(SnapshotContext, metaclass=Logged):
        pass

    with _mode(True):
        context = Context(DumpOptions(max_string_bytes=4))
    assert context._walker is not None
    imports = []
    real_import = builtins.__import__

    def spy(name, *args, **kwargs):
        imports.append(name)
        return real_import(name, *args, **kwargs)

    lookups.clear()
    builtins.__import__ = spy
    try:
        for value in ([1, "abcdefgh"], {"k": (2.5, None)}, "x" * 50):
            context.value(value)
        handbacks = context._walker.handbacks
        Context._text = Descriptor()
        retired = context.value("y" * 50)
    finally:
        builtins.__import__ = real_import
        del Context._text
    assert handbacks == 0 or _traced()
    assert not imports
    # The patched helper retires the walker; only the Python walk, which
    # calls it as a bound method, runs the descriptor.
    assert context._walker is None
    assert lookups == [("__get__", False)]
    assert retired == SnapshotContext(DumpOptions(max_string_bytes=4)).value("y" * 50)


# --- State the context reads through its class ------------------------------


def _logged_property(name, log):
    """A property in place of the state attribute ``name`` that logs each
    access; the value stays in the instance dict under the same name."""

    def get(self):
        log.append(("get", name))
        return vars(self)[name]

    def put(self, value):
        log.append(("set", name))
        vars(self)[name] = value

    return property(get, put)


def _logged_hooks(log):
    """Attribute hooks a context class may define, logging each call."""

    def getattribute(self, name):
        log.append(("get", name))
        return object.__getattribute__(self, name)

    def setattr_(self, name, value):
        log.append(("set", name))
        object.__setattr__(self, name, value)

    def delattr_(self, name):
        log.append(("del", name))
        object.__delattr__(self, name)

    return {
        "__getattribute__": getattribute,
        "__setattr__": setattr_,
        "__delattr__": delattr_,
    }


def _logged_capture(install, when):
    """Capture in each mode on a fresh context subclass that ``install``
    changes, before construction or after a first call, and return the
    pickled results and the log of what ``install`` added."""
    data = {"a": [1, 2, 3, 4], "b": "text" * 10, "c": _Record(1, [2]), "d": (2.5,)}
    outcomes = {}
    for native in (False, True):
        log = []

        class Context(SnapshotContext):
            pass

        if when == "subclass":
            install(Context, log)
        with _mode(native):
            context = Context(
                DumpOptions(max_items=3, max_nodes=12, max_string_bytes=8)
            )
        first = context.value([1, "x" * 20])
        if when == "patched":
            install(Context, log)
        results = (
            first,
            context.value(data),
            context._mapping({"k": data}, 0),
            context.value(data),
            context.value([5]),
        )
        state = (results, list(context._truncation.items()), context._nodes)
        outcomes[native] = (pickle.dumps(state, protocol=5), list(log))
        assert object.__getattribute__(context, "__dict__").get("_walker") is None
    return outcomes


@pytest.mark.parametrize("when", ["subclass", "patched"])
@pytest.mark.parametrize(
    "name",
    [
        "options",
        "objects",
        "_sources",
        "_nodes",
        "_truncation",
        "_layouts",
        "_node_marker",
        "_unlimited",
    ],
)
def test_state_behind_a_property_runs_as_in_python(name, when):
    """A state attribute behind a property, on the class from the start or
    patched on after a call, runs that property exactly as the Python walk
    does: the walker never reads it, and gives the context to Python."""
    outcomes = _logged_capture(
        lambda cls, log: setattr(cls, name, _logged_property(name, log)), when
    )
    assert outcomes[True] == outcomes[False]
    assert outcomes[True][1], "the property never ran"


@pytest.mark.parametrize("when", ["subclass", "patched"])
@pytest.mark.parametrize("hook", ["__getattribute__", "__setattr__", "__delattr__"])
def test_context_attribute_hooks_run_as_in_python(hook, when):
    """The context class's own attribute hooks, defined on it or patched on
    after a call, run exactly as often as in the Python walk: the walker
    reads and clears its own attribute without them. Neither walk deletes
    an attribute, so ``__delattr__`` checks only that the context goes to
    the Python walk."""
    outcomes = _logged_capture(
        lambda cls, log: setattr(cls, hook, _logged_hooks(log)[hook]), when
    )
    assert outcomes[True] == outcomes[False]


@pytest.mark.parametrize(
    "tally", [1.5, _Count(2), True, 2**63 - 1], ids=["float", "int", "bool", "max"]
)
def test_truncation_tally_of_another_type(tally):
    """A truncation tally the walker does not store itself, set between
    calls, is counted on in Python, as the Python walk counts it. An int
    subclass and a bool are controls: the walker counts those on itself."""

    def build():
        return (list(range(12)), list(range(20)))

    def calls(context, first, second):
        def tallied():
            context._truncation["max_items"] = tally
            return context.value(second)

        return [
            lambda: context.value(first),
            tallied,
            lambda: dict(context._truncation),
        ]

    native, _ = _same_in_both_modes(build, DumpOptions(max_items=4), calls)
    assert native.results[-1]["max_items"] == tally + 1


def test_layouts_cache_of_another_class():
    """A layout cache that is not a plain dict is looked up per instance,
    as the Python walk looks it up."""

    class Point(tuple):
        pass

    def build():
        return ([Point((1, 2)), Point((3, 4)), Point((5, 6))],)

    def calls(context, points):
        def replaced():
            context._layouts = _Layouts()
            return context.value(points)

        return [replaced, lambda: context.value([Point((7, 8))])]

    native, _ = _same_in_both_modes(build, DumpOptions(), calls)
    assert native.hooks.count("layouts get") == 4


@pytest.mark.parametrize("then", ["call", "copy"])
@pytest.mark.parametrize("kind", [dict, OrderedDict], ids=["dict", "OrderedDict"])
def test_replaced_sources_keep_the_walker_copies(kind, then):
    """Replacing only ``_sources`` keeps every copy in ``objects`` and the
    originals of earlier copies in the old ``_sources``, so the next capture,
    on the context or a copy of it, reuses them as in the Python walk. The
    copy, whose Python walk reads the dicts as they are, is a control."""

    def build():
        return ([1, [2], {"k": "v"}],)

    def calls(context, shared):
        def replaced():
            old = context._sources
            context._sources = kind()
            if then == "copy":
                target = copy.copy(context)
                result = target.value({"again": shared})
            else:
                result = context.value({"again": shared})
                target = copy.copy(context)
            return (
                result,
                target._nodes,
                len(target.objects),
                list(old.values()),
                list(target._sources.values()),
            )

        return [lambda: context.value(shared), replaced]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_capturing_on_a_context_and_its_copy():
    """A shallow copy shares ``objects`` with its original, so copies either
    one makes later, the original natively and the copy in Python, are the
    other's too, as when both capture in Python."""

    def build():
        return ([1, "a"], {"b": [2, "c"]})

    def calls(context, first, later):
        made = {}

        def copied():
            made["twin"] = copy.copy(context)
            made["original"] = context.value(later)
            return made["original"]

        def on_the_copy():
            made["copy"] = made["twin"].value([later])
            return made["copy"]

        def identities():
            twin = made["twin"]
            again = context.value({"x": later, "y": first})
            return (
                made["copy"][0] is made["original"],
                again["x"] is made["original"],
                twin.objects is context.objects,
                len(context.objects),
                twin._nodes,
            )

        return [lambda: context.value(first), copied, on_the_copy, identities]

    native, _ = _same_in_both_modes(build, DumpOptions(), calls)
    assert native.results[-1][:3] == (True, True, True)


@pytest.mark.parametrize("kind", ["profile", "trace", "monitoring"])
def test_tracing_hooks_see_container_hashes_only_in_python(kind):
    """Hashing a copy of a named tuple runs ``FakeContainer.__hash__``, which
    is Python code: under a profiler, tracer or monitoring tool, a call
    that would hash one goes to the Python walk before the hash runs."""
    point = _Record(1, 2)
    values = [{point}, {point: "v"}, frozenset({point, 3})]
    with _mode(False):
        pure = SnapshotContext(DumpOptions())
        pure.value(point)
        expected = pickle.dumps([pure.value(value) for value in values], protocol=5)
    with _mode(True):
        context = SnapshotContext(DumpOptions())
    context.value(point)
    # Under a test runner's tracer the first call may have handed back too.
    before = context._walker.handbacks
    seen, callers = [], []
    with _recording(kind, seen, callers):
        captured = [context.value(value) for value in values]
    hashed_from = {
        caller
        for code, caller in zip(seen, callers)
        if code is FakeContainer.__hash__.__code__
    }
    assert pickle.dumps(captured, protocol=5) == expected
    assert context._walker.handbacks - before == len(values)
    assert hashed_from == {values_module.__file__}


@pytest.mark.filterwarnings("ignore:non-string key:RuntimeWarning")
def test_context_class_dict_key_that_is_not_str():
    """A context class whose dict holds a key other than an exact str captures
    in Python, and the walker never looks a name up in it: only the Python
    walk's own lookups run the key's ``__eq__``."""
    compared = []

    class Key(str):
        def __hash__(self):
            return hash("_text")

        def __eq__(self, other):
            compared.append(sys._getframe(1).f_code.co_filename)
            return str.__eq__(self, other)

    results = {}
    for native in (False, True):
        Context = type("Context", (SnapshotContext,), {Key("tag"): 1})
        with _mode(native):
            context = Context(DumpOptions(max_string_bytes=4))
        results[native] = pickle.dumps(
            [context.value(["abcdefgh", (1, 2.5)]), context.value("x" * 20)],
            protocol=5,
        )
        assert object.__getattribute__(context, "__dict__").get("_walker") is None
    assert results[True] == results[False]
    assert __file__ not in compared


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="setitimer is unavailable")
def test_signal_handler_capturing_inside_a_native_walk():
    """A signal handler runs inside a native call at the walker's signal
    checks. One that captures on the context being walked is refused with
    RuntimeError; the outer call stays native and its result unchanged."""
    graph = [[i, str(i)] for i in range(100000)]
    options = DumpOptions(max_nodes=None, max_items=None)
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value(graph), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    outcomes = []

    def handler(number, frame):
        inside = _inside_the_walk(context)
        try:
            outcomes.append((inside, context.value([1])))
        except RuntimeError as error:
            outcomes.append((inside, error))

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.0005, 0.0005)
    try:
        captured = context.value(graph)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    inside = [outcome for within, outcome in outcomes if within]
    assert inside, "no signal arrived during the native walk"
    assert all(
        isinstance(error, RuntimeError) and "already capturing" in str(error)
        for error in inside
    )
    assert context._walker is not None
    assert context._walker.handbacks == 0 or _traced()
    assert pickle.dumps(captured, protocol=5) == expected


# --- Copies and hooks the interpreter alone shows ---------------------------


def test_copy_replacing_a_placeholder_the_original_omitted():
    """A copy whose Python walk captures, as a root, a mapping the original
    omitted by depth replaces the placeholder in the ``objects`` they share:
    the original then reuses that copy, as when both capture in Python."""

    def build():
        inner = {"x": {"y": 1}}
        return ({"d": inner}, inner)

    def calls(context, outer, inner):
        return [
            lambda: context.value(outer),
            lambda: copy.copy(context)._mapping(inner, 0),
            lambda: context.value(inner),
            lambda: context.value([inner]),
            lambda: context._mapping(inner, 0),
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=1), calls)


def test_instance_dict_descriptor_of_another_class():
    """A context class whose ``__dict__`` entry is another class's
    descriptor captures as the Python walk does: in Python where that
    descriptor refuses the context, natively where it accepts it, as from
    3.15."""

    class Other:
        pass

    descriptor = vars(Other)["__dict__"]
    results = {}
    for native in (False, True):
        Context = type("Context", (SnapshotContext,), {"__dict__": descriptor})
        with _mode(native):
            context = Context(DumpOptions(max_string_bytes=4))
        captured = [context.value(["abcdefgh", (1, 2.5)]), context.value("x" * 20)]
        state = (captured, list(context._truncation.items()), context._nodes)
        results[native] = pickle.dumps(state, protocol=5)
        try:
            descriptor.__get__(context, Context)
        except TypeError:
            assert object.__getattribute__(context, "_walker") is None
    assert results[True] == results[False]


@contextlib.contextmanager
def _profiled_from_c(calls):
    """Set a profile function from C without an object, as yappi sets one,
    which ``sys.getprofile`` and ``sys.monitoring.get_tool`` cannot see,
    recording each Python call with its caller's file."""
    import ctypes

    profile = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.py_object, ctypes.c_int, ctypes.c_void_p
    )

    def record(obj, frame, what, arg):
        if what == 0:  # PyTrace_CALL
            caller = frame.f_back
            calls.append((frame.f_code, caller.f_code.co_filename if caller else None))
        return 0

    callback = profile(record)
    set_profile = ctypes.pythonapi.PyEval_SetProfile
    set_profile.argtypes = [profile, ctypes.c_void_p]
    set_profile.restype = None
    previous = sys.getprofile()
    set_profile(callback, None)
    try:
        assert sys.getprofile() is None
        yield
    finally:
        set_profile(profile(0), None)
        sys.setprofile(previous)


@contextlib.contextmanager
def _freed_tool(calls):
    """A monitoring tool that freed its id with its events still set, which
    3.12 and 3.13 keep running; ``sys.monitoring.get_tool`` no longer sees
    it."""
    monitoring = sys.monitoring
    tool_id = next(
        (tool for tool in reversed(range(6)) if monitoring.get_tool(tool) is None), None
    )
    if tool_id is None:
        pytest.skip("all monitoring tool ids are in use")

    def record(code, offset):
        caller = sys._getframe(1).f_back
        calls.append((code, caller.f_code.co_filename if caller else None))

    monitoring.use_tool_id(tool_id, "pydumpling-freed-tool-test")
    monitoring.register_callback(tool_id, monitoring.events.PY_START, record)
    monitoring.set_events(tool_id, monitoring.events.PY_START)
    monitoring.free_tool_id(tool_id)
    try:
        assert monitoring.get_tool(tool_id) is None
        yield
    finally:
        monitoring.use_tool_id(tool_id, "pydumpling-freed-tool-test")
        monitoring.set_events(tool_id, 0)
        monitoring.register_callback(tool_id, monitoring.events.PY_START, None)
        monitoring.free_tool_id(tool_id)


@pytest.mark.parametrize("hook", [_profiled_from_c, _freed_tool], ids=["C", "freed"])
def test_hooks_only_the_interpreter_shows_see_helpers_only_in_python(hook):
    """A hook ``sys.getprofile`` and ``sys.monitoring.get_tool`` miss still
    sends a call that runs pydumpling's helpers to the Python walk first:
    the hook sees them called from the Python walk only."""
    data = ["x" * 50, {"k": "y" * 50}]
    options = DumpOptions(max_string_bytes=8)
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value(data), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    calls = []
    with hook(calls):
        captured = context.value(data)
    assert pickle.dumps(captured, protocol=5) == expected
    text = {caller for code, caller in calls if code.co_name == "_text"}
    assert text <= {values_module.__file__}
    if hook is _profiled_from_c or sys.version_info < (3, 14):
        # From 3.14 a freed tool id runs no events.
        assert text, "the hook never ran"
        assert context._walker.handbacks == 1


def test_profiler_set_while_a_native_walk_runs():
    """A profiler that a collection sets while a native walk runs sees
    pydumpling's helpers called from the Python walk only: the walker asks
    for hooks again before each helper, and hands the call back."""
    data = ["x" * 50, *([index] for index in range(200)), "y" * 50, "z" * 50]
    options = DumpOptions(max_string_bytes=8, max_items=None, max_nodes=None)
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value(data), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    calls, inside = [], []

    def profile(frame, event, arg):
        if event == "call":
            calls.append((frame.f_code.co_name, frame.f_back.f_code.co_filename))

    def collected(phase, info):
        # A collection runs at the next Python code, here inside one of the
        # helpers: after the walker asked for hooks before calling it.
        if phase == "start" and _inside_the_walk(context):
            inside.append(phase)
            sys.setprofile(profile)

    previous = sys.getprofile()
    try:
        with _collecting(collected):
            captured = context.value(data)
    finally:
        sys.setprofile(previous)
    assert pickle.dumps(captured, protocol=5) == expected
    text = {caller for name, caller in calls if name == "_text"}
    assert text <= {values_module.__file__}
    if not _traced():
        assert inside, "no collection ran inside the native walk"
        assert text, "the profiler never saw a helper"
        assert context._walker.handbacks == 1


# --- Python code that writes, removes or keys memo entries between calls ----


@pytest.mark.parametrize("how", ["exception", "direct"])
def test_entries_python_adds_to_sources_alone(how):
    """An original Python code keeps in ``_sources`` alone between calls, as
    ``exception`` keeps a builtin exception, stays where it is: both dicts
    hold what the Python walk's hold, in the same order."""

    def build():
        return (["first", [1]], ["second", [2]], ValueError("boom"))

    def calls(context, first, second, error):
        def kept():
            if how == "exception":
                return context.exception(error)["formatted"]
            context._sources[id(error)] = error
            return None

        return [
            lambda: context.value(first),
            kept,
            lambda: context.value(second),
            lambda: _contents(context),
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_entries_python_removes_between_calls():
    """Entries Python code removes from ``objects`` and ``_sources`` between
    calls leave the copies the walker made since, which the Python walk
    wrote at once, in the memo and in both dicts."""

    def build():
        return (["fresh"], [1, ["two"]], {"three": [3]})

    def calls(context, fresh, first, later):
        def added():
            context.objects[id(fresh)] = "<python>"
            context._sources[id(fresh)] = fresh

        def removed():
            del context.objects[id(fresh)]
            del context._sources[id(fresh)]

        def memo():
            twin = copy.copy(context)
            return list(twin.objects.values()), list(twin._sources.values())

        return [
            added,
            lambda: context.value(first),
            removed,
            lambda: context.value({"again": first, "later": later}),
            memo,
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_root_capture_after_sources_were_replaced():
    """A root capture that replaces a depth placeholder the walker wrote
    puts the original into a replaced ``_sources``, as the Python walk
    appends it there."""

    def build():
        inner = {"leaf": {"x": 1}}
        return ({"d": inner}, inner, ["later"])

    def calls(context, outer, inner, later):
        def replaced():
            context._sources = {}

        return [
            lambda: context.value(outer),
            replaced,
            lambda: context.value(later),
            lambda: context._mapping(inner, 0),
            lambda: _contents(context),
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=1), calls)


@pytest.mark.parametrize(
    "where",
    [
        "appended",
        "in place",
        "in place, after a removal",
        "in place, in a replaced dict",
    ],
)
def test_a_placeholder_without_a_reason_raises_only_where_the_walk_reads_it(where):
    """A ``TruncatedValue`` Python code writes into ``objects`` without a
    ``reason``, appended, over the walker's own depth placeholder, after a
    removal, or in a dict Python put in place of ``objects``, raises
    ``AttributeError`` only where ``_mapping`` meets it, in both walks,
    before or after a copy of the context."""
    raised = {}
    for copied in (False, True):
        for native in (False, True):
            with _mode(native):
                context = SnapshotContext(DumpOptions(max_depth=1))
            _check_walker(context, native)
            first, mapping, extra = [0], {"k": 1}, [0]
            context.value(first)
            context.value([mapping])  # the mapping is too deep: a placeholder
            context.value(value=-1)  # a call handed back
            if copied:
                copy.copy(context)  # a copy shares the dicts
            if where.endswith("replaced dict"):
                context.objects = dict(context.objects)
            target = extra if where == "appended" else mapping
            if where == "appended":
                context._sources[id(extra)] = extra
            context.objects[id(target)] = object.__new__(TruncatedValue)
            if where.endswith("removal"):
                del context.objects[id(first)]
            raised[native, copied] = []
            steps = (False, False, True, False, True, None, False, True)
            for step, handed_back in enumerate(steps):
                try:
                    if handed_back is None:
                        context._mapping(target)
                    elif handed_back:
                        context.value(value=step)
                    else:
                        context.value([step])
                except AttributeError:
                    raised[native, copied].append(step)
    assert raised == dict.fromkeys(raised, [5])


def test_a_placeholder_class_whose_getattribute_is_none_raises_where_mapping_meets_it(
    monkeypatch,
):
    """Application code that sets ``TruncatedValue.__getattribute__ = None``
    makes reading any attribute of a placeholder raise ``TypeError``, as
    ``_mapping`` meeting a depth placeholder does in the Python walk: a
    class dict entry of ``None`` is a ``__getattribute__`` other than
    ``object``'s, so the walker reads no reason there either."""

    def build():
        return ({"k": 1},)

    def calls(context, mapping):
        def hooked():
            with monkeypatch.context() as patch:
                patch.setattr(TruncatedValue, "__getattribute__", None)
                return context._mapping(mapping)

        return [
            lambda: context.value([mapping]) and None,  # too deep: a placeholder
            hooked,
            lambda: context.value(mapping),
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=1), calls)


def test_a_fake_class_given_another_base_is_built_as_its_init_builds_it():
    """``FakeClass`` given a base other than ``FakeType`` with the same
    layout, which ``__bases__`` accepts, still gets its copies: the Python
    walk builds each through ``FakeClass.__init__`` and the new base's
    ``__dict__``, and so does the walker, which hands the copy back where
    the instance-dict descriptor ``FakeType`` had at import refuses it."""

    class Other:
        pass

    class Plain:
        def __init__(self):
            self.x = 1

    def build():
        return (Plain(),)

    def calls(context, value):
        def rebased():
            FakeClass.__bases__ = (Other,)
            try:
                return context.value(value)
            finally:
                FakeClass.__bases__ = (FakeType,)

        return [rebased, lambda: context.value([value])]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_a_fake_class_given_another_base_hands_back_before_running_anything():
    """``FakeClass`` given a base whose ``__getattribute__`` and ``__del__``
    are application code: the walker hands the call back before it makes a
    copy it would drop, whose ``__del__`` would run, as the import-time
    instance-dict descriptor refuses it before 3.15, or reads a
    ``FakeClass``'s representation through the base's ``__getattribute__``,
    which the Python walk then runs again."""
    originals = []

    class Other:
        def __getattribute__(self, name):
            if any(self is original for original in originals):
                _called(f"__getattribute__ {name}")
            return object.__getattribute__(self, name)

        def __del__(self):
            _called("__del__")

    def build():
        originals[:] = [FakeClass("<copied snapshot>", {"a": 1})]
        return (originals[0], _Plain())

    def calls(context, copied, plain):
        def rebased(value):
            FakeClass.__bases__ = (Other,)
            try:
                return context.value(value)
            finally:
                FakeClass.__bases__ = (FakeType,)

        return [lambda: rebased(copied), lambda: rebased(plain)]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_a_fake_class_hook_runs_once_when_a_later_value_hands_the_call_back(
    monkeypatch,
):
    """A ``__getattribute__`` application code gave ``FakeClass`` runs where
    ``FakeClass.__repr__`` reads a copied ``FakeClass``'s ``__dict__``: the
    walker hands such a call back before it runs, so a value it hands the
    call back on later does not make it run twice."""
    originals = []

    def hooked(self, name):
        if any(self is original for original in originals):
            _called(f"FakeClass.__getattribute__ {name}")
        return object.__getattribute__(self, name)

    odd_name = _odd_types()[0]  # a str subclass ``__qualname__``: handed back

    def build():
        originals[:] = [FakeClass("<copied snapshot>", {"odd": odd_name()})]
        return (originals[0],)

    def calls(context, copied):
        def hooked_call():
            with monkeypatch.context() as patch:
                patch.setattr(FakeClass, "__getattribute__", hooked)
                return context.value(copied)

        return [hooked_call]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_a_patched_fake_class_repr_names_the_copy(monkeypatch):
    """The Python walk calls ``FakeClass.__repr__`` as the class has it
    then: one application code patched in names the copy of a copied
    ``FakeClass``, and runs once, as the walker hands such a call back."""

    def patched(self):
        _called("FakeClass.__repr__")
        return "<patched>"

    def build():
        return (FakeClass("<copied snapshot>", {"a": 1}),)

    def calls(context, copied):
        def patched_call():
            with monkeypatch.context() as patch:
                patch.setattr(FakeClass, "__repr__", patched)
                return context.value(copied)

        return [patched_call, lambda: context.value(FakeClass("<again>", {}))]

    native, _ = _same_in_both_modes(build, DumpOptions(), calls)
    assert vars(native.results[0])["_FakeClass__repr"] == "<patched>"


@pytest.mark.parametrize("hook", ["__getattribute__", "a property"])
def test_hooks_patched_onto_dump_options_run_as_in_the_python_walk(monkeypatch, hook):
    """The walker reads every budget each call, the Python walk only those a
    call uses: while ``DumpOptions`` holds a ``__getattribute__``, or under
    a budget's name anything but ``None`` or an int, such as a property,
    the walker leaves the context to the Python walk, so the hook runs as
    often, and a property's answer is the budget."""

    def items(options):
        _called("DumpOptions.max_items")
        return 1

    def get(options, name):
        _called(f"DumpOptions.__getattribute__ {name}")
        return object.__getattribute__(options, name)

    def build():
        return ([[1, 2, 3]],)

    def calls(context, value):
        def hooked():
            with monkeypatch.context() as patch:
                if hook == "a property":
                    patch.setattr(DumpOptions, "max_items", property(items))
                else:
                    patch.setattr(DumpOptions, "__getattribute__", get)
                return context.value(value)

        return [lambda: context.value(None), hooked, lambda: context.value(value)]

    _same_in_both_modes(build, DumpOptions(), calls)


_BYTES_WARNING_SCRIPT = """
import os, warnings
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

mixed = {"a": 1, b"a": 2}  # under -b, built with a warning
counts = []
for native in ("0", "1"):
    os.environ["PYDUMPLING_NATIVE"] = native
    context = SnapshotContext(DumpOptions())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BytesWarning)
        context.value(mixed)  # the copy compares b"a" with "a" again
    counts.append(len(caught))
print(vars(context).get("_walker") is not None, counts[0] == counts[1], counts[1] > 0)
"""


@pytest.mark.parametrize("flags", [[], ["-b"]], ids=["plain", "-b"])
def test_under_python_b_a_context_captures_in_python(flags):
    """Under ``python -b`` comparing bytes with str warns: the walker's own
    lookups and inserts, which a call it hands back makes the Python walk
    repeat, would warn where the Python walk does not, or twice, so a
    context captures in Python there, and warns as the Python walk does."""
    process = subprocess.run(
        [sys.executable, *flags, "-c", _BYTES_WARNING_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.split() == (
        ["False", "True", "True"] if flags else ["True", "True", "False"]
    )


class _IndexKey:
    """A memo key that is no id: its ``__index__`` names one."""

    def __init__(self, target):
        self.target = target

    def __index__(self):
        _called("key __index__")
        return id(self.target)

    def __hash__(self):
        _called("key __hash__")
        return 7

    def __eq__(self, other):
        _called("key __eq__")
        return self is other


@pytest.mark.parametrize("key", ["str", "float", "index", "negative"])
def test_memo_keys_that_are_not_ids(key):
    """A key Python code puts in ``objects`` that is not an id, even one
    equal to an id or with methods of its own, is met as the Python walk
    meets it: the key's methods run as often in both walks, and a copy
    Python wrote for an original is the one both return."""

    def build():
        return ([1, "a"], ["victim"])

    def calls(context, first, victim):
        def planted():
            planted_key = {
                "str": "tag",
                "float": float(id(victim)),
                "index": _IndexKey(victim),
                "negative": -1,
            }[key]
            context.objects[planted_key] = "planted"
            context.objects[id(first)] = "<python>"
            context._sources[id(first)] = first

        def memo():
            twin = copy.copy(context)
            if key == "index":
                return twin.objects[id(first)], len(twin.objects), len(twin._sources)
            return _contents(twin)

        return [
            lambda: context.value(first),
            planted,
            lambda: context.value(victim),
            lambda: context.value({"again": victim, "first": first}),
            memo,
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


class _CollidingKey:
    """A ``_sources`` key of another kind hashing like the id of
    ``target``: looking that id up may run its ``__eq__``."""

    def __init__(self, target, log):
        self.target = target
        self.log = log

    def __hash__(self):
        return hash(id(self.target))

    def __eq__(self, other):
        self.log.append("__eq__")
        return other is self


@pytest.mark.parametrize(
    "how", ["added", "added after a copy", "added with a copy", "replaced sources"]
)
@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
def test_a_sources_key_of_another_kind(native, how):
    """A key of another kind Python code puts in ``_sources`` is compared
    natively no more than in the Python walk: the walker, like it, looks no
    id up in ``_sources``, and writes there only the originals it copies."""
    log = []
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    kept, removed, planted = ["kept"], ["removed"], ["planted"]
    first = context.value(kept)
    context.value(removed)
    if how == "added after a copy":
        copy.copy(context)
    else:
        context.value(value="hand-back")
    key = _CollidingKey(planted if how == "added with a copy" else removed, log)
    if how == "replaced sources":
        context._sources = {key: "tag"}
    else:
        context._sources[key] = "tag"
        if how == "added with a copy":
            context.objects[id(planted)] = "<python>"
        # Adding it compared it with the id it collides with, if held.
        log.clear()
        context.value("next")
        del context._sources[id(removed)]
    assert context.value(kept) == first
    assert log == []


@pytest.mark.parametrize("change", ["added", "removed"])
@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
def test_a_copy_finds_a_sources_key_of_another_kind(native, change):
    """Copying a context whose ``_sources`` holds a key of another kind, and
    capturing after it, runs the key's ``__eq__`` no more than in the
    Python walk."""
    log = []
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    kept, unrelated = KeyError("kept"), object()
    # ``exception`` keeps the original in ``_sources`` alone.
    context.exception(kept)
    context.value(["copied"])
    context._sources[_CollidingKey(kept if change == "removed" else unrelated, log)] = (
        "tag"
    )
    log.clear()
    if change == "removed":
        del context._sources[id(kept)]
    copy.copy(context)
    assert context.value(["next"]) == ["next"]
    assert log == []


@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
def test_a_walker_keeps_nothing_of_its_context_alive_between_calls(native):
    """A walker holds none of the originals, dicts or keys of ``objects``
    and ``_sources`` between calls, though Python code holds it: once
    Python code replaces ``_sources``, its originals and keys go, as in the
    Python walk."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    held = context._walker
    original, key = _Plain(), _Plain()
    released = weakref.ref(original), weakref.ref(key)
    context.value(original)
    context._sources[key] = "tag"
    context.value("next")
    context._sources = {}
    del original, key
    gc.collect()
    assert [ref() for ref in released] == [None, None]
    assert (held is not None) is native


def test_copy_made_while_a_call_runs_in_python():
    """A copy of the context that application code makes while the Python
    walk takes a call from the walker shares the memo from then on, so it
    sees the copies the original makes later, as when both capture in
    Python."""
    made = []

    class CopyingName(str):
        def __radd__(self, other):
            if len(made) == 1:
                made.append(copy.copy(made[0]))
            return str.__add__(other, str(self))

    def build():
        Odd = type("Odd", (), {})
        Odd.__qualname__ = CopyingName("Odd")
        return (Odd(), ["shared", ("t", 1)])

    def calls(context, odd, shared):
        made[:] = [context]
        first = []

        def native():
            first.append(context.value(shared))
            return first[0]

        def on_the_copy():
            twin = made[1]
            result = twin.value({"again": shared})
            return result, twin._nodes, result["again"] is first[0]

        return [lambda: context.value(odd), lambda: len(made), native, on_the_copy]

    native, _ = _same_in_both_modes(build, DumpOptions(), calls)
    assert native.results[-1][2] is True


def test_placeholder_of_an_interrupted_tuple_under_a_profiler(monkeypatch):
    """A tuple whose capture an error interrupted keeps its placeholder,
    which a later call returns, natively under a profiler too: naming a
    tuple needs no Python code at any code text limit the walker takes."""

    def build():
        return (("x" * 50, [1]),)

    def calls(context, graph):
        def interrupted():
            def fail(text, limit):
                raise _Error("interrupted")

            with monkeypatch.context() as patch:
                patch.setattr(context_module, "_utf8_prefix", fail)
                return context.value(graph)

        def profiled():
            previous = sys.getprofile()
            sys.setprofile(lambda frame, event, arg: None)
            try:
                return context.value(graph)
            finally:
                sys.setprofile(previous)

        def limited():
            context._code_text_limit = 12

        return [limited, interrupted, profiled, lambda: copy.copy(context).value(graph)]

    native, _ = _same_in_both_modes(build, DumpOptions(max_string_bytes=8), calls)
    assert isinstance(native.results[2], TruncatedValue)


@pytest.mark.parametrize(
    "text, stays",
    [("abcdefgh", True), ("abcdefg\u00e9", False), ("abcdefghi", False)],
    ids=["ascii", "wide", "long"],
)
def test_strings_that_fit_need_no_helper(text, stays):
    """A str no longer than the limit, ASCII and so a byte per character,
    is kept natively, under a profiler too; any other goes to the Python
    helper, which the profiler sees from the Python walk."""
    options = DumpOptions(max_string_bytes=8)
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value([text]), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    previous = sys.getprofile()
    sys.setprofile(lambda frame, event, arg: None)
    try:
        captured = context.value([text])
    finally:
        sys.setprofile(previous)
    assert pickle.dumps(captured, protocol=5) == expected
    assert context._walker.handbacks == (0 if stays else 1)


class _Tally:
    """A truncation tally of no number type, with ``__index__``."""

    def __init__(self, count):
        self.count = count

    def __index__(self):
        _called("tally __index__")
        return self.count

    def __add__(self, other):
        _called("tally __add__")
        return _Tally(self.count + other)

    def __eq__(self, other):
        return isinstance(other, _Tally) and other.count == self.count

    __hash__ = None


def test_truncation_tally_with_an_index():
    """A truncation tally that is no int is counted on in Python with its
    ``__add__``; natively its ``__index__`` never runs."""

    def build():
        return (list(range(12)), list(range(20)))

    def calls(context, first, second):
        def tallied():
            context._truncation["max_items"] = _Tally(2)
            return context.value(second)

        return [
            lambda: context.value(first),
            tallied,
            lambda: context._truncation["max_items"].count,
        ]

    native, _ = _same_in_both_modes(build, DumpOptions(max_items=4), calls)
    assert native.hooks == ["tally __add__"]


def _from_instance_dict(cls, state):
    context = object.__new__(cls)
    context.__dict__.update(state)
    return context


class _DictReduced(SnapshotContext):
    """A context class that pickles and copies its instance dict as it is."""

    def __reduce__(self):
        return (_from_instance_dict, (type(self), dict(self.__dict__)))


class _OwnReduceEx(SnapshotContext):
    def __reduce_ex__(self, protocol):
        return (_from_instance_dict, (type(self), dict(self.__dict__)))


class _OwnGetstate(SnapshotContext):
    def __getstate__(self):
        return self.__dict__.copy()


class _OwnCopy(SnapshotContext):
    def __copy__(self):
        return _from_instance_dict(type(self), self.__dict__)


class _OwnDeepcopy(SnapshotContext):
    def __deepcopy__(self, memo):
        return _from_instance_dict(type(self), copy.deepcopy(self.__dict__, memo))


class _ClassGetattr(SnapshotContext):
    """``__getattr__`` answers ``copy.deepcopy``'s instance lookup."""

    def __getattr__(self, name):
        if name != "__deepcopy__":
            raise AttributeError(name)
        return lambda memo: _from_instance_dict(
            type(self), copy.deepcopy(dict(self.__dict__), memo)
        )


class _OwnGetnewargsEx(SnapshotContext):
    def __getnewargs_ex__(self):
        return (), {}


class _OwnGetnewargs(SnapshotContext):
    def __getnewargs__(self):
        return ()


class _CopyMeta(type):
    def __copy__(cls, context):
        return _from_instance_dict(cls, context.__dict__)


class _MetaCopy(SnapshotContext, metaclass=_CopyMeta):
    pass


class _GetattrMeta(type):
    def __getattr__(cls, name):
        if name != "__copy__":
            raise AttributeError(name)
        return lambda context: _from_instance_dict(cls, context.__dict__)


class _MetaGetattr(SnapshotContext, metaclass=_GetattrMeta):
    pass


class _OwnReplace(SnapshotContext):
    """``copy.replace`` looks ``__replace__`` up on the class."""

    def __replace__(self, **changes):
        return _from_instance_dict(type(self), {**self.__dict__, **changes})


class _DeepReplace(SnapshotContext):
    def __replace__(self, **changes):
        state = copy.deepcopy(dict(self.__dict__))
        return _from_instance_dict(type(self), {**state, **changes})


class _ReplaceMeta(type):
    def __replace__(cls, context, **changes):
        return _from_instance_dict(cls, {**context.__dict__, **changes})


class _MetaReplace(SnapshotContext, metaclass=_ReplaceMeta):
    pass


class _Replacer:
    """A class a context names as its ``__class__``, whose ``__replace__``
    ``copy.replace`` then calls."""

    def __replace__(self, **changes):
        return _from_instance_dict(type(self), {**self.__dict__, **changes})


class _ClassProperty(SnapshotContext):
    """``copy.replace`` takes the class from ``obj.__class__``."""

    __class__ = property(lambda self: _Replacer)


# A plain class attribute, as proxy classes set, answers it too.
_ClassAttribute = type("_ClassAttribute", (SnapshotContext,), {"__class__": _Replacer})


def _state_without_hook(context):
    state = dict(vars(context))
    del state["__getstate__"]
    return state


def _instance_getstate(options):
    """A context whose instance dict holds ``__getstate__``, which
    ``object.__reduce_ex__`` looks up on the instance."""
    context = SnapshotContext(options)
    context.__getstate__ = types.MethodType(_state_without_hook, context)
    return context


@pytest.mark.parametrize("how", ["copy", "deepcopy", "pickle"])
@pytest.mark.parametrize(
    "cls",
    [
        _DictReduced,
        _OwnReduceEx,
        _OwnGetstate,
        _OwnCopy,
        _OwnDeepcopy,
        _ClassGetattr,
        _OwnGetnewargsEx,
        _OwnGetnewargs,
        _MetaCopy,
        _MetaGetattr,
        _instance_getstate,
        _OwnReplace,
        _MetaReplace,
    ],
    ids=[
        "__reduce__",
        "__reduce_ex__",
        "__getstate__",
        "__copy__",
        "__deepcopy__",
        "__getattr__",
        "__getnewargs_ex__",
        "__getnewargs__",
        "metaclass __copy__",
        "metaclass __getattr__",
        "instance __getstate__",
        "__replace__",
        "metaclass __replace__",
    ],
)
def test_context_class_with_a_copy_hook_of_its_own(cls, how):
    """A context class with a copy hook of its own, on the class, the
    instance or, for the class lookups of ``copy.copy`` and
    ``copy.replace``, the metaclass, copies as in the Python walk: whatever
    state the hook builds, ``objects`` holds every copy made so far, and a
    copy given the walker itself shares it."""
    _same_through_a_copy_hook(
        cls,
        {
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
            "pickle": lambda context: pickle.loads(pickle.dumps(context)),
        }[how],
    )


@pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace is new in 3.13")
@pytest.mark.parametrize(
    "cls",
    [_OwnReplace, _DeepReplace, _MetaReplace, _ClassProperty, _ClassAttribute],
    ids=[
        "__replace__",
        "deep __replace__",
        "metaclass __replace__",
        "__class__ property",
        "__class__ attribute",
    ],
)
def test_copy_replace_through_a_hook_of_the_context_class(cls):
    """``copy.replace`` copies through the ``__replace__`` of the class
    ``obj.__class__`` names, which a class attribute of that name can
    answer, looked up through the metaclass: the copy sees the whole memo,
    as in the Python walk."""
    _same_through_a_copy_hook(cls, copy.replace)


@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
def test_a_dataclass_context(native):
    """A dataclass context, which gets a ``__replace__`` from Python 3.13,
    captures natively like any other."""

    @dataclass(init=False, eq=False, repr=False)
    class Context(SnapshotContext):
        pass

    with _mode(native):
        context = Context(DumpOptions())
    assert context.value(["first"]) == ["first"]
    assert (vars(context).get("_walker") is not None) is native


def _same_through_a_copy_hook(cls, copied):
    """Captures, copies the context with ``copied`` and captures on both,
    in each mode, and checks that the outcomes match."""
    data = {"a": [1, 2], "b": ("three", [4])}
    outcomes = {}
    for native in (False, True):
        with _mode(native):
            context = cls(DumpOptions(max_nodes=8))
        first = context.value(data)
        twin = copied(context)
        again = twin.value(data)
        later = context.value({"c": data["a"]})
        counts = (twin._nodes, context._nodes, len(twin.objects), len(context.objects))
        state = (first, again, later, counts, dict(twin._truncation))
        outcomes[native] = pickle.dumps(state, protocol=5)
    assert outcomes[True] == outcomes[False]


def test_a_metaclass_dict_key_of_another_kind():
    """A metaclass dict key other than an exact str, whose ``__eq__`` a
    lookup there may run, is never compared: the walker reads no metaclass
    dict."""
    log = []

    class Key:
        def __hash__(self):
            return hash("__copy__")

        def __eq__(self, other):
            log.append("__eq__")
            return other is self

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Meta = type("Meta", (type,), {Key(): None})
    Context = Meta("Context", (SnapshotContext,), {})
    with _mode(True):
        context = Context(DumpOptions())
    assert context.value(["first"]) == ["first"]
    assert context._walker is not None
    assert log == []


# --- Shared dicts and counting helpers ---------------------------------------


class _CollidingReason:
    """A truncation tally key hashing like a reason, with an ``__eq__``."""

    def __init__(self, reason):
        self.reason = reason

    def __hash__(self):
        return hash(self.reason)

    def __eq__(self, other):
        _called(f"{self.reason} key __eq__")
        return other is self


@pytest.mark.parametrize(
    ("reason", "value"),
    [
        ("max_string_bytes", "\u00e9" * 3),
        ("max_string_bytes", "x" * 40),
        ("max_string_bytes", b""),
        ("max_string_bytes", b"y" * 40),
        ("code_metadata", type("\u00e9" * 1100, (), {})),
        ("max_items", [[1, 2, 3], 2, 3]),
    ],
    ids=["text fits", "long text", "empty bytes", "long bytes", "code text", "items"],
)
def test_a_truncation_key_of_another_kind(reason, value):
    """A ``_truncation`` key not of a plain kind, whose ``__eq__`` looking a
    reason up may run, sends the context to the Python walk: the walker
    reads a helper's tally before the helper runs, where the Python walk
    looks it up only to count."""

    def build():
        return (_CollidingReason(reason),)

    def calls(context, key):
        def keyed():
            context._truncation[key] = 0

        made = value() if isinstance(value, type) else value
        return [lambda: context.value(["first"]), keyed, lambda: context.value(made)]

    options = DumpOptions(max_string_bytes=8, max_items=2)
    _same_in_both_modes(build, options, calls)


@pytest.mark.parametrize("which", ["_sources", "objects"])
def test_one_dict_for_objects_and_sources(which):
    """A context whose ``objects`` and ``_sources`` are one dict captures in
    Python, where each original then stands in for its own copy."""

    def build():
        shared = ["shared", [1]]
        return (shared, [shared, shared])

    def calls(context, shared, pair):
        def aliased():
            if which == "_sources":
                context._sources = context.objects
            else:
                context.objects = context._sources

        return [
            lambda: context.value(["first"]),
            aliased,
            lambda: context.value(pair),
            lambda: context.value(shared),
        ]

    _same_in_both_modes(build, DumpOptions(), calls)


class _AddingName(str):
    """A class name whose concatenation is Python code."""

    def __radd__(self, other):
        return str.__add__(other, str(self))


def test_replacement_a_hand_back_undid_is_forgotten():
    """A call that replaces a written depth placeholder, then goes to the
    Python walk, leaves no replacement for the walker to write later; the
    Python walk makes its own, after the copies made before it, and its
    original, which Python code then removes, stays removed."""

    def build():
        Odd = type("Odd", (), {})
        Odd.__qualname__ = _AddingName("Odd")
        original = _Plain()
        original.x = [1]
        original.odd = Odd()
        attributes = vars(original)
        # Kept alive, so no id is reused once ``_sources`` is replaced.
        return (original, attributes, [[attributes]], ["later"])

    def calls(context, original, attributes, nested, later):
        names = {id(original): "original", id(attributes): "attributes"}

        def replaced():
            context._sources = {}

        def kept():
            return [names.get(key, "other") for key in copy.copy(context)._sources]

        return [
            lambda: context.value(nested),
            replaced,
            lambda: context.value(later),
            lambda: context.value(original),
            kept,
            lambda: context._sources.pop(id(attributes), None) is attributes,
            kept,
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=2), calls)


class _CountedTally(int):
    """A truncation tally that is an int, with an ``__add__`` of its own."""

    def __add__(self, other):
        _called("counted __add__")
        return _CountedTally(int(self) + other)


@pytest.mark.parametrize("helper", ["text", "bytes", "code"])
def test_tally_a_helper_adds_to_once(helper):
    """A tally that one of pydumpling's Python helpers adds to with its own
    ``__add__`` runs it once, in the Python walk, when the call goes there
    after the helper would have run: the walker hands it back first."""
    reason = "code_metadata" if helper == "code" else "max_string_bytes"

    def build():
        long = {
            "text": "s" * 40,
            "bytes": b"b" * 40,
            "code": type("L" * 5000, (), {})(),
        }[helper]
        return ([long, [1, 2, 3, 4]],)

    def calls(context, data):
        def tallied():
            context._truncation[reason] = _CountedTally(0)
            context._truncation["max_items"] = _Tally(0)
            return context.value(data)

        return [tallied]

    options = DumpOptions(max_string_bytes=8, max_items=3)
    native, _ = _same_in_both_modes(build, options, calls)
    assert native.hooks == ["counted __add__", "tally __add__"]


def test_a_hand_back_starts_from_the_dicts_as_the_call_found_them():
    """A call the walker hands back partway first takes out of ``objects``
    and ``_sources`` what it wrote there. Application code the Python walk
    then runs, here a tally's ``__add__``, finds the copies of earlier
    calls and, of this one, those the Python walk has made by then: had the
    walker's stayed, the Python walk would return the unfinished copy it
    found and never count."""

    def build():
        return (["earlier"], [["first"], ["second"], ["third"]])

    def calls(context, earlier, data):
        seen = []

        class Tally(int):
            def __add__(self, other):
                seen.append(_contents(context))
                return int(self) + other

        def counted():
            context._truncation["max_items"] = Tally(0)
            return context.value(data)

        return [
            lambda: context.value(earlier),
            counted,
            lambda: seen,
            lambda: _contents(context),
        ]

    native, _ = _same_in_both_modes(build, DumpOptions(max_items=2), calls)
    assert len(native.results[2]) == 1


@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
def test_a_context_a_class_its_walker_met_refers_to_is_collected(native):
    """Between calls the walker keeps what it learned about the classes its
    last call met, and shows the cyclic collector what that holds: a context
    such a class refers back to is freed by one collection, as in the
    Python walk."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    _check_walker(context, native)

    class Holder:
        pass

    Holder.context = context
    context.value(Holder())
    alive = weakref.ref(context)
    del context, Holder
    gc.collect()
    assert alive() is None
