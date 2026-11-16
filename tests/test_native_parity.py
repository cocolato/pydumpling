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
from deep in the call stack, 120 to 400 frames from the recursion limit, where
the Python walk still has room, so both walks capture the same.  Where the
Python walk reaches the limit, as through the longest chains, the native walk
may capture more, never less, by the provisional definition of
``_captures_less``: no root variable both walks captured reaches more
``'recursion'`` placeholders natively, and ``objects`` keeps no fewer
originals.

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
import copyreg
import cProfile
import functools
import gc
import gzip
import inspect
import json
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
    FakeDict,
    FakeFloat,
    FakeFrame,
    FakeList,
    FakeStr,
    FakeTraceback,
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

    Off, each context built meanwhile gets ``_native_walk = False`` of its
    own as its ``__init__`` ends, and captures in Python for good: contexts
    created inside the walk, such as an exception's metadata context, too.
    """
    if native:
        yield
        return
    init = vars(SnapshotContext)["__init__"]

    def pure(self, *args, **kwargs):
        init(self, *args, **kwargs)
        # Past the class's own attribute hooks, which the walk alone runs.
        if object.__getattribute__(self, "_native_walk"):
            object.__setattr__(self, "_native_walk", False)

    SnapshotContext.__init__ = pure
    try:
        yield
    finally:
        SnapshotContext.__init__ = init


def _check_walker(context, native):
    """The context must capture with the walk its mode asks for."""
    assert context._native_walk is native, f"native={native}"


def _fallbacks():
    """How many calls the native walker has sent to the Python walk."""
    return values_module._STATICS.fallbacks


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
    left: object  # Frames left to push where the ops run, or None.
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
    left = None
    if seed % 10 == 3 and not deep:
        # Far enough from the limit that the Python walk never overflows.
        left = rng.randint(120, 400)
    return _Case(seed, options, ops, left, [stack])


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


@dataclass(repr=False)  # A captured graph can take exponential time to repr.
class _Outcome:
    results: list
    truncation: list
    nodes: int
    hooks: list
    memo: object = None
    blob: bytes = b""
    fallbacks: object = None  # The walker's, compared apart from the blob.

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


def _beneath(left, run):
    """Call ``run`` with ``left`` frames left to push, or directly."""
    if left is None:
        return run()
    return _descend(_frames_left() - left - 2, run)


def _run(case, native):
    """Replay the case's operations on a fresh context in the given mode."""
    with _mode(native):
        context = SnapshotContext(case.options)
        _check_walker(context, native)
        before = _fallbacks()
        with _armed() as calls:
            results = _beneath(
                case.left, lambda: [_apply(context, o) for o in case.ops]
            )
        fallbacks = _fallbacks() - before
        hooks = list(calls)
    return _Outcome(
        results,
        list(context._truncation.items()),
        context._nodes,
        hooks,
        _memo(context),
        fallbacks=fallbacks if native else None,
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
    # The test running, as pytest names it: its node id, then the phase.
    test = os.environ.get("PYTEST_CURRENT_TEST", __file__).rsplit(" ", 1)[0]
    command = f"pytest '{test}'"
    if "::test_walk_parity[" in test:
        command = f"PYDUMPLING_PARITY_SEEDS={case.seed + 1} {command}"
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed is None:
        reproduce = (
            f"reproduce: {command} (PYTHONHASHSEED was unset, so the hash seed "
            "was random: set it to fix one)"
        )
    else:
        reproduce = f"reproduce: PYTHONHASHSEED={hash_seed} {command}"
    raise AssertionError(
        f"native and pure walks differ for seed {case.seed}\n"
        f"options: {case.options}\n"
        f"{pickles}\n"
        f"application hooks ran: native {native.hooks}, pure {pure.hooks}\n"
        f"{reproduce}"
    )


def _overflows(outcome):
    """How often a walk reached the recursion limit: the ``RecursionError``s
    its operations raised, and the ``'recursion'`` placeholders it left."""
    raised = sum(
        type(result) is tuple
        and len(result) == 3
        and [type(part) for part in result] == [str, str, str]
        and result[:2] == ("raised", "RecursionError")
        for result in outcome.results
    )
    return raised, dict(outcome.truncation).get("recursion", 0)


def _recursion_placeholders(root):
    """How many ``TruncatedValue('recursion')`` placeholders a captured value
    reaches, each counted once."""
    count, seen, pending = 0, set(), [root]
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if type(item) is TruncatedValue:
            count += vars(item).get("reason") == "recursion"
            continue
        if isinstance(item, dict):
            pending += dict.keys(item)
            pending += dict.values(item)
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending += item
        try:
            state = object.__getattribute__(item, "__dict__")
        except AttributeError:
            continue
        if type(state) is dict:
            pending += state.values()
    return count


def _frame_copies(captured):
    """The frame copies a captured traceback or frame holds, in chain order."""
    frames, tracebacks = [], [captured]
    while tracebacks:
        entry = tracebacks.pop()
        if isinstance(entry, FakeTraceback):
            tracebacks.append(vars(entry).get("tb_next"))
            frame = vars(entry).get("tb_frame")
            while isinstance(frame, FakeFrame) and frame not in frames:
                frames.append(frame)
                frame = vars(frame).get("f_back")
    return frames


def _root_variables(ops, results):
    """Each root variable of a replay, with the ``'recursion'`` placeholders
    it reaches: the entries of what a ``_mapping``, an ``_attribute_mapping``
    or a ``value`` of a mapping at depth 0 returned, of the state of the
    exception an exception capture holds, and of the namespaces of the
    frames a traceback or frame capture holds, by operation, mapping and
    position."""
    found = {}
    for index, (op, result) in enumerate(zip(ops, results)):
        roots = {}
        if op[0] in ("mapping", "attributes", "value") and op[2] == 0:
            roots["mapping"] = result
        elif op[0] == "exception" and isinstance(result, dict):
            saved = result.get("exc_value")
            if isinstance(saved, BaseException):
                roots["exception"] = vars(saved)
        elif op[0] in ("traceback", "frame"):
            for position, frame in enumerate(_frame_copies(result)):
                roots[f"frame {position} locals"] = vars(frame).get("f_locals")
                roots[f"frame {position} globals"] = vars(frame).get("f_globals")
        seen = set()
        for name, root in roots.items():
            # Frames of one module share their globals' copy.
            if isinstance(root, dict) and id(root) not in seen:
                seen.add(id(root))
                for position, item in enumerate(dict.values(root)):
                    found[index, name, position] = _recursion_placeholders(item)
    return found


def _captures_less(ops, native, pure):
    """How the native outcome captures less than the pure one, where the
    pure walk reached the recursion limit, by a definition the owner has
    yet to approve: (i) a root variable both walks captured reaches more
    ``'recursion'`` placeholders natively, (ii) the native walk keeps fewer
    originals in ``objects``. Not fewer nodes: the pure walk's ``_nodes``
    also counts the copies it dropped where it recovered an overflow at a
    root variable. A variable only the pure walk captured is no shortfall:
    under a node budget, capturing more of an earlier one can leave a later
    one, or a later mapping, with fewer entries."""
    mine = _root_variables(ops, native.results)
    theirs = _root_variables(ops, pure.results)
    shortfalls = [
        f"root variable {key}: {mine[key]} natively, {count} in Python"
        for key, count in theirs.items()
        if mine.get(key, count) > count
    ]
    kept = [len(outcome.memo[0] or ()) for outcome in (native, pure)]
    if kept[0] < kept[1]:
        shortfalls.append(
            f"kept in objects: {kept[0]} natively, {kept[1]} in Python"
            f" (nodes: {native.nodes} natively, {pure.nodes} in Python)"
        )
    return shortfalls


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
        # From deep in the stack, the walker sends no more calls to Python
        # than from where the test runs.
        if case.left is not None:
            shallow = _run(replace(case, left=None), native=True)
    finally:
        case.close()
    reached = _overflows(pure)
    if any(reached):
        # Near the recursion limit the native walk may capture more, never
        # less.
        assert all(n <= p for n, p in zip(_overflows(native), reached)), case.seed
        assert not _captures_less(case.ops, native, pure), case.seed
    else:
        _assert_parity(case, native, pure)
    if case.left is not None:
        assert native.fallbacks == shallow.fallbacks, case.seed


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
# and nodes run in the Python walk in place.


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
    same = native.blob == pure.blob
    assert same, reprlib.repr((native.results, pure.results))
    return native, pure


def _contents(context):
    """What ``objects`` and ``_sources`` hold, in their order: the walker
    writes each entry where and when the Python walk writes it."""
    return (
        list(map(repr, context.objects.values())),
        list(map(repr, context._sources.values())),
    )


def test_class_level_and_bound_calls():
    """``SnapshotContext.value(ctx, ...)`` captures like ``ctx.value(...)``,
    and the bound method is the Python function's in both modes."""
    function = vars(values_module.ValuesMixin)["value"]

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
    assert SnapshotContext.value is function
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
            bound = context.value
            assert bound.__self__ is context
            assert bound.__func__ is function
            assert bound == context.value and hash(bound) == hash(context.value)
            assert bound != SnapshotContext(DumpOptions()).value
            assert bound.__name__ == "value"
            assert bound.__doc__ == function.__doc__
            assert bound.__qualname__ == "ValuesMixin.value"
            assert str(inspect.signature(bound)) == "(value, depth=0)"
            assert inspect.getsource(bound).lstrip().startswith("def value(")
            assert bound.__reduce__() == (getattr, (context, "value"))


class _Redacting(SnapshotContext):
    """A context that overrides a helper the native walker carries out."""

    def _text(self, value):
        return "<redacted>"


class _Composed(SnapshotContext):
    """A context subclass that changes nothing in the walk."""


def test_overriding_a_walk_method_keeps_the_python_walk():
    """A subclass overriding a walk method captures in Python, which calls it:
    the walk is chosen when the class is created."""
    assert _Redacting._native_walk is False
    assert _Composed._native_walk is True
    assert "_native_walk" not in vars(_Composed)
    before = _fallbacks()
    with _mode(True):
        redacting = _Redacting(DumpOptions())
        assert redacting.value({"key": ["secret"]}) == {"<redacted>": ["<redacted>"]}
        composed = _Composed(DumpOptions())
        composed.value([1])
    assert _fallbacks() == before


def test_a_class_past_a_non_cooperative_init_subclass():
    """A context class whose MRO puts a base whose ``__init_subclass__`` does
    not call ``super()`` ahead of ``ValuesMixin`` is judged at its first
    instance: one that overrides a walk method captures in Python, which
    calls it, and one that changes nothing captures natively."""
    registered = []

    class Registering:
        def __init_subclass__(cls, **kwargs):
            registered.append(cls)

    class Redacting(Registering, SnapshotContext):
        def _text(self, value):
            return "<redacted>"

    class Composed(Registering, SnapshotContext):
        pass

    assert registered == [Redacting, Composed]
    assert Redacting._native_walk_class is SnapshotContext
    before = _fallbacks()
    with _mode(True):
        redacting = Redacting(DumpOptions())
        composed = Composed(DumpOptions())
    assert (Redacting._native_walk, Composed._native_walk) == (False, True)
    assert redacting.value({"key": ["secret"]}) == {"<redacted>": ["<redacted>"]}
    composed.value([1])
    assert composed._native_walk
    assert _fallbacks() == before


def _child_environment():
    """The environment of a child interpreter that imports this checkout."""
    package = os.path.dirname(os.path.dirname(os.path.abspath(values_module.__file__)))
    root = os.path.dirname(package)
    path = os.environ.get("PYTHONPATH")
    return dict(os.environ, PYTHONPATH=root if not path else root + os.pathsep + path)


_FRAMES_SCRIPT = """
import json, sys
from pydumpling.fake_types import FakeClass
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext, values

depths = []


class Name(str):
    def __radd__(self, other):  # runs as ``_type_name`` names a Probe
        frame, depth = sys._getframe(), 0
        while frame is not None:
            frame, depth = frame.f_back, depth + 1
        depths.append(depth)
        return str.__add__(other, self)


Probe = type("Probe", (), {})
Probe.__qualname__ = Name("Probe")


class Super(SnapshotContext):
    def value(self, value, depth=0):
        return super().value(value, depth)


class Pure(SnapshotContext):
    _native_walk = False


class Extra:
    pass


def capture(context, rebase=False):
    depths.clear()
    fallbacks = values._STATICS.fallbacks if values._STATICS else 0
    bases = FakeClass.__bases__
    if rebase:
        # Objects are then copied in the Python walk, in place.
        FakeClass.__bases__ = bases + (Extra,)
    try:
        context.value([[Probe()], {"key": (Probe(),)}])
    finally:
        FakeClass.__bases__ = bases
    used = values._STATICS.fallbacks - fallbacks if values._STATICS else 0
    return depths[:], used


instance = SnapshotContext(DumpOptions())
instance._native_walk = False
contexts = {
    "native": (SnapshotContext(DumpOptions()), False),
    "in python in place": (SnapshotContext(DumpOptions()), True),
    "pure instance": (instance, False),
    "pure class": (Pure(DumpOptions()), False),
    "super().value": (Super(DumpOptions()), False),
}
print(json.dumps({name: capture(*how) for name, how in contexts.items()}))
"""


def test_no_walk_adds_a_python_frame():
    """Application code the walk runs sees no more Python frames beneath it
    on any path than in the Python walk alone (``PYDUMPLING_NATIVE=0``):
    the native walk runs it with none of the walk's frames between, a node
    it sends to the Python walk in place starts there with one, and a
    context pure by its instance flag, by its class, or by overriding
    ``value`` and calling ``super().value`` adds no frame either."""
    found = {}
    for native in ("1", "0"):
        environment = _child_environment()
        environment["PYDUMPLING_NATIVE"] = native
        process = subprocess.run(
            [sys.executable, "-c", _FRAMES_SCRIPT],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        assert process.returncode == 0, process.stderr
        found[native] = json.loads(process.stdout)
    pure = {name: depths for name, (depths, _) in found["0"].items()}
    native = {name: depths for name, (depths, _) in found["1"].items()}
    assert len({tuple(depths) for depths in pure.values()} - {()}) == 2
    assert all(len(depths) == 2 for depths in pure.values())
    for name, depths in native.items():
        assert len(depths) == 2, name
        assert all(mine <= theirs for mine, theirs in zip(depths, pure[name])), name
    assert native["native"] < pure["native"]
    assert native["native"] < native["in python in place"] < pure["native"]
    for name in ("pure instance", "pure class", "super().value"):
        assert native[name] == pure[name]
    # Only the stock context reached the walker, which sent each object to
    # the Python walk while ``FakeClass`` had other bases.
    assert {name: sent for name, (_, sent) in found["1"].items()} == {
        "native": 0,
        "in python in place": 2,
        "pure instance": 0,
        "pure class": 0,
        "super().value": 0,
    }


_THREAD_SCRIPT = """
import json, pickle, sys, threading
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext, values

# Built, pickled and freed on the main thread, which has the stack for that.
nodes, outcomes, contexts = {}, {}, []
for levels in (600, 3000):
    node = None
    for _ in range(levels):
        node = [node]
    nodes[levels] = node

def work(levels, native):
    context = SnapshotContext(DumpOptions(max_depth=None, max_nodes=None))
    contexts.append(context)
    if not native:
        context._native_walk = False
    before = values._STATICS.fallbacks
    captured = context.value({"deep": nodes[levels]})
    outcomes[levels, native] = (
        (captured, dict(context._truncation), context._nodes),
        context._truncation["recursion"],
        values._STATICS.fallbacks - before,
    )

threading.stack_size(int(sys.argv[1]) * 1024)
for levels in (600, 3000):
    for native in (True, False):
        thread = threading.Thread(target=work, args=(levels, native))
        thread.start()
        thread.join()
threading.stack_size(0)
same = {
    levels: pickle.dumps(outcomes[levels, True][0], protocol=5)
    == pickle.dumps(outcomes[levels, False][0], protocol=5)
    for levels in (600, 3000)
}
print(json.dumps({
    levels: {
        "same": same[levels],
        "recursion": [outcomes[levels, native][1] for native in (True, False)],
        "fallbacks": outcomes[levels, True][2],
    }
    for levels in (600, 3000)
}))
"""


@pytest.mark.parametrize("stack_kib", [64, 160, 512])
def test_small_thread_stack(stack_kib):
    """A walk on a thread with a small native stack does not crash. With
    less than 192 KiB of it left, each call runs in the Python walk from
    its start. With more, the walk sends the node at its stack budget, and
    everything under it, to the Python walk in place, once. Either way a
    600-level list, where the Python walk has room, is captured as there,
    and a 3000-level one, where it overflows, leaves no more ``'recursion'``
    placeholders."""
    process = subprocess.run(
        [sys.executable, "-c", _THREAD_SCRIPT, str(stack_kib)],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=120,
    )
    assert process.returncode == 0, process.stderr
    found = json.loads(process.stdout)
    assert found["600"]["same"]
    native, pure = found["3000"]["recursion"]
    assert native <= pure == 1
    if stack_kib == 512:
        assert found["600"]["fallbacks"] == 1
    else:
        assert found["600"]["fallbacks"] > 600


class _Nest:
    """Each level is an ``__init__`` call, which Python 3.13+ runs beneath a
    shim frame that counts toward the recursion limit but is not visible."""

    def __init__(self, levels, run):
        self.result = run() if levels <= 0 else _Nest(levels - 1, run).result


def _frames_left():
    """How many frames the caller's thread can still push, counted by pushing
    them: one more than this function's deepest recursion that fits."""

    def push(count):
        if count > 1:
            push(count - 1)

    low, high = 0, sys.getrecursionlimit() + 1
    while high - low > 1:
        middle = (low + high) // 2
        try:
            push(middle)
        except RecursionError:
            high = middle
        else:
            low = middle
    return low + 1


class _Linked:
    """An object whose instance dict holds the next link of a chain."""

    def __init__(self, link):
        self.link = link


_LINKS = {
    "list": lambda inner: [inner],
    "dict": lambda inner: {"k": inner},
    "tuple": lambda inner: (inner,),
    "object": _Linked,
    "mapping": lambda inner: [inner],
}


@pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "max_depth"])
@pytest.mark.parametrize("hidden", [False, True], ids=["frames", "init_frames"])
@pytest.mark.parametrize("shape", list(_LINKS))
def test_near_the_limit_the_walker_captures_at_least_as_much(shape, hidden, bounded):
    """Near the recursion limit the Python walk overflows: it raises, or
    ``_root_value`` leaves a ``'recursion'`` placeholder. The walker pushes
    no Python frames, and bounds the long string at the end of the chain
    itself, so it never raises where the Python walk returns, never leaves
    more placeholders, and from 20 frames below the limit captures what the
    Python walk captures far from it, with no node in the Python walk. A walk
    bounded by the default ``max_depth`` has room in Python from 30 frames
    below the limit, and there both walks capture the same. So beneath
    plain frames and beneath ``__init__`` calls, whose shim frames 3.13+
    counts but hides, for each of the walk's recursive methods."""
    nested = "x" * 5000
    for _ in range(30):
        nested = _LINKS[shape](nested)
    options = DumpOptions() if bounded else DumpOptions(max_depth=None)

    def outcome(left, native):
        with _mode(native):
            context = SnapshotContext(options)
        before = _fallbacks()
        if shape == "mapping":
            capture = functools.partial(context._mapping, {"root": nested})
        else:
            capture = functools.partial(context.value, nested)

        def place():
            return _descend(_frames_left() - left - 1, capture)

        try:
            captured = _Nest(120, place).result if hidden else place()
        except RecursionError:
            return None, None, context._nodes, len(context.objects)
        assert _fallbacks() == before
        state = (captured, dict(context._truncation), context._nodes)
        blob = pickle.dumps(state, protocol=5)
        # Only the root mapping's one variable can hold a 'recursion'
        # placeholder.
        recursion = context._truncation["recursion"]
        return blob, recursion, context._nodes, len(context.objects)

    roomy = outcome(400, False)
    assert outcome(400, True) == roomy
    overflowed = []
    for left in range(2, 260, 3):
        native, pure = outcome(left, True), outcome(left, False)
        assert native[0] is not None or pure[0] is None, left
        if native[0] is not None and pure[0] is not None:
            assert native[1] <= pure[1], left
        # One chain: no copy the pure walk dropped outnumbers the native
        # walk's whole capture, so its nodes too.
        assert native[2] >= pure[2], left
        assert native[3] >= pure[3], left
        if left >= 20:
            assert native == roomy, left
        if pure != roomy:
            overflowed.append(left)
    if bounded:
        assert max(overflowed) < 30, overflowed
    else:
        assert max(overflowed) >= 20, "the Python walk never overflowed"


class _OverflowingName(str):
    """A ``__qualname__`` whose ``+``, application code ``_type_name`` runs,
    raises ``RecursionError`` while ``armed``."""

    armed = False

    def __radd__(self, other):
        _called("overflowing __radd__")
        if _OverflowingName.armed:
            raise RecursionError("maximum recursion depth exceeded")
        return str.__add__(other, str(self))


def test_a_recursion_error_in_application_code_is_recovered_at_the_root_variable():
    """A ``RecursionError`` raised by application code the walk runs while it
    captures a root mapping's variable is recovered there, natively, as
    ``_root_value`` recovers it: the copies made since the variable began
    are forgotten, with their nodes and counters kept, and the variable
    becomes a ``'recursion'`` placeholder. The other variables, and a later
    capture of a forgotten object, are as in the Python walk."""

    def build():
        Deep = type("Deep", (), {})
        Deep.__qualname__ = _OverflowingName("Deep")
        shared = [1, "kept"]
        inner = [2.5, {"k": "x" * 50}, Deep()]
        return {"before": shared, "deep": [shared, inner], "after": [3]}, inner

    def calls(context, namespace, inner):
        def overflowing():
            _OverflowingName.armed = True
            try:
                return context._mapping(namespace)
            finally:
                _OverflowingName.armed = False

        return [overflowing, lambda: context.value(inner)]

    before = _fallbacks()
    native, _ = _same_in_both_modes(build, DumpOptions(max_string_bytes=8), calls)
    root, again = native.results
    assert isinstance(root["deep"], TruncatedValue)
    assert root["deep"].reason == "recursion"
    assert again[:2] == [2.5, {"k": "x" * 8}]
    assert native.hooks == ["overflowing __radd__"] * 2
    assert _fallbacks() == before


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


class _Name(str):
    """A keyword name whose hash and comparison are application code."""

    def __hash__(self):
        _called("_Name.__hash__")
        return str.__hash__(self)

    def __eq__(self, other):
        _called("_Name.__eq__")
        return str.__eq__(self, other)


@pytest.mark.parametrize(
    "form, compared",
    [("bound", 3), ("class", 3), ("mapping", 5), ("receiver", 1)],
)
def test_str_subclass_keyword_names(form, compared):
    """A keyword name that is a str subclass is matched to its parameter by
    CPython alone, in both walks: its ``__eq__`` runs once per parameter it
    is compared with, and its ``__hash__`` never, as the walker sees only the
    arguments."""
    outcomes = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        data = [1, "two"]
        # Built before any hook is collected: building the dict hashes a name.
        if form == "bound":
            names = {_Name("depth"): 1}
            call = functools.partial(context.value, data, **names)
        elif form == "class":
            names = {_Name("depth"): 1}
            call = functools.partial(SnapshotContext.value, context, data, **names)
        elif form == "mapping":
            names = {_Name("counted"): True}
            call = functools.partial(context._mapping, {"key": data}, **names)
        else:
            names = {_Name("self"): context, "value": data}
            call = functools.partial(SnapshotContext.value, **names)
        before = _fallbacks()
        with _armed() as hooks:
            result = call()
        assert _fallbacks() == before
        outcomes[native] = (pickle.dumps(result, protocol=5), list(hooks))
    assert outcomes[True] == outcomes[False]
    assert outcomes[True][1] == ["_Name.__eq__"] * compared


@pytest.mark.parametrize(
    "depth, in_python",
    [
        (-(2**63), True),
        (-(2**63) + 1, True),
        (-(2**62) - 1, True),
        (-(2**62), False),
        (2**62, False),
        (2**62 + 1, True),
    ],
)
def test_a_depth_beyond_two_to_the_62_runs_in_python(depth, in_python):
    """A depth of more than 2**62 in magnitude, which adding to could
    overflow, gives the call and the context to the Python walk before the
    call starts, down to -2**63, whose magnitude no i64 holds; the walker
    takes one within that bound."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=2))
        _check_walker(context, native)
        before = _fallbacks()
        found[native] = pickle.dumps(context.value([[[1]]], depth), 4)
    assert found[True] == found[False]
    assert _fallbacks() - before == (1 if in_python else 0)
    # A depth pydumpling's callers never pass gives the context to Python.
    assert context._native_walk is not in_python


def test_copy_of_a_capturing_context():
    """A shallow copy of a context, which shares its ``objects``, and a deep
    copy, which has its own, capture as the context does."""

    def build():
        shared = [1, [2], {"three": 3}]
        return (shared, {"again": shared})

    def calls(context, shared, holder):
        def copied():
            twin = copy.copy(context)
            deep = copy.deepcopy(context)
            assert twin._native_walk is deep._native_walk is context._native_walk
            return (twin.value(holder), twin._nodes, deep.value([shared]), deep._nodes)

        return [lambda: context.value(shared), copied]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_frame_namespaces_stay_native():
    """A frame's own locals are read natively."""
    with _mode(True):
        context = SnapshotContext(DumpOptions())
        local = {"nested": [1, 2, 3]}  # noqa: F841
        before = _fallbacks()
        context.from_frame(sys._getframe())
        assert _fallbacks() == before


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
assert new._native_walk and old._native_walk
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


def test_errors_and_application_code_inside_native_calls():
    """An error the native walk raises keeps what it did; an odd memo
    entry's reason and a key the private filter hashes run their own code
    natively, as in the Python walk."""

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
            before = _fallbacks()
            for call in calls(context, *build()):
                call()
            assert _fallbacks() == before


def test_type_names_read_natively():
    """Odd but plain class names and modules are read natively."""

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


def test_class_level_calls_on_odd_receivers():
    """A class-level call on an object that is no context, or with odd
    arguments, fails or captures as the function would."""
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
    # ``DumpOptions`` refuses 0 for every budget; forged, it retires the
    # context.
    "max_depth 0": _mutated_options(max_depth=0),
    "max_items 0": _mutated_options(max_items=0),
    "max_nodes 0": _mutated_options(max_nodes=0),
    "max_string_bytes 0": _mutated_options(max_string_bytes=0),
    "max_nodes maxsize": _mutated_options(max_nodes=sys.maxsize),
    "max_nodes beyond 64 bits": _mutated_options(max_nodes=2**70),
    "max_items maxsize": _mutated_options(max_items=sys.maxsize),
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


def test_an_unlimited_context_never_enters_the_walker():
    """An unlimited context runs the Python walk in both modes: ``value``
    reads ``_unlimited`` before the flag, and ``_mapping`` offers a call only
    while ``_unlimited is False``. The walker sends nothing to Python for
    it and writes no flag on it, even where the walk reaches ``_mapping``
    through an object no serializer writes."""
    contexts = []

    def build():
        holder = _Plain()
        holder.items = [1, "two", {"three": 3.0}]
        holder.pending = (step for step in range(3))  # pickled by neither
        return ({"holder": holder, "plain": [4, 5]},)

    def calls(context, data):
        contexts.append(context)
        return [
            lambda: context.value(data),
            lambda: context._mapping({"k": data, "more": [6]}, 0),
            lambda: context._attribute_mapping(vars(data["holder"]), 1),
        ]

    before = _fallbacks()
    native, _ = _same_in_both_modes(build, DumpOptions.unlimited(), calls)
    assert dict(native.truncation)["unserializable"] > 0
    assert _fallbacks() == before
    assert "_native_walk" not in vars(contexts[1])


class _PlainOptions(DumpOptions):
    """An options subclass that reads its budgets as ``DumpOptions`` does."""


class _LoggedOptions(DumpOptions):
    """An options subclass whose budget reads are application code."""

    def __getattribute__(self, name):
        if name.startswith("max_"):
            _called(f"options {name}")
        return object.__getattribute__(self, name)


def test_options_subclasses():
    """An options subclass that leaves its budgets to ``DumpOptions`` is read
    natively; one that reads them with its own code captures in Python, which
    runs that code as often as it reads a budget."""

    def build():
        return ({"key": ["v" * 30, (1, 2)], "more": list(range(9))},)

    def calls(context, data):
        return [lambda: context.value(data), lambda: context.value([data, 1])]

    before = _fallbacks()
    _same_in_both_modes(build, _PlainOptions(max_string_bytes=8), calls)
    assert _fallbacks() == before
    native, _ = _same_in_both_modes(build, _LoggedOptions(max_string_bytes=8), calls)
    assert "options max_items" in native.hooks
    assert _fallbacks() == before + 1


def test_method_objects_pickle_copy_and_weakref():
    """The class attribute pickles and copies as itself, and the bound method
    is the Python walk's own, in both modes."""
    function = vars(values_module.ValuesMixin)["value"]
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

    def calls(context, cls, odd, fake, copies):
        return [
            lambda: context.value([cls]),
            lambda: context.value([cls, odd]),
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
                before = _fallbacks()
                captured = _descend(5000, lambda: context.value(nested))
                assert _fallbacks() == before
                outcomes[native] = pickle.dumps(
                    (captured, dict(context._truncation), context._nodes), protocol=5
                )
        assert outcomes[True] == outcomes[False]
    finally:
        sys.setrecursionlimit(saved)


_ABOVE_THE_LIMIT_SCRIPT = """
import _thread, json, os, sys
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot.values import _STATICS

def deep(frames, run):
    return run() if frames <= 0 else deep(frames - 1, run)

outcomes = {}
for native in (True, False):
    context = SnapshotContext(DumpOptions())
    if not native:
        context._native_walk = False
    before = _STATICS.fallbacks
    parked, go, done = (_thread.allocate_lock() for _ in range(3))
    for lock in (parked, go, done):
        lock.acquire()
    outcome = []

    def park():
        parked.release()
        go.acquire()
        # 500 frames deep, beneath a limit of 100 another thread set.
        try:
            outcome.append(repr(context.value([1, {"a": (2,)}])))
        except RecursionError:
            outcome.append("RecursionError")
        outcome.append(context._native_walk and _STATICS.fallbacks - before)

    def thread():
        try:
            deep(500, park)
        finally:
            done.release()

    _thread.start_new_thread(thread, ())
    parked.acquire()
    sys.setrecursionlimit(100)
    go.release()
    done.acquire()
    sys.setrecursionlimit(1000)
    outcomes["native" if native else "python"] = outcome
print(json.dumps(outcomes))
"""


def test_a_caller_deeper_than_a_lowered_limit():
    """Another thread can lower the recursion limit beneath a thread already
    deeper than it. There the Python walk overflows at its first frame, and
    so does a native capture, whose entry is that frame: the walker is not
    reached."""
    process = subprocess.run(
        [sys.executable, "-c", _ABOVE_THE_LIMIT_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == {
        "native": ["RecursionError", 0],
        "python": ["RecursionError", False],
    }


_DEEP_STACK_SCRIPT = """
import json, os, pickle, sys
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot.values import _STATICS

def deep(level, run):
    items, name = [level, str(level)], f"level-{level}"
    return run() if level == 0 else deep(level - 1, run)

def capture():
    frame = sys._getframe(1)
    outcome = {}
    for native in (True, False):
        context = SnapshotContext(DumpOptions(max_frames=None))
        if not native:
            context._native_walk = False
        before = _STATICS.fallbacks
        snapshot = context.from_frame(frame)
        state = (snapshot, dict(context._truncation), context._nodes)
        outcome[native] = pickle.dumps(state, protocol=5), _STATICS.fallbacks - before
    return outcome

outcome = deep(700, capture)
print(json.dumps([outcome[True][0] == outcome[False][0], outcome[True][1]]))
"""


def test_a_whole_stack_from_700_frames_deep_stays_native():
    """At the default recursion limit, a capture of all 702 frames of a stack
    from 700 frames deep, a few calls per frame, stays native throughout and
    matches the Python walk, which has room there."""
    process = subprocess.run(
        [sys.executable, "-c", _DEEP_STACK_SCRIPT],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout) == [True, 0]


def test_placeholders_replaced_in_place():
    """A depth placeholder replaced in place, by a call the Python walk takes
    or by a native one after the walker wrote it, is the copy later calls
    reuse, natively or in Python."""

    def build():
        return ({"a": {"leaf": [1]}}, {"b": {"leaf": [2]}}, _odd_types()[0]())

    def calls(context, first, second, odd):
        return [
            lambda: context.value([first, second]),
            # A bool depth is not one the walker reads: Python takes the call.
            lambda: context._mapping(first, False),
            lambda: context.value({"again": first}),
            lambda: copy.copy(context) and None,
            lambda: context._mapping(second, 0),
            lambda: context.value([second, odd]),
        ]

    _same_in_both_modes(build, DumpOptions(max_depth=1), calls)


@pytest.mark.parametrize("key", [1, 1.5, b"reason"], ids=["int", "float", "bytes"])
def test_a_placeholder_with_a_key_not_a_str_is_read_as_in_python(key):
    """``_mapping`` meeting a depth placeholder in its memo whose instance
    dict holds a key other than an exact str, a bytes key too, which
    compared with a str warns under ``python -b``, reads its ``reason``
    through the same lookups as the Python walk, natively, and replaces
    it as there."""
    found = {}
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_depth=1))
        _check_walker(context, native)
        inner = {"k": 1}
        (placeholder,) = context.value([inner])
        vars(placeholder)[key] = 1
        before = _fallbacks()
        found[native] = (context._mapping(inner), dict(context._truncation))
    assert found[True] == found[False] == ({"k": 1}, {"max_depth": 1})
    assert _fallbacks() == before


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


@pytest.mark.parametrize("taken", ["natively", "natively, copied"])
def test_a_placeholder_captured_as_itself_and_replaced_in_place(taken):
    """A depth placeholder captured as its own copy is, to the Python walk's
    ``_mapping``, a depth placeholder like any other: it replaces it in
    place before it fails on the placeholder's missing ``items``. The
    walker takes the replacement up and returns it from then on, as the
    Python walk does."""

    def build():
        return (TruncatedValue("max_depth", "dict"),)

    def calls(context, placeholder):
        taking = {
            "natively": [lambda: context.value(placeholder) is placeholder],
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


def _native_call_lines():
    """The lines of ``value`` and ``_mapping`` that call the native walker."""
    lines = {}
    for name in ("value", "_mapping"):
        function = vars(values_module.ValuesMixin)[name]
        source, first = inspect.getsourcelines(function)
        lines[function.__code__] = {
            first + at for at, line in enumerate(source) if "= _native_" in line
        }
    return lines


_NATIVE_CALL_LINES = _native_call_lines()


def _inside_the_walk(context, nodes):
    """Whether a native call on ``context`` is running on this thread.

    Some ``value`` or ``_mapping`` frame on ``context`` is at its call into
    the walker, and the context's ``_nodes`` is still below ``nodes``: the
    call writes its count, ``nodes`` or more, to the context only as it
    returns. The interpreter can run a signal handler or a collection right
    after the call returns, on that same line, and one handler can run
    inside another, so neither the line nor the innermost frame tells alone.
    """
    frame = sys._getframe(1)
    while frame is not None:
        lines = _NATIVE_CALL_LINES.get(frame.f_code)
        if (
            lines is not None
            and frame.f_locals.get("self") is context
            and frame.f_lineno in lines
        ):
            return vars(context)["_nodes"] < nodes
        frame = frame.f_back
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
    walk runs captures as it does outside the walk, and the outer capture is
    unchanged: the context is captured again from inside its own capture as
    in the Python walk. The graph holds no container copy, whose
    ``FakeContainer.__hash__`` is Python code, so a collection runs inside
    the native call only at the walker's signal check, where the node count
    is synced as around application code, and the inner captures count
    toward the outer one's."""
    graph = [[i, str(i)] for i in range(20000)]
    nodes = 1 + 3 * len(graph)  # The list, and each pair, its int and its str.
    results = {}
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
            _check_walker(context, native)
        calls = []

        def callback(phase, info, context=context, calls=calls, native=native):
            if phase != "start" or len(calls) >= 50:
                return
            inside = native and _inside_the_walk(context, nodes)
            before = context._nodes
            inner = context.value(["inner", 2.5])
            calls.append((inside, inner, context._nodes - before))

        with _collecting(callback):
            captured = context.value(graph)
        inside = [outcome for within, outcome, _ in calls if within]
        assert all(inner == ["inner", 2.5] for _, inner, _ in calls)
        assert inside if native else calls and not inside
        assert context._nodes == nodes + sum(counted for _, _, counted in calls)
        results[native] = pickle.dumps(captured, protocol=5)
    assert results[True] == results[False]


def _restore(cls, state):
    """An instance of ``cls`` whose instance dict is updated from ``state``."""
    instance = object.__new__(cls)
    instance.__dict__.update(state)
    return instance


def _reduce(context):
    """Reduce a context from a copy of its instance dict, bypassing its class."""
    return _restore, (type(context), dict(vars(context)))


def _shallow_copy(context, how):
    """A shallow copy of ``context`` sharing the values of its instance dict,
    made with ``copy.copy``, through ``copyreg``, or by hand."""
    if how == "by_hand":
        return _restore(type(context), dict(vars(context)))
    if how == "copyreg":
        copyreg.pickle(SnapshotContext, _reduce)
    try:
        return copy.copy(context)
    finally:
        copyreg.dispatch_table.pop(SnapshotContext, None)


@pytest.mark.parametrize("how", ["copy", "copyreg", "by_hand"])
def test_a_shallow_copy_captures_inside_its_originals_call(how):
    """A shallow copy captures with its original's walk and shares
    ``objects``. A capture through it from a collection callback while the
    original's native call runs captures as it does outside the call, as in
    the Python walk, and the original's capture is unchanged."""
    graph = [[i, str(i)] for i in range(20000)]
    nodes = 1 + 3 * len(graph)  # The list, and each pair, its int and its str.
    results = {}
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
        twin = _shallow_copy(context, how)
        assert twin.objects is context.objects
        _check_walker(context, native)
        _check_walker(twin, native)
        calls = []

        def callback(
            phase, info, context=context, twin=twin, calls=calls, native=native
        ):
            if phase != "start" or len(calls) >= 50:
                return
            inside = native and _inside_the_walk(context, nodes)
            calls.append((inside, twin.value(["inner", 2.5])))

        before = _fallbacks()
        with _collecting(callback):
            captured = context.value(graph)
        assert _fallbacks() == before
        assert all(inner == ["inner", 2.5] for _, inner in calls)
        assert any(inside for inside, _ in calls) if native else calls
        results[native] = pickle.dumps(captured, protocol=5)
    assert results[True] == results[False]


_REBASED_COUNT = 6000


def _rebased_mid_call(native):
    """Capture ``_REBASED_COUNT`` instances while the first collection past
    node 4096 rebases their class, then three new instances in a second
    call. Returns each copy's ``x`` attribute (None where it has none) for
    both calls, the ``unsupported_type`` tally after the first, and whether
    the class changed during the first call."""

    class WithDict:
        pass

    class Planted:
        @property
        def __dict__(self):
            return {}

    class Child(Planted):
        pass

    def instances(count):
        made = [Child() for _ in range(count)]
        for index, instance in enumerate(made):
            instance.x = index
        return made

    first = instances(_REBASED_COUNT)
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
    _check_walker(context, native)
    rebased = []

    def rebase(phase, info):
        if phase != "start" or rebased:
            return
        # The native call counts the list and every instance.
        if (
            _inside_the_walk(context, _REBASED_COUNT)
            if native
            else context._nodes >= 4096
        ):
            Child.__bases__ = (WithDict,)
            rebased.append(True)

    before = _fallbacks()
    with _collecting(rebase):
        captured = context.value(first)
    tally = context._truncation["unsupported_type"]
    assert _fallbacks() == before
    later = context.value(instances(3))
    return (
        [vars(copy).get("x") for copy in captured],
        tally,
        [vars(copy).get("x") for copy in later],
        bool(rebased),
    )


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_a_class_rebased_by_a_collection_mid_call(native):
    """A collection callback past node 4096 rebases a class whose
    ``__dict__`` is a property onto one with the real instance-dict
    descriptor. The instances captured after the collection carry their
    attributes, and only those before it count as ``unsupported_type``, in
    both walks: with no container copy to hash, a collection runs inside a
    native call only at the walker's signal check, after which, as after
    application code, the walker forgets what it read about types. The next
    call reads the attributes too. (``test_native_differences`` pins a
    collection that runs between two sync points.)"""
    xs, tally, later, rebased = _rebased_mid_call(native)
    assert rebased, "no collection ran during the walk"
    hidden = xs.count(None)
    assert 0 < hidden < _REBASED_COUNT
    assert xs == [None] * hidden + list(range(hidden, _REBASED_COUNT))
    assert tally == hidden
    assert later == [0, 1, 2]


def _held_by_a_frame(context):
    """Capture this frame, whose one variable is ``context``, through it."""
    head = context.from_frame(sys._getframe())
    return head.tb_frame.f_locals["context"]


@pytest.mark.parametrize(
    "by, counted", [("itself", 2), ("its own frame", 3), ("another context", 0)]
)
@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
def test_a_context_captured_as_a_value(native, by, counted):
    """A context captured by itself, through ``value`` or a frame that holds
    it, shows the count the walk had reached when it selected the instance
    dict's entries: two nodes past the call's start through ``value``,
    three through a frame namespace. The walker writes its count to the
    instance dict before it reads that dict. Captured by another context,
    it shows the count its last call left. Its copy holds every copy made so
    far, the capturing call's own included."""
    with _mode(native):
        context = SnapshotContext(DumpOptions(max_frames=1))
        other = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    _check_walker(other, native)
    data = [1, "a"]
    context.value(data)
    nodes = context._nodes
    before = _fallbacks()
    if by == "itself":
        copied = context.value(context)
    elif by == "its own frame":
        copied = _held_by_a_frame(context)
    else:
        copied = other.value(context)
    assert _fallbacks() == before
    attrs = vars(copied)
    earlier = [id(data), id(data[0]), id(data[1])]
    objects, sources = list(attrs["objects"]), list(attrs["_sources"])
    if counted:
        assert objects[:3] == sources[:3] == earlier
        assert {id(context), id(vars(context))} <= set(objects)
    else:
        assert objects == sources == earlier
    assert attrs["_nodes"] == nodes + counted


_REENTRY_SCRIPT = """
import copy, gc, pickle, sys, threading
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext, values

mode = sys.argv[1]
context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
assert context._native_walk
entered, done = threading.Event(), threading.Event()
errors, independent, captures = [], [], []
fired = False

def capture_same():
    try:
        if mode == 'copy':
            copy.copy(context)
        elif mode == 'deepcopy':
            copy.deepcopy(context)
        elif mode == 'pickle':
            pickle.loads(pickle.dumps(context))
        else:
            captures.append(context.value([1]))
    except RuntimeError as error:
        errors.append(str(error))
    finally:
        done.set()

native_lines = {
    values.ValuesMixin.value.__code__: next(
        number for number, line in enumerate(open(values.__file__), 1)
        if "captured = _native_value(" in line
    )
}

def inside():
    # The native call runs beneath its entry's frame, at that call.
    frame = sys._getframe(1)
    while frame is not None:
        if native_lines.get(frame.f_code) == frame.f_lineno:
            return True
        frame = frame.f_back
    return False

def collected(phase, info):
    # The graph holds no container copy, so a collection runs inside the
    # native call at one of its signal checks.
    global fired
    if phase != 'start' or fired or not inside():
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

graph = [[i, str(i)] for i in range(20000)]
gc.callbacks.append(collected)
gc.set_threshold(1, 1, 1)
captured = context.value(graph)
gc.set_threshold(700, 10, 10)
assert len(captured) == len(graph)
if mode == 'other':
    entered.set()
    thread.join(5)
    assert not thread.is_alive()
assert fired, 'no collection ran inside the native call'
assert errors == [], errors
if mode == 'separate':
    assert independent == [[1]]
elif mode in ('same', 'other'):
    assert captures == [[1]], captures
print('ok')
"""


@pytest.mark.parametrize(
    "mode", ["same", "other", "copy", "deepcopy", "pickle", "separate"]
)
def test_unanticipated_capture_reentry(mode):
    """A capture through the context from a collection inside its native
    call, or from another thread meanwhile, captures as in the Python walk;
    so do independent contexts, and copying or pickling the context, which
    captures nothing."""
    process = subprocess.run(
        [sys.executable, "-c", _REENTRY_SCRIPT, mode],
        capture_output=True,
        text=True,
        env=_child_environment(),
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "ok"


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


def test_a_deleted_helper_a_capture_never_calls(monkeypatch):
    """A missing helper need not fail a capture that never calls it."""
    for native in (False, True):
        with _mode(native):
            context = SnapshotContext(DumpOptions())
        context.value(1)
        with monkeypatch.context() as patch:
            patch.delattr(SnapshotContext, "_text")
            assert context.value(42) == 42


@contextlib.contextmanager
def _hooked(kind, on_call=None):
    """Set a tracer, a profiler, ``cProfile`` or a ``sys.monitoring`` tool for
    the block, and restore a test runner's hook after it. The list it yields
    gets the code of each Python function the hook sees start, and
    ``on_call`` is called with that code as it starts; ``cProfile`` lists
    what it saw only when the block ends, and calls nothing."""
    seen = []

    def record(code):
        seen.append(code)
        if on_call is not None:
            on_call(code)

    if kind == "cprofile":
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            yield seen
        finally:
            profiler.disable()
        seen.extend(entry.code for entry in profiler.getstats())
        return
    if kind == "monitoring":
        monitoring = sys.monitoring
        tool = next(
            (tool for tool in reversed(range(6)) if monitoring.get_tool(tool) is None),
            None,
        )
        if tool is None:
            pytest.skip("all monitoring tool ids are in use")
        start = monitoring.events.PY_START
        monitoring.use_tool_id(tool, "pydumpling-test")
        monitoring.register_callback(tool, start, lambda code, offset: record(code))
        monitoring.set_events(tool, start)
        try:
            yield seen
        finally:
            monitoring.set_events(tool, 0)
            monitoring.register_callback(tool, start, None)
            monitoring.free_tool_id(tool)
        return
    getter, setter = {
        "profile": (sys.getprofile, sys.setprofile),
        "trace": (sys.gettrace, sys.settrace),
    }[kind]

    def hook(frame, event, arg):
        if event == "call":
            record(frame.f_code)
        return hook

    previous = getter()
    setter(hook)
    try:
        yield seen
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
            # A native call may run no collection: then one after it shares.
            gc.collect()
        assert inner and inner[0] is captured[0]
        assert context.value(graph[0]) is captured[0]
        copy.copy(context)
        assert context._nodes == len(context.objects) <= 20


def _helpers_case():
    """A case whose capture in Python runs pydumpling's helpers, which the
    native walk does itself: ``_layout`` for container and scalar
    subclasses, and ``_text`` and ``_bounded_bytes`` for long data; and
    ``FakeContainer.__hash__`` for a named tuple key, which a dict insert
    runs in both walks."""
    point = _Record(1, "two")
    values = [
        OrderedDict(a=1, b=[2]),
        defaultdict(list, k=[3]),
        Counter("hello"),
        point,
        _Color.HUGE,
        _Mode.WIDE,
        "x" * 10_000,
        bytearray(b"y" * 5_000),
        {point: "keyed"},
        {point, 4},
    ]
    ops = [
        ("value", values, 0),
        ("mapping", {"values": values, "point": point}, 0, (), False, dict),
    ]
    return _Case(-1, DumpOptions(), ops, None, [])


@pytest.mark.parametrize("kind", ["trace", "profile", "cprofile", "monitoring"])
def test_parity_under_hooks(kind):
    """Under a tracer, a profiler, ``cProfile`` or a ``sys.monitoring`` tool
    the walker takes every call it takes without one, and its snapshot is
    the Python walk's byte for byte: the hook sees none of pydumpling's
    helpers, which the walker does itself, and changes nothing either walk
    captures. (Seed 3 runs from deep in the stack, found by overflowing on
    purpose, which unsets a Python trace function.)"""
    cases = [_case(seed) for seed in (0, 1, 2, 4, 5)] + [_helpers_case()]
    try:
        for case in cases:
            _run(case, native=False)
            unhooked = _run(case, native=True)
            with _hooked(kind) as seen:
                native = _run(case, native=True)
            with _hooked(kind):
                pure = _run(case, native=False)
            _assert_parity(case, native, pure)
            assert native.fallbacks == unhooked.fallbacks, case.seed
    finally:
        for case in cases:
            case.close()
    assert native.fallbacks == 0
    helpers = {
        code.co_name
        for code in seen
        if getattr(code, "co_filename", None)
        in (values_module.__file__, context_module.__file__)
    }
    # The context's own, the two methods the case calls the walker by, and
    # the node check ``value`` makes before its entry.
    assert helpers <= {"__init__", "value", "_mapping", "_has_nodes"}


class _HookedName(str):
    """A ``__qualname__`` whose ``+`` is application code ``_type_name``
    runs, and a hook can see start."""

    def __radd__(self, other):
        return str.__add__(other, str(self))


@pytest.mark.parametrize("kind", ["profile", "trace", "monitoring"])
def test_a_hook_capturing_inside_application_code_matches_python(kind):
    """A profiler, tracer or monitoring callback that captures through the
    context while the walk runs application code (here a ``__qualname__``'s
    ``__radd__``) gets the copy the call made already, from the memo, and
    captures a value not copied yet, counting its node on from the walk's
    count, which it reads in ``_nodes`` as in the Python walk. The outer
    capture then counts on from the hook's, and is the same either way."""
    radd = _HookedName.__radd__.__code__
    outcomes = {}
    for native in (False, True):
        Named = type("Named", (), {})
        Named.__qualname__ = _HookedName("Named")
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_string_bytes=8))
        shared, fresh, inner = [], [1], []

        def capture(code, context=context, inner=inner, values=(shared, fresh)):
            if code is radd and not inner:
                inner.append(context._nodes)
                inner.extend(context.value(value) for value in values)
                inner.append(context._nodes)

        before = _fallbacks()
        with _hooked(kind, capture):
            captured = context.value([shared, "x" * 100, Named()])
        assert _fallbacks() == before
        assert len(inner) == 4
        assert inner[1] is captured[0]
        assert inner[2] == [1]
        state = (inner, captured, dict(context._truncation), context._nodes)
        outcomes[native] = pickle.dumps(state, protocol=5)
    assert outcomes[True] == outcomes[False]


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 is unavailable")
def test_signal_handlers_and_asyncio_keep_the_native_walk():
    """Installing a Python signal handler, as ``asyncio.run`` installs one for
    SIGINT, leaves the walker in use and the handler installed: a handler
    runs inside a native call where Python code runs, which the walker's
    signal checks make sure of every 4,096 nodes."""
    with _mode(True):
        context = SnapshotContext(DumpOptions())

    def handler(number, frame):
        pass

    before = _fallbacks()
    previous = signal.signal(signal.SIGUSR1, handler)
    try:
        assert context.value([1, 2]) == [1, 2]
        assert signal.getsignal(signal.SIGUSR1) is handler
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert _fallbacks() == before

    async def main():
        with _mode(True):
            inner = SnapshotContext(DumpOptions())
        assert inner.value({"a": [1]}) == {"a": [1]}
        return inner._native_walk

    assert asyncio.run(main())
    assert _fallbacks() == before


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

    before = _fallbacks()
    with _mode(True):
        hooked = Hooked(DumpOptions())
    assert hooked.value([1, "x"]) == [1, "x"]
    assert hooked._native_walk
    assert _fallbacks() == before
    assert lookups == []

    class Context(SnapshotContext, metaclass=Logged):
        pass

    with _mode(True):
        context = Context(DumpOptions(max_string_bytes=4))
    assert context._native_walk
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
        fallbacks = _fallbacks() - before
        Context._text = Descriptor()
        # In a list: ``value`` captures an exact str itself, before the
        # walker's entry, through the class.
        patched = context.value(["y" * 50])
    finally:
        builtins.__import__ = real_import
        del Context._text
    assert fallbacks == 0
    assert not imports
    # The class changed after it was created, which a native context does
    # not see: the walker bounds the text itself.
    assert context._native_walk
    assert lookups == []
    expected = SnapshotContext(DumpOptions(max_string_bytes=4)).value(["y" * 50])
    assert patched == expected


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

    def getattr_(self, name):
        log.append(("missing", name))
        raise AttributeError(name)

    def setattr_(self, name, value):
        log.append(("set", name))
        object.__setattr__(self, name, value)

    def delattr_(self, name):
        log.append(("del", name))
        object.__delattr__(self, name)

    return {
        "__getattribute__": getattribute,
        "__getattr__": getattr_,
        "__setattr__": setattr_,
        "__delattr__": delattr_,
    }


def _logged_capture(attributes, when):
    """Capture in each mode on a fresh context subclass given the class
    attributes ``attributes(log)`` returns, in its body (``subclass``) or set
    on it after a first call (``patched``), and return the pickled results
    and the log of what those attributes ran."""
    data = {"a": [1, 2, 3, 4], "b": "text" * 10, "c": _Record(1, [2]), "d": (2.5,)}
    outcomes = {}
    for native in (False, True):
        log = []
        added = attributes(log)
        body = {} if when == "patched" else added
        Context = type("Context", (SnapshotContext,), body)
        with _mode(native):
            context = Context(
                DumpOptions(max_items=3, max_nodes=12, max_string_bytes=8)
            )
        first = context.value([1, "x" * 20])
        if when == "patched":
            for name, attribute in added.items():
                setattr(Context, name, attribute)
        results = (
            first,
            context.value(data),
            context._mapping({"k": data}, 0),
            context.value(data),
            context.value([5]),
        )
        state = (results, list(context._truncation.items()), context._nodes)
        outcomes[native] = (pickle.dumps(state, protocol=5), list(log))
        # The walk is chosen when the class is created.
        assert context._native_walk is (native and when == "patched")
    return outcomes


_STATE_NAMES = (
    "options",
    "objects",
    "_sources",
    "_nodes",
    "_truncation",
    "_layouts",
    "_node_marker",
    "_unlimited",
)


@pytest.mark.parametrize("name", _STATE_NAMES)
def test_state_behind_a_property_runs_as_in_python(name):
    """A context class with a state attribute behind a property captures in
    Python, which runs that property exactly as the Python walk does."""
    outcomes = _logged_capture(
        lambda log: {name: _logged_property(name, log)}, "subclass"
    )
    assert outcomes[True] == outcomes[False]
    assert outcomes[True][1], "the property never ran"


@pytest.mark.parametrize(
    "hook", ["__getattribute__", "__getattr__", "__setattr__", "__delattr__"]
)
def test_context_attribute_hooks_run_as_in_python(hook):
    """A context class with attribute hooks of its own captures in Python,
    which runs them exactly as often as the Python walk does. Neither walk
    deletes an attribute or reads one that is missing here, so
    ``__delattr__`` and ``__getattr__`` check only that the context goes to
    the Python walk (``test_a_context_class_with_getattr``)."""
    assert hook in values_module._STATE_ACCESS
    outcomes = _logged_capture(lambda log: {hook: _logged_hooks(log)[hook]}, "subclass")
    assert outcomes[True] == outcomes[False]


def test_the_state_access_hooks_are_every_attribute_hook():
    """Reading, writing and deleting an instance attribute goes through
    ``__getattribute__``, then ``__getattr__`` for a missing one,
    ``__setattr__`` and ``__delattr__`` (and descriptors under the state
    names, which ``_NATIVE_STATE`` covers): a class with any of them of its
    own captures in Python."""
    assert set(values_module._STATE_ACCESS) == set(_logged_hooks([]))


@pytest.mark.parametrize("missing", ["_layouts", "_code_text_limit", "_node_marker"])
def test_a_context_class_with_getattr(missing):
    """A context class whose ``__getattr__`` answers a state attribute its
    instance lacks captures in Python, which runs the hook at every read of
    it, as the Python walk does, not once per call."""
    Pair = namedtuple("Pair", "left right")

    def attributes(log):
        def getattr_(self, name):
            log.append(name)
            if name != missing:
                raise AttributeError(name)
            return {
                "_layouts": lambda: vars(self).setdefault("_kept", {}),
                "_code_text_limit": lambda: 12,
                "_node_marker": lambda: TruncatedValue("max_nodes"),
            }[name]()

        return {"__getattr__": getattr_}

    outcomes = {}
    for native in (False, True):
        log = []
        Context = type("Context", (SnapshotContext,), attributes(log))
        assert Context._native_walk is False
        with _mode(native):
            context = Context(DumpOptions(max_nodes=12))
        delattr(context, missing)
        held = _Plain()
        held.a_name_past_twelve_bytes = [Pair(1, 2), Pair(3, 4)]
        results = (
            context.value([Pair(5, 6), held, Pair(7, 8)]),
            context._mapping({"k": [Pair(9, 10)] * 3}),
            context.value(list(range(20))),
            [context.value([index]) for index in range(3)],
        )
        state = (results, list(context._truncation.items()), context._nodes)
        outcomes[native] = (pickle.dumps(state, protocol=5), log)
    assert outcomes[True] == outcomes[False]
    assert outcomes[True][1].count(missing) > 1


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


class _LoggedList(list):
    """A list whose iteration, truth and subscripts run Python code."""

    def __init__(self, label, items):
        super().__init__(items)
        self.label = label

    def __iter__(self):
        _called(f"{self.label} __iter__")
        return list.__iter__(self)

    def __bool__(self):
        _called(f"{self.label} __bool__")
        return True

    def __getitem__(self, index):
        _called(f"{self.label} __getitem__ {index}")
        return list.__getitem__(self, index)


class _LoggedEntry:
    """A ``(name, member)`` entry of a layout, unpacked through its own
    ``__iter__``."""

    def __init__(self, name, member):
        self.name, self.member = name, member

    def __iter__(self):
        _called(f"entry {self.name}")
        return iter((self.name, self.member))


class _LoggedMember:
    """A layout member whose ``__get__`` is Python code; one holding None
    is unset."""

    def __init__(self, name, held):
        self.name, self.held = name, held

    def __get__(self, instance, owner):
        _called(f"__get__ {self.name}")
        if self.held is None:
            raise AttributeError(self.name)
        return self.held


class _HashedName(str):
    """A slot name whose hash is Python code."""

    def __hash__(self):
        _called(f"__hash__ {str(self)}")
        return str.__hash__(self)


def test_a_layout_python_code_wrote():
    """A layout Python code wrote into ``_layouts`` runs its own code where
    ``_fill_copy`` and ``_filled_slot`` read it: the layout's subscripts,
    iterating its members, unpacking each entry, each member's
    ``__get__``, the fields' truth and iteration, and a slot name's hash,
    the same number of times and in the same order in both walks."""

    class Point(tuple):
        pass

    class Number(int):
        pass

    class Bag(list):
        __slots__ = ("a",)

    def build():
        bag = Bag([1])
        bag.a = "slot"
        return (Point((1, 2)), Number(5), bag)

    def calls(context, point, number, bag):
        def planted():
            members = [
                _LoggedEntry("unset", _LoggedMember("unset", None)),
                _LoggedEntry("held", _LoggedMember("held", [3])),
            ]
            context._layouts[id(Point)] = _LoggedList(
                "layout",
                [Point, _LoggedList("members", members), _LoggedList("fields", "xy")],
            )
            context._layouts[id(Number)] = (
                Number,
                _LoggedList("number members", members),
                (),
            )
            name = _HashedName("a")
            context._layouts[id(Bag)] = (Bag, [(name, vars(Bag)["a"])], ())
            return context.value([point, [number], bag])

        def unpacked(entry):
            context._layouts[id(Point)] = (Point, [entry], ())
            return context.value(Point((7, 8)))

        return [
            planted,
            lambda: context.value(Point((5, 6))),
            lambda: unpacked(["x", _LoggedMember("x", 1), "extra"]),
            lambda: unpacked(("x",)),
            lambda: unpacked(5),
        ]

    native, _ = _same_in_both_modes(build, DumpOptions(max_depth=2), calls)
    assert "entry held" in native.hooks and "__hash__ a" in native.hooks
    assert [result[1] for result in native.results[2:]] == ["ValueError"] * 2 + [
        "TypeError"
    ]


class _LoggedName(str):
    """A ``__qualname__`` whose slicing, length, encoding and formatting
    run Python code."""

    def __getitem__(self, index):
        _called("name __getitem__")
        return _LoggedName(str.__getitem__(self, index))

    def __len__(self):
        _called("name __len__")
        return str.__len__(self)

    def encode(self, *args):
        _called("name encode")
        return str.encode(self, *args)

    def __str__(self):
        _called("name __str__")
        return str.__str__(self)


@pytest.mark.parametrize("length", [5, 5000])
def test_a_type_name_that_is_a_str_subclass(length):
    """A class in ``builtins`` whose ``__qualname__`` is a str subclass
    names its instances with that object: bounding it as code text slices,
    measures and encodes it through its own code, and formatting the copy's
    representation runs its ``__str__`` when it fits, in both walks."""

    def build():
        Named = type("Named", (), {"__module__": "builtins"})
        Named.__qualname__ = _LoggedName("N" * length)
        return ([Named(), Named()],)

    def calls(context, values):
        return [lambda: context.value(values)]

    native, _ = _same_in_both_modes(build, DumpOptions(), calls)
    assert "name encode" in native.hooks
    assert ("name __str__" in native.hooks) is (length == 5)


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
        assert Context._native_walk is False
    assert results[True] == results[False]
    assert __file__ not in compared


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="setitimer is unavailable")
def test_signal_handler_capturing_inside_a_native_walk():
    """A signal handler runs inside a native call at the walker's signal
    checks. One that captures on the context being walked captures as it
    does outside the call, as the interpreter may run it right after the
    call returns, reading and counting on from the walk's node count; the
    outer call stays native and its result unchanged."""
    graph = [[i, str(i)] for i in range(100000)]
    nodes = 1 + 3 * len(graph)  # The list, and each pair, its int and its str.
    options = DumpOptions(max_nodes=None, max_items=None)
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value(graph), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    before = _fallbacks()
    outcomes = []

    def handler(number, frame):
        inside = _inside_the_walk(context, nodes)
        outcomes.append((inside, context.value([1])))

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, 0.0005, 0.0005)
    try:
        captured = context.value(graph)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert _fallbacks() == before
    inside = [outcome for within, outcome in outcomes if within]
    assert inside, "no signal arrived during the native walk"
    assert all(copy == [1] for _, copy in outcomes)
    # Each handler's list is a node of its own; its int is the graph's.
    assert context._nodes == nodes + len(outcomes)
    assert context._native_walk
    assert pickle.dumps(captured, protocol=5) == expected


# --- Copies and their placeholders ------------------------------------------


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
    descriptor, as a plain mixin gives it before 3.15, captures natively,
    as the Python walk does: both read the state with attribute access,
    which never looks ``__dict__`` up."""

    class Other:
        pass

    descriptor = vars(Other)["__dict__"]
    results = {}
    for native in (False, True):
        Context = type("Context", (SnapshotContext,), {"__dict__": descriptor})
        assert Context._native_walk is True
        with _mode(native):
            context = Context(DumpOptions(max_string_bytes=4))
        _check_walker(context, native)
        before = _fallbacks()
        captured = [context.value(["abcdefgh", (1, 2.5)]), context.value("x" * 20)]
        state = (captured, list(context._truncation.items()), context._nodes)
        results[native] = pickle.dumps(state, protocol=5)
        if native:
            assert _fallbacks() == before
    assert results[True] == results[False]


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
            context.value(_odd_types()[0]())  # a str-subclass ``__qualname__``
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
            for step, odd in enumerate(steps):
                try:
                    if odd is None:
                        context._mapping(target)
                    elif odd:
                        context.value([step, _odd_types()[0]()])
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


def test_a_fake_class_given_another_base_runs_objects_in_python():
    """``FakeClass`` given a base whose ``__getattribute__`` and ``__del__``
    are application code: the walker runs every object in the Python walk,
    before it makes a copy it would drop, whose ``__del__`` would run, as
    the import-time instance-dict descriptor refuses it before 3.15, or
    reads a ``FakeClass``'s representation through the base's
    ``__getattribute__``, which the Python walk runs a different number of
    times."""
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


def test_a_fake_class_hook_runs_once_where_its_repr_reads_the_dict(monkeypatch):
    """A ``__getattribute__`` application code gave ``FakeClass`` runs where
    ``FakeClass.__repr__`` reads a copied ``FakeClass``'s ``__dict__``,
    once: the walker reads it through the same attribute lookup, as
    application code, and natively names the copy of a str-subclass
    ``__qualname__`` after it."""
    originals = []

    def hooked(self, name):
        if any(self is original for original in originals):
            _called(f"FakeClass.__getattribute__ {name}")
        return object.__getattribute__(self, name)

    odd_name = _odd_types()[0]  # a str subclass ``__qualname__``

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
    ``FakeClass``, and runs once, as the walker runs an exact ``FakeClass``
    in the Python walk in place once its ``__repr__`` is replaced."""

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


@pytest.mark.parametrize("change", ["another base", "another __repr__"])
def test_an_object_the_walker_runs_in_python_in_place(monkeypatch, change):
    """While ``FakeClass`` has another base, the walker runs every object in
    the Python walk; while ``FakeClass.__repr__`` is replaced, every exact
    ``FakeClass``. Inside a call that has started, such an object runs
    there in place, with nothing undone: the snapshot and the hooks are the
    Python walk's, and only that one node goes to the Python walk."""

    class Other:
        def __getattribute__(self, name):
            _called(f"Other.__getattribute__ {name}")
            return object.__getattribute__(self, name)

    def replaced(self):
        _called("FakeClass.__repr__")
        return "<replaced>"

    sent = []

    def build():
        held = _Plain() if change == "another base" else FakeClass("<copy>", {})
        held.field = [1]
        return ([1, held, "after"],)

    def calls(context, value):
        def changed():
            # ``monkeypatch`` cannot restore ``__bases__``, which is no entry
            # of the class dict.
            if change == "another base":
                FakeClass.__bases__ = (Other,)
            else:
                monkeypatch.setattr(FakeClass, "__repr__", replaced)
            try:
                before = _fallbacks()
                captured = context.value(value)
                sent.append(_fallbacks() - before)
                return captured
            finally:
                FakeClass.__bases__ = (FakeType,)
                monkeypatch.undo()

        return [changed]

    native, _ = _same_in_both_modes(build, DumpOptions(), calls)
    assert native.hooks
    assert sent == [0, 1, 0]  # The Python walk, the walker, the Python walk.


_BYTES_WARNING_SCRIPT = """
import os, warnings
from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext

mixed = {"a": 1, b"a": 2}  # under -b, built with a warning
counts = []
for native in (False, True):
    context = SnapshotContext(DumpOptions())
    if not native:
        context._native_walk = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BytesWarning)
        context.value(mixed)  # the copy compares b"a" with "a" again
    counts.append(len(caught))
print(context._native_walk, counts[0] == counts[1], counts[1] > 0)
"""


@pytest.mark.parametrize("flags", [[], ["-b"]], ids=["plain", "-b"])
def test_under_python_b_a_context_captures_in_python(flags):
    """Under ``python -b`` comparing bytes with str warns, and a warning the
    walker's own lookups and inserts raised would be attributed to the line
    that called it, so a context captures in Python there, and warns as the
    Python walk does."""
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
        context.value(_odd_types()[0]())  # a str-subclass ``__qualname__``
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
    """The walker holds none of the originals, dicts or keys of ``objects``
    and ``_sources`` between calls: once Python code replaces ``_sources``,
    its originals and keys go, as in the Python walk."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
    _check_walker(context, native)
    original, key = _Plain(), _Plain()
    released = weakref.ref(original), weakref.ref(key)
    context.value(original)
    context._sources[key] = "tag"
    context.value("next")
    context._sources = {}
    del original, key
    gc.collect()
    assert [ref() for ref in released] == [None, None]


def test_copy_made_by_application_code_mid_call():
    """A copy of the context that application code the walker runs makes
    mid-call shares the memo from then on, so it sees the copies the
    original makes later, as when both capture in Python."""
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


class _FailingName(str):
    """A ``__qualname__`` whose ``+``, application code ``_type_name`` runs,
    raises while ``armed``."""

    armed = False

    def __radd__(self, other):
        if _FailingName.armed:
            raise _Error("interrupted")
        return str.__add__(other, str(self))


def test_placeholder_of_an_interrupted_tuple_under_a_profiler():
    """A tuple whose capture an error interrupted keeps its placeholder,
    which a later call returns, natively under a profiler too: naming a
    tuple needs no Python code at any code text limit the walker takes."""

    def build():
        Failing = type("Failing", (), {})
        Failing.__qualname__ = _FailingName("Failing")
        return (("x" * 50, [1], Failing()),)

    def calls(context, graph):
        def interrupted():
            _FailingName.armed = True
            try:
                return context.value(graph)
            finally:
                _FailingName.armed = False

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
    "text", ["abcdefgh", "abcdefg\u00e9", "abcdefghi"], ids=["ascii", "wide", "long"]
)
def test_strings_need_no_helper(text):
    """A str is bounded natively, whether it fits or not: a profiler sees
    no helper of pydumpling's run, and the copy is the Python walk's."""
    options = DumpOptions(max_string_bytes=8)
    with _mode(False):
        expected = pickle.dumps(SnapshotContext(options).value([text]), protocol=5)
    with _mode(True):
        context = SnapshotContext(options)
    before = _fallbacks()
    with _hooked("profile") as seen:
        captured = context.value([text])
    assert pickle.dumps(captured, protocol=5) == expected
    assert _fallbacks() == before
    assert SnapshotContext._bounded_text.__code__ not in seen
    assert SnapshotContext._text.__code__ not in seen


# Around a budget of 8 bytes: under, at and over it, and cut inside a two-,
# three- and four-byte character and a lone surrogate's three bytes.
_BOUNDED_TEXTS = (
    "abcdefg",
    "abcdefgh",
    "abcdefghi",
    "abcdefg\u00e9",
    "abcdef\u9903",
    "abcdefg\U0001f600",
    "abcdefg\ud800",
    "\ud800\udc00\ud800",
    "\u00e9" * 4,
    "",
)
_BOUNDED_DATA = (b"", b"abcdefg", b"abcdefgh", b"abcdefghi")


def test_bounded_text_bytes_and_layouts_match_python():
    """Strings and bytes around the byte budget, names around the code text
    limit (attribute names and a type name, cut inside a character), and
    the layouts written for slotted and named tuple classes: the copies,
    tallies, memo and ``_layouts`` are the Python walk's, byte for byte."""
    Point = namedtuple("Point", "x y")

    class Slotted(list):
        __slots__ = ("a", "b")

    class Count(int):
        pass

    def build():
        held = _Plain()
        for name in ("short", "a_name_past_twelve_bytes", "na\u00efve_\u540d\u524d"):
            setattr(held, name, 1)
        vars(held)["abcdefghijk\ud800"] = 2
        Named = type("Named", (), {"__module__": "builtins"})
        Named.__qualname__ = "abcdefghijk\u00e9"
        slotted = Slotted([1])
        slotted.a = "set"
        data = [*_BOUNDED_DATA, *map(bytearray, _BOUNDED_DATA)]
        return (
            list(_BOUNDED_TEXTS),
            data,
            [held, Named()],
            [Point(1, 2), slotted, Count(3)],
        )

    def calls(context, texts, data, objects, subclasses):
        def limited():
            context._code_text_limit = 12

        def layouts():
            return [
                (cls.__qualname__, [name for name, _ in members], fields)
                for cls, members, fields in context._layouts.values()
            ]

        return [
            limited,
            lambda: context.value(texts),
            lambda: context.value(data),
            lambda: context.value(objects),
            lambda: context.value(subclasses),
            layouts,
        ]

    native, _ = _same_in_both_modes(build, DumpOptions(max_string_bytes=8), calls)
    assert dict(native.truncation)["max_string_bytes"] == 8  # six texts, two bytes
    assert [name for name, _, _ in native.results[-1]] == [
        "Point",
        "test_bounded_text_bytes_and_layouts_match_python.<locals>.Slotted",
        "test_bounded_text_bytes_and_layouts_match_python.<locals>.Count",
    ]


class _PlainText(str):
    pass


class _PlainData(bytes):
    pass


def _long_attribute_name():
    held = _Plain()
    setattr(held, "名" * 2000, 1)
    return held


# What the walk slices, or keeps whole, with each budget below: str and
# bytes long and short, ASCII and not, their subclasses, any bytearray, and
# a name past the code text limit. A budget past 256, the last cached small
# int, is a new int in each slice.
_SLICED = {
    "ascii str": "x" * 5000,
    "wide str": "é" * 300,
    "wider str": "é餃" * 2000,
    "lone surrogates": "\ud800" * 3000,
    "str subclass": _PlainText("y" * 5000),
    "long bytes": b"y" * 5000,
    "short bytes": b"y" * 200,
    "one byte": b"z",
    "empty bytes": b"",
    "bytes subclass": _PlainData(b"y" * 5000),
    "short bytearray": bytearray(b"ab"),
    "long bytearray": bytearray(5000),
    "long attribute name": _long_attribute_name(),
}


def _blocks_per_capture(native, options, item, runs=300):
    """``sys.getallocatedblocks()`` growth per capture of ``[item]`` on a
    new context: the least over batches of ``runs``. A capture makes no
    cyclic garbage, so the collector stays off meanwhile. What the free
    lists keep fills them within a few batches (a collection would empty
    them again), while a leak grows every batch, so batches run until one
    grows nothing, ten at most."""
    with _mode(native):

        def once():
            context = SnapshotContext(options)
            _check_walker(context, native)
            context.value([item])

        enabled = gc.isenabled()
        gc.disable()
        try:
            least = float("inf")
            for _ in range(10):
                before = sys.getallocatedblocks()
                for _ in range(runs):
                    once()
                least = min(least, (sys.getallocatedblocks() - before) / runs)
                if least <= 0:
                    break
        finally:
            if enabled:
                gc.enable()
    return least


@pytest.mark.parametrize("budget", [4096, 257, None], ids=lambda budget: f"{budget}")
def test_captures_of_sliced_values_keep_nothing_allocated(budget):
    """Repeated captures of what the walk slices keep no more memory
    allocated natively than in Python: a leaked object per capture is a
    block per capture."""
    options = DumpOptions(max_string_bytes=budget)
    grown = {
        name: (
            _blocks_per_capture(True, options, item),
            _blocks_per_capture(False, options, item),
        )
        for name, item in _SLICED.items()
    }
    assert {name: g for name, g in grown.items() if g[0] > g[1] + 0.25} == {}


# Each budget at 1, its least valid value, alone and all at once. A forged 0
# is among ``_STATE_CHANGES``.
_BUDGETS = ("max_items", "max_nodes", "max_string_bytes", "max_depth")
_AT_ONE = {name: {name: 1} for name in _BUDGETS} | {"all": dict.fromkeys(_BUDGETS, 1)}

_AT_ONE_SOURCE = """
def holder(values, text):
    local = [text, values]
    yield sys._getframe()
"""


@pytest.mark.parametrize("budgets", list(_AT_ONE))
def test_each_budget_at_one(budgets):
    """With a budget at 1, alone or all at once, each kind of value, a root
    namespace with priority names, a nested mapping, an attribute mapping
    with a private name and a frame, each captured on a context of its own,
    give the Python walk's copies, tallies, node counts and memo, byte for
    byte."""
    options = DumpOptions(**_AT_ONE[budgets])

    def build():
        held = _Plain()
        held.name, held.items = "é", [1, 2]
        slotted = _Slotted()
        slotted.first = b"ab"
        label = _SlotText("ab")
        label.child = "c"
        values = [
            *("", "a", "é", "ab", b"", b"a", b"ab", bytearray(b"ab")),
            *(0, 255, 256, -129, 2**70, True, None, 1.5),
            *(_Count(300), _Ratio(2.5), label, _Record(1, 2), held, slotted),
            *([], [1], [1, [2, [3]]], (), (1, 2), frozenset({1}), {1, 2}),
            *({}, {"a": 1}, {"a": {"b": 1}, "c": 2}, OrderedDict(a=1, b=2)),
            Counter("ab"),
        ]
        namespace = {"self": held, "a": values[1:4], "b": "text", "c": 2**9}
        scope = {"sys": sys, "__name__": "at_one"}
        exec(_AT_ONE_SOURCE, scope)
        generator = scope["holder"](values[:4], "été")
        return values, namespace, generator, next(generator)

    def calls(context, values, namespace, generator, frame):
        def alone(capture):
            def call():
                fresh = SnapshotContext(options)
                captured = capture(fresh)
                truncation = list(fresh._truncation.items())
                return captured, truncation, fresh._nodes, _memo(fresh)

            return call

        return [
            lambda: context.value(values),
            *(alone(lambda fresh, value=value: fresh.value(value)) for value in values),
            alone(lambda fresh: fresh._mapping(namespace, priority=("b", "self"))),
            alone(lambda fresh: fresh._mapping(namespace, 1)),
            alone(lambda fresh: fresh._attribute_mapping(namespace, 1, private="b")),
            alone(lambda fresh: fresh.from_frame(frame)),
        ]

    native, _ = _same_in_both_modes(build, options, calls)
    # Each budget at 1 cuts something.
    reasons = {reason for reason, _ in native.truncation}
    for _, tallies, _, _ in native.results[1:]:
        reasons.update(reason for reason, _ in tallies)
    assert set(_AT_ONE[budgets]) <= reasons


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
    copy captures with its original's walk."""
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
    _check_walker(context, native)


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
    assert context._native_walk
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


def test_a_replaced_placeholder_s_original_removed_by_python_code():
    """A call that replaces a written depth placeholder, here an instance's
    attributes captured too deep before, and runs application code on the
    way (a class name's ``__radd__``), makes its replacement where the
    Python walk does, after the copies made before it, and its original,
    which Python code then removes, stays removed."""

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
def test_a_tally_a_bound_adds_to_runs_once(helper):
    """A tally that the bound of a string, bytes or a class name adds to,
    with an ``__add__`` of its own, runs it once, natively as in the Python
    walk, and so does a later ``max_items`` tally's."""
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


def test_application_code_mid_call_finds_the_copies_made_so_far():
    """Application code the walker runs, here a tally's ``__add__``, finds
    in ``objects`` and ``_sources`` the copies of earlier calls and, of
    this one, those made by then, as in the Python walk."""

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
def test_a_context_a_class_it_captured_refers_to_is_collected(native):
    """The walker keeps nothing of a context, or of the classes of the
    values a call met, once the call ends, so a context that such a class
    refers back to is freed by one collection, as in the Python walk."""
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


def test_capturing_the_context_s_own_dicts():
    """A walk that captures ``objects`` and ``_sources`` themselves copies
    each key as a value. Both walks keep an entry under two int objects, one
    in each dict, so both count the same nodes there, leave the same dicts,
    and truncate a later capture at the same value under a node budget."""

    def build():
        return ([[index, str(index)] for index in range(5)],)

    def calls(context, data):
        def captured():
            # The copies are keyed by ids, which differ from one run to the next.
            copies = context.value([context.objects, context._sources])
            return [
                (len(copy), sorted(type(item).__name__ for item in copy.values()))
                for copy in copies
            ]

        return [
            lambda: context.value(data),
            captured,
            lambda: (context._nodes, len(context.objects), len(context._sources)),
            lambda: context.value([[index, float(index)] for index in range(40)]),
        ]

    for nodes in (10000, 150):
        native, _ = _same_in_both_modes(build, DumpOptions(max_nodes=nodes), calls)
        assert native.results[2][0] > 2 * 16  # each key counted in both dicts


def _has_instance_dict(context):
    """Whether ``context``'s instance dict exists. Reading ``__dict__``
    makes it, and from then on attribute access on the context costs more
    (3.12, 3.14)."""
    return any(
        type(referent) is dict and "objects" in referent
        for referent in gc.get_referents(context)
    )


@pytest.mark.parametrize("native", [False, True], ids=["pure", "native"])
@pytest.mark.parametrize("holder", ["attributes", "dict", "context"])
def test_a_nodes_key_leaves_the_context_s_dict_unmade(native, holder):
    """A captured mapping with a ``_nodes`` key holding the very int the
    context holds (an object's attributes, a dict, another context) leaves
    the context's instance dict unmade, as in the Python walk. Only the
    context's own dict, or a copy made from it, sends the walker to read
    ``ctx.__dict__``. Small ints are one object each, so one of the counts
    below is the one the context holds while the walk selects that key."""
    with _mode(native):
        context = SnapshotContext(DumpOptions())
        others = [SnapshotContext(DumpOptions()) for _ in range(32)]
    _check_walker(context, native)
    held = []
    for count, other in enumerate(others):
        if holder == "attributes":
            value = _Plain()
            value._nodes = count
            value.count = 1
        elif holder == "dict":
            value = {"_nodes": count, "count": 1}
        else:
            value = other
            value._nodes = count
        held.append(value)
    assert not _has_instance_dict(context)
    context.value([[1, 2], held])
    assert not _has_instance_dict(context)
    vars(context)
    assert _has_instance_dict(context)


def test_namespaces_after_application_code_mid_call():
    """A call that runs application code, a tally's ``__add__``, holds in
    ``_namespaces`` the attribute dicts of the objects captured by earlier
    calls and by this one, as in the Python walk."""

    def build():
        made = [_Plain() for _ in range(4)]
        for index, instance in enumerate(made):
            instance.field = index
        return (made[0], made[1:])

    def calls(context, earlier, later):
        class Tally(int):
            def __add__(self, other):
                return int(self) + other

        def tallied():
            context._truncation["max_items"] = Tally(0)
            return context.value(later)

        def namespaces():
            held = [earlier, *later]
            return (
                [id(vars(instance)) in context._namespaces for instance in held],
                len(context._namespaces),
            )

        return [lambda: context.value(earlier), tallied, namespaces]

    native, _ = _same_in_both_modes(build, DumpOptions(max_items=2), calls)
    assert native.results[2] == ([True, True, True, False], 3)


# --- A class no ``__init__`` judged -------------------------------------------


def _unjudged(log):
    """A context class created past ``ValuesMixin.__init_subclass__``, by a
    base ahead of it whose own does not call ``super()``, that overrides a
    walk method: nothing judged it, and it inherits ``_native_walk``."""

    class Registering:
        def __init_subclass__(cls, **kwargs):
            pass

    class Masking(Registering, SnapshotContext):
        def _text(self, value):
            log.append(value)
            return "<masked>"

    assert Masking._native_walk_class is not Masking
    return Masking


@pytest.mark.parametrize("how", ["__class__", "object.__new__", "copy"])
def test_a_class_no_init_judged_captures_in_python(how):
    """An instance that reaches a class no ``__init__`` judged, by assigning
    ``__class__``, through ``object.__new__`` or as a copy, captures in
    Python, which calls the class's own walk method: the walker's entries
    find the class was not judged and give the context to the Python walk
    before anything of the class runs."""
    data = {"key": ["secret", "x" * 20], "other": [1, (2,)]}
    outcomes = {}
    for native in (False, True):
        log = []
        Masking = _unjudged(log)
        with _mode(native):
            plain = SnapshotContext(DumpOptions(max_string_bytes=8))
        if how == "object.__new__":
            context = object.__new__(Masking)
            vars(context).update(vars(plain))
        else:
            plain.__class__ = Masking
            context = plain if how == "__class__" else copy.copy(plain)
        assert context._native_walk is native
        before = _fallbacks()
        results = (context.value(data), context._mapping({"k": ["more"]}))
        state = (results, list(context._truncation.items()), context._nodes)
        outcomes[native] = (pickle.dumps(state, protocol=5), list(log))
        assert context._native_walk is False
        assert Masking._native_walk_class is not Masking
        if native:
            assert _fallbacks() == before + 1
    assert outcomes[True] == outcomes[False]
    assert "secret" in outcomes[True][1]


_UNJUDGED_MODULE = """
from pydumpling.snapshot import SnapshotContext

LOG = []


class Registering:
    def __init_subclass__(cls, **kwargs):
        pass


class Masking(Registering, SnapshotContext):
    def _text(self, value):
        LOG.append(value)
        return "<masked>"
"""

_UNPICKLED_SCRIPT = """
import json, pickle, sys
sys.path.insert(0, sys.argv[1])
import masking
from pydumpling.options import DumpOptions
from pydumpling.snapshot import values

if sys.argv[2] == "write":
    with open(sys.argv[3], "wb") as file:
        pickle.dump(masking.Masking(DumpOptions(max_string_bytes=8)), file)
    raise SystemExit
with open(sys.argv[3], "rb") as file:
    context = pickle.load(file)
judged = vars(masking.Masking).get("_native_walk_class") is masking.Masking
offered = context._native_walk
captured = context.value({"key": ["secret", "x" * 20], "other": [1, (2,)]})
print(json.dumps({
    "native": values._native is not None,
    "judged": judged,
    "offered": offered,
    "captured": repr(captured),
    "log": masking.LOG,
    "truncation": list(context._truncation.items()),
    "nodes": context._nodes,
}))
"""


def test_a_context_unpickled_in_a_fresh_process_of_a_class_no_init_judged(
    tmp_path,
):
    """A context pickled in one process and loaded in another, where its
    class was created past ``ValuesMixin.__init_subclass__`` and no
    ``__init__`` judged it, captures there as in the Python walk."""
    (tmp_path / "masking.py").write_text(_UNJUDGED_MODULE)
    blob = tmp_path / "context.pickle"
    found = {}
    for mode in ("write", "1", "0"):
        environment = _child_environment()
        environment["PYDUMPLING_NATIVE"] = "1" if mode == "write" else mode
        process = subprocess.run(
            [sys.executable, "-B", "-c", _UNPICKLED_SCRIPT]
            + [str(tmp_path), mode, str(blob)],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        assert process.returncode == 0, process.stderr
        if mode != "write":
            found[mode] = json.loads(process.stdout)
    native, pure = found["1"], found["0"]
    assert native["native"] and native["offered"] and not native["judged"]
    assert pure["native"] is False
    for key in ("captured", "log", "truncation", "nodes"):
        assert native[key] == pure[key], key
    assert "secret" in native["log"]


# --- Inserts that compare with application keys -------------------------------


def _capturing_name(hashed, holder, log):
    """A key of another kind that hashes like ``hashed``: its ``__eq__``,
    which an insert or lookup of an equal-hashing key runs, captures
    through the context in ``holder`` once armed, and logs the count it
    found. It holds no attributes, so a capture of it holds no context."""

    class CapturingName:
        def __hash__(self):
            return hash(hashed)

        def __eq__(self, other):
            context = holder[0]
            if context is None:
                return False
            log.append(("enter", type(other).__name__, context._nodes))
            context.value([[10], [11]])
            log.append(("leave", context._nodes))
            return False

        def __reduce__(self):
            # A slot name stays the copy's key itself.
            return (str, ("capturing name",))

    return CapturingName()


class _SlotList(list):
    __slots__ = ("a",)


def _colliding_attribute(holder, log):
    """A snapshot copy captured again whose attribute dict holds a key of
    another kind and then a plain name that hashes like it."""
    copy_ = FakeClass("<copy>", {})
    vars(copy_)[_capturing_name("zz", holder, log)] = 5
    vars(copy_)["zz"] = 6
    return copy_


def _colliding_slot(holder, log):
    """A list subclass with a slot under a name of another kind that hashes
    like the copy's type key, which goes into the copy's namespace last."""
    name = _capturing_name(FakeList._type_key, holder, log)
    cls = type("Odd", (_SlotList,), {"__slots__": (), name: _SlotList.a})
    instance = cls([1])
    instance.a = [2]
    return instance


@pytest.mark.filterwarnings("ignore:non-string key:RuntimeWarning")
@pytest.mark.parametrize("where", ["attribute dict", "slot name"])
@pytest.mark.parametrize("nodes", [1000, 16])
def test_an_insert_that_compares_with_an_application_key(where, nodes):
    """Where the walk inserts a plain key into a dict that holds a key of
    another kind, hashing alike runs that key's ``__eq__``. The walker runs
    it as application code, with ``_nodes`` synced around it, so a capture
    from there counts toward the budget as in the Python walk. (A key
    planted in the context's own ``_layouts`` is not synced:
    ``test_native_differences`` pins it.)"""
    outcomes = {}
    for native in (False, True):
        log, holder = [], [None]
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=nodes))
        if where == "attribute dict":
            target = _colliding_attribute(holder, log)
        else:
            target = _colliding_slot(holder, log)
        holder[0] = context
        captured = context.value([[1], [2], target, [3], [4]])
        holder[0] = None
        state = (
            captured,
            list(context._truncation.items()),
            context._nodes,
            len(context.objects),
        )
        outcomes[native] = (pickle.dumps(state, protocol=5), log)
    assert outcomes[True] == outcomes[False]
    assert outcomes[True][1], "the key's __eq__ never ran"
