"""Seeded A/B parity between the native walker and the pure-Python walk.

Every seed draws random ``DumpOptions``, builds a random object graph and call
stack, and runs one random sequence of capture operations twice on the same
live objects: on a context served by ``pydumpling._native`` and on a pure
Python one.  The pickled results, the truncation counters in key order, and
the node count must match byte for byte, and the native walk must run exactly
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

import builtins
import contextlib
import copy
import copyreg
import functools
import gc
import gzip
import inspect
import os
import pickle
import pickletools
import random
import subprocess
import sys
import threading
import types
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
    """Python 3.15+ writes an int too long to convert as a placeholder with its
    address, so an exception message would differ between any two captures."""
    if sys.version_info < (3, 15):
        yield
        return
    saved = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(saved)


# --- Random options and values ---------------------------------------------

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


def _describe(op):
    kind, *arguments = op
    target = type(arguments[0]).__name__ if arguments else ""
    if kind in ("value", "mapping", "attributes"):
        details = ", ".join(repr(argument) for argument in arguments[1:])
        return f"{kind}({target}, {details})"
    return f"{kind}({target})"


# --- Running both walks ----------------------------------------------------


@dataclass
class _Outcome:
    results: list
    truncation: list
    nodes: int
    hooks: list
    blob: bytes = b""

    def __post_init__(self):
        try:
            self.blob = pickle.dumps(
                (self.results, self.truncation, self.nodes), protocol=5
            )
        except Exception as error:
            self.blob = f"unpicklable: {type(error).__name__}: {error}".encode()


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
    return _Outcome(results, list(context._truncation.items()), context._nodes, hooks)


# --- Diagnosis -------------------------------------------------------------

# Pickle writes these by value, never as a memo reference to an earlier copy.
_UNSHARED = (int, float, bool, type(None))
_BY_REFERENCE = (
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.ModuleType,
    types.MethodDescriptorType,
)


def _brief(value):
    """A short description that never runs a huge or recursive repr."""
    cls = type(value)
    if isinstance(value, (str, bytes, bytearray)):
        more = "..." if len(value) > 24 else ""
        return f"{value[:24]!r}{more} ({cls.__name__} of {len(value)})"
    if cls is int and value.bit_length() > 64:
        return f"<int of {value.bit_length()} bits>"
    if cls in (int, float, bool, complex, type(None), TruncatedValue):
        return repr(value)
    if cls is FakeClass:
        return f"FakeClass {str(value.__dict__.get('_FakeClass__repr'))[:40]!r}"
    if cls in (list, tuple, dict, set, frozenset) or issubclass(cls, FakeContainer):
        return f"{cls.__name__} of {len(value)}"
    return f"<{cls.__name__}>"


def _same_atom(left, right):
    if type(left) is float:
        return repr(left) == repr(right)
    if type(left) is complex:
        return (repr(left.real), repr(left.imag)) == (
            repr(right.real),
            repr(right.imag),
        )
    return left == right


def _parts(value, keep):
    """The labelled parts pickle writes for ``value``, in its order."""
    cls = type(value)
    if cls is list or cls is tuple:
        return [(f"[{index}]", item) for index, item in enumerate(value)]
    if cls is set or cls is frozenset:
        return [(f"{{#{index}}}", item) for index, item in enumerate(value)]
    if cls is dict:
        parts = []
        for index, (key, item) in enumerate(value.items()):
            parts.append((f"<key #{index}>", key))
            label = repr(key) if type(key) is str and len(key) <= 40 else _brief(key)
            parts.append((f"[{label}]", item))
        return parts
    reducer = copyreg.dispatch_table.get(cls)
    reduced = reducer(value) if reducer is not None else value.__reduce_ex__(5)
    keep.append(reduced)
    if isinstance(reduced, str):
        return []
    labels = (".<callable>", ".<args>", ".<state>", ".<items>", ".<entries>")
    parts = []
    for label, part in zip(labels, reduced):
        if label in (".<items>", ".<entries>") and part is not None:
            part = list(part)
            keep.append(part)
        parts.append((label, part))
    return parts


def _first_difference(roots):
    """The first path where two captures differ in type, value or sharing.

    ``roots`` holds ``(label, native, pure)`` triples, walked in order with
    one sharing map: pickle writes a second reference to an object, a str
    included, as a memo lookup, so objects must pair one to one across the
    two sides, between the results and the truncation keys as well.
    """
    keep = []
    pairs = {}
    reverse = {}
    paths = {}
    pending = list(reversed(roots))
    while pending:
        path, a, b = pending.pop()
        if type(a) is not type(b):
            return f"{path}: type {_brief(a)} != {_brief(b)}"
        if type(a) in _UNSHARED:
            if not _same_atom(a, b):
                return f"{path}: {_brief(a)} != {_brief(b)}"
            continue
        if id(a) in pairs or id(b) in reverse:
            if pairs.get(id(a)) != id(b) or reverse.get(id(b)) != id(a):
                return (
                    f"{path}: identity sharing differs: native object first seen "
                    f"at {paths.get(('a', id(a)), 'nowhere')}, pure object at "
                    f"{paths.get(('b', id(b)), 'nowhere')}"
                )
            continue
        pairs[id(a)] = id(b)
        reverse[id(b)] = id(a)
        paths[("a", id(a))] = paths[("b", id(b))] = path
        keep.extend((a, b))
        # One object on both sides still pairs its parts with themselves.
        if isinstance(a, _BY_REFERENCE):
            if a is not b:
                return f"{path}: {a!r} is not {b!r}"
            continue
        if type(a) in (str, bytes, bytearray, complex):
            if not _same_atom(a, b):
                return f"{path}: {_brief(a)} != {_brief(b)}"
            continue
        left_parts = _parts(a, keep)
        right_parts = _parts(b, keep)
        if len(left_parts) != len(right_parts):
            return (
                f"{path}: {_brief(a)} has {len(left_parts)} parts, "
                f"{_brief(b)} has {len(right_parts)}"
            )
        for (label, x), (other, y) in reversed(list(zip(left_parts, right_parts))):
            if label != other:
                return f"{path}: part {label} != {other}"
            pending.append((path + label, x, y))
    return None


def _pickle_window(blob, offset):
    """The pickle opcodes around ``offset``."""
    lines = []
    try:
        for opcode, argument, position in pickletools.genops(blob):
            if position is not None and offset - 120 <= position <= offset + 40:
                shown = "" if argument is None else _brief(argument)
                lines.append(f"    {position:7}: {opcode.name} {shown}")
    except Exception as error:
        lines.append(f"    <{type(error).__name__}: {error}>")
    return lines[-12:]


def _diagnosis(case, native, pure):
    lines = [
        f"native and pure walks differ for seed {case.seed}",
        f"options: {case.options}",
        "operations: " + ", ".join(map(_describe, case.ops)),
    ]
    if case.margin is not None:
        lines.append(f"run {case.margin} frames below the recursion limit")
    difference = _first_difference(
        [
            ("results", native.results, pure.results),
            ("truncation", native.truncation, pure.truncation),
            ("nodes", native.nodes, pure.nodes),
        ]
    )
    if difference is not None:
        lines.append("first difference (native vs pure): " + difference)
    elif native.blob != pure.blob:
        offset = next(
            (
                index
                for index, (x, y) in enumerate(zip(native.blob, pure.blob))
                if x != y
            ),
            min(len(native.blob), len(pure.blob)),
        )
        lines.append(f"no structural difference; pickles first differ at {offset}")
        for label, outcome in (("native", native), ("pure", pure)):
            lines.append(f"  {label} pickle opcodes:")
            lines.extend(_pickle_window(outcome.blob, offset))
    if native.hooks != pure.hooks:
        lines.append(f"application hooks ran: native {native.hooks}, pure {pure.hooks}")
    lines.append(f"truncation native: {native.truncation}")
    lines.append(f"truncation pure:   {pure.truncation}")
    lines.append(f"nodes native={native.nodes} pure={pure.nodes}")
    lines.append(
        f"reproduce: PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', '<unset>')} "
        f"pytest {__file__} -k 'test_walk_parity and [{case.seed}]'"
    )
    return "\n".join(lines)


def _assert_parity(case, native, pure):
    if native.blob != pure.blob or native.hooks != pure.hooks:
        raise AssertionError(_diagnosis(case, native, pure))


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
    """A broken comparison must not pass silently."""
    for seed in (0, 9, 13):
        case = _case(seed)
        try:
            first = _run(case, native=False)
            second = _run(case, native=False)
        finally:
            case.close()
        _assert_parity(case, first, second)
        second.results.append("extra")
        with pytest.raises(AssertionError, match="results: .* has"):
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
        (_Outcome(changed, same, 3, []), r"results\[0\]\[2\]\['key'\]: 1.5 != 2.5"),
        (_Outcome(unshared, same_too, 3, []), r"results\[0\]\[1\]: identity sharing"),
        (_Outcome(retyped, also_same, 3, []), r"results\[1\]: type"),
        (_Outcome(copied, reordered[::-1], 3, []), r"truncation\[0\]\[0\]"),
        (_Outcome(*copy(), 4, []), "nodes: 3 != 4"),
        # The reason a placeholder shares with the counter's key.
        (
            _Outcome(results, [("".join(("max_", "depth")), 1), truncation[1]], 3, []),
            "sharing",
        ),
        (_Outcome(*copy(), 3, ["__repr__"]), "application hooks ran"),
    ]
    for outcome, message in perturbations:
        with pytest.raises(AssertionError, match=message):
            _assert_parity(case, original, outcome)


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
    if native[1] is not None and pure[1] is not None:
        left, right = dill.loads(native[1]), dill.loads(pure[1])
        lines.append(
            "first difference (native vs pure): "
            + str(_first_difference([("payload", left, right)]))
        )
        for label, payload in (("native", left), ("pure", right)):
            metadata = payload.get("snapshot_meta", {})
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
                results, list(context._truncation.items()), context._nodes, list(hooks)
            )
    native, pure = outcomes[True], outcomes[False]
    assert native.hooks == pure.hooks
    assert native.blob == pure.blob, (native.results, pure.results)
    return native, pure


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
    """A subclass overriding a walk method captures in Python, which calls it."""
    with _mode(True):
        redacting = _Redacting(DumpOptions())
        assert vars(redacting).get("_walker") is None
        assert redacting.value({"key": ["secret"]}) == {"<redacted>": ["<redacted>"]}
        assert type(vars(_Composed(DumpOptions())).get("_walker")).__name__ == "Walker"


def _child_environment():
    """The environment of a child interpreter that imports this checkout."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(values_module.__file__)))
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


class _Colliding(str):
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
        odd = type("Odd", (), {_Colliding("tag"): 1, _Key("other"): 2})
        big = type("Big", (int,), {_Colliding("tag"): 1})
        space = type("Space", (dict,), {_Colliding("tag"): 1})
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


def test_copy_of_a_capturing_context():
    """A copy of a native context captures in Python from the walker's memo.

    The Python walk's copy shares its ``objects`` dict with the original, so
    later captures on the original see the copy's; the native original does
    not, so only the copy's own captures are compared.
    """

    def build():
        shared = [1, [2], {"three": 3}]
        return (shared, {"again": shared})

    def calls(context, shared, holder):
        def copied():
            twin = copy.copy(context)
            assert vars(twin).get("_walker") is None
            deep = copy.deepcopy(context)
            return (twin.value(holder), twin._nodes, deep.value([shared]), deep._nodes)

        return [lambda: context.value(shared), copied]

    _same_in_both_modes(build, DumpOptions(), calls)


def test_frame_namespaces_stay_native():
    """A frame's own locals are read natively, without a hand-back."""
    with _mode(True):
        context = SnapshotContext(DumpOptions())
        local = {"nested": [1, 2, 3]}  # noqa: F841
        context.from_frame(sys._getframe())
        assert context._walker.handbacks == 0


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
from pydumpling.fake_types import FakeDict
from pydumpling.options import DumpOptions as NewOptions
from pydumpling.snapshot import SnapshotContext as New

class Space(dict):
    pass

new = New(NewOptions(max_depth=None))
old = Old(DumpOptions(max_depth=None))
assert type(vars(new)["_walker"]).__name__ == "Walker"
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


def test_memo_entries_written_between_calls():
    """Copies Python memoizes in ``objects`` between calls are adopted in order."""

    def build():
        withheld = {"secret": [1]}
        return (withheld, {"cfg": withheld, "other": [2]})

    def calls(context, withheld, holder):
        def seed():
            context.objects[id(withheld)] = "<withheld>"
            context._sources[id(withheld)] = withheld
            return context.value(holder)

        return [lambda: context.value([3]), seed, lambda: context.value(withheld)]

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


_STATE_CHANGES = {
    "truncation dict": lambda c: setattr(c, "_truncation", {}),
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
            lambda: context.value(nested),
            lambda: context._mapping({"k": nested}, 0),
        ]

    _same_in_both_modes(build, DumpOptions(max_nodes=40, max_items=8), calls)


@pytest.mark.parametrize("change", ["set to None", "deleted", "moved", "objects reset"])
def test_walker_removed_or_moved_keeps_its_copies(change):
    """A walker taken off its context, or moved to another one, leaves every
    copy it made in ``objects``, as the Python walk writes each at once."""

    def build():
        return (["a", ["b"], {"c": 1}],)

    def calls(context, shared):
        def switch():
            # Nothing else may hold a removed walker: it writes its copies
            # when it is released.
            if change == "set to None":
                context._walker = None
            elif change == "deleted":
                vars(context).pop("_walker", None)
            elif change == "moved":
                other = SnapshotContext(context.options)
                if "_walker" in vars(context):
                    other._walker = context._walker
                return other.value([shared]), other._nodes
            else:
                context.objects = {}
            return context.value({"again": shared})

        return [
            lambda: context.value({"first": shared}),
            switch,
            lambda: context.value({"third": shared}),
            lambda: context.value(shared) is context.value(shared),
        ]

    _same_in_both_modes(build, DumpOptions(max_nodes=12), calls)


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
            assert vars(context).get("_walker") is None
            results[native] = pickle.dumps(
                (context.value(data), dict(context._truncation), context._nodes)
            )
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
                    assert context._walker.handbacks == 0
                outcomes[native] = pickle.dumps(
                    (captured, dict(context._truncation), context._nodes), protocol=5
                )
        assert outcomes[True] == outcomes[False]
    finally:
        sys.setrecursionlimit(saved)


_AUDIT_SCRIPT = """
import os, pickle, sys
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
        print("walker", type(walker).__name__, walker.handbacks)
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
    assert process.stdout.split() == ["walker", "Walker", "0", "same"]


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


def _inside_the_walk(context):
    """Whether a native walk on ``context`` is running on this thread."""
    try:
        context._walker.handbacks
    except RuntimeError:
        return True
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
    """A collection that captures on the same context during a native walk
    takes the Python walk, and the outer capture is unchanged."""
    graph = [[i, str(i)] for i in range(20000)]
    results = {}
    for native in (True, False):
        with _mode(native):
            context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
            _check_walker(context, native)
        inner, inside = [], []

        def callback(phase, info, context=context, inner=inner, inside=inside):
            if phase == "start" and len(inner) < 3:
                inside.append(native and _inside_the_walk(context))
                inner.append(context.value(["inner", 2.5]))

        with _collecting(callback):
            captured = context.value(graph)
        assert inner and all(copy == ["inner", 2.5] for copy in inner)
        if native:
            assert any(inside)
        results[native] = pickle.dumps(captured, protocol=5)
    assert results[True] == results[False]


def test_another_thread_capturing_the_same_context():
    """A second thread may not capture on a context mid-walk."""
    graph = [[i, str(i)] for i in range(20000)]
    with _mode(True):
        context = SnapshotContext(DumpOptions(max_nodes=None, max_items=None))
    entered, done = threading.Event(), threading.Event()
    errors = []

    def callback(phase, info):
        if phase == "start" and not entered.is_set() and _inside_the_walk(context):
            entered.set()
            done.wait(30)

    def other():
        entered.wait(30)
        try:
            context.value([1])
        except RuntimeError as error:
            errors.append(str(error))
        finally:
            done.set()

    thread = threading.Thread(target=other)
    thread.start()
    with _collecting(callback):
        captured = context.value(graph)
    thread.join()
    assert len(captured) == len(graph)
    assert errors and "already capturing" in errors[0]


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
