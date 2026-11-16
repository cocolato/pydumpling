"""Bounded capture of application values: containers, scalars, attributes."""

import builtins
import itertools
import os
import sys
import types
from collections import OrderedDict

from ..fake_types import (
    FakeBytes,
    FakeClass,
    FakeDict,
    FakeFloat,
    FakeFrozenSet,
    FakeInt,
    FakeList,
    FakeSet,
    FakeStr,
    FakeTuple,
    TruncatedValue,
)
from ..options import DumpOptions  # noqa: F401  # the native walker's table reads it
from . import identity as _identity
from .identity import _CLASS_DICT, _CLASS_MRO, _derives, _type_mro

# Each builtin container base maps to the copy type of its exact instances and
# the snapshot subclass for instances of its subclasses.
_CONTAINER_TYPES = [
    (dict, dict, FakeDict),
    (list, list, FakeList),
    (tuple, tuple, FakeTuple),
    (set, set, FakeSet),
    (frozenset, frozenset, FakeFrozenSet),
]
if hasattr(builtins, "frozendict"):
    # Python 3.15+.  frozendict is not a dict subclass, and every copy is a
    # FakeDict so a snapshot taken on 3.15 still loads on older Pythons.
    _CONTAINER_TYPES.append((builtins.frozendict, FakeDict, FakeDict))
_CONTAINER_BASES = {id(entry[0]): entry for entry in _CONTAINER_TYPES}
# Exact scalars are kept as they are; each scalar base maps to the snapshot
# subclass for instances of its subclasses, such as IntEnum or StrEnum members.
_SCALAR_COPIES = {
    id(copy_type._base): copy_type
    for copy_type in (FakeStr, FakeBytes, FakeInt, FakeFloat)
}

# Key types whose hashing and comparison run only builtin code.
_PLAIN_KEY_TYPES = frozenset(
    map(id, (str, int, float, complex, bool, bytes, types.NoneType))
)


def _plain_key(key):
    """Whether hashing or comparing ``key`` runs no application code."""
    if type(key) is tuple:
        return all(id(type(item)) in _PLAIN_KEY_TYPES for item in key)
    return id(type(key)) in _PLAIN_KEY_TYPES


def _key_order(source, view, ordered):
    """A mapping's keys in its own order."""
    # move_to_end changes an OrderedDict's link order but not dict.items.
    # CPython's OrderedDict iterator hashes every key it passes, so callers
    # request link order only for plain keys.
    return OrderedDict.__iter__(source) if ordered else (key for key, _ in view)


def _container_base(value):
    """The builtin container ``value`` stores entries in, and its copy types."""
    for cls in _type_mro(value):
        entry = _CONTAINER_BASES.get(id(cls))
        if entry is not None:
            return entry
    return None, None, None


def _scalar_copy_type(value):
    """The snapshot subclass for a str, bytes, int, or float subclass instance."""
    for cls in _type_mro(value):
        copy_type = _SCALAR_COPIES.get(id(cls))
        if copy_type is not None:
            return copy_type
    return None


# The value memo legitimately stores None; only this sentinel means "new".
_UNMEMOIZED = object()


class ValuesMixin:
    """Value capture shared by the composed :class:`SnapshotContext`."""

    # Whether ``value`` and ``_mapping`` offer a call to the native walker,
    # and the class that was chosen for (see below the class).
    _native_walk = False
    _native_walk_class = None

    def _layout(self, value):
        """Member descriptors and named tuple fields of ``type(value)``, cached."""
        cls = type(value)
        layout = self._layouts.get(id(cls))
        if layout is None:
            members = []
            fields = ()
            seen = set()
            for owner in _type_mro(value):
                for name, member in _CLASS_DICT.__get__(owner, type(owner)).items():
                    # The first definition in the MRO wins, as in attribute
                    # lookup, so a subclass property hides a base slot.
                    if name in seen:
                        continue
                    seen.add(name)
                    if type(member) is types.MemberDescriptorType:
                        members.append((name, member))
                    elif name == "_fields" and type(member) is tuple:
                        if all(type(field) is str for field in member):
                            fields = member
            # Keeping the type alive keeps its id from being reused.
            layout = self._layouts[id(cls)] = (cls, members, fields)
        return layout[1], layout[2]

    def _attribute_mapping(self, attrs, depth, *, priority=(), private=None):
        """Capture an attribute dict, whose keys are names, not string data.

        ``private`` is the bookkeeping attribute of a snapshot copy captured
        again; its caller restores it, so it is not converted as data.
        """
        if private is not None:
            attrs = {
                key: item
                for key, item in dict.items(attrs)
                if type(key) is not str or key != private
            }
        self._namespaces.add(id(attrs))
        return self._mapping(attrs, depth, priority=priority)

    def _fill_copy(self, value, copy, depth):
        """Keep a subclass's type name and attributes beside its entries or value.

        Instance attributes, filled slots, struct sequence fields outside the
        tuple part, and named tuple fields all become attributes of the copy.
        """
        namespace = copy.__dict__
        cls = type(value)
        own = cls is type(copy)
        key = type(copy)._type_key
        attrs = self._attributes(value) or {}
        type_name = attrs.get(key) if own else self._type_name(value)
        limit = self.options.max_depth
        if limit is not None and depth >= limit:
            # Only a scalar copy gets here: its value is kept at any depth,
            # like a plain value, but its attributes and slots are nested
            # data and follow the depth budget.
            # Only an exact str name is compared: a str subclass attribute name
            # could run its own ``__ne__``, so any other name counts as data.
            if any(
                type(name) is not str or name != key for name in attrs
            ) or self._filled_slot(value):
                self._truncation["max_depth"] += 1
            namespace[key] = type_name
            return
        if attrs:
            namespace.update(
                self._attribute_mapping(attrs, depth + 1, private=key if own else None)
            )
        members, fields = self._layout(value)
        for name, member in members:
            try:
                item = member.__get__(value, cls)
            except Exception:
                continue  # An unset slot holds nothing.
            if (
                self.options.max_items is not None
                and len(namespace) >= self.options.max_items
            ):
                self._truncation["max_items"] += 1
                break
            if not self._has_nodes():
                self._truncation["max_nodes"] += 1
                break
            namespace[name] = self.value(item, depth + 1)
        if fields and type(copy) is FakeTuple:
            namespace.update(zip(fields, copy))
        namespace[key] = type_name

    def _filled_slot(self, value):
        """Whether any slot of ``value`` holds a value; runs no application code."""
        cls = type(value)
        for _, member in self._layout(value)[0]:
            try:
                member.__get__(value, cls)
            except Exception:
                continue
            return True
        return False

    def value(self, value, depth=0):
        cached = self.objects.get(id(value), _UNMEMOIZED)
        if cached is not _UNMEMOIZED:
            return cached
        if not self._has_nodes():
            self._truncation["max_nodes"] += 1
            return self._node_marker
        self._nodes += 1
        value_type = type(value)
        if value_type is TruncatedValue:
            return self._remember(value, value)
        if (
            value_type is types.NoneType
            or value_type is bool
            or value_type is float
            or value_type is complex
        ):
            return self._remember(value, value)
        if value_type is int:
            limit = self.options.max_string_bytes
            if limit is not None and value.bit_length() > limit * 8:
                return self._remember(value, self._omit("max_string_bytes", value))
            return self._remember(value, value)
        if value_type is str:
            return self._remember(value, self._text(value))
        if value_type is bytes or value_type is bytearray:
            return self._remember(
                value,
                self._bounded_bytes(
                    value, self.options.max_string_bytes, "max_string_bytes"
                ),
            )
        if self._unlimited:
            if self._serializable(value):
                return self._remember(value, value)
            self._truncation["unserializable"] += 1
        elif self._native_walk:
            # A C call, in this frame: no Python frame is added, so wherever
            # the call goes on below, it has the room this walk would have.
            # The exact scalars above cost this walk less than the entry
            # does; an unlimited context is never offered.
            captured = _native_value(self, value, depth)
            if captured is not _IN_PYTHON:
                return captured
        copy_type = _scalar_copy_type(value)
        if copy_type is not None:
            # Like an exact scalar, the value is kept at any depth.
            return self._scalar(value, copy_type, depth)
        limit = self.options.max_depth
        if limit is not None and depth >= limit:
            return self._remember(value, self._omit("max_depth", value))
        base, plain, snapshot_type = _container_base(value)
        # A subclass keeps its entries in a snapshot subclass of its base,
        # with its type name and attributes alongside.
        target = plain if value_type is base else snapshot_type
        if snapshot_type is FakeDict:  # dict, and frozendict on Python 3.15+
            captured = self._mapping(value, depth, counted=True, container=target)
            if target is snapshot_type:
                self._fill_copy(value, captured, depth)
            return captured
        if base is not None:
            sequence = base is list or base is tuple
            mutable = base is list or base is set
            captured = target() if mutable else ([] if sequence else set())
            self._remember(
                value,
                captured
                if mutable
                else TruncatedValue(_RECURSIVE_IMMUTABLE, self._type_name(value)),
            )
            count = 0
            # Iterate through the base type: a subclass ``__iter__`` must be
            # neither executed nor able to hide entries.
            for item in base.__iter__(value):
                if (
                    self.options.max_items is not None
                    and count >= self.options.max_items
                ):
                    self._truncation["max_items"] += 1
                    break
                if not self._has_nodes() and id(item) not in self.objects:
                    self._truncation["max_nodes"] += 1
                    break
                converted = self.value(item, depth + 1)
                captured.append(converted) if sequence else captured.add(converted)
                count += 1
            if not mutable:
                captured = target(captured)
                self._remember(value, captured)
            if target is snapshot_type:
                self._fill_copy(value, captured, depth)
            return captured
        representation = (
            FakeClass.__repr__(value)
            if value_type is FakeClass
            else "<%s snapshot>" % self._type_name(value)
        )
        captured = FakeClass(representation, {})
        self._remember(value, captured)
        attrs = self._attributes(value)
        if attrs is not None:
            converted = self._attribute_mapping(
                attrs,
                depth + 1,
                private="_FakeClass__repr" if value_type is FakeClass else None,
            )
            captured.__dict__.update(converted)
            captured.__dict__["_FakeClass__repr"] = representation
        else:
            self._truncation["unsupported_type"] += 1
        return captured

    def _scalar(self, value, copy_type, depth):
        """Copy a str, bytes, int, or float subclass instance and its attributes.

        The value is read through the base type, so no application
        ``__getitem__``, ``__len__``, ``__int__``, or ``bit_length`` runs, and it
        has the budget of an exact instance.
        """
        base = copy_type._base
        limit = self.options.max_string_bytes
        if base is int:
            if limit is not None and int.bit_length(value) > limit * 8:
                return self._remember(value, self._omit("max_string_bytes", value))
            plain = int.__int__(value)
        elif base is float:
            plain = float.__float__(value)
        else:
            # The base slice copies at most one item past the budget into an
            # exact instance: enough to see the budget was exceeded.
            head = base.__getitem__(value, slice(None if limit is None else limit + 1))
            plain = (
                self._text(head)
                if base is str
                else self._bounded_bytes(head, limit, "max_string_bytes")
            )
        captured = self._remember(value, copy_type(plain))
        self._fill_copy(value, captured, depth)
        return captured

    @staticmethod
    def _attributes(value):
        # Call only Python's real instance-dictionary descriptor.  In
        # particular, never invoke an application property named __dict__.
        for cls in _type_mro(value):
            descriptor = _CLASS_DICT.__get__(cls, type(cls)).get("__dict__")
            if type(descriptor) is types.GetSetDescriptorType:
                try:
                    attrs = descriptor.__get__(value, type(value))
                    return attrs if type(attrs) is dict else None
                except Exception:
                    return None
            if descriptor is not None:
                return None
        return None

    def _mapping(self, source, depth=0, *, priority=(), counted=False, container=dict):
        if self._native_walk and self._unlimited is False:
            captured = _native_mapping(
                self, source, depth, priority, counted, container
            )
            if captured is not _IN_PYTHON:
                return captured
        if id(source) in self.objects:
            cached = self.objects[id(source)]
            if type(cached) is dict or type(cached) is FakeDict:
                return cached
            # A mapping root may fit even when a deeper reference was omitted.
            if not isinstance(cached, TruncatedValue) or cached.reason != "max_depth":
                self._truncation["unsupported_type"] += 1
                return {}
        if not counted:
            if not self._has_nodes():
                self._truncation["max_nodes"] += 1
                return {}
            self._nodes += 1
        if self.options.max_depth is not None and depth >= self.options.max_depth:
            self._truncation["max_depth"] += 1
            return self._remember(source, container())
        captured = self._remember(source, container())
        # Frame variable names are identifiers, not application string values,
        # so a string budget must not rewrite them.
        names = id(source) in self._namespaces
        count = 0
        preferred = set(priority)
        try:
            # The builtin base's items skip a subclass ``items`` override and
            # run no key hash or equality method.
            reader = _container_base(source)[0]
            view = reader.items(source) if reader is not None else source.items()
            ordered = _derives(source, OrderedDict)
            if ordered and not all(map(_plain_key, dict.keys(source))):
                # Reading link order would run a key's own __hash__; keep
                # insertion order instead and say so.
                ordered = False
                self._truncation["mapping_order"] += 1
            # Select the bounded keys first: preferred names first, then the
            # rest, each group in the mapping's own order.
            keys = (
                key
                for key in _key_order(source, view, ordered)
                if type(key) is not str or key != "__builtins__"
            )
            if preferred:
                keys = itertools.chain(
                    (
                        key
                        for key in _key_order(source, view, ordered)
                        if type(key) is str
                        and key in preferred
                        and key != "__builtins__"
                    ),
                    (
                        key
                        for key in keys
                        if type(key) is not str or key not in preferred
                    ),
                )
            limit, reason = self.options.max_items, "max_items"
            nodes = self.options.max_nodes
            if nodes is not None and not names and (limit is None or nodes < limit):
                # Each entry not already memoized costs a node, so the node
                # budget also bounds the selection and its temporary memory.
                limit, reason = nodes, "max_nodes"
            # One extra key lets the loop below report truncation.
            try:
                selected = itertools.islice(keys, None if limit is None else limit + 1)
            except ValueError:
                # islice refuses a stop past sys.maxsize; no mapping holds that
                # many keys.
                if type(limit) is not int or limit < sys.maxsize:
                    raise
                selected = keys
            chosen = dict.fromkeys(id(key) for key in selected)
            remaining = len(chosen)
            for key, item in view:
                if not remaining:
                    break
                if chosen.get(id(key), ()) is None:
                    chosen[id(key)] = (key, item)
                    remaining -= 1
            items = (entry for entry in chosen.values() if entry is not None)
            for key, item in items:
                if limit is not None and count >= limit:
                    self._truncation[reason] += 1
                    break
                if not self._has_nodes():
                    self._truncation["max_nodes"] += 1
                    break
                converted_key = (
                    self._code_text(key)
                    if names and type(key) is str
                    else self.value(key, depth + 1)
                )
                if depth:
                    converted_value = self.value(item, depth + 1)
                else:
                    converted_value = self._root_value(item)
                try:
                    captured[converted_key] = converted_value
                except TypeError:
                    self._truncation["unsupported_key"] += 1
                count += 1
        except RecursionError:
            # Not a concurrent mutation: let the enclosing root mapping, or
            # the writer's minimal fallback, handle the overflow.
            raise
        except RuntimeError:
            self._truncation["concurrent_mutation"] += 1
        return captured

    def _root_value(self, item):
        """Capture a root mapping's entry, recovering from a recursion overflow.

        Only a root mapping, such as a frame namespace, is far enough from the
        interpreter's limit to recover, so one value nested too deeply does
        not cost the rest of the frame.
        """
        mark = len(self.objects)
        try:
            return self.value(item, 1)
        except RecursionError:
            # Forget the copies the overflow left half built, so other
            # references capture those objects afresh.
            for identity in list(itertools.islice(self.objects, mark, None)):
                del self.objects[identity]
            return self._remember(item, self._omit("recursion", item))

    def __init_subclass__(cls, **kwargs):
        """Choose the walk of a new context class (``_choose_walk``)."""
        super().__init_subclass__(**kwargs)
        _choose_walk(cls)


# The native walker.  Kept below ``ValuesMixin``, whose methods read these
# names only when called: a line number past 256 costs a tracer such as
# coverage's C tracer an int allocation per line event, so this code above
# the class made the Python walk 9-14% slower under it (3.13, 3.14).

_native = None
if (
    sys.version_info >= (3, 12)
    and os.environ.get("PYDUMPLING_NATIVE") != "0"
    and not sys.flags.bytes_warning
):
    # The optional native walker; this module stays the reference walk.  The
    # extension is built for the stable ABI of Python 3.12 and later, so 3.11
    # never loads it.  ``PYDUMPLING_NATIVE=0`` at import leaves the module out
    # entirely.  So does ``python -b``, a flag fixed for the life of the
    # process: comparing bytes with str warns there, and a warning the
    # walker's own dict and set lookups and inserts raise is attributed to the
    # line that called the walker, not to the line of this walk that compares,
    # so filters and the once-per-location registry would treat it otherwise.
    # A free-threaded build is left out too: a stable-ABI extension does not
    # fit its object layout, and one built for it would run without the GIL
    # the walker relies on.  A build whose configuration does not load, as a
    # frozen application without its sysconfig data, captures in Python as
    # well.
    import sysconfig

    try:
        if not sysconfig.get_config_var("Py_GIL_DISABLED"):
            from .. import _native
    except ImportError:
        pass


# The reason of an immutable container's placeholder.  One shared object, so
# the native walker's placeholders pickle to the same bytes as these.
_RECURSIVE_IMMUTABLE = "recursive immutable reference"

# What a native entry returns when this call runs in the Python walk instead,
# in the frame that called it.
_IN_PYTHON = object()
# The walker's table and its native entries, bound below when the extension
# loads.
_STATICS = _native_value = _native_mapping = _native_stock = None

# The context methods the native walker carries out itself.  A context class
# that overrides any of them captures in Python, which calls the override.
_NATIVE_WALK = (
    "value",
    "_mapping",
    "_attribute_mapping",
    "_root_value",
    "_fill_copy",
    "_filled_slot",
    "_scalar",
    "_layout",
    "_attributes",
    "_remember",
    "_has_nodes",
    "_omit",
    "_type_name",
    "_bounded_text",
    "_bounded_bytes",
    "_text",
    "_code_text",
)


# How the Python walk reads and writes the context's own state, which the
# native walker reads and writes with the same attribute access, once per
# call: every hook of an instance's attribute access, ``__getattr__`` for an
# attribute that is missing included.  Attribute access never looks
# ``__dict__`` up, so a class whose ``__dict__`` entry is another class's, as
# a plain mixin gives it before 3.15, still captures natively.
_STATE_ACCESS = ("__getattribute__", "__getattr__", "__setattr__", "__delattr__")

# The context state the native walker reads or writes.  The stock class
# holds none of them, so attribute access finds them in the instance; a
# class attribute of one of these names, such as a property, runs at each
# read in the Python walk and at most once per call in the walker, so such a
# class captures in Python.
_NATIVE_STATE = (
    "options",
    "_code_text_limit",
    "_unlimited",
    "_truncation",
    "_namespaces",
    "objects",
    "_sources",
    "_nodes",
    "_node_marker",
    "_layouts",
)

# ``ValuesMixin._native_walk`` says whether ``value`` and ``_mapping`` offer
# a call to the native walker, which mirrors this module step for step on
# the context's own state.  True on the composed context class when the
# extension loaded (``_stock``), and on each subclass whose walk attributes
# are that class's; False on a subclass that overrides a walk method, an
# attribute hook or a state attribute, whose walk only this module runs, and
# on its subclasses.  ``_native_walk_class`` names the class it was chosen for.
# The walker's entries check it and choose the walk, at its first native
# call, of a class whose ``__init_subclass__`` chain never reached
# ``ValuesMixin``'s, so ``SnapshotContext.__init__`` checks nothing.  An
# instance set to False captures in Python: the walker sets it so for a
# context whose state or arguments it does not mirror.  Changes to
# a class after it was chosen are not seen.  An unlimited context, where
# probing each original with the serializer dominates the cost, is never
# offered to the walker (``value`` reads ``_unlimited`` first, ``_mapping``
# offers only ``_unlimited is False``), and nothing sets the flag on it: once
# any instance holds ``_native_walk`` of its own, the name joins the class's
# shared instance keys (3.13+), and every later load of the class's flag, on
# any context, misses its specialization, 3 to 6 ns more a load.

# The walk attributes of the context class the native walker stands in for,
# recorded by ``_stock``; None while the extension is not loaded.
_STOCK_WALK = None


def _walk_methods(cls):
    """The class attributes of ``cls`` the native walker relies on, or None.

    For each walk method, attribute access hook and state name, the class
    attribute an instance of ``cls`` finds, or None, read from the class
    dicts along the MRO without calling its ``__get__`` or any metaclass
    hook.  None if a class dict holds a key other than an exact str, whose
    own ``__eq__`` looking a name up there may run.
    """
    namespaces = [
        _CLASS_DICT.__get__(base, type(base))
        for base in _CLASS_MRO.__get__(cls, type(cls))
    ]
    if not all(type(key) is str for namespace in namespaces for key in namespace):
        return None
    return tuple(
        next((namespace[name] for namespace in namespaces if name in namespace), None)
        for name in _NATIVE_WALK + _STATE_ACCESS + _NATIVE_STATE
    )


def _stock(cls):
    """Make ``cls``, the composed context class, capture natively.

    Its walk attributes are what the native walker stands in for; each
    subclass is compared with them when it is created.  The walker's
    entries take the class itself as judged without looking it up.
    """
    global _STOCK_WALK
    if _native is not None:
        _STOCK_WALK = _walk_methods(cls)
        cls._native_walk = True
        cls._native_walk_class = cls
        _native_stock(cls)


def _choose_walk(cls):
    """Choose the walk of a context class, once.

    A class captures natively only if every walk attribute it finds is the
    one the composed class finds, compared by identity.  One that sets
    ``_native_walk`` to False itself keeps it, as its subclasses inherit it;
    any other value it sets is replaced.  ``_native_walk_class`` records
    that the class was judged.  Class dicts are read through
    ``type.__dict__``, and both are set through ``type``, so no hook of a
    metaclass, nor the value's own code, runs.  Returns whether the class
    keeps the walk it inherits, which the walker's entries, judging a class
    at its first native call, read as "go on natively".
    """
    own = _CLASS_DICT.__get__(cls, type(cls)).get("_native_walk", _UNMEMOIZED)
    walk = _walk_methods(cls) if _STOCK_WALK is not None else None
    stock = walk is not None and all(
        mine is theirs for mine, theirs in zip(walk, _STOCK_WALK)
    )
    if not stock:
        type.__setattr__(cls, "_native_walk", False)
    elif own is not _UNMEMOIZED and own is not False and own is not True:
        type.__setattr__(cls, "_native_walk", True)
    type.__setattr__(cls, "_native_walk_class", cls)
    return stock and own is not False


def _frame_locals_type():
    """A function's locals type, if the runtime permits frame inspection."""
    try:
        frame = sys._getframe()
    except Exception:
        # Audit policies can forbid frame access. Only the locals-proxy fast
        # path needs this type; other values can still be captured natively.
        return None
    try:
        return type(frame.f_locals)
    finally:
        del frame


if _native is not None:
    try:
        # The walker's table of this module's classes and of ``value``, which
        # it calls for a node it runs in this walk, so a reloaded or vendored
        # copy of the package has its own, its two entries, which ``value``
        # and ``_mapping`` call, and where ``_stock`` names the composed
        # context class.
        _STATICS = _native.Statics(globals(), _identity)
        _native_value, _native_mapping = _STATICS.value, _STATICS.mapping
        _native_stock = _STATICS.stock
    except Exception as error:
        # An extension built for another version of this module, which lacks
        # what it reads here or has other entries: the package it only
        # speeds up still imports.
        import warnings

        warnings.warn(
            "pydumpling's native walker does not match this version of "
            f"pydumpling and is not used ({error!r}); rebuild it with "
            "rust/build_native.py",
            RuntimeWarning,
            stacklevel=1,
        )
        _native = None
        _STATICS = _native_value = _native_mapping = _native_stock = None
