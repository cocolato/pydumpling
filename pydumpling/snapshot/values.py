"""Bounded capture of application values: containers, scalars, attributes."""

import builtins
import itertools
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
from .identity import _CLASS_DICT, _derives, _type_mro

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
                else TruncatedValue(
                    "recursive immutable reference", self._type_name(value)
                ),
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
            chosen = dict.fromkeys(
                id(key)
                for key in itertools.islice(keys, None if limit is None else limit + 1)
            )
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
