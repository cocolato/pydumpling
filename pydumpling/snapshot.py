"""Bounded, shared snapshots of real frames and application values."""

import builtins
import copy
import itertools
import os
import pickle
import reprlib
import sys
import traceback
import types
from collections import Counter, OrderedDict, deque
from dataclasses import asdict, replace

import dill

from .fake_types import (
    FakeClass,
    FakeCode,
    FakeDict,
    FakeFrame,
    FakeFrozenSet,
    FakeList,
    FakeSet,
    FakeTraceback,
    FakeTuple,
    TruncatedValue,
)
from .options import DumpOptions

_CLASS_DICT = type.__dict__["__dict__"]
_CLASS_MRO = type.__dict__["__mro__"]
_CLASS_MODULE = type.__dict__["__module__"]
_CLASS_QUALNAME = type.__dict__["__qualname__"]
# Code metadata (file names, function and class names, variable and name
# tables) is diagnostic structure, not application data. Truncating it yields
# frame headers that name the wrong file or function, so it has its own fixed
# bound instead of following ``max_string_bytes``.  Bytecode has the same
# problem but a much larger natural size, so it gets a separate bound; the
# ``outer`` reduction stage still drops it entirely when bytes are tight.
# ``DumpOptions.unlimited()`` disables both bounds.
_CODE_TEXT_LIMIT = 4096
_CODE_BYTES_LIMIT = 64 * 1024
# Types are compared by identity throughout.  ``isinstance`` falls back to an
# instance's ``__class__`` attribute, ``x in tuple`` calls a metaclass
# ``__eq__``, and a set lookup calls a metaclass ``__hash__``; application code
# can override any of them.
#
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
SNAPSHOT_FORMAT_VERSION = 1
_BUILTIN_EXCEPTIONS = {
    id(value): value
    for value in builtins.__dict__.values()
    if isinstance(value, type) and issubclass(value, BaseException)
}


def _type_mro(value):
    """A value's type MRO, bypassing instance and metaclass attribute hooks."""
    cls = type(value)
    return _CLASS_MRO.__get__(cls, type(cls))


def _derives(value, base):
    """Whether ``base`` is in the type MRO of ``value``, compared by identity."""
    return any(cls is base for cls in _type_mro(value))


def _container_base(value):
    """The builtin container ``value`` stores entries in, and its copy types."""
    for cls in _type_mro(value):
        entry = _CONTAINER_BASES.get(id(cls))
        if entry is not None:
            return entry
    return None, None, None


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


class SnapshotContext:
    def __init__(self, options: DumpOptions, *, serializer="dill", minimal=False):
        if not isinstance(options, DumpOptions):
            raise TypeError("options must be a DumpOptions instance")
        if serializer not in ("dill", "pickle"):
            raise ValueError("serializer must be dill or pickle")
        self.options = options
        self.serializer = serializer
        self.minimal = minimal
        self.frames = {}
        self.codes = {}
        self.objects = {}
        self._sources = {}
        self._truncation = Counter()
        self._nodes = 0
        self._original_frames = 0
        self._capturing_frames = False
        self._code_sources = {}
        self._payload_codes = set()
        self._exception_copies = {}
        self._node_marker = TruncatedValue("max_nodes")
        # Frame namespaces keep their variable names whichever path reaches
        # them first; see ``_build``.
        self._namespaces = set()
        self._layouts = {}
        self._code_text_limit, self._code_bytes_limit = (
            (None, None)
            if options.is_unlimited
            else (_CODE_TEXT_LIMIT, _CODE_BYTES_LIMIT)
        )

    def metadata(self):
        return {
            "limits": asdict(self.options),
            "python_version": sys.version.split()[0],
            "original_frames": self._original_frames,
            "captured_frames": len(self.frames),
            "truncation": dict(self._truncation),
        }

    def _omit(self, reason, value=None):
        self._truncation[reason] += 1
        return TruncatedValue(
            reason, self._type_name(value) if value is not None else "object"
        )

    def _type_name(self, value):
        cls = type(value)
        # Direct type descriptors also bypass metaclass properties.  Merely
        # calling type.__getattribute__ would still execute those properties.
        module = _CLASS_MODULE.__get__(cls, type(cls))
        name = _CLASS_QUALNAME.__get__(cls, type(cls))
        module = module if type(module) is str else "?"
        return self._code_text(name if module == "builtins" else module + "." + name)

    def _bounded_text(self, value, limit, reason):
        if limit is None:
            return value
        # Slice before encoding so a multi-gigabyte string is never copied.
        prefix = value[:limit]
        if (
            len(value) == len(prefix)
            and len(prefix.encode("utf-8", "surrogatepass")) <= limit
        ):
            return value
        self._truncation[reason] += 1
        return _utf8_prefix(value, limit)

    def _bounded_bytes(self, value, limit, reason):
        if limit is not None and len(value) > limit:
            self._truncation[reason] += 1
        return value[:limit] if limit is not None else value[:]

    def _text(self, value):
        return self._bounded_text(
            value, self.options.max_string_bytes, "max_string_bytes"
        )

    def _code_text(self, value):
        """Bound code metadata by its own limit, never by ``max_string_bytes``."""
        return self._bounded_text(value, self._code_text_limit, "code_metadata")

    @staticmethod
    def _absolute_filename(filename):
        """Resolve a source path, tolerating a working directory that is gone."""
        if filename.startswith("<") and filename.endswith(">"):
            return filename
        try:
            return os.path.abspath(filename)
        except OSError:
            # A deleted working directory must not cost the whole snapshot.
            return filename

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

    def _fill_container(self, value, copy, depth):
        """Keep a container subclass's type name and attributes beside its entries.

        Instance attributes, filled slots, struct sequence fields outside the
        tuple part, and named tuple fields all become attributes of the copy.
        """
        namespace = copy.__dict__
        cls = type(value)
        own = cls is type(copy)
        attrs = self._attributes(value) or {}
        if attrs:
            namespace.update(
                self._attribute_mapping(
                    attrs, depth + 1, private="_FakeContainer__type" if own else None
                )
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
        namespace["_FakeContainer__type"] = (
            attrs.get("_FakeContainer__type") if own else self._type_name(value)
        )

    def _remember(self, original, captured):
        self.objects[id(original)] = captured
        # Transient locals proxies and __dict__ views must not have their ids
        # recycled while this capture is still using its identity memo.
        self._sources[id(original)] = original
        return captured

    def _has_nodes(self):
        limit = self.options.max_nodes
        return limit is None or self._nodes < limit

    def value(self, value, depth=0):
        cached = self.objects.get(id(value))
        if id(value) in self.objects:
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
        if self.options.is_unlimited and not self.minimal:
            serializer = dill if self.serializer == "dill" else pickle
            try:
                serializer.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                return self._remember(value, value)
            except Exception:
                self._truncation["unserializable"] += 1
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
                self._fill_container(value, captured, depth)
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
                self._fill_container(value, captured, depth)
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
        metadata = SnapshotContext(limits, serializer=self.serializer, minimal=True)
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
        if len(converted) == 1 and type(converted[0]) is str:
            return converted[0]
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
        if self.options.is_unlimited and not self.minimal:
            serializer = dill if self.serializer == "dill" else pickle
            try:
                serializer.dumps((type(exc), exc), protocol=pickle.HIGHEST_PROTOCOL)
                extra.update(exc_type=type(exc), exc_value=exc)
                return extra
            except Exception:
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


def _utf8_prefix(text, limit):
    return (
        text[:limit].encode("utf-8", "surrogatepass")[:limit].decode("utf-8", "ignore")
    )


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
    elif stage == "minimal" and truncation.get("minimal_snapshot"):
        reduced = dict(payload)
    else:
        reduced = _reduce_values(payload, options, minimal=stage == "minimal")
        metadata = reduced["snapshot_meta"]
        truncation = Counter(metadata["truncation"])
    extra = reduced.get("exc_extra")
    if extra and extra.get("formatted"):
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
    reduced_metadata = context.metadata()
    truncation.update(reduced_metadata["truncation"])
    metadata.update(
        truncation=dict(truncation), effective_limits=reduced_metadata["limits"]
    )
    reduced["snapshot_meta"] = metadata
    return reduced
