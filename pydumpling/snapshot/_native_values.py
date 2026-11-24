"""Custom value nodes for the native engine's iterative traversal."""

from ..fake_types import FakeClass, FakeDict, FakeTuple, TruncatedValue
from .values import (
    _RECURSIVE_IMMUTABLE,
    _attribute_namespace,
    _container_base,
    _copy_attributes,
    _filled_members,
    _object_copy,
    _scalar_copy,
    _scalar_copy_type,
    _slot_fits,
)


def _fill_copy(ctx, value, copy, depth):
    try:
        namespace = copy.__dict__
        attributes = _copy_attributes(ctx, value, copy, depth, namespace)
        if attributes is None:
            return
        key, type_name, attrs, private = attributes
        if attrs:
            source = _attribute_namespace(ctx, attrs, private)
            converted = yield "mapping", source, depth + 1, (), False, dict
            namespace.update(converted)
        members, fields = ctx._layout(value)
        for name, item in _filled_members(value, members):
            if not _slot_fits(ctx, namespace):
                break
            namespace[name] = yield "value", item, depth + 1
        if fields and type(copy) is FakeTuple:
            namespace.update(zip(fields, copy))
        namespace[key] = type_name
    except StopIteration as error:
        yield "raise", error


def custom_value(ctx, value, depth):
    """Capture one node after the engine's memo check and node charge.

    Children are requests to the engine. A suspended generator keeps the
    parent's insertion order and partial copy without adding a native call
    frame per level of the application graph. An explicit raise request
    carries application StopIteration across generator boundaries before
    PEP 479 can turn it into a RuntimeError.
    """
    try:
        value_type = type(value)
        copy_type = _scalar_copy_type(value)
        if copy_type is not None:
            copy, fill = _scalar_copy(ctx, value, copy_type)
            if fill:
                yield from _fill_copy(ctx, value, copy, depth)
            return copy

        limit = ctx.options.max_depth
        if limit is not None and depth >= limit:
            return ctx._remember(value, ctx._omit("max_depth", value))
        # The engine captures the exact builtin containers itself.
        base, _, target = _container_base(value)
        if target is FakeDict:
            copy = yield "mapping", value, depth, (), True, target
        elif base is not None:
            sequence = base is list or base is tuple
            mutable = base is list or base is set
            copy = target() if mutable else ([] if sequence else set())
            ctx._remember(
                value,
                copy
                if mutable
                else TruncatedValue(_RECURSIVE_IMMUTABLE, ctx._type_name(value)),
            )
            for count, item in enumerate(base.__iter__(value)):
                if ctx.options.max_items is not None and count >= ctx.options.max_items:
                    ctx._truncation["max_items"] += 1
                    break
                if not ctx._has_nodes() and id(item) not in ctx.objects:
                    ctx._truncation["max_nodes"] += 1
                    break
                converted = yield "value", item, depth + 1
                if sequence:
                    copy.append(converted)
                else:
                    copy.add(converted)
            if not mutable:
                copy = target(copy)
                ctx._remember(value, copy)
        else:
            copy, representation, attrs = _object_copy(ctx, value, value_type)
            if attrs is not None:
                private = "_FakeClass__repr" if value_type is FakeClass else None
                source = _attribute_namespace(ctx, attrs, private)
                converted = yield "mapping", source, depth + 1, (), False, dict
                copy.__dict__.update(converted)
                copy.__dict__["_FakeClass__repr"] = representation
            return copy

        yield from _fill_copy(ctx, value, copy, depth)
        return copy
    except StopIteration as error:
        yield "raise", error
