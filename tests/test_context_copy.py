"""Context copies keep the normal instance state, including subclass slots."""

import copy
import pickle

import dill
import pytest

from pydumpling.options import DumpOptions
from pydumpling.snapshot import SnapshotContext
from pydumpling.snapshot import values as values_module


class _SlottedContext(SnapshotContext):
    __slots__ = ("tag", "__private", "unset")


class _ChildContext(_SlottedContext):
    __slots__ = ("extra",)


class _WalkerSlotContext(_ChildContext):
    __slots__ = ("_walker",)


def _pickle_copy(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _dill_copy(value):
    return dill.loads(dill.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize(
    "context_type", [_SlottedContext, _ChildContext, _WalkerSlotContext]
)
@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, _pickle_copy, _dill_copy])
def test_context_copy_preserves_slots(monkeypatch, native, context_type, clone):
    if native and values_module._native is None:
        pytest.skip("the optional native walker is unavailable")
    monkeypatch.setenv("PYDUMPLING_NATIVE", "1" if native else "0")
    context = context_type(DumpOptions())
    source = [1, [2]]
    captured = context.value(source)
    walker = getattr(context, "_walker", None)
    # A slot for the walker, released before the instance dict, is not the
    # stock layout, so that context captures in Python.
    assert (walker is not None) is (native and context_type is not _WalkerSlotContext)

    context.tag = ["slot data"]
    context.alias = context.tag
    context._SlottedContext__private = "private data"
    if isinstance(context, _ChildContext):
        context.extra = "inherited slots"

    copied = clone(context)

    assert copied.tag == ["slot data"]
    assert copied.alias is copied.tag
    assert copied._SlottedContext__private == "private data"
    assert not hasattr(copied, "unset")
    if isinstance(context, _ChildContext):
        assert copied.extra == "inherited slots"
    assert getattr(copied, "_walker", None) is None
    assert getattr(context, "_walker", None) is walker
    assert context.objects[id(source)] is captured
    if clone is copy.copy:
        assert copied.tag is context.tag
        assert copied.objects is context.objects
        assert copied.value(source) is captured
    else:
        assert copied.tag is not context.tag
    assert copied.value("new value") == "new value"
