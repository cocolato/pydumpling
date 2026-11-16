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


def _pickle_copy(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _dill_copy(value):
    return dill.loads(dill.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _fallbacks():
    statics = values_module._STATICS
    return statics.fallbacks if statics is not None else 0


@pytest.mark.parametrize("native", [False, True], ids=["python", "native"])
@pytest.mark.parametrize("context_type", [_SlottedContext, _ChildContext])
@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, _pickle_copy, _dill_copy])
def test_context_copy_preserves_slots(native, context_type, clone):
    if native and values_module._native is None:
        pytest.skip("the optional native walker is unavailable")
    context = context_type(DumpOptions())
    if not native:
        context._native_walk = False
    # Slots for names the walk does not use leave the stock walk.
    assert context._native_walk is native
    before = _fallbacks()
    source = [1, [2]]
    captured = context.value(source)

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
    # Every copy captures with its original's walk.
    assert copied._native_walk is native
    assert context.objects[id(source)] is captured
    if clone is copy.copy:
        assert copied.tag is context.tag
        assert copied.objects is context.objects
        assert copied.value(source) is captured
    else:
        assert copied.tag is not context.tag
    assert copied.value("new value") == "new value"
    assert _fallbacks() == before
