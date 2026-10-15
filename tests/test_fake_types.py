import builtins
import inspect
import pickle
import warnings

import dill
import pytest

from pydumpling.fake_types import (
    FakeClass,
    FakeCode,
    FakeFrame,
    FakeTraceback,
    FakeType,
)


def test_fake_class_preserves_attributes():
    value = FakeClass("<saved instance>", {"answer": 42})
    assert value.answer == 42
    assert repr(value) == "<saved instance>"


@pytest.mark.parametrize("serializer", [pickle, dill])
def test_real_code_positions_survive_serialization(serializer):
    def example(value):
        return value + 1

    original = example.__code__
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        captured = FakeCode(original)
    restored = serializer.loads(serializer.dumps(captured))
    assert list(restored.co_lines()) == list(original.co_lines())
    assert list(restored.co_positions()) == list(original.co_positions())
    assert restored.co_name == original.co_name


def test_old_code_state_supports_saved_position_methods_and_legacy_lines():
    code = (lambda: None).__code__
    restored = object.__new__(FakeCode)
    restored.__setstate__({"co_positions": code.co_positions})
    assert list(restored.co_positions()) == list(code.co_positions())
    restored = object.__new__(FakeCode)
    restored.__setstate__(
        {
            "co_firstlineno": 10,
            "co_lnotab": b"\x02\x02",
            "co_code": b"\x00" * 6,
        }
    )
    assert list(restored.co_lines()) == [(0, 2, 10), (2, 6, 12)]


def test_legacy_frame_constructor_and_builtins_restoration():
    answer = 42
    original = inspect.currentframe()
    try:
        frame = FakeFrame(original)
        assert frame.f_locals["answer"] == answer
        assert isinstance(frame.f_code, FakeCode)
        restored = pickle.loads(pickle.dumps(frame))
        assert restored.f_builtins is builtins.__dict__
    finally:
        del original


def test_empty_traceback_and_safe_repr_compatibility():
    empty = FakeTraceback()
    assert (empty.tb_frame, empty.tb_next, empty.tb_lineno, empty.tb_lasti) == (
        None,
        None,
        None,
        0,
    )

    class BadRepresentation:
        def __repr__(self):
            raise ValueError("broken representation")

    assert FakeType._safe_repr(BadRepresentation()).startswith("repr error")
