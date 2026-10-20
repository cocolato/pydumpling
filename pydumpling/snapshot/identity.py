"""Identity-based type introspection that runs no application hooks."""

# Direct type descriptors also bypass metaclass properties.  Merely calling
# type.__getattribute__ would still execute those properties.  Types are
# compared by identity throughout: ``isinstance`` falls back to an instance's
# ``__class__`` attribute, ``x in tuple`` calls a metaclass ``__eq__``, and a
# set lookup calls a metaclass ``__hash__``; application code can override
# any of them.
_CLASS_DICT = type.__dict__["__dict__"]
_CLASS_MRO = type.__dict__["__mro__"]
_CLASS_MODULE = type.__dict__["__module__"]
_CLASS_QUALNAME = type.__dict__["__qualname__"]


def _type_mro(value):
    """A value's type MRO, bypassing instance and metaclass attribute hooks."""
    cls = type(value)
    return _CLASS_MRO.__get__(cls, type(cls))


def _derives(value, base):
    """Whether ``base`` is in the type MRO of ``value``, compared by identity."""
    return any(cls is base for cls in _type_mro(value))
