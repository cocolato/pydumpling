"""Resource limits for capturing a debugging snapshot."""

from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class DumpOptions:
    """Limits shared by every frame and value in a dump.

    ``None`` disables an individual limit.  :meth:`unlimited` also opts into
    best-effort serialization of original objects and their custom hooks.
    """

    max_frames: Optional[int] = 64
    max_depth: Optional[int] = 5
    max_items: Optional[int] = 100
    max_string_bytes: Optional[int] = 4096
    max_nodes: Optional[int] = 10000
    max_dump_bytes: Optional[int] = 16 * 1024 * 1024

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("%s must be a positive integer or None" % field.name)

    @classmethod
    def unlimited(cls):
        """Disable all budgets and allow original-object serialization."""
        return cls(**{field.name: None for field in fields(cls)})

    @property
    def is_unlimited(self):
        return all(getattr(self, field.name) is None for field in fields(self))
