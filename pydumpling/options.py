"""Resource limits for capturing a debugging snapshot."""

from dataclasses import dataclass, field, fields
from typing import Optional


def _budget_fields():
    return [item for item in fields(DumpOptions) if item.metadata.get("budget", True)]


@dataclass(frozen=True)
class DumpOptions:
    """Limits shared by every frame and value in a dump.

    ``None`` disables an individual budget.  :meth:`unlimited` also opts into
    best-effort serialization of original objects and their custom hooks.
    ``embed_source_lines`` is a count, not a budget: zero (the default)
    embeds no source, a positive number embeds that many lines of context on
    each side of every frame's current line, and ``None`` embeds whole files.
    """

    max_frames: Optional[int] = 64
    max_depth: Optional[int] = 5
    max_items: Optional[int] = 100
    max_string_bytes: Optional[int] = 4096
    max_nodes: Optional[int] = 10000
    max_dump_bytes: Optional[int] = 16 * 1024 * 1024
    embed_source_lines: Optional[int] = field(default=0, metadata={"budget": False})

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.metadata.get("budget", True):
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(
                        "%s must be a positive integer or None" % item.name
                    )
            elif value is not None and (type(value) is not int or value < 0):
                raise ValueError("%s must be a nonnegative integer or None" % item.name)

    @classmethod
    def unlimited(cls):
        """Disable all capture budgets and allow original-object serialization.

        Source embedding stays at its own setting; budgets do not imply it.
        """
        return cls(**{item.name: None for item in _budget_fields()})

    @property
    def is_unlimited(self):
        return all(getattr(self, item.name) is None for item in _budget_fields())
