"""Bounded, shared snapshots of real frames and application values."""

from .context import SnapshotContext
from .reduce import reduce_snapshot

# Format 2 adds the str, bytes, int, and float subclass copies; it only adds
# types, so format 1 dumps load unchanged.
SNAPSHOT_FORMAT_VERSION = 2

__all__ = ["SNAPSHOT_FORMAT_VERSION", "SnapshotContext", "reduce_snapshot"]
