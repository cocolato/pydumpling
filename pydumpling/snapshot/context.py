"""The composed snapshot context: budgets, metadata, and serialization."""

import os
import pickle
import socket
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

import dill

from ..fake_types import TruncatedValue
from ..options import DumpOptions
from .exceptions import ExceptionsMixin
from .frames import FramesMixin
from .identity import _CLASS_MODULE, _CLASS_QUALNAME
from .values import ValuesMixin

# Code metadata (file names, function and class names, variable and name
# tables) is diagnostic structure, not application data. Truncating it yields
# frame headers that name the wrong file or function, so it has its own fixed
# bound instead of following ``max_string_bytes``.  Bytecode has the same
# problem but a much larger natural size, so it gets a separate bound; the
# ``outer`` reduction stage still drops it entirely when bytes are tight.
# ``DumpOptions.unlimited()`` disables both bounds.
_CODE_TEXT_LIMIT = 4096
_CODE_BYTES_LIMIT = 64 * 1024
# Process context is diagnostic metadata, not application data; entries past
# this many command-line arguments are dropped and reported as truncated.
_ARGV_LIMIT = 256


def _utf8_prefix(text, limit):
    return (
        text[:limit].encode("utf-8", "surrogatepass")[:limit].decode("utf-8", "ignore")
    )


class SnapshotContext(ValuesMixin, FramesMixin, ExceptionsMixin):
    def __init__(self, options: DumpOptions, *, serializer="dill", minimal=False):
        if not isinstance(options, DumpOptions):
            raise TypeError("options must be a DumpOptions instance")
        if serializer not in ("dill", "pickle"):
            raise ValueError("serializer must be dill or pickle")
        self.options = options
        self.serializer = serializer
        self.minimal = minimal
        # Evaluated once: the dataclass property walks every field per call.
        self._unlimited = options.is_unlimited and not minimal
        self.frames = {}
        # Every line a traceback entry or frame reported, per captured frame.
        self._frame_lines = {}
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
        # Environment context is gathered lazily, so throwaway contexts used
        # to convert a single value never pay for it.
        self._started = time.monotonic()
        self._created = datetime.now(timezone.utc)
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
            "created_at": self._created.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "duration_ms": round((time.monotonic() - self._started) * 1000),
            "pid": os.getpid(),
            "hostname": self._hostname(),
            "cwd": self._cwd(),
            "argv": self._argv(),
            "thread_name": threading.current_thread().name,
            "original_frames": self._original_frames,
            "captured_frames": len(self.frames),
            "truncation": dict(self._truncation),
        }

    @staticmethod
    def _hostname():
        """The machine name, or None where the platform cannot provide it."""
        try:
            return socket.gethostname()
        except OSError:
            return None

    def _cwd(self):
        """The working directory, bounded like code metadata."""
        try:
            return self._code_text(os.getcwd())
        except OSError:
            # A deleted working directory must not cost the whole snapshot.
            return None

    def _argv(self):
        """The command line, bounded like code metadata."""
        argv = getattr(sys, "argv", None)
        if type(argv) is not list:
            return None
        if len(argv) > _ARGV_LIMIT:
            self._truncation["argv_entries"] += len(argv) - _ARGV_LIMIT
            argv = argv[:_ARGV_LIMIT]
        return [
            self._code_text(entry) if type(entry) is str else None for entry in argv
        ]

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

    def _remember(self, original, captured):
        self.objects[id(original)] = captured
        # Transient locals proxies and __dict__ views must not have their ids
        # recycled while this capture is still using its identity memo.
        self._sources[id(original)] = original
        return captured

    def _has_nodes(self):
        limit = self.options.max_nodes
        return limit is None or self._nodes < limit

    def _serializable(self, value):
        """Whether the writer can keep this original object as it is."""
        # Probe with the serializer that writes the snapshot.  pickle and dill
        # accept different objects: a value only pickle can write costs a
        # failed dill attempt, and next to one only dill can write, both
        # writers fail and the snapshot falls back to minimal.
        serializer = dill if self.serializer == "dill" else pickle
        try:
            serializer.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception:
            return False
