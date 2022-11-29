from __future__ import absolute_import, division, print_function, unicode_literals

import gzip
import pdb
import dill
import pickle
from distutils.version import StrictVersion
import inspect
import types
from .fake_types import FakeFrame, FakeTraceback, FakeCode


def load_dumpling(filename):
    with gzip.open(filename, "rb") as f:
        try:
            return dill.load(f)
        except Exception:
            return pickle.load(f)


def debug_dumpling(dump_file, pdb=pdb):
    inspect.isframe = lambda obj: isinstance(
        obj, types.FrameType) or obj.__class__ == FakeFrame
    inspect.istraceback = lambda obj: isinstance(
        obj, types.TracebackType) or obj.__class__ == FakeTraceback
    inspect.iscode = lambda obj: isinstance(
        obj, types.CodeType) or obj.__class__ == FakeCode
    dumpling = load_dumpling(dump_file)
    if not StrictVersion("0.0.1") <= StrictVersion(dumpling["version"]) < StrictVersion("1.0.0"):
        raise ValueError("Unsupported dumpling version: %s" %
                         dumpling["version"])
    tb = dumpling["traceback"]
    pdb.post_mortem(tb)
