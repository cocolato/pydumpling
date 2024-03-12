from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import dill


class FakeType(object):
    @classmethod
    def _safe_repr(cls, v):
        try:
            return repr(v)
        except Exception as e:
            return "repr error: %s" % str(e)

    @classmethod
    def _convert_dict(cls, v):
        return {cls._convert(k): cls._convert(i) for k, i in v.items()}

    @classmethod
    def _convert_obj(cls, obj):
        try:
            return FakeClass(cls._safe_repr(obj), cls._convert_dict(obj.__dict__))
        except Exception:
            return cls._convert(obj)

    @classmethod
    def _convert_seq(cls, v):
        return map(cls._convert, v)

    @classmethod
    def _convert(cls, v):

        if v is None:
            return v

        if dill is not None:
            try:
                dill.dumps(v)
                return v
            except Exception:
                return cls._safe_repr(v)
        else:
            from datetime import date, time, datetime, timedelta

            BUILTIN = (str, unicode, int, long, float, date, time, datetime, timedelta) if sys.version_info.major == 2 \
                else (str, int, float, date, time, datetime, timedelta)  # noqa: F821

            if type(v) in BUILTIN:
                return v

            if isinstance(v, (tuple, list, set)):
                return type(v)(cls._convert_seq(v))

            if isinstance(v, dict):
                return cls._convert_dict(v)

            return cls._safe_repr(v)


class FakeTraceback(FakeType):

    def __init__(self, traceback=None):
        self.tb_frame = FakeFrame(
            traceback.tb_frame) if traceback and traceback.tb_frame else None
        self.tb_lineno = traceback.tb_lineno if traceback else None
        self.tb_next = FakeTraceback(
            traceback.tb_next) if traceback and traceback.tb_next else None
        self.tb_lasti = traceback.tb_lasti if traceback else 0


class FakeFrame(FakeType):
    def __init__(self, frame):
        self.f_code = FakeCode(frame.f_code)
        self.f_locals = self._convert_dict(frame.f_locals)
        if "self" in frame.f_locals:
            self.f_locals["self"] = self._convert_obj(frame.f_locals["self"])
        self.f_globals = self._convert_dict(frame.f_globals)
        self.f_lineno = frame.f_lineno
        self.f_back = FakeFrame(frame.f_back) if frame.f_back else None
        self.f_lasti = frame.f_lasti
        self.f_builtins = frame.f_builtins


class FakeClass(FakeType):

    def __init__(self, repr, vals):
        self.__repr = repr
        self.__dict__.update(vars)

    def __repr__(self):
        return self.__repr


class FakeCode(FakeType):

    def __init__(self, code):
        self.co_filename = os.path.abspath(code.co_filename)
        self.co_name = code.co_name
        self.co_argcount = code.co_argcount
        self.co_consts = tuple(FakeCode(c) if hasattr(
            c, "co_filename") else c for c in code.co_consts)
        self.co_firstlineno = code.co_firstlineno
        self.co_lnotab = code.co_lnotab
        self.co_varnames = code.co_varnames
        self.co_flags = code.co_flags
        self.co_code = code.co_code
        self._co_lines = list(code.co_lines()) if hasattr(
            code, "co_lines") else []
        if hasattr(code, "co_kwonlyargcount"):
            self.co_kwonlyargcount = code.co_kwonlyargcount
        if hasattr(code, "co_positions"):
            self.co_positions = code.co_positions

    def co_lines(self):
        return iter(self._co_lines)
