from __future__ import absolute_import, division, print_function, unicode_literals

import gzip
import sys
import dill
import pickle
import warnings
import inspect
from .fake_types import FakeFrame, FakeTraceback

__version__ = "0.1.1"


def save_dumping(filename=None, tb=None):
    try:
        if tb is None:
            exc_type, exc_value, exc_tb = sys.exc_info()

        if filename is None:
            filename = "%s:%d.dump" % (
                exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_frame.f_lineno)

        fake_tb = FakeTraceback(exc_tb)
        dumpling = {
            "traceback": fake_tb,
            "version": __version__,
            "exc_extra": {
                "exc_type": exc_type,
                "exc_value": exc_value,
            },
            "dump_type": "DILL"
        }
        with gzip.open(filename, "wb") as f:
            try:
                dill.dump(dumpling, f, protocol=dill.HIGHEST_PROTOCOL)
            except Exception:
                dumpling["dump_type"] = "PICKLE"
                pickle.dump(dumpling, f, protocol=dill.HIGHEST_PROTOCOL)
    except Exception as e:
        err_msg = "Unexpected error: %s when dumping traceback" % str(e)
        warnings.warn(err_msg, RuntimeWarning)


def dump_current_traceback(filename=None):
    try:
        fake_tb = gen_tb_from_frame(inspect.currentframe())
        dumpling = {
            "traceback": fake_tb,
            "version": __version__,
            "dump_type": "DILL"
        }
        if filename is None:
            filename = "%s:%d.dump" % (
                fake_tb.tb_frame.f_code.co_filename, fake_tb.tb_frame.f_lineno)
        with gzip.open(filename, "wb") as f:
            try:
                dill.dump(dumpling, f, protocol=dill.HIGHEST_PROTOCOL)
            except Exception:
                dumpling["dump_type"] = "PICKLE"
                pickle.dump(dumpling, f, protocol=dill.HIGHEST_PROTOCOL)
    except Exception as e:
        err_msg = "Unexpected error: %s when dumping traceback" % str(e)
        warnings.warn(err_msg, RuntimeWarning)


def gen_tb_from_frame(f):
    tb = FakeTraceback()
    tb.tb_frame = FakeFrame(f)
    tb.tb_lasti = f.f_lasti
    tb.tb_lineno = f.f_lineno
    queue = []
    f = f.f_back
    if f is None:
        return tb
    while f:
        tb = FakeTraceback()
        tb.tb_frame = FakeFrame(f)
        tb.tb_lasti = f.f_lasti
        tb.tb_lineno = f.f_lineno
        queue.append(tb)
        f = f.f_back

    for i in range(len(queue)-1, 0, -1):
        queue[i].tb_next = queue[i-1]
    queue[0].tb_next = None
    return queue[-1]
