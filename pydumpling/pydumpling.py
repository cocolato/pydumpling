from __future__ import absolute_import, division, print_function, unicode_literals

import gzip
import sys
import dill
import pickle
import warnings
from .fake_types import FakeTraceback

__version__ = "0.0.1"


def save_dumping(filename=None, tb=None):
    try:
        if tb is None:
            tb = sys.exc_info()[2]

        if filename is None:
            filename = "%s:%d.dump" % (
                tb.tb_frame.f_code.co_filename, tb.tb_frame.f_lineno)

        fake_tb = FakeTraceback(tb)
        dumpling = {
            "traceback": fake_tb,
            "version": __version__,
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
