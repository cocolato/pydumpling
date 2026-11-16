"""Suite-wide checks."""

import importlib.util
import sys
from pathlib import Path

import pytest

from pydumpling.snapshot import values

_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    """Refuse to run on a native walker built from other sources.

    ``rust/build_native.py`` builds the extension into the package and leaves
    it there, git-ignored, across checkouts and edits of the crate.  One built
    from other sources still imports, and captures as the crate no longer
    does, so every test would exercise the wrong walker; or ``values`` drops
    it with a warning, and every native test would be skipped.
    """
    # The module ``values`` imported, whether it kept it or not.
    native = sys.modules.get("pydumpling._native")
    script = _ROOT / "rust" / "build_native.py"
    if native is None or not script.is_file():
        return
    if Path(native.__file__).resolve().parent != _ROOT / "pydumpling":
        return  # Installed elsewhere, not built here.
    spec = importlib.util.spec_from_file_location("build_native", script)
    build_native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_native)
    if getattr(native, "fingerprint", None) != build_native.fingerprint():
        raise pytest.UsageError(
            f"{native.__file__} was not built from the sources in rust/: "
            "rebuild it with `python rust/build_native.py`, or set "
            "PYDUMPLING_NATIVE=0 to run the Python walk alone"
        )
    if values._native is None:
        raise pytest.UsageError(
            f"{native.__file__} matches the sources in rust/, but "
            "pydumpling.snapshot.values does not use it: see its RuntimeWarning"
        )
