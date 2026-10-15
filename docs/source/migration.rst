Migration to 0.2
================

Python and tooling
------------------

Version 0.2 requires Python 3.11 or newer. Use uv and the committed ``uv.lock``
for development. Old PDM requirements and lockfiles, tox and Flake8 configuration,
and Hatch environment scripts have been removed. Hatchling remains the build
backend. Ruff handles lint and formatting; pytest runs directly in the uv
environment. The ``doc`` extra remains available.

The package, documentation, and distribution metadata share the version in
``pydumpling/_version.py``. Installation provides a ``pydumpling`` command as well
as the existing ``python -m pydumpling`` entry point.

Existing callers
----------------

The existing public capture, load, and debug function names remain available.
Capture is bounded by default: 64 frames, 5 nested levels, 100 items per container,
4,096 bytes per string, 10,000 visited values, and 16 MiB of uncompressed
serialized data. Some values may therefore be summarized rather than preserved
in full. Pass ``options=DumpOptions.unlimited()`` to request unrestricted capture.

Writer failures continue to emit warnings by default. Applications that need a
capture failure to propagate can set ``strict=True``. The exception hook retains
normal exception reporting even when snapshot capture fails.

Automatic filenames now use ``source-name-line.dump`` in the current working
directory, rather than writing alongside the source file. Explicit paths are
unchanged.

Loading and debugging use a 64 MiB decompressed-data limit. For a trusted file
that legitimately exceeds it, pass ``max_bytes=None`` or use
``--max-load-mib 0``. This setting does not relax the trusted-source requirement.

``pydumpling.debug_dumpling.mock_inspect()`` is now a context manager. Calling it
alone no longer patches ``inspect``. Direct callers integrating third-party
debuggers must wrap the debugging session so that host state is restored::

    import ipdb
    from pydumpling import load_dumpling
    from pydumpling.debug_dumpling import mock_inspect

    snapshot = load_dumpling("example.dump")
    with mock_inspect():
        ipdb.post_mortem(snapshot["traceback"])

``debug_dumpling`` and ``r_post_mortem`` manage this context automatically.

Dump compatibility
------------------

Old 0.1 dumps remain readable when their serialized objects are compatible with
the running Python interpreter. Cross-Python-version compatibility is not
guaranteed. Use the matching interpreter and source checkout for inspection.
Snapshots cannot resume execution of the original process.

New dumps carry ``format_version=1`` independently of the package version.
Unversioned dumps are read as legacy format 0; unknown schema versions are
rejected. Serialization retries reuse the captured graph, and byte-budget
failures omit globals and outer-frame locals before reducing fault-frame values.
Oversized exception reports become bounded summaries with omission counts.
Format 1 uses the schema version for compatibility; the historical package-version
range applies only to legacy format 0. See
:doc:`api` for retry metadata and concurrency limits of embedded debuggers.

New capability
--------------

``capture_pid`` and ``--capture-pid`` request live snapshots from another
CPython 3.14+ process. Both processes need matching CPython versions, same-user
local access, shared process and filesystem namespaces, and updated Pydumpling
in the target. A timed-out request can execute later; its request script is
retained. See :doc:`tutorial` for examples and :doc:`api` for parameters.
