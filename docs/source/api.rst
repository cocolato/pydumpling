API reference
=============

Capture options
---------------

Import ``DumpOptions`` from ``pydumpling``. Pass an options object to
``save_dumping``, ``dump_current_traceback``, ``catch_any_exception``, or
``capture_pid``. Omitting it applies bounded defaults.

.. list-table:: Default capture budgets
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Default
   * - ``max_frames``
     - 64 frames
   * - ``max_depth``
     - 5 nested levels
   * - ``max_items``
     - 100 items per container
   * - ``max_string_bytes``
     - 4,096 bytes per captured string value
   * - ``max_nodes``
     - 10,000 visited values
   * - ``max_dump_bytes``
     - 16 MiB of uncompressed serialized data

Values that exceed traversal budgets may be summarized. Writers capture once,
then try dill and pickle on the same captured graph with separate output buffers.
If neither fits the byte budget, reductions first omit frame globals, then omit
outer-frame locals while retaining the innermost frame's arguments and locals.
The latter step also drops code constants and bytecode, retaining argument names
and source location tables. These steps preserve the retained values and their
shared references. If that
still does not fit, one reduction halves depth, item, string, and node budgets
(with a minimum of one); finally, minimal mode drops all variables. Disabled
traversal budgets use half their default value for this last reduction.
Steps that omit nothing are skipped. All reductions operate on the captured
snapshot. Capture fails only if the minimal snapshot also cannot be saved.
Serialization failures unrelated to size proceed directly to minimal.

Use ``DumpOptions.unlimited()`` to remove
all capture limits explicitly. An unrestricted snapshot can consume substantial
memory, CPU time, and disk space, and permits custom object serialization hooks
to execute. Individual limits can be disabled with ``None``; all other values
must be positive integers.
Original objects in unrestricted mode may depend on their original environment
or custom loading hooks; not every object can be round-tripped successfully.
They may also retain live references or be mutated by serialization hooks, so
reuse across serializers cannot freeze them. Bounded capture copies supported
values, but traversing a running process is still not an atomic, cross-thread
snapshot. Only a failure to build the initial snapshot triggers a fresh minimal
capture of live frames.

Basic built-in exception diagnostics have a small reserved traversal budget, so
filenames and syntax locations survive even when application values exhaust
their budget. Captured diagnostic fields still use ``max_string_bytes``; the
combined formatted report is bounded by ``max_dump_bytes``.
During fallback, oversized diagnostic text is replaced with an annotated summary
whose combined type name, message, and formatted text use at most one quarter of
the byte budget, capped at 4,096 UTF-8 bytes. The summary retains short diagnostic
prefixes and reports omitted characters. Metadata records ``exception_summary``;
``globals_omitted`` and ``outer_locals_omitted`` count omitted frame mappings.
``code_payload_omitted`` counts code objects whose constants or bytecode were omitted.
Budgets too small for even frame structure and metadata still cause capture to fail.

.. code-block:: python

   from pydumpling import DumpOptions, dump_current_traceback

   options = DumpOptions(max_frames=16, max_depth=3, max_items=25)
   dump_current_traceback("small.dump", options=options, strict=True)

Writers and exception hook
--------------------------

``save_dumping(filename=None, exc_info=None, *, options=None, strict=False)``
   Capture the current exception or an explicit ``(type, value, traceback)``
   tuple. Use it inside an exception handler when ``exc_info`` is omitted.

``dump_current_traceback(filename=None, *, options=None, strict=False)``
   Capture the current caller stack without requiring an exception.

Both writers accept paths, return ``None``, and report failures with
``RuntimeWarning`` by default. Set ``strict=True`` to raise errors instead.
An omitted filename uses an automatically generated name in the current
directory. An existing output file is replaced atomically after successful
serialization.

``catch_any_exception(*, options=None)``
   Install a main-thread ``sys.excepthook`` that saves unhandled exceptions,
   then invokes the previous hook for normal reporting. Call it once at
   application startup. Thread and asyncio exception handlers are not installed.

Loading and debugging
---------------------

``load_dumpling(filename, *, max_bytes=64 * 1024 * 1024)``
   Read a trusted compressed dump and return the snapshot dictionary. The byte
   limit applies to decompressed data. ``max_bytes=None`` disables it.
   Invalid gzip, undecodable payloads, unsupported formats, and invalid snapshot
   structures raise ``pydumpling.debug_dumpling.DumpFormatError``. Filesystem
   errors such as missing files or denied access keep their original types.

New snapshots expose capture budgets, Python version, frame counts, and omission
reasons in ``snapshot_meta``. Inspect ``snapshot_meta["truncation"]`` to see which
budgets or unsupported values limited the capture. CLI printing and debugging
also display a snapshot notice when values were summarized.
``snapshot_meta["attempts"]`` lists failed stages, serializers (when applicable),
and exception class names. Omission counts accumulate across reductions;
``limits`` retains the requested budgets and ``effective_limits`` records the
reduced traversal budgets when used. A serializer change alone does not discard
captured variables.

``format_version`` versions the complete snapshot schema, including FakeCode's
line and position tables, independently of the package ``version``. New writers
emit format 1. Missing format versions are treated as legacy format 0; unknown
versions are rejected. Format 1 compatibility depends on the schema version;
the producer's package version must be well formed but may belong to another
release series. Only legacy format 0 uses the historical package-version range.
Incompatible changes to internal encodings must increment
this format version and provide an explicit reader migration.

``debug_dumpling(dump_file, pdb=pdb, *, max_bytes=64 * 1024 * 1024)``
   Load a trusted snapshot and start post-mortem inspection with pdb or a
   compatible supplied debugger. The same loading limit applies.

``r_post_mortem(dump_file, addr="127.0.0.1", port=4444, *, timeout=None, max_bytes=64 * 1024 * 1024)``
   Open a remote pdb session. Keep the listener on loopback and connect through
   an SSH tunnel; the session has no built-in authentication or encryption.
   ``timeout`` limits how long to wait for a client connection.

Debugger sessions temporarily patch process-wide ``inspect`` predicates and
restore tracing and SIGINT state on exit. Remote commands also redirect process
standard streams. A shared reentrant lock serializes these operations: a session
in another thread waits until the active session exits, and the connection
``timeout`` does not bound that wait. Other application threads do not acquire
this lock and can observe the temporary global state. Prefer debugging dumps in
a separate process when the application must keep serving concurrent work.

.. warning::

   dill and pickle can execute code during deserialization. Loading limits
   constrain data size and do not make untrusted dumps safe.

Running-process capture
-----------------------

``capture_pid(pid, filename, *, options=None, timeout=30) -> Path``
   Request a snapshot from another CPython 3.14+ process and return its output
   path. Controller and target must use the same CPython version on the same
   machine and under the same user, with shared process and filesystem
   namespaces. The target needs the updated Pydumpling installed and must permit
   CPython remote execution. Operating system permissions can prevent attachment.

Timeout stops waiting rather than cancelling a dispatched request. The request
script is retained and may execute later; the CLI prints its directory. Avoid
submitting repeated requests while one is pending. Remove the retained directory
only after its ``status.json`` confirms completion or the target has exited.

Command line
------------

Both ``pydumpling`` and ``python -m pydumpling`` expose the same interface.
Choose exactly one action: ``--print``, ``--debug``, ``--rdebug``, or
``--capture-pid PID``. The first three take a dump filename; PID capture takes
``--output FILE`` and optionally ``--timeout SECONDS`` or ``--unlimited``.

``--host`` and ``--port`` configure remote debugging; ``--timeout`` also limits
waiting for a remote client. ``--max-load-mib`` sets
the decompressed loading limit in MiB; zero disables it. Use ``--help`` for all
options and ``--version`` for the installed version.
