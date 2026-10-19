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

``embed_source_lines`` is a count, not a budget: zero (the default) embeds no
source, a positive number embeds that many lines of context on each side of
every captured frame's current line, and ``None`` embeds whole files. Embedded
lines are stored per file in the dump's ``source`` mapping and each line is
bounded like code metadata. ``DumpOptions.unlimited()`` disables the budgets
but leaves source embedding at its own setting.

Values that exceed traversal budgets may be summarized. Writers capture once,
then try pickle and dill on the same captured graph, each attempt streaming
through gzip into its own temporary file, so the full serialized form never
sits in memory and a failed attempt changes no existing dump.
A bounded snapshot holds only builtin and pydumpling types, so the much faster
pickle goes first; ``DumpOptions.unlimited()`` keeps original objects that may
need dill, so dill goes first there. Dumps are gzip-compressed at level 6.
If neither fits the byte budget, reductions first omit embedded source, then
frame globals, then outer-frame locals while retaining the innermost frame's
arguments and locals.
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

Code metadata is not application data and does not follow ``max_string_bytes``.
File names, function and class names, variable and name tables, bytecode,
frame variable names, and attribute names keep separate fixed bounds, so a small
string budget cannot rewrite the file or function named in a frame header,
rename a local or an attribute, or make ``pp`` fail to find it. That holds even
when a local such as ``globals()`` reaches a frame namespace first. Metadata
truncation is reported as
``code_metadata``; ``DumpOptions.unlimited()`` removes these bounds too.

When ``max_depth`` is ``None``, a value nested too deeply for the interpreter is
replaced by a placeholder and reported as ``recursion``; the other variables of
its frame are kept.

.. code-block:: python

   from pydumpling import DumpOptions, dump_current_traceback

   options = DumpOptions(max_frames=16, max_depth=3, max_items=25)
   dump_current_traceback("small.dump", options=options, strict=True)

Writers and exception hooks
---------------------------

``save_dumping(filename=None, exc_info=None, *, options=None, strict=False)``
   Capture the current exception or an explicit ``(type, value, traceback)``
   tuple. Use it inside an exception handler when ``exc_info`` is omitted.

``dump_current_traceback(filename=None, *, options=None, strict=False)``
   Capture the current caller stack without requiring an exception.

Both writers accept paths, return ``None``, and report failures with
``RuntimeWarning`` by default. Set ``strict=True`` to raise errors instead.
An omitted filename uses an automatically generated name in the current
directory: the source file, the failing line, a UTC timestamp in milliseconds,
the pid, and a per-process sequence, so repeated failures never overwrite an
earlier snapshot. An existing output file is
replaced atomically after successful serialization.

``catch_any_exception(*, options=None, include_threads=False, manager=None)``
   Install a main-thread ``sys.excepthook`` that saves unhandled exceptions,
   then invokes the previous hook for normal reporting. Call it once at
   application startup. ``include_threads=True`` also hooks
   ``threading.excepthook`` for exceptions that end a thread. ``manager`` is a
   ``DumpManager`` used to place and prune every saved dump; give capture
   budgets to the manager instead of ``options``. Re-running updates budgets
   without replacing installed hooks.

``install_asyncio_handler(loop=None, *, options=None)``
   Save snapshots for exceptions an asyncio event loop reports. Call it from
   async code to hook the running loop, or pass ``loop`` explicitly. The
   loop's previous handler, or its default, still reports each exception,
   even if saving fails.

``restore_exception_hooks()``
   Remove every hook installed by the functions above, restoring the hooks
   that were active at installation. Hooks another package installed on top
   of ours stay installed.

``DumpManager(directory, *, retention=50, options=None)``
   Save into one directory with unique names and bounded retention.
   ``save_exception(exc_info=None, *, options=None, strict=False)`` and
   ``save_current(*, options=None, strict=False)`` mirror the writers and
   return the written path, or ``None`` when a non-strict save failed;
   ``target()`` builds a fresh managed path, and ``prune()`` deletes the
   oldest dumps beyond ``retention`` and returns them (``retention=None``
   keeps everything). Per-call ``options`` override the manager's budgets.
   Pruning follows file modification time, with names breaking ties.

Loading and debugging
---------------------

``load_dumpling(filename, *, max_bytes=64 * 1024 * 1024)``
   Read a trusted compressed dump and return the snapshot dictionary. The byte
   limit applies to decompressed data. ``max_bytes=None`` disables it.
   Invalid gzip, undecodable payloads, unsupported formats, and invalid snapshot
   structures raise ``pydumpling.DumpFormatError``. Filesystem
   errors such as missing files or denied access keep their original types.

New snapshots expose capture budgets, Python version, frame counts, and omission
reasons in ``snapshot_meta``, plus process context: ``created_at`` (UTC
ISO 8601), ``duration_ms``, ``pid``, ``hostname``, ``cwd``, ``argv`` (bounded
like code metadata, past 256 entries reported as ``argv_entries``), and the
capturing ``thread_name``. Environment fields are ``None`` where the
environment cannot provide them. Inspect ``snapshot_meta["truncation"]`` to see which
budgets or unsupported values limited the capture. CLI printing and debugging
also display a snapshot notice when values were summarized.
``snapshot_meta["attempts"]`` lists failed stages, serializers (when applicable),
and exception class names. Omission counts accumulate across reductions;
``limits`` retains the requested budgets and ``effective_limits`` records the
reduced traversal budgets when used. A serializer change alone does not discard
captured variables. Instances of ``dict``, ``list``, ``tuple``, ``set``, and
``frozenset`` subclasses such as ``Counter``, ``OrderedDict``, ``defaultdict``,
or ``namedtuple`` become snapshot subclasses of the same base (``FakeDict``,
``FakeList``, ``FakeTuple``, ``FakeSet``, ``FakeFrozenSet``). They keep every
entry, show the original type name in their ``repr``, and keep instance
attributes, filled ``__slots__``, named tuple fields, and struct sequence fields
such as ``os.stat_result.st_mtime`` as attributes, so ``pp record.x`` still
works. ``OrderedDict`` entries keep their own order, including ``move_to_end``
changes. CPython hashes every key while reading that order, so when a key's type
could run application code during hashing, the entries keep insertion order
instead and ``mapping_order`` is recorded. On Python 3.15+, ``frozendict``,
which is not a ``dict`` subclass, and its subclasses are captured the same way
as ``FakeDict`` copies whose ``repr`` names the original type, so such a
snapshot still loads on older Pythons.
Tuple and frozenset copies hash like the plain values they equal, so
``pp cache[(1, 2)]`` finds a named tuple key; other copies, and those holding
unhashable entries, hash by identity. Every copy therefore stays usable as a set
member or mapping key, and ``pp`` lays copies out across lines. Container
entries are read through the base type, so an application ``__iter__`` or
``items`` override is neither called nor able to hide entries. Types are
compared by identity, so an overridden ``__class__`` or
a metaclass ``__eq__`` or ``__hash__`` is never run.
Instances of ``str``, ``bytes``, ``int``, and ``float`` subclasses, such as
``IntEnum`` and ``StrEnum`` members, become ``FakeStr``, ``FakeBytes``,
``FakeInt``, or ``FakeFloat`` copies. They keep the value, the original type
name in their ``repr``, and instance attributes and filled slots, so
``pp member._name_`` still works, and they compare and hash like the plain value
they equal. The value is read through the base type, so no application
``__str__``, ``__int__``, or ``__getitem__`` runs, and it has the budget of a
plain value: kept at any depth, with text and bytes cut to ``max_string_bytes``
and an integer larger than that replaced by a placeholder. Its attributes and
slots are nested data and follow ``max_depth``.

``format_version`` versions the complete snapshot schema, including FakeCode's
line and position tables, independently of the package ``version``. New writers
emit format 2, which adds the ``FakeStr``, ``FakeBytes``, ``FakeInt``, and
``FakeFloat`` copies. Format 2 only adds types, so readers load format 1 dumps
unchanged; Pydumpling 0.2.0 reads only format 1, rejecting format 2 dumps or,
when they contain the new copies, failing to decode them. Missing format
versions are treated as legacy format 0; unknown versions are rejected. Format
compatibility depends on the schema version; the producer's package version
must be well formed but may belong to another release series. Only legacy
format 0 uses the historical package-version range. Incompatible changes to
internal encodings must increment this format version and provide an explicit
reader migration.

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
Inside a session, ``list`` and ``ll`` show source embedded with
``embed_source_lines`` for files missing on this machine; files that still
exist locally keep showing their local source, and the injection is removed
when the session ends.

Embedded source is stored as a ``source`` mapping of
``{filename: {"lines": {lineno: text}, "total": file_lines}}``. Each line is
bounded like code metadata. When the byte budget is exceeded, the ``source``
reduction stage drops the mapping (reported as ``source_omitted``) before any
captured variables are reduced. The ``minimal`` fallback also omits source.
The ``--html`` reporter prefers embedded source when drawing a frame's
context and falls back to the local file.

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

Timeout stops waiting rather than cancelling a dispatched request and raises
``pydumpling.CaptureTimeoutError``; an interrupted wait raises
``pydumpling.CaptureInterruptedError``. Both carry ``request_directory``. The
request script is retained and may execute later; the CLI prints its directory. Avoid
submitting repeated requests while one is pending. Remove the retained directory
only after its ``status.json`` confirms completion or the target has exited.

Command line
------------

Both ``pydumpling`` and ``python -m pydumpling`` expose the same interface.
Choose exactly one action: ``--print``, ``--json``, ``--info``, ``--html
FILE``, ``--debug``, ``--rdebug``, or ``--capture-pid PID``. The first five
take a dump filename; PID capture takes ``--output FILE`` and optionally
``--timeout SECONDS`` or ``--unlimited``.

``--print`` writes the stored traceback and exception text to standard output,
so the report survives a pipe or a redirect; the snapshot notice and other
diagnostics stay on standard error. Characters the output encoding cannot
represent, such as lone surrogates from undecodable paths, are escaped as
standard error would escape them. If the reader closes the pipe early, as
``head`` does, the command stops quietly with status 141; if only the notice's
reader is gone, the report is still written.

``--json`` prints the same snapshot as an indented JSON object: top-level
capture metadata (format and package versions, Python version, ``created_at``,
``pid``, ``thread_name``, ``duration_ms``), the exception (``type``,
``message``, and ``formatted`` text, or ``null`` for a stack snapshot), a
``frames`` list from outermost to innermost, and the ``truncation`` mapping.
Variable values are guarded ``repr`` strings. ``--globals`` adds each frame's
globals to the JSON output (locals are always included). ``--info`` prints the stored metadata as
a human-readable summary: creation time, versions, capture budgets, effective
budgets after reductions, truncation reasons, and failed save attempts.

``--host`` and ``--port``
configure remote debugging; ``--timeout`` also limits
waiting for a remote client. ``--max-load-mib`` sets
the decompressed loading limit in MiB; zero disables it. Use ``--help`` for all
options and ``--version`` for the installed version.
