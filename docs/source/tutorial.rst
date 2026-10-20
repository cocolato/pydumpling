Tutorial
========

Save an exception
-----------------

Save the following as ``example.py`` and run ``python example.py``:

.. code-block:: python

   from pydumpling import save_dumping


   def calculate():
       quantity = 3
       price = "12.50"
       return quantity + price


   try:
       calculate()
   except TypeError:
       save_dumping("error.dump", strict=True)

Print the snapshot or start pdb:

.. code-block:: console

   pydumpling --print error.dump
   pydumpling --debug error.dump

Use ``bt``, ``u``, ``d``, ``list``, ``ll``, and ``pp price`` to inspect the stack
and variables. A dump is a saved snapshot, not a resumable Python process.
Keep the matching Python interpreter and source checkout available.

Save the current stack
----------------------

.. code-block:: python

   from pydumpling import dump_current_traceback


   def calculate():
       quantity = 3
       price = 12.5
       dump_current_traceback("current.dump", strict=True)
       return quantity * price


   calculate()

For unhandled main-thread exceptions, call ``catch_any_exception()`` once during
application startup. It retains the previous ``sys.excepthook`` for normal
reporting. Pass ``include_threads=True`` to also capture exceptions that end a
thread, and ``restore_exception_hooks()`` to stop saving snapshots again. For
asyncio, call ``install_asyncio_handler()`` inside async code. To keep every
snapshot in one directory with bounded retention, build a
``DumpManager("dumps", retention=20)`` and pass it as ``manager`` to either
hook function; hooks and plain saves then write uniquely named files, with
older dumps pruned by file modification time. Without a manager, hooks keep
adding dumps to the working directory.

See :doc:`api` for capture budgets, strict failures, and loading limits.

Capture a running process
-------------------------

Run this target program with CPython 3.14+ and updated Pydumpling installed:

.. code-block:: python

   import os
   import time

   import pydumpling

   print(f"PID: {os.getpid()}", flush=True)
   counter = 0
   while True:
       counter += 1
       time.sleep(0.1)

In another terminal, replace ``12345`` with the printed PID:

.. code-block:: console

   pydumpling --capture-pid 12345 --output live.dump --timeout 30
   pydumpling --print live.dump

The controller and target must use the same CPython version, be on the same
machine under the same user, and share process and filesystem namespaces. The
target must have updated Pydumpling installed and permit CPython remote
execution. Operating system permissions may also prevent attachment.

A timeout ends the controller's wait; it cannot cancel an already dispatched
request. The target may execute the retained request script later. The CLI prints
the request directory on timeout. Avoid submitting repeated pending requests.
Remove retained request files only after ``status.json`` confirms completion or
the target has exited.
Use ``--unlimited`` to disable capture budgets explicitly.

Remote pdb through SSH
----------------------

Run the debugger on the server, then forward the loopback port from your
workstation:

.. code-block:: console

   # Server:
   pydumpling --rdebug error.dump --host 127.0.0.1 --port 4444
   # Workstation:
   ssh -L 4444:127.0.0.1:4444 user@server
   # A second local terminal:
   nc 127.0.0.1 4444

The server defaults to ``127.0.0.1:4444``. Remote pdb allows Python expression
execution and has no built-in authentication or encryption, so keep it on
loopback and use an SSH tunnel.

Trust and privacy
-----------------

Only deserialize trusted dumps. The default 64 MiB decompressed loading limit
does not prevent dill or pickle from executing code. Snapshot values may include
credentials and personal data; control file permissions, retention, and
sharing. Third-party debugger and cross-Python-version compatibility are not
guaranteed.

