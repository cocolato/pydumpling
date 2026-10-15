Pydumpling
==========

Pydumpling saves exception or live Python stack snapshots to compressed dump
files. Print a traceback, inspect saved variables with pdb, or use a debugger
over an SSH tunnel. The project is derived from
`elifiner/pydump <https://github.com/elifiner/pydump>`_.

Requires Python 3.11+. Capturing a running process by PID requires CPython 3.14+.
Snapshots preserve captured state for inspection; they cannot resume the
original program.

.. warning::

   Only load trusted dump files. Deserialization with dill or pickle can execute
   code, and snapshots may contain credentials or personal data. Size budgets
   limit resource use, not deserialization privileges.

.. toctree::
   :maxdepth: 2

   installation
   tutorial
   api
   migration
