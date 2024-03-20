Welcome to Pydumpling
=======================

.. image:: https://img.shields.io/pypi/dm/pydumpling
   :alt: PyPI - Downloads
   :target: https://pypi.org/project/pydumpling/

.. image:: https://img.shields.io/pypi/v/pydumpling
   :alt: PyPI - Version
   :target: https://pypi.org/project/pydumpling/

.. image:: https://img.shields.io/github/stars/cocolato/pydumpling
   :alt: GitHub Repo stars
   :target: https://github.com/cocolato/pydumpling/



It's a fork/optimized version from `elifiner/pydump`_ .The main optimization points are:

* Save the ``Python traceback`` anywhere, not just when it's an exception.
* Optimize code structure && remove redundant code
* fix bug in python2.7 && support python3.10+
* supported more pdb command
* provides command line tools

.. _elifiner/pydump: https://github.com/elifiner/pydump

Pydumpling writes the ``python current traceback`` into a file and
can later load it in a Python debugger. It works with the built-in
pdb and with other popular debuggers (`pudb`_, `ipdb`_ and `pdbpp`_).

.. _pudb: https://github.com/inducer/pudb
.. _ipdb: https://github.com/gotcha/ipdb
.. _pdbpp: https://github.com/pdbpp/pdbpp

Why use Pydumpling?
-------------------

* We usually use ``try... except ... `` to catch exceptions that occur in our programs, but do we really know why they occur?
* When your project is running online, you suddenly get an unexpected exception that causes the process to exit. How do you reproduce this problem?
* Not enough information in the logs to help us pinpoint online issues?
* If we were able to save the exception error and then use the debugger to recover the traceback at that time, we could see the entire stack variables along the traceback as if you had caught the exception at the local breakpoint.


User's Guide
------------

Get started with :doc:`installation`
and then get an overview with the :doc:`tutorial` that shows how to use.

.. toctree::
   :maxdepth: 4

   installation
   tutorial