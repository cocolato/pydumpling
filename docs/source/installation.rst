Installation and development
============================

Install
-------

Use Python 3.11 or newer:

.. code-block:: console

   python -m pip install pydumpling
   pydumpling --version
   python -m pydumpling --help

This documentation describes version 0.2.0. To try the current source checkout,
run ``python -m pip install .`` from the repository root.

Development
-----------

Use uv and the committed lockfile:

.. code-block:: console

   uv sync --locked --dev --extra doc
   uv run --locked ruff check .
   uv run --locked ruff format --check .
   uv run --locked pytest --cov=pydumpling --cov-report=term-missing
   uv run --locked --extra doc sphinx-build -W --keep-going -b html docs/source docs/build/html
   uv build --sdist
   uv build --wheel dist/pydumpling-0.2.0.tar.gz
   uv run --locked twine check --strict dist/*

Run ``uv lock`` after deliberately changing dependency constraints and commit the
updated lockfile. Hatchling builds packages; uv manages development environments.

CI verifies Python 3.11–3.15 on Linux and Python 3.14 on Windows and macOS.
It checks lint and formatting, enforces at least 85% coverage, builds documentation
with warnings treated as errors, and installs the wheel outside the checkout to
check both CLI entry points. It does not publish releases automatically.
