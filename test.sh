#! /usr/bin/env bash

set -e

pip install pytest pytest-order flake8 six

flake8 ./tests ./pydumpling
python -m pytest