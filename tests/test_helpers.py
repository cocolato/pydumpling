import pytest
from argparse import ArgumentTypeError

from pydumpling.helpers import validate_file_name


def test_validate_file_name():
    dump_file = "./tests/dump/validate_file_name.dump"
    assert validate_file_name(dump_file) == dump_file

    with pytest.raises(ArgumentTypeError, match="File must be .dump file"):
        validate_file_name("test.txt")

    with pytest.raises(ArgumentTypeError, match="File missing.dump not found"):
        validate_file_name("missing.dump")
