import argparse
import os
from unittest.mock import patch, mock_open

import pytest

from pydumpling.cli import main, parser
from pydumpling.helpers import validate_file_name


@patch('os.path.exists')
def test_validate_file_name_valid(mock_exists):
    mock_exists.return_value = True
    result = validate_file_name("test.dump")
    assert result == "test.dump"
    mock_exists.assert_called_once_with("test.dump")


def test_validate_file_name_invalid():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        validate_file_name("test.txt")
    assert str(exc.value) == "File must be .dump file"


def test_validate_file_name_empty():
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        validate_file_name("")
    assert str(exc.value) == "File must be .dump file"


@patch('os.path.exists')
@patch('pydumpling.cli.load_dumpling')
@patch('pydumpling.cli.print_traceback_and_except')
def test_main_print(mock_print, mock_load, mock_exists):
    mock_exists.return_value = True
    test_args = ['--print', 'test.dump']
    with patch('sys.argv', ['pydumpling'] + test_args):
        main()
        mock_load.assert_called_once_with('test.dump')
        mock_print.assert_called_once_with(mock_load.return_value)


@patch('os.path.exists')
@patch('pydumpling.cli.debug_dumpling')
def test_main_debug(mock_debug, mock_exists):
    mock_exists.return_value = True
    test_args = ['--debug', 'test.dump']
    with patch('sys.argv', ['pydumpling'] + test_args):
        main()
        mock_debug.assert_called_once_with('test.dump')


@patch('os.path.exists')
@patch('pydumpling.cli.r_post_mortem')
def test_main_rdebug(mock_rpost, mock_exists):
    mock_exists.return_value = True
    test_args = ['--rdebug', 'test.dump']
    with patch('sys.argv', ['pydumpling'] + test_args):
        main()
        mock_rpost.assert_called_once_with('test.dump')


def test_parser_no_args():
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_multiple_actions():
    with pytest.raises(SystemExit):
        parser.parse_args(['--print', '--debug', 'test.dump'])


def test_parser_missing_filename():
    with pytest.raises(SystemExit):
        parser.parse_args(['--print'])


@patch('os.path.exists')
def test_parser_invalid_filename(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(SystemExit):
        parser.parse_args(['--print', 'test.txt'])


def test_parser_help(capsys):
    with pytest.raises(SystemExit):
        parser.parse_args(['--help'])
    captured = capsys.readouterr()
    assert 'pydumpling cli tools' in captured.out


@patch('os.path.exists')
def test_validate_file_name_not_found(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(argparse.ArgumentTypeError) as exc:
        validate_file_name("test.dump")
    assert str(exc.value) == "File test.dump not found"
