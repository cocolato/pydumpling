# Python post-mortem debugging

English | [简体中文](README_zh.md)

It's a fork/optimized version from [elifiner/pydump](https://github.com/elifiner/pydump).The main optimization points are：
* Save the `Python traceback` anywhere, not just when it's an exception.
* Optimize code structure && remove redundant code
* fix bug in python3.10+
* supported more pdb commnd
* a useful command line tool for debug


Pydumpling writes the `python current traceback` into a file and 
can later load it in a Python debugger. It works with the built-in 
pdb and with other popular debuggers (pudb, ipdb and pdbpp).

## Why use pydumpling?

* We usually use `try... except ... ` to catch exceptions that occur in our programs, but do we really know why they occur?
* When your project is running online, you suddenly get an unexpected exception that causes the process to exit. How do you reproduce this problem?
* Not enough information in the logs to help us pinpoint online issues?
* If we were able to save the exception error and then use the debugger to recover the traceback at that time, we could see the entire stack variables along the traceback as if you had caught the exception at the local breakpoint.

## Install pydumpling
Python version：>=3.7

```
pip install pydumpling
```

## How to use pydumpling


### Save the python traceback anywhere.
```python
from pydumpling import dump_current_traceback
from inspect import currentframe


def inner():
    a = 1
    b = "2"
    dump_current_traceback("test.dump")
    c = str(a) + b


def outer():
    d = 4
    inner()

```

### Save the exception traceback.

In the code, find the place where we need to do the `try ... except ...` and use `save_dumpling()`. When we save the dump file, it will default to `${exception filename}:${error lineno}.dump`.

```python
from pydumpling import save_dumping

def inner():
    a = 1
    b = "2"
    c = a + b


def outer():
    inner()


if __name__ == "__main__":
    try:
        outer()
    except Exception:
        save_dumping("test.dump")

```

Now we have the `test.dump` file, which we can use `debub_dumpling` to do pdb debug:
```python     
Python 3.10.6 (main, Aug  1 2022, 20:38:21) [GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from pydumpling import debug_dumpling
>>> debug_dumpling("test.dump")
> /home/loyd/vscodeFiles/pydumpling/test.py(6)inner()
-> c = a + b
(Pdb) list 1,17
  1     from pydumpling import save_dumping
  2  
  3     def inner():
  4  >>     a = 1
  5         b = "2"
  6  ->     c = a + b
  7  
  8  
  9     def outer():
 10         inner()
 11  
 12  
 13     if __name__ == "__main__":
 14         try:
 15             outer()
 16         except Exception:
 17             save_dumping("test.dump")
(Pdb) ll
  3     def inner():
  4  >>     a = 1
  5         b = "2"
  6  ->     c = a + b
(Pdb) bt
  /home/loyd/vscodeFiles/pydumpling/test.py(15)<module>()
-> outer()
  /home/loyd/vscodeFiles/pydumpling/test.py(10)outer()
-> inner()
> /home/loyd/vscodeFiles/pydumpling/test.py(6)inner()
-> c = a + b
(Pdb) pp a
1
(Pdb) pp b
'2'
(Pdb) u
> /home/loyd/vscodeFiles/pydumpling/test.py(10)outer()
-> inner()
(Pdb) ll
  9     def outer():
 10  ->     inner()
(Pdb) 
```

### Use Command Line

Use command line to print the traceback:
`python -m pydumpling --print test.deump`

It will print:
```python
Traceback (most recent call last):
  File "/workspaces/pydumpling/tests/test_dump.py", line 20, in test_dumpling
    outer()
  File "/workspaces/pydumpling/tests/test_dump.py", line 14, in outer
    inner()
  File "/workspaces/pydumpling/tests/test_dump.py", line 10, in inner
    c = a + b  # noqa: F841
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```


Use command line to do pdb debug:
`python -m pydumpling --debug test.deump`

It will open the pdb window:
```python
-> c = a + b
(Pdb) 
```

## TODO
- []
