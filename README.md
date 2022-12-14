# Python post-mortem debugging

English | [简体中文](README_zh.md)

It's a fork/optimized version from [elifiner/pydump](https://github.com/elifiner/pydump).The main optimization points are：
* Optimize code structure && remove redundant code
* fix bug in python2.7 && support python3.10+
* supported more pdb commnd


Pydump writes the traceback of an exception into a file and 
can later load it in a Python debugger. It works with the built-in 
pdb and with other popular debuggers (pudb, ipdb and pdbpp).

## Why use pydumpling?

* We usually use `try... except ... ` to catch exceptions that occur in our programs, but do we really know why they occur?
* When your project is running online, you suddenly get an unexpected exception that causes the process to exit. How do you reproduce this problem?
* Not enough information in the logs to help us pinpoint online issues?
* If we were able to save the exception error and then use the debugger to recover the traceback at that time, we could see the entire stack variables along the traceback as if you had caught the exception at the local breakpoint.

## Install pydumpling
Python version：>= 2.7, >=3.6

Not published in pypi，so use the `.whl` file install pydumpling in the dist path.
```
pip install dist/pydumpling-0.1.0-py2.py3-none-any.whl
```

## How to use pydumpling

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
## TODO
- []
