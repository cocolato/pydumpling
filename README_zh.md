# Pydumpling

[English](README.md) | 简体中文

Pydumpling 将 Python 异常现场或当前调用栈保存为压缩的 `.dump` 文件，
之后可以打印 traceback、通过 pdb 查看保存的变量，或启动远程调试会话。
项目源自 [elifiner/pydump](https://github.com/elifiner/pydump)。

运行环境为 Python 3.11+；按 PID 捕获运行中的进程需要 CPython 3.14+。

## 安装

~~~sh
python -m pip install pydumpling
pydumpling --version
~~~

以下示例对应 0.2.0。

## 保存异常现场

将下面的完整示例保存为 `example.py`，运行 `python example.py`：

~~~python
from pydumpling import save_dumping


def calculate():
    quantity = 3
    price = "12.50"
    return quantity + price


try:
    calculate()
except TypeError:
    save_dumping("error.dump", strict=True)
~~~

~~~sh
pydumpling --print error.dump
pydumpling --debug error.dump
# 也可以通过 Python 模块入口运行：
python -m pydumpling --print error.dump
~~~

`--print` 会把 traceback 和异常文本写到标准输出，因此可以正常用于管道和重定向；
省略提示仍然输出到标准错误。

常用 pdb 命令包括 `bt`、`u`、`d`、`list`、`ll` 和 `pp price`。
快照用于检查保存下来的状态，无法恢复或继续执行原来的程序。

## 保存当前调用栈

~~~python
from pydumpling import dump_current_traceback


def calculate():
    quantity = 3
    price = 12.5
    dump_current_traceback("current.dump", strict=True)
    return quantity * price


calculate()
~~~

在应用启动阶段调用一次 `catch_any_exception()`，可以保存主线程的未处理异常。
它安装 `sys.excepthook`，并保留之前的 hook 来正常打印异常，
不会安装线程或 asyncio 的异常处理器。

## 控制快照开销

`DumpOptions()` 的默认预算为：

| 参数 | 默认值 |
| --- | --- |
| `max_frames` | 64 个栈帧 |
| `max_depth` | 5 层嵌套 |
| `max_items` | 每个容器 100 项 |
| `max_string_bytes` | 每个捕获的字符串值 4,096 字节 |
| `max_nodes` | 总计 10,000 个被访问的值 |
| `max_dump_bytes` | 16 MiB 未压缩的序列化数据 |

达到上限时，部分值会以摘要形式保存。可以调整预算，也可以显式取消限制：

~~~python
from pydumpling import DumpOptions, dump_current_traceback

dump_current_traceback(
    "small.dump",
    options=DumpOptions(max_frames=16, max_depth=3, max_items=25),
    strict=True,
)
dump_current_traceback("unlimited.dump", options=DumpOptions.unlimited(), strict=True)
~~~

无限制捕获可能消耗大量资源，并允许执行对象自定义的序列化钩子。
加载后的 `snapshot_meta` 字典记录实际预算和省略原因。
无限制模式保存的原始对象可能依赖原运行环境或自定义加载钩子，
无法保证所有对象都能成功还原；默认的可读快照模式不受此影响。

`save_dumping(filename=None, exc_info=None, *, options=None, strict=False)` 和
`dump_current_traceback(filename=None, *, options=None, strict=False)`
默认通过 warning 报告捕获或写入失败；传入 `strict=True` 可直接抛出异常。
`catch_any_exception(*, options=None)` 支持相同的捕获预算。

加载和调试默认最多读取 64 MiB 解压后的 dump 数据。
使用 `load_dumpling(path, max_bytes=None)` 或
`debug_dumpling(path, max_bytes=None)` 可取消读取限制；
命令行对应 `--max-load-mib 0`。

## 按 PID 捕获运行中的进程

在安装了新版 Pydumpling 的 CPython 3.14+ 环境中运行下面的目标程序：

~~~python
import os
import time

import pydumpling

print(f"PID: {os.getpid()}", flush=True)
counter = 0
while True:
    counter += 1
    time.sleep(0.1)
~~~

在另一个终端，将 `12345` 替换为目标输出的 PID：

~~~sh
pydumpling --capture-pid 12345 --output live.dump --timeout 30
pydumpling --print live.dump
~~~

Python API 为 `capture_pid(pid, filename, *, options=None, timeout=30) -> Path`。
控制端和目标端必须使用相同的 CPython 版本，位于同一台机器，
使用相同用户身份，并共享进程与文件系统命名空间。
目标环境需要安装新版 Pydumpling 并允许 CPython 远程执行；
操作系统权限也可能阻止附加。

超时仅停止等待结果，已经发送的请求仍可能稍后在目标进程中执行。
超时时命令行会打印保留的请求目录。前一次请求尚未结束时，应避免反复提交。
传入 `--unlimited` 可显式取消捕获预算。
确认目录内的 `status.json` 已记录执行完成，或目标进程已经退出后，再清理请求文件。

## 远程查看

调试器默认监听 `127.0.0.1:4444`：

~~~sh
# 服务器上：
pydumpling --rdebug error.dump --host 127.0.0.1 --port 4444

# 本地工作站上：
ssh -L 4444:127.0.0.1:4444 user@server
# 再打开一个本地终端：
nc 127.0.0.1 4444
~~~

远程 pdb 会话可以执行 Python 表达式，没有内置身份验证和加密。
请保持监听回环地址，并通过 SSH 隧道访问。

## dump 文件的使用边界

只加载可信来源的 dump：dill 和 pickle 反序列化可能执行代码。
体积限制无法让不可信的文件变得安全。快照可能包含局部变量、凭据、
个人数据和应用状态，请控制文件权限、保留期限和共享范围。

查看 dump 时使用匹配的 Python 解释器与源码版本。
跨 Python 版本加载以及第三方调试器兼容性没有保证。
旧版 0.1 dump 中的对象与当前解释器兼容时仍可读取。
升级变化见 [0.2 迁移说明](CHANGELOG.md)。

## 开发

~~~sh
uv sync --locked --dev --extra doc
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest --cov=pydumpling --cov-report=term-missing
uv run --locked --extra doc sphinx-build -W --keep-going -b html docs/source docs/build/html
uv build --sdist
uv build --wheel dist/pydumpling-0.2.0.tar.gz
uv run --locked twine check --strict dist/*
~~~

仓库提交 `uv.lock`。主动修改依赖后运行 `uv lock` 并一并提交新的锁文件。
CI 检查 Linux 上的 Python 3.11–3.15，以及 Windows/macOS 上的 Python 3.14；
执行 85% 覆盖率检查、严格文档构建，并在源码目录外验证安装后的 wheel。
Hatchling 只负责构建发行包，开发环境统一由 uv 管理。
