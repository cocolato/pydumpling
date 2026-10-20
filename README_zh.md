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

## 机器可读与元数据视图

~~~sh
pydumpling --json error.dump                # 缩进 JSON 报告
pydumpling --json --globals error.dump       # 额外包含每帧 globals
pydumpling --info error.dump                 # 版本、预算、截断信息
~~~

JSON 对象包含捕获元数据（`created_at`、`pid`、`thread_name`、`duration_ms`、
各版本号）、异常（`type`、`message`、`formatted` 文本；栈快照为 `null`）、
从最外层到最内层的 `frames` 列表，以及 `truncation` 映射；
变量值是带保护的 `repr` 字符串，可直接接入 `jq` 和 CI 工具链。

## 嵌入源码与 HTML 报告

~~~python
from pydumpling import DumpOptions, save_dumping

try:
    risky_step()
except ValueError:
    save_dumping("error.dump", options=DumpOptions(embed_source_lines=5), strict=True)
~~~

`embed_source_lines` 会在 dump 内保存每帧当前行上下各 N 行源码
（`None` 表示整个文件）。本地文件缺失或已与之不一致时，`--print`、`--debug`、
`--rdebug` 和 `--html` 都显示运行时的源码；本地文件未改动时仍显示完整的本地版本。
超出 dump 字节预算时，先舍弃内嵌源码，再缩减捕获的变量。

~~~sh
pydumpling --html report.html error.dump   # 单文件 HTML，无脚本依赖
~~~

HTML 报告展示异常、每帧的源码窗口（按同样规则选取，并高亮当前行）
和 locals 表格。

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
它安装 `sys.excepthook`，并保留之前的 hook 来正常打印异常。
传入 `include_threads=True` 可同时捕获结束线程的异常（线程内的 `SystemExit`
属于正常退出，不保存快照），`restore_exception_hooks()` 可停用已安装的全部钩子；
其他库套在外层的钩子会保留，被套住的本库钩子只负责转发。
asyncio 场景在异步代码内调用 `install_asyncio_handler()`（或显式传入 `loop=`），
捕获事件循环上报的异常。不使用管理器时，每个钩子都会在工作目录新建 dump 且从不清理，
长期运行的服务应传入 `DumpManager`。

## 管理快照目录

`DumpManager` 把快照统一保存到一个目录，文件名唯一并限制保留数量：

~~~python
from pydumpling import DumpManager

manager = DumpManager("dumps", retention=20)
try:
    risky_step()
except ValueError:
    manager.save_exception()  # 当前栈用 save_current()
~~~

默认文件名现在包含源码行号、UTC 时间戳、pid 和进程内序号，
同一处反复出错也不会覆盖之前的快照。管理器按文件修改时间清理旧快照；
保存间隔短到文件系统给出相同时间时，按文件名里的时间戳和序号保持写入顺序。
`target()` 会先创建管理目录，再返回新路径。
钩子可直接接入管理器：`catch_any_exception(manager=manager, include_threads=True)`、
`install_asyncio_handler(manager=manager)`。
每份快照的 `snapshot_meta` 记录创建时间、耗时、pid、主机名、工作目录、
命令行和捕获线程。

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
`catch_any_exception(*, options=None, include_threads=False, manager=None)`
支持相同的捕获预算，`install_asyncio_handler(loop=None, *, options=None, manager=None)`
覆盖事件循环；两者都保留原有钩子的正常上报。

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
