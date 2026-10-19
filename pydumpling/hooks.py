"""Installation of the optional exception and event-loop hooks."""

import sys
import threading
import weakref

from .dump_manager import DumpManager
from .options import DumpOptions
from .pydumpling import save_dumping

# Loops whose exception handler this package installed, so
# restore_exception_hooks can find them again; loops are weakly referenced.
_asyncio_loops = weakref.WeakSet()


def catch_any_exception(
    *,
    options: DumpOptions | None = None,
    include_threads: bool = False,
    manager: DumpManager | None = None,
) -> None:
    """Install snapshot hooks and always preserve the previous hooks.

    Covers unhandled main-thread exceptions and, with ``include_threads``,
    exceptions that end a thread. ``manager`` places each dump in its managed
    directory and applies retention; give capture budgets to the manager (or
    to ``options`` when no manager is used), never to both.
    """
    if options is not None and not isinstance(options, DumpOptions):
        raise TypeError("options must be a DumpOptions instance or None")
    if manager is not None and not isinstance(manager, DumpManager):
        raise TypeError("manager must be a DumpManager instance or None")
    if manager is not None and options is not None:
        raise ValueError("pass capture budgets through the manager, not options")
    _install_excepthook(options, manager)
    if include_threads:
        _install_thread_hook(options, manager)


def _save_via(manager, exc_info, options):
    if manager is not None:
        manager.save_exception(exc_info)
    else:
        save_dumping(exc_info=exc_info, options=options)


def _install_excepthook(options, manager) -> None:
    current = sys.excepthook
    if getattr(current, "_pydumpling_hook", False):
        current._pydumpling_options = options
        current._pydumpling_manager = manager
        return

    def hook(exc_type, exc_value, exc_tb):
        try:
            _save_via(
                hook._pydumpling_manager,
                (exc_type, exc_value, exc_tb),
                hook._pydumpling_options,
            )
        finally:
            hook._pydumpling_previous(exc_type, exc_value, exc_tb)

    hook._pydumpling_hook = True
    hook._pydumpling_options = options
    hook._pydumpling_manager = manager
    hook._pydumpling_previous = current
    sys.excepthook = hook


def _install_thread_hook(options, manager) -> None:
    current = threading.excepthook
    if getattr(current, "_pydumpling_hook", False):
        current._pydumpling_options = options
        current._pydumpling_manager = manager
        return

    def hook(args):
        try:
            if args.exc_traceback is not None:
                _save_via(
                    hook._pydumpling_manager,
                    (args.exc_type, args.exc_value, args.exc_traceback),
                    hook._pydumpling_options,
                )
        finally:
            hook._pydumpling_previous(args)

    hook._pydumpling_hook = True
    hook._pydumpling_options = options
    hook._pydumpling_manager = manager
    hook._pydumpling_previous = current
    threading.excepthook = hook


def install_asyncio_handler(loop=None, *, options: DumpOptions | None = None) -> None:
    """Save snapshots for exceptions an asyncio loop reports.

    Call it from async code to hook the running loop, or pass ``loop``
    explicitly. The loop's previous handler, or its default, still reports
    each exception afterwards. See :func:`restore_exception_hooks` for
    removal.
    """
    if options is not None and not isinstance(options, DumpOptions):
        raise TypeError("options must be a DumpOptions instance or None")
    import asyncio

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError(
                "install_asyncio_handler requires a running event loop; "
                "pass loop= explicitly outside async code"
            ) from error
    current = loop.get_exception_handler()
    if getattr(current, "_pydumpling_handler", False):
        current._pydumpling_options = options
        return

    def handler(loop, context):
        try:
            exc = context.get("exception")
            if isinstance(exc, BaseException) and exc.__traceback__ is not None:
                save_dumping(
                    exc_info=(type(exc), exc, exc.__traceback__),
                    options=handler._pydumpling_options,
                )
        finally:
            previous = handler._pydumpling_previous
            if previous is None:
                loop.default_exception_handler(context)
            else:
                previous(loop, context)

    handler._pydumpling_handler = True
    handler._pydumpling_options = options
    handler._pydumpling_previous = current
    loop.set_exception_handler(handler)
    try:
        _asyncio_loops.add(loop)
    except TypeError:  # pragma: no cover - a loop type without weakref support
        pass


def restore_exception_hooks() -> None:
    """Remove the hooks this package installed, keeping later third-party ones.

    Restores the ``sys.excepthook`` and ``threading.excepthook`` that were
    active when :func:`catch_any_exception` ran, and the previous handler of
    every loop :func:`install_asyncio_handler` covered. Hooks another package
    installed on top of ours stay installed.
    """
    current = sys.excepthook
    if getattr(current, "_pydumpling_hook", False):
        sys.excepthook = current._pydumpling_previous
    thread_hook = threading.excepthook
    if getattr(thread_hook, "_pydumpling_hook", False):
        threading.excepthook = thread_hook._pydumpling_previous
    for loop in list(_asyncio_loops):
        try:
            handler = loop.get_exception_handler()
            if getattr(handler, "_pydumpling_handler", False):
                loop.set_exception_handler(handler._pydumpling_previous)
        except Exception:
            pass  # A closed loop's handler can no longer run anyway.
    _asyncio_loops.clear()
