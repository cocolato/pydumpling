"""Installation of the optional exception and event-loop hooks."""

import sys
import threading
import weakref

from .dump_manager import DumpManager
from .options import DumpOptions
from .pydumpling import save_dumping

# The one hook per hook point that saves snapshots. A hook another package
# installed on top of ours cannot be unwound without dropping that package's
# hook, so restoring or reinstalling deactivates ours instead: an inactive
# hook only forwards to the hook it replaced.
_active_hooks = {}
# The active handler this package installed on each loop; loops are weak keys.
_asyncio_handlers = weakref.WeakKeyDictionary()


def catch_any_exception(
    *,
    options: DumpOptions | None = None,
    include_threads: bool = False,
    manager: DumpManager | None = None,
) -> None:
    """Install snapshot hooks and always preserve the previous hooks.

    Covers unhandled main-thread exceptions and, with ``include_threads``,
    exceptions that end a thread; ``SystemExit`` in a thread is a normal exit
    and saves nothing. ``manager`` places each dump in its managed directory
    and applies retention; give capture budgets to the manager (or to
    ``options`` when no manager is used), never to both. Calling it again
    applies the new settings to every hook it already installed.
    """
    _check_settings(options, manager)
    _install(sys, _excepthook, options, manager)
    if include_threads:
        _install(threading, _thread_hook, options, manager)
    elif threading in _active_hooks:
        _configure(_active_hooks[threading], options, manager)


def _check_settings(options, manager) -> None:
    if options is not None and not isinstance(options, DumpOptions):
        raise TypeError("options must be a DumpOptions instance or None")
    if manager is not None and not isinstance(manager, DumpManager):
        raise TypeError("manager must be a DumpManager instance or None")
    if manager is not None and options is not None:
        raise ValueError("pass capture budgets through the manager, not options")


def _configure(hook, options, manager, registered=None) -> None:
    """Make ``hook`` the one saving hook of its hook point, with these settings.

    ``registered`` is the hook of ours that was saving there before; when it
    is another one, a later package wrapped it, so it now only forwards.
    """
    if registered is not None and registered is not hook:
        registered._pydumpling_active = False
    hook._pydumpling_active = True
    hook._pydumpling_options = options
    hook._pydumpling_manager = manager


def _save_via(manager, exc_info, options):
    if manager is not None:
        manager.save_exception(exc_info)
    else:
        save_dumping(exc_info=exc_info, options=options)


def _owns(hook) -> bool:
    """Whether this package installed ``hook`` itself.

    The marker refers to the hook it was set on. ``functools.wraps`` copies it
    onto a third-party wrapper, where it then refers to another function.
    """
    return hook is not None and getattr(hook, "_pydumpling_hook", None) is hook


def _chained(current, factory):
    """``current`` when it is ours, else a new hook of ours forwarding to it."""
    if _owns(current):
        return current
    hook = factory()
    hook._pydumpling_hook = hook
    hook._pydumpling_previous = current
    return hook


def _install(module, factory, options, manager) -> None:
    """Make one pydumpling hook active on ``module.excepthook``."""
    hook = module.excepthook = _chained(module.excepthook, factory)
    _configure(hook, options, manager, _active_hooks.get(module))
    _active_hooks[module] = hook


def _excepthook():
    def hook(exc_type, exc_value, exc_tb):
        try:
            if hook._pydumpling_active:
                _save_via(
                    hook._pydumpling_manager,
                    (exc_type, exc_value, exc_tb),
                    hook._pydumpling_options,
                )
        finally:
            hook._pydumpling_previous(exc_type, exc_value, exc_tb)

    return hook


def _thread_hook():
    def hook(args):
        try:
            # Like the main thread, where SystemExit never reaches
            # sys.excepthook, a thread calling sys.exit() is not a failure.
            if (
                hook._pydumpling_active
                and args.exc_traceback is not None
                and not (
                    isinstance(args.exc_type, type)
                    and issubclass(args.exc_type, SystemExit)
                )
            ):
                _save_via(
                    hook._pydumpling_manager,
                    (args.exc_type, args.exc_value, args.exc_traceback),
                    hook._pydumpling_options,
                )
        finally:
            hook._pydumpling_previous(args)

    return hook


def install_asyncio_handler(
    loop=None,
    *,
    options: DumpOptions | None = None,
    manager: DumpManager | None = None,
) -> None:
    """Save snapshots for exceptions an asyncio loop reports.

    Call it from async code to hook the running loop, or pass ``loop``
    explicitly. The loop's previous handler, or its default, still reports
    each exception afterwards. ``manager`` and ``options`` work as for
    :func:`catch_any_exception`. See :func:`restore_exception_hooks` for
    removal.
    """
    _check_settings(options, manager)
    import asyncio

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError(
                "install_asyncio_handler requires a running event loop; "
                "pass loop= explicitly outside async code"
            ) from error
    handler = _chained(loop.get_exception_handler(), _asyncio_handler)
    loop.set_exception_handler(handler)
    try:
        registered = _asyncio_handlers.get(loop)
        _asyncio_handlers[loop] = handler
    except TypeError:  # pragma: no cover - a loop type without weakref support
        registered = None
    _configure(handler, options, manager, registered)


def _asyncio_handler():
    def handler(loop, context):
        try:
            exc = context.get("exception")
            if (
                handler._pydumpling_active
                and isinstance(exc, BaseException)
                and exc.__traceback__ is not None
            ):
                _save_via(
                    handler._pydumpling_manager,
                    (type(exc), exc, exc.__traceback__),
                    handler._pydumpling_options,
                )
        finally:
            previous = handler._pydumpling_previous
            if previous is None:
                loop.default_exception_handler(context)
            else:
                previous(loop, context)

    return handler


def _unwrapped(hook):
    """The first hook below this package's own at the top of a chain."""
    while _owns(hook):
        hook = hook._pydumpling_previous
    return hook


def restore_exception_hooks() -> None:
    """Stop every snapshot hook this package installed.

    Restores the ``sys.excepthook`` and ``threading.excepthook`` that were
    active when :func:`catch_any_exception` ran, and the previous handler of
    every loop :func:`install_asyncio_handler` covered. A hook another
    package installed on top of ours stays installed; ours below it stops
    saving snapshots and only forwards to the hook it replaced.
    """
    for hook in (*_active_hooks.values(), *_asyncio_handlers.values()):
        hook._pydumpling_active = False
    _active_hooks.clear()
    for module in (sys, threading):
        current = module.excepthook
        previous = _unwrapped(current)
        if previous is not current:
            module.excepthook = previous
    for loop in list(_asyncio_handlers):
        try:
            current = loop.get_exception_handler()
            previous = _unwrapped(current)
            if previous is not current:
                loop.set_exception_handler(previous)
        except Exception:
            pass  # A closed loop's handler can no longer run anyway.
    _asyncio_handlers.clear()
