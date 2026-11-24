"""Dispatch one stream of snapshot roots to the installed capture engine."""

import sys

pydumpling_native = None
if sys.implementation.name == "cpython" and (3, 12) <= sys.version_info[:2] <= (3, 14):
    import sysconfig

    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        try:
            import pydumpling_native
        except ImportError:
            # A missing or unloadable optional binary leaves the Python
            # reader and writer available. No capture has started here.
            pass


def frame_child(request):
    """Propagate a child's StopIteration past the frame generator unchanged."""
    try:
        return (yield request)
    except StopIteration as error:
        # Let the driver raise it outside the generator (PEP 479).
        yield "raise", error


def run_python(context, program):
    """Resolve the frame/exception adapter's requests using the Python walk."""
    try:
        request = next(program)
        while request[0] != "raise":
            try:
                if request[0] == "value":
                    _, value, depth = request
                    captured = context.value(value, depth)
                else:
                    _, source, depth, priority, counted, container = request
                    captured = context._mapping(
                        source,
                        depth,
                        priority=priority,
                        counted=counted,
                        container=container,
                    )
            except Exception as error:
                request = program.throw(error)
            else:
                request = program.send(captured)
    except StopIteration as done:
        return done.value
    raise request[1]


def run(context, program):
    from ..options import DumpOptions
    from .context import SnapshotContext

    if (
        pydumpling_native is not None
        and type(context) is SnapshotContext
        and type(context.options) is DumpOptions
        and not context._unlimited
        and not context.minimal
    ):
        return pydumpling_native.capture(context, program)
    return run_python(context, program)
