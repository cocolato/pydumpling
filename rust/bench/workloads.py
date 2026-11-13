"""Exception workloads for benchmarking pydumpling's value walk.

The module imports a handful of modules and defines constants, a dataclass and
helpers, like a typical application module, so frame globals look realistic.
It must not import pydumpling: the benchmark imports it into the legacy copy,
the pure-Python repo code and the native walker alike.

Imports and locals that are never read exist so the capture has them to walk.
"""

# ruff: noqa: F401, F841

import argparse
import importlib.machinery
import json
import linecache
import logging
import os
import re
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
RETRY_CODES = (429, 500, 502, 503)
SERVICE_NAME = "orders-api"
FEATURE_FLAGS = {"fast_path": True, "audit": False, "region": "eu-west-1"}
_PATTERN = re.compile(r"^(?P<kind>[a-z]+)-(?P<id>\d+)$")


@dataclass
class Order:
    order_id: int
    customer: str
    items: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Repository:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = OrderedDict()
        self.hits = 0

    def load(self, order_id):
        self.hits += 1
        return self.cache.get(order_id)


# ----------------------------------------------------------------- (a) typical


def _validate(order, limits):
    total = sum(price * qty for _, price, qty in order.items)
    threshold = limits["max_total"]
    currency = order.meta.get("currency", "EUR")
    if total > threshold:
        raise ValueError(f"order {order.order_id} total {total} exceeds {threshold}")
    return total


def _price(order, repo, limits):
    discount = 0.15
    tags = ["priority", "b2b", "eu"]
    attempt = 2
    return _validate(order, limits) * (1 - discount)


def _handle(request, repo):
    order = Order(
        request["id"],
        request["customer"],
        [("sku-1", 19.99, 3), ("sku-2", 5.5, 10), ("sku-3", 120.0, 1)],
        {"currency": "EUR", "channel": "web"},
    )
    limits = {"max_total": 100, "max_items": 50}
    user_agent = "Mozilla/5.0 (X11; Linux x86_64)"
    started = 1730800000.123
    return _price(order, repo, limits)


def typical_exc_info():
    """4 frames with a dozen small locals."""
    repo = Repository("/srv/orders")
    request = {"id": 4711, "customer": "ACME GmbH", "path": "/v1/orders/4711"}
    try:
        _handle(request, repo)
    except ValueError:
        return sys.exc_info()


# -------------------------------------------------------------- (b) heavy locals


def _heavy_inner(numbers, table, nested):
    checksum = len(numbers) + len(table)
    raise KeyError(f"missing shard for checksum {checksum}")


def _heavy_outer():
    numbers = list(range(50_000))
    table = {f"key-{i:05d}": [i, i + 1, i + 2] for i in range(20_000)}
    nested = {
        f"region-{a}": {
            f"zone-{b}": {
                f"rack-{c}": {"load": c * 0.5, "ok": c % 3 == 0} for c in range(10)
            }
            for b in range(10)
        }
        for a in range(10)
    }
    blob = "x" * 100_000
    return _heavy_inner(numbers, table, nested)


def heavy_exc_info():
    """3 frames; a 50k-int list, a 20k str->list dict, a nested dict, 100 KB str."""
    try:
        _heavy_outer()
    except KeyError:
        return sys.exc_info()


# --------------------------------------------------------------- (c) deep stack


def _deep(level, payload):
    name = f"level-{level}"
    weight = level * 1.5
    flags = [level % 2 == 0, level % 3 == 0]
    if level == 0:
        raise RuntimeError(f"bottom reached with {payload['tag']}")
    return _deep(level - 1, payload)


def deep_exc_info(depth=60):
    """depth + 2 frames (62 by default), 3-4 small locals each."""
    try:
        _deep(depth, {"tag": "deep", "n": depth})
    except RuntimeError:
        return sys.exc_info()


# ------------------------------------------------- (d) frames in large source files


def _parse_port(text):
    port = int(text)
    if port > 1024:
        # Not an ArgumentTypeError: argparse lets it propagate through its frames.
        raise RuntimeError(f"port {port} must be privileged")
    return port


def _json_hook(obj):
    if "port" in obj:
        parser = argparse.ArgumentParser(prog="svc")
        parser.add_argument("--port", type=_parse_port)
        parser.add_argument("--host", default="localhost")
        return parser.parse_args(["--port", str(obj["port"])])
    return obj


def stdlib_exc_info():
    """A failure raised through argparse.py and json/*.py frames."""
    try:
        json.loads('{"service": {"port": 8080}}', object_hook=_json_hook)
    except RuntimeError:
        return sys.exc_info()


# ---------------------------------------------- (e) namespace-heavy module globals

# Module imports and imported names at the top of every generated layer.
_LAYER_IMPORTS = (
    "abc",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "hashlib",
    "itertools",
    "json",
    "logging",
    "math",
    "operator",
    "os",
    "pathlib",
    "re",
    "sys",
    "time",
    "types",
    "typing",
)
_LAYER_FROM_IMPORTS = (
    ("collections", ("OrderedDict", "defaultdict", "deque", "namedtuple")),
    ("dataclasses", ("dataclass", "field")),
    ("datetime", ("datetime", "timedelta", "timezone")),
    ("enum", ("Enum", "IntEnum")),
    ("functools", ("lru_cache", "partial", "wraps")),
    ("pathlib", ("Path",)),
    ("typing", ("Any", "Optional")),
)
# One definition kind per generated block, cycled; 26 cycles of 9 kinds plus
# the fixed header and footer give each layer about 300 globals.
_LAYER_BLOCKS = 26


def _layer_block(kind, k):
    """Source for one module-level definition of the given kind."""
    if kind == 0:
        return f"def helper_{k:03d}(value, scale=1):\n    return value * scale + {k}\n"
    if kind == 1:
        return (
            f'@_route("/v1/items/{k}")\n'
            f"def view_{k:03d}(request):\n"
            f'    return {{"id": {k}, "request": request}}\n'
        )
    if kind == 2:
        return (
            f"class Model{k:03d}:\n"
            f'    table = "model_{k:03d}"\n\n'
            "    def __init__(self, pk):\n"
            "        self.pk = pk\n\n"
            "    def save(self):\n"
            "        return self.pk\n"
        )
    if kind == 3:
        return (
            f"def check_{k:03d}(record):\n"
            f'    return record.get("layer") == LAYER and {k} >= 0\n'
        )
    if kind == 4:
        return (
            "@dataclass\n"
            f"class Item{k:03d}:\n"
            "    id: int = 0\n"
            '    name: str = ""\n'
            "    tags: tuple = ()\n"
        )
    if kind == 5:
        return f"scaled_{k:03d} = partial(helper_{k - 5:03d}, scale=2)\n"
    if kind == 6:
        return (
            "@lru_cache(maxsize=32)\n"
            f"def cached_{k:03d}(key):\n"
            f"    return hash((key, {k}))\n"
        )
    if kind == 7:
        return f'instance_{k:03d} = Model{k - 5:03d}("pk-{k}")\n'
    return f"class Status{k:03d}(IntEnum):\n    OK = 0\n    RETRY = 1\n    FAILED = 2\n"


def _layer_source(index):
    """Source of one generated application module, about 300 globals."""
    parts = [f'"""Generated layer {index} of the namespace-heavy workload."""\n']
    parts.extend(f"import {name}\n" for name in _LAYER_IMPORTS)
    parts.extend(
        f"from {module} import {', '.join(names)}\n"
        for module, names in _LAYER_FROM_IMPORTS
    )
    parts.append(
        "\nlog = logging.getLogger(__name__)\n"
        f"LAYER = {index}\n"
        f'SERVICE = "svc-layer-{index:02d}"\n'
        "TIMEOUT = 2.5\n"
        "RETRIES = (1, 2, 4, 8)\n"
        f'SETTINGS = {{"layer": {index}, "debug": False, "tags": ["a", "b"]}}\n'
        "_ROUTE_PREFIX = re.compile(r'^/v[0-9]+/')\n"
        "\n\n"
        "def _route(path):\n"
        "    def decorate(func):\n"
        "        @wraps(func)\n"
        "        def wrapper(*args, **kwargs):\n"
        "            return func(*args, **kwargs)\n\n"
        "        wrapper.route = path\n"
        '        wrapper.methods = ("GET", "POST")\n'
        "        return wrapper\n\n"
        "    return decorate\n\n\n"
    )
    for k in range(_LAYER_BLOCKS * 9):
        parts.append(_layer_block(k % 9, k) + "\n\n")
    parts.append(
        "class Registry:\n"
        "    def __init__(self):\n"
        "        self.handlers = {}\n"
        "        self.hits = 0\n\n"
        "    def register(self, name, func):\n"
        "        self.handlers[name] = func\n"
        "        return func\n\n\n"
        "REGISTRY = Registry()\n"
        "REGISTRY.register('helper', helper_000)\n"
        "REGISTRY.register('view', view_001)\n"
        "DEFAULT_ITEM = Item004(id=LAYER, name=SERVICE, tags=('default',))\n\n\n"
        "def enter(payload, depth):\n"
        "    handler = Model002(payload['path'])\n"
        '    record = {"layer": LAYER, "depth": depth, "service": SERVICE}\n'
        "    if check_003(record) and REGISTRY.handlers:\n"
        '        log.debug("enter %s", record)\n'
        "    if NEXT is None:\n"
        "        raise LookupError(f\"no route for {payload['path']} at {SERVICE}\")\n"
        "    return NEXT(payload, depth + 1)\n\n\n"
        "NEXT = None\n"
    )
    return "".join(parts)


def _make_layer(index):
    name = f"bench_namespace.layer{index:02d}"
    # A path that does not exist; linecache holds the source, as for a
    # module loaded from a zip or a frozen bundle.
    filename = f"/bench_namespace/layer{index:02d}.py"
    source = _layer_source(index)
    module = types.ModuleType(name)
    loader = importlib.machinery.SourceFileLoader(name, filename)
    module.__file__ = filename
    module.__loader__ = loader
    module.__package__ = "bench_namespace"
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader, origin=filename)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    # Registered like an imported module, which also keeps it alive.
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def namespace_exc_info(layers=20):
    """layers + 1 frames, each layer in its own module of about 300 globals.

    The globals are mostly functions (some decorated, so their ``__dict__``
    holds ``__wrapped__``), classes, imported modules and a few instances:
    nearly every value captures as a FakeClass.
    """
    modules = [_make_layer(index) for index in range(layers)]
    for outer, inner in zip(modules, modules[1:]):
        outer.NEXT = inner.enter
    payload = {"path": "/v1/items/42", "user": "bench", "attempt": 1}
    try:
        modules[0].enter(payload, 0)
    except LookupError:
        return sys.exc_info()


# ------------------------------------------------------------------ registry

# DumpOptions keyword arguments with the four value budgets lifted.
_NO_VALUE_LIMITS = {
    "max_items": None,
    "max_nodes": None,
    "max_depth": None,
    "max_string_bytes": None,
}


def registry():
    """(name, description, exc_info factory, DumpOptions kwargs), in run order.

    A function rather than a module global, so frame globals of the workloads
    above do not carry the table into their own captures.
    """
    return [
        (
            "a_typical",
            "4 frames, ~12 small locals, realistic module globals; defaults",
            typical_exc_info,
            {},
        ),
        (
            "b_heavy_default",
            "50k-int list, 20k str->list dict, nested dict, 100 KB str; defaults",
            heavy_exc_info,
            {},
        ),
        (
            "b2_heavy_nolimits",
            "same locals; max_items/max_nodes/max_depth/max_string_bytes=None",
            heavy_exc_info,
            dict(_NO_VALUE_LIMITS),
        ),
        (
            "c_deep",
            "62 frames, 3-4 small locals each; defaults",
            deep_exc_info,
            {},
        ),
        (
            "d_stdlib_embedall",
            "14 frames through argparse.py and json/*.py; embed_source_lines=None",
            stdlib_exc_info,
            {"embed_source_lines": None},
        ),
        (
            "e_namespace_heavy",
            "21 frames, 20 modules of ~300 functions/classes/modules each; defaults",
            namespace_exc_info,
            {},
        ),
    ]
