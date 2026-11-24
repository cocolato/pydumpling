"""Tracebacks for comparing capture engines. Keep measured frames free of harness state."""


class Order:
    def __init__(self, number, lines):
        self.number = number
        self.lines = lines


def small(request_id):
    if request_id == "r-42":
        raise ValueError("invalid request")


def shared(batch, remaining):
    if remaining:
        return shared(batch, remaining - 1)
    raise ValueError("batch write failed")


def mixed(order):
    raise ValueError("order rejected")


def budget(request_id, rows):
    raise ValueError("page timeout")


def exception_case(name):
    """Build the input outside the traceback and before either timer starts."""
    if name == "small":
        function, args, options = small, ("r-42",), {}
    elif name == "shared":
        batch = [
            {"sku": f"part-{i}", "price": i + 1000, "stock": [i, i + 1]}
            for i in range(500)
        ]
        function, args = shared, (batch, 11)
        options = {"max_items": 600, "max_depth": 12}
    elif name == "mixed":
        nested = {
            "lines": [Order(i, [(i, str(i)), (i + 1, str(i + 1))]) for i in range(60)]
        }
        for level in range(8):
            nested = {"level": level, "children": [nested]}
        function, args, options = mixed, (Order(42, nested),), {"max_depth": 32}
    elif name == "budget":
        rows = [{"sku": f"part-{i}", "stock": i + 1000} for i in range(1000)]
        function, args, options = (
            budget,
            ("r-42", rows),
            {"max_items": 1000, "max_nodes": 40},
        )
    else:
        raise ValueError(name)
    try:
        function(*args)
    except ValueError as error:
        # The harness frame would otherwise capture benchmark setup and options.
        return (type(error), error, error.__traceback__.tb_next), options
    raise AssertionError(f"{name} did not raise")
