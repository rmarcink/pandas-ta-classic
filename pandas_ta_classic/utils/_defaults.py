"""Read an indicator's defaults and bounds from its validation calls.

The defaults live in the body (``length = _pos_int(length, 10, "length", gt=1)``),
so :func:`defaults` parses the indicator's own source once and reports what
those calls say, without running the indicator.
"""

import ast
import functools
import inspect
from collections.abc import Callable
from typing import Any

from pandas_ta_classic.utils._core import _require_input

# helper -> (kind, the lower bound it applies when none is given)
_HELPERS = {
    "_pos_int": ("int", 0),
    "_pos_float": ("float", 0),
    "_number": ("float", None),
    "_bool_param": ("bool", None),
    "_str_param": ("str", None),
}


def _option_name(node: ast.expr) -> str | None:
    """``length`` in ``_pos_int(length, ...)``, ``angle`` in ``_bool_param(kwargs.pop("angle", None), ...)``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("pop", "get") and node.args:
        first = node.args[0]
        return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else None
    return None


def _choices(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ma" and not node.args:
        from pandas_ta_classic.overlap.ma import ma  # the moving averages; ma imports this package's indicators

        return tuple(ma())
    try:
        return tuple(ast.literal_eval(node))
    except ValueError:
        return None


@functools.cache
def _parse(fn: Callable) -> dict[str, dict[str, Any]]:
    module = inspect.getmodule(fn)
    if module is None:
        raise ValueError(f"defaults() cannot find the module of {fn!r}")
    tree = ast.parse(inspect.getsource(module))
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__)
    found: dict[str, dict[str, Any]] = {}
    for node in ast.walk(body):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) in (*_HELPERS, "get_drift", "get_offset") and node.args):
            continue
        option = _option_name(node.args[0])
        if option is None or option in found:
            continue
        helper = node.func.id  # type: ignore[attr-defined]
        if helper in ("get_drift", "get_offset"):
            found[option] = {"kind": "int", "default": 1 if helper == "get_drift" else 0}
            continue
        kind, gt = _HELPERS[helper]
        entry: dict[str, Any] = {"kind": kind}
        try:
            entry["default"] = ast.literal_eval(node.args[1])
        except ValueError:  # computed from another parameter (adx: lensig defaults to length)
            entry["default"] = None
            entry["computed"] = ast.unparse(node.args[1])
        keywords = {k.arg: k.value for k in node.keywords}
        if kind in ("int", "float"):
            for bound, start in (("gt", gt), ("ge", None), ("lt", None)):
                if bound in keywords:
                    try:
                        entry[bound] = ast.literal_eval(keywords[bound])
                    except ValueError:  # depends on another parameter (squeeze: gt=1 if lazybear else 0)
                        entry[bound] = None
                        entry[f"{bound}_computed"] = ast.unparse(keywords[bound])
                else:
                    entry[bound] = start
        if kind == "str" and "choices" in keywords:
            entry["choices"] = _choices(keywords["choices"])
        found[option] = entry
    return found


def defaults(indicator: str | Callable) -> dict[str, dict[str, Any]]:
    """{parameter: {"kind", "default", "gt", "ge", "lt", "choices"}} of an indicator, read from
    its validation calls without running it: ``ta.defaults("hma")["length"]`` gives
    ``{"kind": "int", "default": 10, "gt": 1, "ge": None, "lt": None}``.

    A default computed from another parameter reads ``None`` with the expression under
    ``"computed"``; a bound that depends on another parameter likewise under ``"gt_computed"``.
    Options validated in a helper function rather than the indicator itself are not listed.
    """
    _require_input(indicator, "defaults", "indicator")
    if isinstance(indicator, str):
        import pandas_ta_classic as ta  # the package namespace; this module is imported by it

        fn = getattr(ta, indicator, None)
        if fn is None:
            raise ValueError(f"defaults() indicator {indicator!r} is not an indicator of pandas_ta_classic")
    else:
        fn = indicator
    return {k: dict(v) for k, v in _parse(inspect.unwrap(fn)).items()}
