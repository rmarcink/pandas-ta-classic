"""Every parameter an indicator docstring documents is one the indicator reads.

A fresh-eyes audit found 27 documented parameters that no code read, so
passing them changed nothing and raised nothing (``**kwargs`` swallowed
them): leftovers of an older signature (``kvo`` ``long``/``length_sig``,
``stc`` ``tclen``), a wrong spelling (``open`` for ``open_``), inputs the
indicator never took (``aroon`` ``close``, ``stochrsi`` ``high``/``low``) and
EMA options copied into indicators that never reach an EMA (``sma``
``adjust``/``presma``).

A documented name is accepted when it is in the signature, is read from
``kwargs`` (``pop``/``get``/``setdefault``/``[]``/``in``), or is accepted by a
package function the indicator hands its ``kwargs`` to (``ema(close, **kwargs)``,
``attach_signals(..., kwargs=kwargs)``), followed recursively.
"""

import ast
import functools
import inspect
import re
import textwrap

import pytest

import pandas_ta_classic as ta
from pandas_ta_classic._indicator_loader import _find_indicator_func

_ENTRY = re.compile(r"^\s{4}(\w+)\s*\(", re.MULTILINE)
_SECTION = re.compile(r"\n(?=\w[\w ]*:\s*\n)")
_KWARG_READ = re.compile(r'kwargs(?:\.(?:pop|get|setdefault)\(|\[)"(\w+)"|"(\w+)" in kwargs')
_FILL = {"fillna", "fill_method"}  # read by apply_fill, which every indicator calls


def _documented(doc: str | None) -> set:
    names: set = set()
    for section in _SECTION.split(doc or ""):
        if section.split("\n", 1)[0].strip() in ("Args:", "Kwargs:"):
            names |= set(_ENTRY.findall(section))
    return names


def _forwards_kwargs(call: ast.Call) -> bool:
    return any(k.arg is None or (isinstance(k.value, ast.Name) and k.value.id == "kwargs") for k in call.keywords)


def _resolve(func: ast.expr, namespace: dict):
    if isinstance(func, ast.Name):
        return namespace.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return getattr(namespace.get(func.value.id), func.attr, None)
    return None


@functools.cache
def _accepted(fn) -> frozenset:
    fn = inspect.unwrap(fn)
    source = textwrap.dedent(inspect.getsource(fn))
    names = set(inspect.signature(fn).parameters) - {"kwargs"}
    names |= {a or b for a, b in _KWARG_READ.findall(source)}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _forwards_kwargs(node):
            target = _resolve(node.func, fn.__globals__)
            target = inspect.unwrap(target) if callable(target) else None
            if inspect.isfunction(target) and target.__module__.startswith("pandas_ta_classic") and target is not fn:
                names |= _accepted(target)
    return frozenset(names)


INDICATORS = sorted(i for names in ta.Category.values() for i in names)


@pytest.mark.parametrize("name", INDICATORS)
def test_documented_parameters_are_read(name):
    func = _find_indicator_func(name)
    phantom = sorted(_documented(func.__doc__) - _accepted(func) - _FILL)
    assert not phantom, f"{name}() documents parameters it never reads: {phantom}"


def test_the_scan_sees_the_parameters():
    """Guard the guard: a broken section parser would pass everything."""
    assert {"close", "length", "offset", "fillna"} <= _documented(ta.rsi.__doc__)
    assert {"xa", "signal_indicators"} <= _accepted(_find_indicator_func("rsi"))  # through attach_signals(kwargs=kwargs)
    assert "adjust" in _accepted(_find_indicator_func("tema"))  # through ema(close, **kwargs)
    assert "adjust" not in _accepted(_find_indicator_func("sma"))
