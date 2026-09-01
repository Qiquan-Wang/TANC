"""_compat.py — accessors that tolerate being called.

The toolkit follows one rule: an accessor is a **property** when it is a cheap
lookup, and a **method** when it takes arguments or does real work.  The rule is
easy to state and impossible to guess from a name, and getting it wrong in the
"property called as a method" direction fails in a particularly unhelpful way::

    >>> pop.trained()
    TypeError: 'TrainedPopulation' object is not callable

The value was already computed; only the parentheses were wrong.  The types here
let that call succeed, returning the same object with a ``DeprecationWarning``,
so a wrong guess costs a warning rather than a traceback.

This is a transitional courtesy, not a second supported spelling.  The plan is to
keep it for one release, then drop it — after which the call fails loudly, which
is the right end state.

Wrapping is only worth doing for accessors people actually reach for.  Anything
returning a large object, or where ``__call__`` would collide with real meaning,
is left alone.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

__all__ = ["CallableList", "CallableArray", "callable_property", "warn_called"]


def warn_called(name: str) -> None:
    """Emit the standard "you added parentheses" warning."""
    warnings.warn(
        f"{name!r} is a property, not a method — drop the parentheses. "
        f"Calling it returns the same value for now, but will stop working "
        f"in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


class CallableList(list):
    """A ``list`` that returns itself when called.

    Used for property returns such as ``result.errors`` so that
    ``result.errors()`` keeps working through one deprecation cycle.  Every
    ordinary list operation is unaffected.
    """

    _accessor_name = "this property"

    def __call__(self) -> "CallableList":
        warn_called(self._accessor_name)
        return self


class CallableArray(np.ndarray):
    """An ``ndarray`` that returns itself when called.

    Built with ``np.asarray(x).view(CallableArray)``.  Slices and arithmetic
    results are ordinary arrays again, which is deliberate: only the accessor's
    own return value needs to tolerate a stray ``()``.
    """

    _accessor_name = "this property"

    def __call__(self) -> "CallableArray":
        warn_called(self._accessor_name)
        return self


def callable_property(name: str, value: Any) -> Any:
    """Wrap *value* so that calling it warns and returns it unchanged.

    Falls back to the unwrapped value for types that cannot be subclassed
    cheaply, so a caller never loses data to this helper.
    """
    if isinstance(value, np.ndarray):
        out = value.view(CallableArray)
        out._accessor_name = name
        return out
    if isinstance(value, list):
        out = CallableList(value)
        out._accessor_name = name
        return out
    return value
