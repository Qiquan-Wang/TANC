"""_serialization.py — pickle-based save/load for toolkit result objects.

Every result container in the toolkit (``ModelSnapshot``, ``TrainingView``,
``GraphBundle``, ``TopoResult``) exposes a ``.save(path)`` instance method and a
matching ``.load(path)`` classmethod.  Both delegate here so the on-disk format
is identical across stages.

Format
------
A ``.tda`` file is a pickle of a small *envelope* dict::

    {
        "magic": "tanc",
        "format_version": 1,
        "object_type": "TopoResult",   # informational
        "object": <the saved object>,
    }

The envelope lets :func:`load_tda` give a clear error on a non-toolkit file or a
future/older format version, instead of a cryptic unpickling traceback.

.. warning::
    Pickle executes arbitrary code on load.  Only load ``.tda`` files you
    created or trust.  The format is Python-only and not guaranteed stable
    across major dependency upgrades (numpy / networkx pickles).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

_MAGIC = "tanc"
_TDA_FORMAT_VERSION = 1
_SUFFIX = ".tda"


def _coerce_path(path: Any, *, for_read: bool) -> Path:
    """Normalise *path* to a ``Path`` and default the suffix to ``.tda``."""
    p = Path(path)
    if p.suffix == "":
        candidate = p.with_suffix(_SUFFIX)
        # On read, only adopt the .tda suffix if that file actually exists;
        # otherwise keep the bare name so the missing-file error names what the
        # caller actually asked for.
        if not for_read or candidate.exists():
            p = candidate
    return p


def save_tda(obj: Any, path: Any) -> Path:
    """Pickle *obj* into a versioned ``.tda`` envelope at *path*.

    Returns the resolved :class:`pathlib.Path` (with a ``.tda`` suffix added
    when *path* had none).
    """
    p = _coerce_path(path, for_read=False)
    p.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "magic": _MAGIC,
        "format_version": _TDA_FORMAT_VERSION,
        "object_type": type(obj).__name__,
        "object": obj,
    }
    with p.open("wb") as fh:
        pickle.dump(envelope, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_tda(path: Any, *, expected_type: type | None = None) -> Any:
    """Load an object previously written by :func:`save_tda`.

    Parameters
    ----------
    path
        File to read (``.tda`` suffix added automatically if omitted).
    expected_type
        When given, raise :class:`TypeError` if the loaded object is not an
        instance of this type.  ``Klass.load`` passes ``expected_type=Klass``
        so ``TopoResult.load`` never silently returns a ``GraphBundle``.
    """
    p = _coerce_path(path, for_read=True)
    if not p.exists():
        raise FileNotFoundError(f"No such tanc save file: {p}")
    with p.open("rb") as fh:
        envelope = pickle.load(fh)

    if not isinstance(envelope, dict) or envelope.get("magic") != _MAGIC:
        raise ValueError(
            f"{p} is not a tanc save file (missing format header)."
        )
    version = envelope.get("format_version")
    if version != _TDA_FORMAT_VERSION:
        raise ValueError(
            f"{p} was written with tda save-format v{version}, but this "
            f"version of tanc reads v{_TDA_FORMAT_VERSION}."
        )

    obj = envelope["object"]
    if expected_type is not None and not isinstance(obj, expected_type):
        raise TypeError(
            f"{p} contains a {type(obj).__name__}, but "
            f"{expected_type.__name__}.load() was called.  Load it with "
            f"{type(obj).__name__}.load(...) instead."
        )
    return obj


class SaveLoadMixin:
    """Mixin adding ``.save(path)`` / ``.load(path)`` to a result container.

    A result type gains persistence simply by listing this mixin as a base::

        @dataclass
        class TopoResult(SaveLoadMixin):
            ...

        result.save("run.tda")
        result = TopoResult.load("run.tda")
    """

    def save(self, path: Any) -> Path:
        """Pickle this object to *path* (``.tda`` suffix added if omitted).

        Returns the resolved path written.
        """
        return save_tda(self, path)

    @classmethod
    def load(cls, path: Any):
        """Load an object of this type previously written by :meth:`save`.

        Raises :class:`TypeError` if the file holds a different toolkit type.
        """
        return load_tda(path, expected_type=cls)

