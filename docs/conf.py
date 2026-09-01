"""Sphinx configuration for the TANC local docs build.

Build with:
    cd docs
    pip install -r requirements.txt
    make html               # produces docs/_build/html/index.html
    # On Windows: .\\make.bat html
"""

from __future__ import annotations

import os
import sys

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath(".."))

# ── Paper-reproduction notebooks ────────────────────────────────────────────
# Copy paper_reproduce/*.ipynb into docs/notebooks/ so myst-nb can render them
# (with their stored outputs — never executed), and generate the landing page.
import pathlib
import shutil

_HERE = pathlib.Path(__file__).resolve().parent
_NB_SRC = _HERE.parent / "paper_reproduce"
_NB_DST = _HERE / "notebooks"
_NB_DST.mkdir(exist_ok=True)
_nb_stems = []
for _nb in sorted(_NB_SRC.glob("*.ipynb")):
    shutil.copy2(_nb, _NB_DST / _nb.name)
    _nb_stems.append(_nb.stem)

# Copy the hands-on tutorial into the docs tree (rendered from its stored
# outputs; nb_execution_mode="off" means the build never re-runs it).
shutil.copy2(_HERE.parent / "TUTORIAL.ipynb", _HERE / "tutorial.ipynb")

_pr = [
    "Preset recipes (shortcuts)",
    "==========================",
    "",
    "Each preset is just a **pre-filled point** along the four axes "
    "(:doc:`composing`): a fixed *space*, *construction*, *method*, and *output* "
    "chosen to match a published method.  ``TDAPipeline.from_paper(\"<name>\")`` "
    "returns the same kind of pipeline you would assemble by hand — see "
    ":doc:`composing` for how to build your own instead.",
    "",
    "One runnable notebook per preset below, rendered with its **stored outputs** "
    "(the build never re-executes).  Each trains a small model on the laptop GPU "
    "(or generates data), runs the preset, and shows the paper's headline result.  "
    "See ``paper_reproduce/README.md`` for dataset notes and the model-first vs. "
    "explicit-extraction guide.",
    "",
    ".. toctree::",
    "   :maxdepth: 1",
    "",
] + [f"   notebooks/{_s}" for _s in _nb_stems] + [""]
(_HERE / "paper_reproductions.rst").write_text("\n".join(_pr))

# ── Project metadata ────────────────────────────────────────────────────────
project = "TANC"
author = "TANC contributors"
copyright = f"2026, {author}"
release = "0.1.0"

# ── Extensions ──────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # NumPy / Google docstring style
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_nb",                      # MyST .md files + rendered .ipynb notebooks
]

# Render notebooks from their stored outputs only — never execute during the
# build (the docs env mocks torch/gudhi, and executing would train models).
nb_execution_mode = "off"
nb_merge_streams = True

autosummary_generate = True
autodoc_default_options = {
    "members":           True,
    "undoc-members":     False,
    "show-inheritance":  True,
    "member-order":      "bysource",
}
# Render a docstring's ``Attributes`` section as :ivar: fields rather than as
# separate attribute directives. Without this, every dataclass field documented
# in an Attributes block is registered twice — once by napoleon, once by
# autodoc's member scan — which is where the duplicate-object warnings came from.
napoleon_use_ivar = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Don't error out if optional deps (torch, gudhi, persim, …) aren't installed
# in the docs-build environment — autodoc will skip those members.
autodoc_mock_imports = [
    "torch", "tensorflow",
    "gudhi", "ripser", "persim", "giotto", "gtda",
    "networkx", "scipy", "sklearn", "matplotlib",
    "umap",
]

# ── HTML output ─────────────────────────────────────────────────────────────
html_theme = "furo"                 # sidebar nav, client-side search, dark mode
html_static_path = ["_static"]
html_title = "TANC"

html_theme_options = {
    "source_repository": "https://github.com/Qiquan-Wang/TANC/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# .rst is handled by Sphinx; myst_nb registers .md (MyST) and .ipynb (notebooks).
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}