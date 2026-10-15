"""Sphinx configuration sharing the package's version source."""

from pathlib import Path
from runpy import run_path

project = "pydumpling"
author = "cocolato"
copyright = "2026, cocolato"
release = run_path(
    str(Path(__file__).resolve().parents[2] / "pydumpling" / "_version.py")
)["__version__"]
version = release

extensions = ["sphinx_rtd_theme", "sphinx_tabs.tabs", "sphinx_copybutton"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_title = "Pydumpling Documentation"
html_show_sourcelink = False
html_context = {
    "display_github": True,
    "github_user": "cocolato",
    "github_repo": "pydumpling",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}
