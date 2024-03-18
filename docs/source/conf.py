# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pydumpling'
copyright = '2024, cocolato'
author = 'cocolato'
release = '0.1.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "cocolato",  # Username
    "github_repo": "pydumpling",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/source/",  # Path in the checkout to the docs root
}

extensions = [
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinx_copybutton"
]

templates_path = ['_templates']
exclude_patterns = []


html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_title = "Pydumpling Documentation"
html_show_sourcelink = False
