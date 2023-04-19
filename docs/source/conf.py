# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys_path = sys.path.append(os.path.join(os.path.dirname(__name__), "..", "src"))

project = "simpleopt"
copyright = "2023, Luis Garcia Ramos"
author = "Luis Garcia Ramos"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]
# exclude_patterns = [_build]
templates_path = ["_templates"]
master_doc = "index"
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# src_path = sys.path.append(
#    os.path.join(os.path.dirname(__name__), "..", "src", "simpleopt")

# #sys.path.insert(
#     0,
#     os.path.abspath(
#         sys.path.append(
#             os.path.join(os.path.dirname(__name__), "..", "src", "simpleopt")
#         )
#     ),
# )
# )
