# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import metadata
qklearn_metadata = metadata('qklearn')

project = 'qukit-learn'
copyright = "2025, " + qklearn_metadata["Author"]
author = qklearn_metadata["Author"]
release = version = qklearn_metadata["Version"]
summary = qklearn_metadata["Summary"]
rst_prolog = f".. |summary| replace:: {summary}"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
