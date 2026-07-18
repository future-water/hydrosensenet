"""Sphinx configuration for the hydrosensenet documentation."""
from importlib.metadata import version as _pkg_version

project = "hydrosensenet"
author = "Jeil Oh, John Lee, Matthew Bartos"
copyright = "2026, Jeil Oh, John Lee, Matthew Bartos"

release = _pkg_version("hydrosensenet")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = ["colon_fence", "deflist"]

# API docs are written in numpydoc style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "hydrosensenet"
html_show_sphinx = False  # drop the "Sphinx" part of the footer credit
html_css_files = ["custom.css"]  # hides the rest (see _static/custom.css)
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/future-water/hydrosensenet/",
    "source_branch": "main",
    "source_directory": "docs/",
}
