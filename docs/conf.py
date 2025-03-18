# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "KinoML"
copyright = "2021, OpenKinome"
author = "OpenKinome"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    # "sphinxemoji.sphinxemoji",
    "sphinx-prompt",
    "sphinx_copybutton",
    # "notfound.extension",
    "myst_parser",
    # "sphinxcontrib.httpdomain",
    "autoapi.extension",
    "nbsphinx",
    "nbsphinx_link",
    # "sphinx_last_updated_by_git",
    "sphinx_panels",
    "IPython.sphinxext.ipython_console_highlighting",
]

autosectionlabel_prefix_document = True

nbsphinx_execute = "auto"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

sphinxemoji_style = "twemoji"


autoapi_dirs = ["../kinoml"]
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_ignore = [
    "*migrations*",
    "*_version*",
    "*tests*",
    "*/data/*",
]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    # "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_keep_files = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "sphinx-notfound-page",
    ".ipynb_checkpoints/*",
    "__pycache__",
    "kinoml/data",
    "developers",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_material

# Choose the material theme
html_theme = "sphinx_material"
# Get the them path
html_theme_path = sphinx_material.html_theme_path()
# Register the required helpers for the html context
html_context = sphinx_material.get_html_context()
version_dropdown = False

# Material theme options (see theme.conf for more information)
html_theme_options = {
    "nav_title": "KinoML",
    "repo_url": "https://github.com/openkinome/kinoml/",
    "repo_name": "KinoML",
    "logo_icon": "&#xe6dd",
    "base_url": "https://openkinome.org/kinoml/",
    # "google_analytics_account": "UA-XXXXX",
    "html_minify": False,
    "html_prettify": True,
    "css_minify": True,
    "repo_type": "github",
    "globaltoc_depth": 3,
    "color_primary": "#3f51b5",
    "color_accent": "blue",
    "touch_icon": "images/custom_favicon.png",
    "theme_color": "#3f51b5",
    "master_doc": False,
    "nav_links": [
        {"href": "index", "internal": True, "title": "User guide"},
        {"href": "api/kinoml/index", "internal": True, "title": "API Reference"},
        {
            "href": "https://openkinome.org",
            "internal": False,
            "title": "OpenKinome",
        },
    ],
    "heroes": {
        "index": "Structure-informed machine learning for kinase modeling",
    },
    "version_dropdown": False,
    "version_json": "_static/versions.json",
    "version_info": {
        "Release": "",
        "Development": "",
        "Release (rel)": "",
        "Development (rel)": "",
    },
    "table_classes": ["plain"],
}

# globaltoc seems it's not added by default
html_sidebars = {
    "**": [
        "globaltoc.html",
        "localtoc.html",
        "searchbox.html",
    ]
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_favicon = "_static/images/custom_favicon.png"

# -------
# MyST
# -------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
]

myst_update_mathjax = False
mathjax3_config = {
    "tex2jax": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "processRefs": False,
        "processEnvironments": False,
    }
}
