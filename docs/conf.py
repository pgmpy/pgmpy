#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../pgmpy_notebooks/notebooks"))

# -- General configuration ------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "sphinx_immaterial",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxext.opengraph",
    "sphinx_sitemap",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "pgmpy"
copyright = "2025, Ankur Ankan"
author = "Ankur Ankan, Abinash Panda"

version = "dev"
release = "1.0.0"
language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
nbsphinx_execute = "never"
pygments_style = "default"
todo_include_todos = True

# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_immaterial"

html_title = "pgmpy"
html_favicon = "logo_favi.ico"
html_logo = "logo.png"
html_static_path = ["_static"]
html_extra_path = ["robots.txt", "llms.txt", "llms-full.txt"]

html_theme_options = {
    "repo_url": "https://github.com/pgmpy/pgmpy",
    "repo_name": "pgmpy/pgmpy",
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "features": [
        "navigation.tabs",
        "navigation.tabs.sticky",
        "navigation.sections",
        "navigation.top",
        "search.highlight",
        "search.share",
        "toc.follow",
        "content.tabs.link",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "indigo",
            "toggle": {
                "icon": "material/weather-night",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "black",
            "accent": "indigo",
            "toggle": {
                "icon": "material/weather-sunny",
                "name": "Switch to light mode",
            },
        },
    ],
    "version_dropdown": True,
    "version_info": [
        {"version": "https://pgmpy.org", "title": "1.0.0 (stable)", "aliases": []},
        {"version": "https://pgmpy.org/dev", "title": "dev", "aliases": []},
    ],
    "toc_title_is_page_title": True,
    "globaltoc_collapse": True,
}

html_search_language = "en"
htmlhelp_basename = "pgmpydoc"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, "pgmpy.tex", "pgmpy Documentation", "Ankur Ankan", "manual")
]

# -- Options for manual page output ---------------------------------------

man_pages = [(master_doc, "pgmpy", "pgmpy Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "pgmpy",
        "pgmpy Documentation",
        author,
        "pgmpy",
        "Python Library for Causal and Probabilistic Modeling using Bayesian Networks.",
        "Miscellaneous",
    )
]

# To suppress autosummary warnings while building docs.
numpydoc_show_class_members = False

# -- SEO configuration -----------------------------------------------------

# Base URL for sitemap and canonical links
html_baseurl = "https://pgmpy.org"

# Open Graph metadata (social sharing previews)
ogp_site_url = "https://pgmpy.org"
ogp_site_name = "pgmpy"
ogp_image = "https://pgmpy.org/_static/logo.png"
ogp_description_length = 200
ogp_type = "website"
ogp_custom_meta_tags = [
    '<meta name="description" content="pgmpy: A Python library for causal inference and probabilistic inference using Directed Acyclic Graphs (DAGs) and Bayesian Networks." />',
    '<meta name="keywords" content="pgmpy, Bayesian Networks, causal inference, probabilistic graphical models, structure learning, parameter estimation, Python" />',
    '<meta name="twitter:card" content="summary" />',
]
