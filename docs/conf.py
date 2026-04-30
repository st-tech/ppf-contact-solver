import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ZOZO's Contact Solver 🫶"
author = "Ryoichi Ando"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_togglebutton",
]

# Start collapsibles closed by default; reader clicks to expand.
togglebutton_hint = "Click to expand"
togglebutton_hint_hide = "Click to collapse"

# Heavy third-party runtime deps of the frontend package. Mocking them
# lets autodoc walk frontend/ without installing the full scientific-Python
# stack into the docs venv.  The Blender addon does NOT go through autodoc
# (see generate_blender_api_reference.py), so bpy / mathutils are not listed.
#
# numpy is intentionally NOT mocked: its types appear in union annotations
# (e.g. ``list[float] | np.ndarray``), and MockObject doesn't support the
# ``|`` operator, so mocking numpy makes autodoc fail at import time.
# numpy is declared in docs/requirements.txt instead.
autodoc_mock_imports = [
    "numba",
    "tqdm",
    "PIL",
    "psutil",
    "pythreejs",
    "IPython",
    "pytetwild",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "dollarmath",
    "amsmath",
    "attrs_inline",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", ".venv", "**/.venv",
    "blender_addon/debug/**",
]

language = "en"

# The frontend package has a handful of same-named attributes across
# classes (``path``, ``scene``, ``name``) that autodoc cannot
# disambiguate when a docstring mentions them unqualified.  These warnings
# predate the ``-W`` switch in build.sh; suppressing the category lets
# docstring-level warnings (napoleon parse errors, bad RST in
# ``Example:`` blocks, etc.) still fail the build while this known
# ambiguity does not.
suppress_warnings = ["ref.python"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_theme_options = {
    # Shibuya ships with light / dark / auto colour modes and a header
    # toggle; no extra options needed to enable them.
    "github_url": "https://github.com/st-tech/ppf-contact-solver",
    "accent_color": "blue",
}
html_context = {
    "source_type": "github",
    "source_user": "st-tech",
    "source_repo": "ppf-contact-solver",
}
html_static_path = ["_static"]
html_extra_path = ["blender_addon/images/gallery", "blender_addon/gallery_blends"]
html_css_files = ["custom.css"]

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration
