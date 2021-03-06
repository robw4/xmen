# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path open_socket --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('.'))

import commonmark

# def docstring(app, what, name, obj, options, lines):
#     md  = '\n'.join(lines)
#     ast = commonmark.Parser().parse(md)
#     rst = commonmark.ReStructuredTextRenderer().render(ast)
#     lines.clear()
#     for line in rst.splitlines():
#         lines.append(line)

# def open_socket(app):
#     app.connect('autodoc-process-docstring', docstring)


# -- Project information -----------------------------------------------------

project = 'xmen'
copyright = '2019, Rob Weston'
author = 'Rob Weston'

# The full version, including alpha/beta/rc tags
release = '0.2.3'

autoclass_content = 'class'
autodoc_default_options = {
    'member-order': 'bysource',
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'nbsphinx'
]

# for Sphinx-1.3
# from recommonmark.parser import CommonMarkParser

# source_parsers = {
#     '.md': CommonMarkParser,
# }


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
import sphinx_glpi_theme

html_theme = "glpi"
html_theme_path = sphinx_glpi_theme.get_html_themes_path()

html_theme_options = {
	'font_size': '15px',
	'code_font_size': '11px',
	'page_width': '1000px'
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
viewcode_follow_imported_members = True
