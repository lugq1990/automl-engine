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
import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = u'automl-engine'
copyright = u'2020, gqianglu'
author = u'gqianglu'

# The full version, including alpha/beta/rc tags
# The short X.Y version.
version = u''
release = u''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon'
]
source_suffix = [".rst", ".md"]
master_doc = 'index'
autoclass_content = 'both'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
pygments_style = 'sphinx'
todo_include_todos = False

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

htmlhelp_basename = 'automl-enginedoc'
latex_documents = [
    (master_doc, 'automl-engine.tex', u'automl-engine Documentation',
     u'gqianglu', 'manual'),
]

man_pages = [
    (master_doc, 'automl-engine', u'automl-engine Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'automl-engine', u'automl-engine Documentation',
     author, 'automl-engine', '3 lines of code for automate machine learning for classification and regression.',
     'Miscellaneous'),
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True