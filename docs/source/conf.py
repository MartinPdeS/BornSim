"""BornSim documentation configuration."""

from bornsim import __version__

project = "BornSim"
author = "Martin Poinsinet de Sivry-Houle"
copyright = "2026, Martin Poinsinet de Sivry-Houle"
release = __version__
version = release
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]
html_theme = "pydata_sphinx_theme"
exclude_patterns = []
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "capture_repr": (),
    "image_scrapers": ("matplotlib",),
    "abort_on_example_error": True,
    "download_all_examples": False,
    "backreferences_dir": None,
}
