"""BornSim documentation configuration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

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
    "reset_argv": "gallery_config.gallery_arguments",
    "subsection_order": [
        "../examples/random_media",
        "../examples/structured_media",
        "../examples/born_orders",
        "../examples/results",
        "../examples/validation",
    ],
}


def write_interactive_media(app, exception):
    """Export optional browser views without requiring Plotly for docs builds."""
    if exception is not None or app.builder.format != "html":
        return
    from pathlib import Path

    output = Path(app.outdir) / "_static"
    output.mkdir(exist_ok=True)
    try:
        import plotly  # noqa: F401
    except ImportError:
        message = (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Interactive random medium</title><body>"
            "<p>This documentation build includes Matplotlib views. "
            "To enable interactive volumes, install BornSim[visualization] "
            "and rebuild the documentation.</p></body></html>"
        )
        for correlation in ("gaussian", "exponential"):
            (output / f"random-medium-{correlation}.html").write_text(message, encoding="utf-8")
        return

    from bornsim import Grid, RandomMedium

    grid = Grid(
        shape=(16, 16, 16),
        spacing=25e-9,
    )
    for correlation in ("gaussian", "exponential"):
        medium = RandomMedium(
            index_std=0.01,
            correlation_length=75e-9,
            correlation=correlation,
        )
        volume = medium.to_volume(
            grid=grid,
            seed=42,
        )
        figure = volume.plot_3d(
            backend="plotly",
            mode="volume",
            field="index",
            surface_count=16,
            opacity=0.2,
            opacity_scale="increasing",
        )
        figure.update_layout(title=f"{correlation.capitalize()} random medium · higher index is more opaque")
        figure.write_html(
            file=str(output / f"random-medium-{correlation}.html"),
            include_plotlyjs=True,
        )


def setup(app):
    app.connect("build-finished", write_interactive_media)
