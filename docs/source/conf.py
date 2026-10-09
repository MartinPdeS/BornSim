"""BornSim documentation configuration."""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent))

from bornsim import __version__
from bornsim.units import ureg

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
    "sphinx_design",
]

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_logo = "_static/logo.png"

html_favicon = "_static/favicon.png"

html_theme_options = {
    "logo": {"alt_text": "BornSim"},
    "show_nav_level": 0,
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/MartinPdeS/BornSim",
            "icon": "fa-brands fa-github",
        },
    ],
    "show_prev_next": False,
    "show_version_warning_banner": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "github-dark",
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/MartinPdeS/BornSim/documentation_page/version_switcher.json",
        "version_match": os.getenv("tag", "latest"),
    },
    # The deployment regenerates this remote manifest after the build.
    "check_switcher": False,
}

html_context = {
    "github_url": "https://github.com",
    "github_user": "MartinPdeS",
    "github_repo": "BornSim",
    "github_version": "master" if os.getenv("tag", "latest") == "latest" else os.environ["tag"],
    "doc_path": "docs/source",
    "default_mode": "dark",
}

exclude_patterns = []

autodoc_typehints = "description"

napoleon_google_docstring = False

napoleon_numpy_docstring = True


sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "capture_repr": (),
    "image_scrapers": ("matplotlib", "gallery_config.plotly_scraper"),
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
    """Export browser volume views for HTML documentation builds."""

    if exception is not None or app.builder.format != "html":
        return

    from pathlib import Path

    output = Path(app.outdir) / "_static"

    output.mkdir(exist_ok=True)

    from bornsim import Grid
    from bornsim.medium.random_medium import GaussianMedium, ExponentialMedium

    grid = Grid(
        shape=(16, 16, 16),
        spacing=2.5e-08 * ureg.meter,
    )

    for correlation, medium_type in (("gaussian", GaussianMedium), ("exponential", ExponentialMedium)):
        medium = medium_type(
            refractive_index_std=0.01,
            correlation_length=7.5e-08 * ureg.meter,
            background_refractive_index=1.33,
        )

        volume = medium.to_volume(
            grid=grid,
            seed=42,
        )

        figure = volume.plot_3d(
            backend="plotly",
            mode="volume",
            field="refractive_index",
            surface_count=16,
            opacity=0.2,
            opacity_scale="increasing",
        )

        figure.update_layout(title=f"{correlation.capitalize()} random medium · higher refractive index is more opaque")

        figure.write_html(
            file=str(output / f"random-medium-{correlation}.html"),
            include_plotlyjs=True,
        )


def setup(app):
    app.connect("build-finished", write_interactive_media)
