"""BornSim documentation configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bornsim import __version__
from bornsim.units import ureg
from martinpdes_docs import configure

project = "BornSim"
package = "BornSim"
repository = "MartinPdeS/BornSim"
author = "Martin Poinsinet de Sivry-Houle"
release = __version__
version = release


globals().update(
    configure(
        project=project,
        package=package,
        author=author,
        repository=repository,
        logo="_static/logo.png",
        favicon="_static/favicon.png",
        gallery_options={
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
        },
        html_css_files=["layout.css"],
    )
)

def write_interactive_media(app, exception):
    """Export browser volume views for HTML documentation builds."""

    if exception is not None or app.builder.format != "html":
        return

    output = Path(app.outdir) / "_static"
    output.mkdir(exist_ok=True)

    from bornsim import Grid
    from bornsim.medium.random_medium import ExponentialMedium, GaussianMedium

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
