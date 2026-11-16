"""Export the README figures from BornSim's Plotly visualization methods.

Static image export requires Kaleido. The checked-in images were generated
with Kaleido 0.2.1; install it with ``python -m pip install kaleido==0.2.1``
and run this file from the repository root.
"""

from pathlib import Path

from bornsim import AngularSampling, EnsembleSampling, Grid, Solver, Source
from bornsim.medium.random_medium import GaussianMedium
from bornsim.units import ureg


OUTPUT_DIRECTORY = Path(__file__).parent


def main() -> None:
    """Generate paired Plotly views from one seeded random medium."""

    grid = Grid(
        shape=(14, 14, 14),
        spacing=35 * ureg.nanometer,
    )

    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.018,
        correlation_length=70 * ureg.nanometer,
    )

    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )

    source = Source(
        wavelength=532 * ureg.nanometer,
    )

    sampling = AngularSampling(
        n_points=49,
        azimuth_samples=20,
    )

    solver = Solver(
        source=source,
        sampling=sampling,
        order=2,
    )

    ensemble_sampling = EnsembleSampling(
        realizations=6,
        seed=42,
    )

    result = solver.ensemble(
        medium=medium,
        grid=grid,
        ensemble_sampling=ensemble_sampling,
    )

    medium_figure = volume.plot_3d(
        field="refractive_index",
        mode="isosurface",
        surface_count=8,
        opacity=0.32,
    )

    phase_figure = result.plot_phase_function(
        view="3d",
        order=2,
    )

    # Keep the README images focused on the data while leaving the live figures
    # and their controls untouched for users of the interactive plot.
    medium_figure.update_layout(sliders=[])

    phase_figure.write_image(
        file=OUTPUT_DIRECTORY / "bornsim_phase_function_3d.png",
        width=1400,
        height=1000,
        scale=1.5,
    )

    medium_figure.write_image(
        file=OUTPUT_DIRECTORY / "bornsim_medium_isosurfaces_3d.png",
        width=1400,
        height=1000,
        scale=1.5,
    )


if __name__ == "__main__":
    main()
