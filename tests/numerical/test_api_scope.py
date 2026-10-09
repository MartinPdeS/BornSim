"""Keep the supported API numerical and reject removed compatibility paths."""

from bornsim.medium.random_medium import GaussianMedium

import importlib.util
import json
import numpy as np
import pytest
import bornsim

from bornsim import AngularSampling, Grid, Result, Solver, Source
from bornsim.units import ureg


@pytest.mark.parametrize(
    "name",
    [
        "AnalyticalMedium",
        "RandomMedium",
        "angular_scattering",
        "optical_properties",
        "random_volume",
        "BornSeries",
        "ensemble_scattering",
    ],
)
def test_removed_top_level_exports_are_unavailable(name):
    assert not hasattr(bornsim, name)

    assert name not in bornsim.__all__


@pytest.mark.parametrize("module", ["bornsim.model", "bornsim.media"])
def test_removed_modules_are_not_installed(module):
    assert importlib.util.find_spec(module) is None


def test_only_current_archive_schema_is_supported(tmp_path):
    grid = Grid(
        shape=(2, 2, 2),
        spacing=50 * ureg.nanometer,
    )

    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=75 * ureg.nanometer,
    )

    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )

    sampling = AngularSampling(
        n_points=3,
        polar_samples=16,
        azimuth_samples=4,
    )

    solver = Solver(
        source=Source(wavelength=633 * ureg.nanometer),
        sampling=sampling,
        order=1,
    )

    result = solver.solve(target=volume)

    path = result.save(path=tmp_path / "result.npz")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    metadata = json.loads(str(arrays["metadata"].item()))

    assert metadata["schema_version"] == 2

    assert "directional_differential" not in metadata["units"]

    metadata["schema_version"] = 1

    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="Unsupported result archive schema"):
        Result.load(path=path)
