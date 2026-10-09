import json

import pytest

from bornsim import AnalyticalMedium, Layer, RandomMedium, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg


def test_random_statistics_preserve_existing_metadata_keys():
    for medium in (
        RandomMedium(
            correlation_length=80 * ureg.nanometer,
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation="matern",
            smoothness=1.5,
        ),
        AnalyticalMedium(
            correlation_length=80 * ureg.nanometer,
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation="gaussian",
            smoothness=1.5,
        ),
    ):
        metadata = medium.metadata

        assert set(metadata) == {
            "background_refractive_index",
            "refractive_index_std",
            "correlation_length_m",
            "correlation",
            "smoothness",
        }

        assert metadata["correlation_length_m"] == pytest.approx(80e-9)

        assert json.loads(json.dumps(metadata)) == metadata

        metadata["smoothness"] = -1

        assert medium.metadata["smoothness"] == 1.5


def test_solver_uses_medium_metadata_polymorphically():
    class TaggedRandom(RandomMedium):
        @property
        def metadata(self):
            return {**super().metadata, "label": "custom medium"}

    medium = TaggedRandom(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    )

    volume = medium.to_volume(
        shape=(2, 2, 2),
        spacing=50e-9 * ureg.meter,
    )

    single = solver.solve(target=volume)

    ensemble = solver.ensemble(
        medium=medium,
        shape=(2, 2, 2),
        realizations=2,
        angles=[0, 1] * ureg.radian,
        azimuth_samples=4,
        polar_samples=16,
        spacing=50e-9 * ureg.meter,
    )

    assert single.provenance["medium"] == medium.metadata

    assert ensemble.provenance["medium"] == medium.metadata


def test_structured_metadata_preserves_order_units_and_independence():
    medium = StructuredMedium(
        regions=[
            Layer(
                lower=-1e-07 * ureg.meter,
                upper=0 * ureg.meter,
                refractive_index=1.34,
            ),
            Sphere(
                radius=50 * ureg.nanometer,
                refractive_index=1.35,
            ),
        ],
        background_refractive_index=1.33,
    )

    metadata = medium.metadata

    assert metadata["background_refractive_index"] == 1.33

    layer, sphere = metadata["regions"]

    assert layer == {"type": "Layer", "lower_m": -100e-9, "upper_m": 0, "refractive_index": 1.34, "axis": 2}

    assert sphere["radius_m"] == pytest.approx(50e-9)

    assert sphere["centre_m"] == [0, 0, 0]

    assert json.loads(json.dumps(metadata)) == metadata

    sphere["centre_m"][0] = 1

    metadata["regions"].reverse()

    assert medium.metadata["regions"][0]["type"] == "Layer"

    assert medium.metadata["regions"][1]["centre_m"] == [0, 0, 0]
