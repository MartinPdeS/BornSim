"""Physical configuration is chosen by callers, rather than supplied implicitly."""

from bornsim.medium.random_medium import GaussianMedium, WhittleMaternMedium

import pytest
from bornsim import Grid, Rotation, Source, StructuredMedium, Volume
from bornsim.units import ureg


@pytest.mark.parametrize(
    "constructor, configuration, omitted_parameter",
    [
        (Source, {"wavelength": 633 * ureg.nanometer}, "wavelength"),
        (Grid, {"shape": (2, 2, 2), "spacing": 50 * ureg.nanometer}, "shape"),
        (Grid, {"shape": (2, 2, 2), "spacing": 50 * ureg.nanometer}, "spacing"),
        (Rotation, {"axis": (0, 0, 1), "angle": 30 * ureg.degree}, "axis"),
        (Rotation, {"axis": (0, 0, 1), "angle": 30 * ureg.degree}, "angle"),
    ]
    + [
        (
            constructor,
            {
                "background_refractive_index": 1.33,
                "refractive_index_std": 0.01,
                "correlation_length": 100 * ureg.nanometer,
            },
            parameter,
        )
        for constructor in (GaussianMedium,)
        for parameter in ("background_refractive_index", "refractive_index_std", "correlation_length")
    ],
)
def test_omitted_physical_constructor_inputs_are_rejected(constructor, configuration, omitted_parameter):
    arguments = {name: value for name, value in configuration.items() if name != omitted_parameter}

    with pytest.raises(TypeError, match=omitted_parameter):
        constructor(**arguments)


def test_volume_requires_explicit_background_refractive_index():
    import numpy as np

    grid = Grid(
        shape=(2, 2, 2),
        spacing=50 * ureg.nanometer,
    )

    with pytest.raises(TypeError, match="background_refractive_index"):
        Volume(
            delta_refractive_index=np.zeros(grid.shape),
            grid=grid,
        )


@pytest.mark.parametrize("settings", [{}, {"shape": (2, 2, 2)}, {"spacing": 50 * ureg.nanometer}])
def test_volume_generation_never_invents_missing_grid_settings(settings):
    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100 * ureg.nanometer,
    )

    with pytest.raises(TypeError):
        medium.to_volume(**settings)


def test_structured_builder_requires_background_before_voxelization():
    import numpy as np

    grid = Grid(
        shape=(2, 2, 2),
        spacing=50 * ureg.nanometer,
    )

    medium = StructuredMedium()

    with pytest.raises(ValueError, match="add_background"):
        medium.to_volume(grid=grid)

    medium.add_background(refractive_index=1.5)

    volume = medium.to_volume(grid=grid)

    assert volume.background_refractive_index == 1.5

    np.testing.assert_array_equal(volume.delta_refractive_index, np.zeros(grid.shape))


def test_matern_smoothness_is_required_only_for_matern_covariance():
    configuration = {
        "background_refractive_index": 1.33,
        "refractive_index_std": 0.01,
        "correlation_length": 100 * ureg.nanometer,
    }

    with pytest.raises(TypeError, match="smoothness"):
        WhittleMaternMedium(
            **configuration,
        )

    medium = GaussianMedium(
        **configuration,
    )

    assert medium.metadata["smoothness"] is None
