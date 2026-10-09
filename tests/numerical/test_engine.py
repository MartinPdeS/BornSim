from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bornsim import RandomMedium, Volume
from bornsim.series import BornSeries
from bornsim.ensemble import ensemble_scattering
from bornsim.green import GreenOperator
from bornsim.units import ureg
from bornsim import Directions


def test_engine_reuses_operator_and_starts_each_volume_independently():
    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    first = medium.to_volume(
        shape=(3, 3, 3),
        spacing=3e-08 * ureg.meter,
        seed=12,
    )

    second = medium.to_volume(
        shape=(3, 3, 3),
        spacing=3e-08 * ureg.meter,
        seed=13,
    )

    directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    engine = BornSeries(
        shape=first.delta_refractive_index.shape,
        spacing=first.spacing.to("meter"),
        background_refractive_index=first.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions),
        order=3,
    )

    operator = engine.operator

    initial = engine.solve(volume=first)

    original = initial.amplitudes.copy()

    initial.amplitudes[:] = 0  # Returned arrays must not alter engine state.

    other = engine.solve(volume=second)

    repeat = engine.solve(volume=first)

    np.testing.assert_array_equal(repeat.amplitudes, original)

    expected = BornSeries(
        shape=second.delta_refractive_index.shape,
        spacing=second.spacing.to("meter"),
        background_refractive_index=second.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions),
        order=3,
    ).solve(volume=second)

    np.testing.assert_array_equal(other.amplitudes, expected.amplitudes)

    np.testing.assert_array_equal(other.field_norms, expected.field_norms)

    assert engine.operator is operator

    assert not np.array_equal(original, other.amplitudes)

    directions[:] = 0

    np.testing.assert_array_equal(engine.solve(volume=first).amplitudes, original)

    assert not engine.directions.vectors.flags.writeable

    with pytest.raises(FrozenInstanceError):
        engine.wavelength = 500e-9


@pytest.mark.parametrize(
    "volume",
    [
        Volume(
            delta_refractive_index=np.zeros((3, 2, 2)),
            spacing=5e-08 * ureg.meter,
            background_refractive_index=1.33,
        ),
        Volume(
            delta_refractive_index=np.zeros((2, 2, 2)),
            spacing=4e-08 * ureg.meter,
            background_refractive_index=1.33,
        ),
        Volume(
            delta_refractive_index=np.zeros((2, 2, 2)),
            spacing=5e-08 * ureg.meter,
            background_refractive_index=1.5,
        ),
    ],
)
def test_engine_rejects_incompatible_volume(volume):
    engine = BornSeries(
        shape=(2, 2, 2),
        spacing=5e-08 * ureg.meter,
        background_refractive_index=1.33,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=[[0, 0, 1]]),
    )

    with pytest.raises(ValueError, match="must match"):
        engine.solve(volume=volume)

    with pytest.raises(TypeError, match="Volume"):
        engine.solve(volume=object())


def test_ensemble_constructs_one_green_operator(monkeypatch):
    operators = []

    class CountingGreen(GreenOperator):
        def __init__(self, *, shape, spacing, wavelength, background_refractive_index):
            operators.append(self)

            super().__init__(
                shape=shape,
                spacing=spacing,
                wavelength=wavelength,
                background_refractive_index=background_refractive_index,
            )

    monkeypatch.setattr("bornsim.series.GreenOperator", CountingGreen)

    result = ensemble_scattering(
        medium=RandomMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="matern",
            smoothness=1.5,
        ),
        wavelength=6.33e-07 * ureg.meter,
        shape=(3, 3, 3),
        spacing=3e-08 * ureg.meter,
        order=3,
        realizations=4,
        seed=12,
        angles=[0, 1] * ureg.radian,
        azimuth_samples=4,
        polar_samples=16,
    )

    assert len(operators) == 1

    assert result["field_norms"].shape == (4, 3)

    assert np.all(np.isfinite(result["stderr"]))

    assert np.any(result["stderr"] > 0)
