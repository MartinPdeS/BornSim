from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bornsim import RandomMedium, Volume
from bornsim.series import BornSeries
from bornsim.ensemble import ensemble_scattering
from bornsim.green import GreenOperator


def test_engine_reuses_operator_and_starts_each_volume_independently():
    medium = RandomMedium()
    first = medium.to_volume(
        shape=(3, 3, 3),
        spacing=30e-9,
        seed=12,
    )
    second = medium.to_volume(
        shape=(3, 3, 3),
        spacing=30e-9,
        seed=13,
    )
    directions = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    engine = BornSeries(
        shape=first.delta_index.shape,
        spacing=first.spacing,
        background_index=first.background_index,
        wavelength=633e-9,
        directions=directions,
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
        shape=second.delta_index.shape,
        spacing=second.spacing,
        background_index=second.background_index,
        wavelength=633e-9,
        directions=directions,
        order=3,
    ).solve(volume=second)
    np.testing.assert_array_equal(other.amplitudes, expected.amplitudes)
    np.testing.assert_array_equal(other.field_norms, expected.field_norms)
    assert engine.operator is operator
    assert not np.array_equal(original, other.amplitudes)
    directions[:] = 0
    np.testing.assert_array_equal(engine.solve(volume=first).amplitudes, original)
    assert not engine.directions.flags.writeable
    with pytest.raises(FrozenInstanceError):
        engine.wavelength = 500e-9


@pytest.mark.parametrize(
    "volume",
    [
        Volume(
            delta_index=np.zeros((3, 2, 2)),
            spacing=50e-9,
        ),
        Volume(
            delta_index=np.zeros((2, 2, 2)),
            spacing=40e-9,
        ),
        Volume(
            delta_index=np.zeros((2, 2, 2)),
            spacing=50e-9,
            background_index=1.5,
        ),
    ],
)
def test_engine_rejects_incompatible_volume(volume):
    engine = BornSeries(
        shape=(2, 2, 2),
        spacing=50e-9,
        background_index=1.33,
        wavelength=633e-9,
        directions=[[0, 0, 1]],
    )
    with pytest.raises(ValueError, match="must match"):
        engine.solve(volume=volume)
    with pytest.raises(TypeError, match="Volume"):
        engine.solve(volume=object())


def test_ensemble_constructs_one_green_operator(monkeypatch):
    operators = []

    class CountingGreen(GreenOperator):
        def __init__(self, *, shape, spacing, wavelength, background_index):
            operators.append(self)
            super().__init__(
                shape=shape,
                spacing=spacing,
                wavelength=wavelength,
                background_index=background_index,
            )

    monkeypatch.setattr("bornsim.series.GreenOperator", CountingGreen)
    result = ensemble_scattering(
        medium=RandomMedium(),
        wavelength=633e-9,
        shape=(3, 3, 3),
        spacing=30e-9,
        order=3,
        realizations=4,
        seed=12,
        angles=[0, 1],
        azimuth_samples=4,
        polar_samples=16,
    )
    assert len(operators) == 1
    assert result["field_norms"].shape == (4, 3)
    assert np.all(np.isfinite(result["stderr"]))
    assert np.any(result["stderr"] > 0)
