"""Validate editable composition against independent voxel references."""

from bornsim.medium.random_medium import WhittleMaternMedium

from bornsim import Grid
import json
import numpy as np
import pytest
from bornsim import Layer, Sphere, StructuredMedium
from bornsim.units import ureg


def test_background_and_structures_update_in_place_with_last_structure_winning():
    medium = StructuredMedium(background_refractive_index=1.33)

    assert medium.add_background(refractive_index=1.3) is None

    layer = Layer(
        lower=-1e-07 * ureg.meter,
        upper=1e-07 * ureg.meter,
        refractive_index=1.31,
    )

    sphere = Sphere(
        radius=2e-08 * ureg.meter,
        refractive_index=1.32,
    )

    returned = medium.add_structures(layer, sphere)

    assert returned is None

    volume = medium.to_volume(
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        )
    )

    expected = np.full((3, 3, 3), 1.31 - 1.3)

    expected[1, 1, 1] = 1.32 - 1.3

    np.testing.assert_array_equal(volume.delta_refractive_index, expected)

    assert medium.metadata["background_refractive_index"] == 1.3

    assert [region["type"] for region in medium.metadata["regions"]] == ["Layer", "Sphere"]


def test_random_background_seed_and_structure_replacement():
    statistics = WhittleMaternMedium(
        background_refractive_index=1.3,
        refractive_index_std=0.005,
        correlation_length=1e-07 * ureg.meter,
        smoothness=1.5,
    )

    medium = StructuredMedium(background_refractive_index=1.33)

    medium.add_background(medium=statistics)

    sphere = Sphere(
        radius=2e-08 * ureg.meter,
        refractive_index=1.31,
    )

    medium.add_structures(sphere)

    background = statistics.to_volume(
        seed=42,
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        ),
    )

    first = medium.to_volume(
        seed=42,
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        ),
    )

    repeated = medium.to_volume(
        seed=42,
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        ),
    )

    expected = background.delta_refractive_index.copy()

    expected[1, 1, 1] = 1.31 - 1.3

    np.testing.assert_array_equal(first.delta_refractive_index, expected)

    np.testing.assert_array_equal(repeated.delta_refractive_index, first.delta_refractive_index)

    changed = medium.to_volume(
        seed=43,
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        ),
    )

    assert not np.array_equal(changed.delta_refractive_index, first.delta_refractive_index)

    assert changed.delta_refractive_index[1, 1, 1] == first.delta_refractive_index[1, 1, 1]

    assert medium.metadata["background"] == statistics.metadata

    assert json.loads(json.dumps(medium.metadata)) == medium.metadata


def test_replacing_background_retains_absolute_structure_indices_and_old_volumes():
    medium = StructuredMedium(background_refractive_index=1.33)

    sphere = Sphere(
        radius=2e-08 * ureg.meter,
        refractive_index=1.34,
    )

    medium.add_structures(sphere)

    statistics = WhittleMaternMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
        smoothness=1.5,
    )

    medium.add_background(medium=statistics)

    previous = medium.to_volume(
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        )
    )

    snapshot = previous.delta_refractive_index.copy()

    medium.add_background(refractive_index=1.3)

    current = medium.to_volume(
        grid=Grid(
            shape=(3, 3, 3),
            spacing=5e-08 * ureg.meter,
        )
    )

    expected = np.zeros((3, 3, 3))

    expected[1, 1, 1] = 1.34 - 1.3

    np.testing.assert_array_equal(current.delta_refractive_index, expected)

    np.testing.assert_array_equal(previous.delta_refractive_index, snapshot)

    assert previous.background_refractive_index == 1.33

    assert "background" not in medium.metadata


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({}, ValueError),
        ({"refractive_index": 0}, ValueError),
        ({"refractive_index": np.nan}, ValueError),
        ({"refractive_index": 1 * ureg.meter}, ValueError),
        ({"medium": object()}, TypeError),
        (
            {
                "refractive_index": 1.3,
                "medium": WhittleMaternMedium(
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=1e-07 * ureg.meter,
                    smoothness=1.5,
                ),
            },
            ValueError,
        ),
    ],
)
def test_invalid_background_is_atomic(arguments, error):
    medium = StructuredMedium(background_refractive_index=1.33)

    before = medium.metadata

    with pytest.raises(error):
        medium.add_background(**arguments)

    assert medium.metadata == before


def test_invalid_structure_batch_is_atomic_and_empty_batch_preserves_state():
    medium = StructuredMedium(background_refractive_index=1.33)

    sphere = Sphere(
        radius=1 * ureg.meter,
        refractive_index=1.34,
    )

    medium.add_structures(sphere)

    before = medium.metadata

    with pytest.raises(TypeError, match="structures must"):
        medium.add_structures(sphere, object(), sphere)

    assert medium.metadata == before

    assert medium.add_structures() is None

    assert medium.metadata == before

    with pytest.raises(TypeError):
        medium.add_background(1.3)

    with pytest.raises(TypeError):
        medium.add_structures(structures=[sphere])
