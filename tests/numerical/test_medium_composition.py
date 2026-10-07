"""Validate editable composition against independent voxel references."""

import json

import numpy as np
import pytest

from bornsim import Layer, RandomMedium, Sphere, StructuredMedium
from bornsim.units import ureg


def test_background_and_structures_update_in_place_with_last_structure_winning():
    medium = StructuredMedium()
    assert medium.add_background(index=1.3 * ureg.dimensionless) is None
    layer = Layer(
        lower=-100e-9,
        upper=100e-9,
        index=1.31,
    )

    sphere = Sphere(
        radius=20e-9,
        index=1.32,
    )

    returned = medium.add_structures(
        layer,
        sphere,
    )
    assert returned is None
    volume = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
    )
    expected = np.full((3, 3, 3), 1.31 - 1.3)
    expected[1, 1, 1] = 1.32 - 1.3
    np.testing.assert_array_equal(volume.delta_index, expected)
    assert medium.metadata["background_index"] == 1.3
    assert [region["type"] for region in medium.metadata["regions"]] == ["Layer", "Sphere"]


def test_random_background_seed_and_structure_replacement():
    statistics = RandomMedium(
        background_index=1.3,
        index_std=0.005,
    )
    medium = StructuredMedium()
    medium.add_background(medium=statistics)
    sphere = Sphere(
        radius=20e-9,
        index=1.31,
    )

    medium.add_structures(
        sphere,
    )
    background = statistics.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
        seed=42,
    )
    first = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
        seed=42,
    )
    repeated = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
        seed=42,
    )
    expected = background.delta_index.copy()
    expected[1, 1, 1] = 1.31 - 1.3
    np.testing.assert_array_equal(first.delta_index, expected)
    np.testing.assert_array_equal(repeated.delta_index, first.delta_index)
    changed = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
        seed=43,
    )
    assert not np.array_equal(changed.delta_index, first.delta_index)
    assert changed.delta_index[1, 1, 1] == first.delta_index[1, 1, 1]
    assert medium.metadata["background"] == statistics.metadata
    assert json.loads(json.dumps(medium.metadata)) == medium.metadata


def test_replacing_background_retains_absolute_structure_indices_and_old_volumes():
    medium = StructuredMedium()
    sphere = Sphere(
        radius=20e-9,
        index=1.34,
    )

    medium.add_structures(
        sphere,
    )
    statistics = RandomMedium()
    medium.add_background(medium=statistics)
    previous = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
    )
    snapshot = previous.delta_index.copy()
    medium.add_background(index=1.3)
    current = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
    )
    expected = np.zeros((3, 3, 3))
    expected[1, 1, 1] = 1.34 - 1.3
    np.testing.assert_array_equal(current.delta_index, expected)
    np.testing.assert_array_equal(previous.delta_index, snapshot)
    assert previous.background_index == 1.33
    assert "background" not in medium.metadata


@pytest.mark.parametrize(
    "arguments, error",
    [
        ({}, ValueError),
        ({"index": 0}, ValueError),
        ({"index": np.nan}, ValueError),
        ({"index": 1 * ureg.meter}, ValueError),
        ({"medium": object()}, TypeError),
        ({"index": 1.3, "medium": RandomMedium()}, ValueError),
    ],
)
def test_invalid_background_is_atomic(arguments, error):
    medium = StructuredMedium()
    before = medium.metadata
    with pytest.raises(error):
        medium.add_background(**arguments)
    assert medium.metadata == before


def test_invalid_structure_batch_is_atomic_and_empty_batch_preserves_state():
    medium = StructuredMedium()
    sphere = Sphere(
        radius=1,
        index=1.34,
    )
    medium.add_structures(
        sphere,
    )
    before = medium.metadata
    with pytest.raises(TypeError, match="structures must"):
        medium.add_structures(
            sphere,
            object(),
            sphere,
        )
    assert medium.metadata == before
    assert medium.add_structures() is None
    assert medium.metadata == before
    with pytest.raises(TypeError):
        medium.add_background(1.3)
    with pytest.raises(TypeError):
        medium.add_structures(structures=[sphere])
