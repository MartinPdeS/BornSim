import numpy as np
import pytest
from bornsim import Box, Rotation, Sphere
from bornsim.rotation import RotationInput, RotationMatrix
from bornsim.units import ureg


def test_rotation_matrix_uses_active_right_handed_convention():
    rotation = Rotation(
        axis=(0, 0, 1),
        angle=90 * ureg.degree,
    )

    expected = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )

    np.testing.assert_allclose(rotation.matrix, expected, atol=1e-15)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param(
            (
                (0, -1, 0),
                (1, 0, 0),
                (0, 0, 1),
            ),
            id="nested-sequence",
        ),
        pytest.param(
            np.array(
                [
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                ]
            ),
            id="numpy-array",
        ),
        pytest.param(
            np.array(
                [
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                ]
            )
            * ureg.dimensionless,
            id="dimensionless-quantity",
        ),
    ],
)
def test_shape_normalizes_rotation_input_to_immutable_matrix(matrix: RotationInput):
    box = Box(
        size=(4, 2, 2) * ureg.meter,
        refractive_index=1.34,
        rotation=matrix,
    )

    expected: RotationMatrix = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    assert box.rotation == expected

    positions = np.array([[1.5, 0, 0], [0, 1.5, 0]]) * ureg.meter

    np.testing.assert_array_equal(box.mask(positions=positions), [False, True])


def test_transform_preserves_orientation_and_rotates_about_global_pivot():
    box = Box(
        size=(4, 2, 2) * ureg.meter,
        refractive_index=1.34,
    )

    rotation = Rotation(
        axis=(0, 0, 1),
        angle=90 * ureg.degree,
    )

    rotated = box.rotated(rotation=rotation)

    positions = np.array([[1.5, 0, 0], [0, 1.5, 0]]) * ureg.meter

    np.testing.assert_array_equal(box.mask(positions=positions), [True, False])

    np.testing.assert_array_equal(rotated.mask(positions=positions), [False, True])

    moved = rotated.translated(offset=(3, 0, 0) * ureg.meter)

    np.testing.assert_array_equal(
        moved.mask(positions=(np.array([[1.5, 0, 0], [0, 1.5, 0]]) + [3, 0, 0]) * ureg.meter),
        [False, True],
    )

    orbited = moved.rotated(
        rotation=rotation,
        about=(0, 0, 0) * ureg.meter,
    )

    np.testing.assert_allclose(orbited.centre.to(ureg.meter).magnitude, (0, 3, 0), atol=1e-14)


def test_sphere_rotation_moves_its_centre_without_adding_orientation():
    sphere = Sphere(
        radius=1 * ureg.meter,
        refractive_index=1.35,
        centre=(3, 0, 0) * ureg.meter,
    )

    rotated = sphere.rotated(
        rotation=Rotation(
            axis=(0, 0, 1),
            angle=90 * ureg.degree,
        ),
        about=(0, 0, 0) * ureg.meter,
    )

    np.testing.assert_allclose(rotated.centre.to(ureg.meter).magnitude, (0, 3, 0), atol=1e-14)

    assert "rotation" not in Sphere.__dataclass_fields__


def test_rotation_rejects_improper_matrices():
    box = Box(
        size=(4, 2, 2) * ureg.meter,
        refractive_index=1.34,
    )

    with pytest.raises(ValueError, match="orthonormal"):
        box.rotated(rotation=np.diag([1, 1, -1]))
