"""Voxel-centre geometry for finite structured dielectric samples.

Lengths use SI metres or compatible quantities. Shapes set absolute refractive index,
not dielectric contrast; Volume retains the solver's linearized constitutive
law. Interfaces are sampled at cell centres without subvoxel averaging.
"""

from dataclasses import dataclass, field, replace
import numpy as np
from numpy.typing import NDArray
from typing import Any
from .units import _refractive_index_values, Quantity, validate_units, ureg
from .rotation import Rotation, RotationInput, RotationMatrix
from .material import Material


def _refractive_index(*, value: float) -> float:
    refractive_index = _refractive_index_values(
        value=value,
        name="refractive_index",
        scalar=True,
    )

    if not np.isfinite(refractive_index) or refractive_index <= 0:
        raise ValueError("refractive_index must be finite and positive.")

    return float(refractive_index)


def _axis(*, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2 (x, y, z).")

    return int(value)


class _RotationUnset:
    pass


_ROTATION_UNSET = _RotationUnset()


@dataclass(frozen=True, kw_only=True)
class _Transformable:
    """Return transformed immutable shapes in global SI coordinates."""

    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    refractive_index: float | None = None
    material: Material | None = None

    def _validate_transform(
        self,
        *,
        rotation: RotationInput | _RotationUnset = _ROTATION_UNSET,
    ) -> None:
        material = Material._resolve(material=self.material, refractive_index=self.refractive_index)

        object.__setattr__(self, "material", material)

        object.__setattr__(self, "refractive_index", material.refractive_index)

        validate_units(
            self.centre,
            unit="meter",
            name="centre",
            scalar=False,
        )

        if np.shape(self.centre) != (3,):
            raise ValueError("centre must contain three coordinates.")

        if np.any(~np.isfinite(self.centre)):
            raise ValueError("centre must be finite.")

        if not isinstance(rotation, _RotationUnset):
            matrix = Rotation._matrix(rotation=rotation)

            object.__setattr__(
                self,
                "rotation",
                (
                    (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
                    (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
                    (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
                ),
            )

    def _replaced(self, *, settings: dict[str, Any]) -> "_Transformable":
        """Clone geometry while retaining unit-bearing lengths."""

        return replace(self, **settings)

    def translated(self, *, offset: Quantity) -> "_Transformable":
        """Return a translated shape; offset uses metres or length quantities."""

        validate_units(
            offset,
            unit="meter",
            name="offset",
            scalar=False,
        )

        if np.shape(offset) != (3,):
            raise ValueError("offset must contain three coordinates.")

        if np.any(~np.isfinite(offset)):
            raise ValueError("offset must be finite.")

        displacement = offset

        centre = self.centre + displacement

        return self._replaced(settings={"centre": centre, "refractive_index": None})

    def _rotation_settings(self, *, matrix: NDArray[np.float64]) -> dict[str, RotationMatrix]:
        return {}

    def rotated(self, *, rotation: RotationInput, about: Quantity | None = None) -> "_Transformable":
        """Return a rotated shape around its centre or an explicit global point.

        See :class:`bornsim.rotation.Rotation` for the rotation convention.
        """

        matrix = Rotation._matrix(rotation=rotation)

        centre = self.centre

        if about is not None:
            validate_units(
                about,
                unit="meter",
                name="about",
                scalar=False,
            )

            if np.shape(about) != (3,):
                raise ValueError("about must contain three coordinates.")

            if np.any(~np.isfinite(about)):
                raise ValueError("about must be finite.")

            point = about

            centre = point + matrix @ (centre - point)

        settings: dict[str, Any] = {"centre": centre, "refractive_index": None}

        settings.update(self._rotation_settings(matrix=matrix))

        return self._replaced(settings=settings)


class _RotatableTransformable(_Transformable):
    rotation: RotationInput

    def _rotation_settings(self, *, matrix: NDArray[np.float64]) -> dict[str, RotationMatrix]:
        rotated = matrix @ np.asarray(self.rotation)

        return {
            "rotation": (
                (float(rotated[0, 0]), float(rotated[0, 1]), float(rotated[0, 2])),
                (float(rotated[1, 0]), float(rotated[1, 1]), float(rotated[1, 2])),
                (float(rotated[2, 0]), float(rotated[2, 1]), float(rotated[2, 2])),
            )
        }


@dataclass(frozen=True, kw_only=True)
class Layer(_RotatableTransformable):
    """A slab with absolute refractive index inside ``lower <= local[axis] < upper``.

    Bounds are SI lengths in the slab local frame, offset by centre and
    oriented by rotation. The default axis is
    z (2); x and y are 0 and 1. The half-open interval gives adjacent slabs an
    unambiguous interface. The slab spans the voxel box in transverse axes;
    it represents a finite sample, not an infinite planar background.

    Use :meth:`from_center_and_thickness` to specify a slab by its centre and
    full thickness instead of its local lower and upper bounds.
    """

    lower: Quantity
    upper: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    axis: int = 2
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    rotation: RotationInput = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    @classmethod
    def from_center_and_thickness(
        cls,
        *,
        thickness: Quantity,
        centre: Quantity,
        refractive_index: float | None = None,
        material: Material | None = None,
        axis: int = 2,
        rotation: RotationInput = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ) -> "Layer":
        """Create a slab from its centre and full thickness along ``axis``."""

        validate_units(
            thickness,
            unit="meter",
            name="thickness",
            scalar=True,
        )

        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("thickness must be finite and positive.")

        return cls(
            lower=-thickness / 2,
            upper=thickness / 2,
            centre=centre,
            refractive_index=refractive_index,
            material=material,
            axis=axis,
            rotation=rotation,
        )

    def __post_init__(self) -> None:
        validate_units(
            self.lower,
            unit="meter",
            name="lower",
            scalar=True,
        )

        if np.any(~np.isfinite(self.lower)):
            raise ValueError("lower must be finite.")

        validate_units(
            self.upper,
            unit="meter",
            name="upper",
            scalar=True,
        )

        if np.any(~np.isfinite(self.upper)):
            raise ValueError("upper must be finite.")

        object.__setattr__(self, "axis", _axis(value=self.axis))

        self._validate_transform(rotation=self.rotation)

        if self.lower >= self.upper:
            raise ValueError("lower must be smaller than upper.")

    def mask(self, *, positions: Quantity) -> np.ndarray:
        """Return membership at unit-bearing voxel positions, shape (..., 3)."""

        validate_units(
            positions,
            unit="meter",
            name="positions",
        )

        coordinate = ((positions - self.centre) @ np.asarray(self.rotation))[..., self.axis]

        return (coordinate >= self.lower) & (coordinate < self.upper)


@dataclass(frozen=True, kw_only=True)
class Sphere(_Transformable):
    """Set absolute refractive index where ``sum((r-centre)**2) <= radius**2``.

    Radius is positive; centre is a three-component SI length or quantity.
    The closed surface is sampled at voxel centres, giving a staircase boundary.
    """

    radius: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)

    def __post_init__(self) -> None:
        validate_units(
            self.radius,
            unit="meter",
            name="radius",
            scalar=True,
        )

        if np.any(~np.isfinite(self.radius)) or np.any(self.radius <= 0):
            raise ValueError("radius must be finite and positive.")

        self._validate_transform()

    def mask(self, *, positions: Quantity) -> np.ndarray:
        """Return membership at unit-bearing voxel positions, shape (..., 3)."""

        validate_units(
            positions,
            unit="meter",
            name="positions",
        )

        return np.sum((positions - self.centre) ** 2, axis=-1) <= self.radius**2


@dataclass(frozen=True, kw_only=True)
class Ellipsoid(_RotatableTransformable):
    """Set absolute refractive index where ``sum((local/radii)**2) <= 1``.

    Three positive semi-axis radii align with local x, y, z. All lengths use SI or
    compatible quantities. rotation maps local semi-axes into global coordinates.
    """

    radii: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    rotation: RotationInput = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self) -> None:
        validate_units(
            self.radii,
            unit="meter",
            name="radii",
            scalar=False,
        )

        if np.shape(self.radii) != (3,):
            raise ValueError("radii must contain three coordinates.")

        if np.any(~np.isfinite(self.radii)) or np.any(self.radii <= 0):
            raise ValueError("radii must be finite and positive.")

        self._validate_transform(rotation=self.rotation)

    def mask(self, *, positions: Quantity) -> np.ndarray:
        """Return membership at unit-bearing voxel positions, shape (..., 3)."""

        validate_units(
            positions,
            unit="meter",
            name="positions",
        )

        local = (positions - self.centre) @ np.asarray(self.rotation)

        return np.sum((local / self.radii) ** 2, axis=-1) <= 1


@dataclass(frozen=True, kw_only=True)
class Box(_RotatableTransformable):
    """Set absolute refractive index where ``abs(local) <= size/2`` in every axis.

    Size contains three positive full side lengths along the local axes.
    All lengths use SI or compatible quantities; the boundary is closed.
    """

    size: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    rotation: RotationInput = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self) -> None:
        validate_units(
            self.size,
            unit="meter",
            name="size",
            scalar=False,
        )

        if np.shape(self.size) != (3,):
            raise ValueError("size must contain three coordinates.")

        if np.any(~np.isfinite(self.size)) or np.any(self.size <= 0):
            raise ValueError("size must be finite and positive.")

        self._validate_transform(rotation=self.rotation)

    def mask(self, *, positions: Quantity) -> np.ndarray:
        """Return membership at unit-bearing voxel positions, shape (..., 3)."""

        validate_units(
            positions,
            unit="meter",
            name="positions",
        )

        local = (positions - self.centre) @ np.asarray(self.rotation)

        return np.all(np.abs(local) <= self.size / 2, axis=-1)


@dataclass(frozen=True, kw_only=True)
class Cylinder(_RotatableTransformable):
    """A finite circular cylinder with absolute refractive index and closed boundary.

    Membership requires ``abs(local[axis]) <= height/2`` and squared
    transverse distance <= radius**2. Radius and full height are positive SI
    lengths or quantities; axis is 0, 1, or 2, default z. Centre has three lengths.
    """

    radius: Quantity
    height: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    axis: int = 2
    rotation: RotationInput = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self) -> None:
        validate_units(
            self.radius,
            unit="meter",
            name="radius",
            scalar=True,
        )

        if np.any(~np.isfinite(self.radius)) or np.any(self.radius <= 0):
            raise ValueError("radius must be finite and positive.")

        validate_units(
            self.height,
            unit="meter",
            name="height",
            scalar=True,
        )

        if np.any(~np.isfinite(self.height)) or np.any(self.height <= 0):
            raise ValueError("height must be finite and positive.")

        object.__setattr__(self, "axis", _axis(value=self.axis))

        self._validate_transform(rotation=self.rotation)

    def mask(self, *, positions: Quantity) -> np.ndarray:
        """Return membership at unit-bearing voxel positions, shape (..., 3)."""

        validate_units(
            positions,
            unit="meter",
            name="positions",
        )

        displacement = (positions - self.centre) @ np.asarray(self.rotation)

        transverse = [i for i in range(3) if i != self.axis]

        return (np.abs(displacement[..., self.axis]) <= self.height / 2) & (
            np.sum(displacement[..., transverse] ** 2, axis=-1) <= self.radius**2
        )


def __getattr__(name):
    """Preserve the historical StructuredMedium import."""
    if name == "StructuredMedium":
        from .medium.structured.composite import StructuredMedium

        return StructuredMedium

    raise AttributeError(name)
