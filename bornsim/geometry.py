"""Voxel-centre geometry for finite structured dielectric samples.

Lengths use SI metres or compatible quantities. Shapes set absolute refractive index,
not dielectric contrast; Volume retains the solver's linearized constitutive
law. Interfaces are sampled at cell centres without subvoxel averaging.
"""

from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from typing import Any
from .units import _refractive_index_values, Quantity, validate_units, ureg
from .medium import Medium
from .medium.random_medium import RandomFieldMedium
from .volume import Volume
from .grid import Grid
from .rotation import Rotation, RotationInput, RotationMatrix
from .material import Material
import warnings


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


@dataclass(kw_only=True)
class StructuredMedium(Medium):
    """Compose ordered material regions in a uniform background.

    Parameters
    ----------
    regions : sequence of Layer, Sphere, Ellipsoid, Box, or Cylinder
        Regions set absolute refractive index. Overlap precedence follows the
        overlap policy; indices are not added. Default is empty.
    background_refractive_index : float
        Explicit positive uniform background refractive index. It fills uncovered
        voxels and extends outside the finite voxel box.

    overlap : {"replace", "preserve", "error"}
        Later regions win, earlier regions win, or reject shared voxel centres.
    warn_on_clipping : bool
        Warn when geometry extends beyond the voxel box; default False.
    fluctuations : RandomFieldMedium, configured with :meth:`add_fluctuations`
        Optional additive field applied over the background and every region.

    Notes
    -----
    Configure this mutable builder with ``add_background``, ``add_structures``,
    and optionally ``add_fluctuations``.
    An empty builder has no background; voxelization requires an explicit one.
    Medium remains abstract; random statistics and geometry shapes remain frozen.
    Calls update the builder and return None. Previously generated volumes do
    not change. Constructor regions are still accepted for existing callers.
    Voxelization returns ``delta_refractive_index(r) = n(r) - background_refractive_index``. A
    configured additive fluctuation field is added after the absolute-index regions are composed.
    The numerical solver uses ``epsilon_r = n0**2 + 2*n0*delta_refractive_index``, omitting
    ``delta_refractive_index**2`` even at higher Born orders. Strong refractive index differences
    can invalidate this constitutive approximation or Born iteration.
    Regions crossing the box are clipped. Layers therefore have finite lateral
    extent; this is not a transfer-matrix or layered-background Green solver.
    """

    regions: tuple = ()
    background_refractive_index: float | None = None
    overlap: str = "replace"
    warn_on_clipping: bool = False
    _background: RandomFieldMedium | None = field(default=None, init=False, repr=False)
    _fluctuations: RandomFieldMedium | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        regions = tuple(self.regions)

        if any(not isinstance(region, (Layer, Sphere, Ellipsoid, Box, Cylinder)) for region in regions):
            raise TypeError("regions must contain Layer, Sphere, Ellipsoid, Box, or Cylinder objects.")

        if self.overlap not in ("replace", "preserve", "error"):
            raise ValueError("overlap must be replace, preserve, or error.")

        if not isinstance(self.warn_on_clipping, bool):
            raise ValueError("warn_on_clipping must be a bool.")

        self.regions = regions

        if self.background_refractive_index is not None:
            self.background_refractive_index = _refractive_index(value=self.background_refractive_index)

    def add_background(self, *, refractive_index=None, medium=None, material=None):
        """Set the background in place, retaining all existing structures.

        Supply exactly one of ``refractive_index`` (a positive uniform refractive index)
        or ``medium`` (RandomFieldMedium statistics). A random background is sampled
        with to_volume(seed=...) and fills only voxels outside structures.
        The exterior scattering background remains uniform at background_refractive_index.
        Replacing a background changes the reference n0, not structures'
        absolute refractive indices. Validation precedes any state change.
        Returns None. Random statistics are copied when added.
        """

        if material is not None:
            resolved = Material._resolve(material=material, refractive_index=refractive_index)

            refractive_index = resolved.refractive_index

        if (refractive_index is None) == (medium is None):
            raise ValueError("Supply exactly one of refractive_index or medium.")

        if medium is not None:
            if not isinstance(medium, RandomFieldMedium):
                raise TypeError("medium must be a RandomFieldMedium.")

            background = replace(medium)

            background_refractive_index = background.background_refractive_index
        else:
            background = None

            background_refractive_index = _refractive_index(value=refractive_index)

        if (
            self._fluctuations is not None
            and background_refractive_index != self._fluctuations.background_refractive_index
        ):
            raise ValueError("background_refractive_index must match the configured fluctuations.")

        if background is not None and self._fluctuations is not None:
            raise ValueError("A random background cannot be combined with additive fluctuations.")

        self.background_refractive_index = background_refractive_index

        self._background = background

    def add_fluctuations(self, *, medium):
        """Set an additive random field applied over the background and all structures.

        The statistical medium must share the configured uniform background.
        This replaces any previously configured additive fluctuation field.
        Random backgrounds and additive fluctuations are mutually exclusive.
        Returns None. Generated volumes record the medium and seed as provenance.
        """

        if not isinstance(medium, RandomFieldMedium):
            raise TypeError("medium must be a RandomFieldMedium.")

        fluctuations = replace(medium)

        if self.background_refractive_index is None:
            raise ValueError("Set a uniform background before adding fluctuations.")

        if self._background is not None:
            raise ValueError("Additive fluctuations cannot be combined with a random background.")

        if fluctuations.background_refractive_index != self.background_refractive_index:
            raise ValueError("fluctuation medium background_refractive_index must match the structured medium.")

        self._fluctuations = fluctuations

    def add_structures(self, *structures):
        """Append material regions in argument order; return None.

        Accepts any number of positional Layer, Sphere, Ellipsoid, Box, or
        Cylinder objects. Later arguments replace earlier material in overlaps
        and replace a random background inside their masks. Separately configured
        additive fluctuations remain present over every region. All arguments
        are validated before editing the builder; an empty call is a no-op.
        Existing voxel samples are independent snapshots.
        """

        if any(not isinstance(structure, (Layer, Sphere, Ellipsoid, Box, Cylinder)) for structure in structures):
            raise TypeError("structures must be Layer, Sphere, Ellipsoid, Box, or Cylinder objects.")

        self.regions = (*self.regions, *structures)

    @property
    def is_random(self):
        """Random backgrounds or additive fluctuations supply independent realizations."""

        return self._background is not None or self._fluctuations is not None

    @property
    def metadata(self):
        """Return independent SI geometry descriptions in application order.

        Region type, absolute refractive index, axis where applicable, and dimensions are
        recorded. Length keys have the ``_m`` suffix, and vectors are JSON lists.
        The list order preserves replacement precedence in overlapping regions.
        Configured additive fluctuation statistics are recorded under
        ``fluctuations``. Generated volumes with fluctuations retain a snapshot
        of this metadata and their seed.
        """

        length_fields = {"lower", "upper", "radius", "height", "centre", "radii", "size"}

        regions = []

        for region in self.regions:
            description: dict[str, object] = {"type": type(region).__name__}

            for name, value in asdict(region).items():
                if name == "material":
                    continue

                if name == "rotation" and np.array_equal(value, np.eye(3)):
                    continue

                if isinstance(region, Layer) and name == "centre" and np.all(value == 0):
                    continue

                if name in length_fields:
                    magnitude = value.to("meter").magnitude

                    value = np.asarray(magnitude).tolist()

                key = f"{name}_m" if name in length_fields else name

                description[key] = list(value) if isinstance(value, tuple) else value

            regions.append(description)

        metadata: dict[str, object] = {**super().metadata, "regions": regions}

        if self.overlap != "replace":
            metadata["overlap"] = self.overlap

        if self._background is not None:
            metadata["background"] = self._background.metadata

        if self._fluctuations is not None:
            metadata["fluctuations"] = self._fluctuations.metadata

        return metadata

    def to_volume(self, *, grid: Grid, seed: int = 0) -> Volume:
        """Voxelize this medium on an explicit Grid.

        An explicit background is required. Ordered shapes set absolute refractive
        indices; overlap follows replace, preserve, or error. A random background
        uses the supplied seed and is replaced by structures inside their masks.
        Additive fluctuations use the same seed and are applied over all regions."""

        if self.background_refractive_index is None:
            raise ValueError("Set a background with add_background before generating a volume.")

        grid = Grid._resolve(grid=grid)

        if self.overlap not in ("replace", "preserve", "error"):
            raise ValueError("overlap must be replace, preserve, or error.")

        positions = grid.positions

        if self._background is None:
            contrast = np.zeros(grid.shape)
        else:
            background = self._background.to_volume(
                grid=grid,
                seed=seed,
            )

            contrast = background.delta_refractive_index.copy()

        occupied = np.zeros(grid.shape, dtype=bool)

        for region_index, region in enumerate(self.regions):
            if self.warn_on_clipping and self._is_clipped(region=region, grid=grid):
                warnings.warn(
                    f"Region {region_index} ({type(region).__name__}) crosses the voxel box and is clipped.",
                    UserWarning,
                    stacklevel=2,
                )

            mask = region.mask(positions=positions)

            if self.overlap == "error" and np.any(mask & occupied):
                raise ValueError(f"Region {region_index} overlaps existing structures at voxel centres.")

            selected = mask & ~occupied if self.overlap == "preserve" else mask

            contrast[selected] = region.refractive_index - self.background_refractive_index

            occupied |= mask

        if self._fluctuations is not None:
            fluctuation_field = self._fluctuations.to_volume(
                grid=grid,
                seed=seed,
            )

            contrast += fluctuation_field.delta_refractive_index

        return Volume(
            delta_refractive_index=contrast,
            grid=grid,
            background_refractive_index=self.background_refractive_index,
            medium=deepcopy(self) if self._fluctuations is not None else None,
            seed=seed if self._fluctuations is not None else None,
        )

    @staticmethod
    def _is_clipped(*, region, grid):
        half_box = np.asarray(grid.shape) * grid.spacing / 2

        centre = region.centre

        if isinstance(region, Sphere):
            half_extent = np.ones(3) * region.radius
        else:
            matrix = np.asarray(region.rotation)

            if isinstance(region, Layer):
                normal = matrix[:, region.axis]

                reach = np.sum(np.abs(normal) * half_box)

                shift = np.dot(centre, normal)

                return region.lower + shift < -reach or region.upper + shift > reach

            if isinstance(region, Box):
                half_extent = np.abs(matrix) @ (region.size / 2)
            elif isinstance(region, Ellipsoid):
                half_extent = np.sqrt(matrix**2 @ region.radii**2)
            else:
                normal = matrix[:, region.axis]

                half_extent = np.abs(normal) * region.height / 2 + region.radius * np.sqrt(np.maximum(0, 1 - normal**2))

        return bool(np.any(np.abs(centre) + half_extent > half_box * (1 + 1e-12)))
