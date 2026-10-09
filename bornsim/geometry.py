"""Voxel-centre geometry for finite structured dielectric samples.

Lengths use SI metres or compatible quantities. Shapes set absolute refractive index,
not dielectric contrast; Volume retains the solver's linearized constitutive
law. Interfaces are sampled at cell centres without subvoxel averaging.
"""

from dataclasses import asdict, dataclass, field, replace
import numpy as np
from typing import Any
from .units import _refractive_index_values, Quantity, validate_units, ureg
from .media import Medium, RandomMedium
from .volume import Volume
from .grid import Grid
from .rotation import Rotation
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


@dataclass(frozen=True, kw_only=True)
class _Transformable:
    """Return transformed immutable shapes in global SI coordinates."""

    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    refractive_index: float | None = None
    material: Material | None = None

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

    def rotated(self, *, rotation: Rotation | np.ndarray, about: Quantity | None = None) -> "_Transformable":
        """Return a rotated shape around its centre or an explicit global point.

        Rotation is active: R maps local column coordinates to the global
        frame. Successive world rotations compose as R_new @ R_existing.
        Membership uses local row coordinates (positions-centre) @ R.
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

        if hasattr(self, "rotation"):
            settings["rotation"] = tuple(tuple(row) for row in matrix @ np.asarray(self.rotation))

        return self._replaced(settings=settings)


@dataclass(frozen=True, kw_only=True)
class Layer(_Transformable):
    """A slab with absolute refractive index inside ``lower <= local[axis] < upper``.

    Bounds are SI lengths in the slab local frame, offset by centre and
    oriented by rotation. The default axis is
    z (2); x and y are 0 and 1. The half-open interval gives adjacent slabs an
    unambiguous interface. The slab spans the voxel box in transverse axes;
    it represents a finite sample, not an infinite planar background.
    """

    lower: Quantity
    upper: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    axis: int = 2
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

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

        material = Material._resolve(material=self.material, refractive_index=self.refractive_index)

        object.__setattr__(self, "material", material)

        object.__setattr__(self, "refractive_index", material.refractive_index)

        object.__setattr__(self, "axis", _axis(value=self.axis))

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

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

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

    def mask(self, *, positions: Quantity) -> np.ndarray:
        """Return membership at unit-bearing voxel positions, shape (..., 3)."""

        validate_units(
            positions,
            unit="meter",
            name="positions",
        )

        return np.sum((positions - self.centre) ** 2, axis=-1) <= self.radius**2


@dataclass(frozen=True, kw_only=True)
class Ellipsoid(_Transformable):
    """Set absolute refractive index where ``sum((local/radii)**2) <= 1``.

    Three positive semi-axis radii align with local x, y, z. All lengths use SI or
    compatible quantities. rotation maps local semi-axes into global coordinates.
    """

    radii: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

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

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

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
class Box(_Transformable):
    """Set absolute refractive index where ``abs(local) <= size/2`` in every axis.

    Size contains three positive full side lengths along the local axes.
    All lengths use SI or compatible quantities; the boundary is closed.
    """

    size: Quantity
    refractive_index: float | None = None
    material: Material | None = None
    centre: Quantity = field(default_factory=lambda: np.zeros(3) * ureg.meter)
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

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

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

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
class Cylinder(_Transformable):
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
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

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

        object.__setattr__(self, "axis", _axis(value=self.axis))

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

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

    Notes
    -----
    Configure this mutable builder with ``add_background`` and ``add_structures``.
    An empty builder has no background; voxelization requires an explicit one.
    Medium remains abstract; random statistics and geometry shapes remain frozen.
    Calls update the builder and return None. Previously generated volumes do
    not change. Constructor regions are still accepted for existing callers.
    Voxelization returns ``delta_refractive_index(r) = n(r) - background_refractive_index``. The
    numerical solver uses ``epsilon_r = n0**2 + 2*n0*delta_refractive_index``, omitting
    ``delta_refractive_index**2`` even at higher Born orders. Strong refractive index differences
    can invalidate this constitutive approximation or Born iteration.
    Regions crossing the box are clipped. Layers therefore have finite lateral
    extent; this is not a transfer-matrix or layered-background Green solver.
    """

    regions: tuple = ()
    background_refractive_index: float | None = None
    overlap: str = "replace"
    warn_on_clipping: bool = False
    _background: RandomMedium | None = field(default=None, init=False, repr=False)

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
        or ``medium`` (RandomMedium statistics). A random background is sampled
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
            if not isinstance(medium, RandomMedium):
                raise TypeError("medium must be a RandomMedium.")

            background = replace(medium)

            background_refractive_index = background.background_refractive_index
        else:
            background = None

            background_refractive_index = _refractive_index(value=refractive_index)

        self.background_refractive_index = background_refractive_index

        self._background = background

    def add_structures(self, *structures):
        """Append material regions in argument order; return None.

        Accepts any number of positional Layer, Sphere, Ellipsoid, Box, or
        Cylinder objects. Later arguments replace earlier material in overlaps,
        including random background fluctuations. All arguments are validated
        before editing the builder; an empty call is a no-op. Existing voxel
        samples are independent snapshots.
        """

        if any(not isinstance(structure, (Layer, Sphere, Ellipsoid, Box, Cylinder)) for structure in structures):
            raise TypeError("structures must be Layer, Sphere, Ellipsoid, Box, or Cylinder objects.")

        self.regions = (*self.regions, *structures)

    @property
    def is_random(self):
        """A random background supplies independent composed realizations."""

        return self._background is not None

    @property
    def metadata(self):
        """Return independent SI geometry descriptions in application order.

        Region type, absolute refractive index, axis where applicable, and dimensions are
        recorded. Length keys have the ``_m`` suffix, and vectors are JSON lists.
        The list order preserves replacement precedence in overlapping regions.
        Voxelization returns a manual volume; retain this description separately
        when reconstructing a structured sample.
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

        return metadata

    def to_volume(
        self,
        *,
        grid: Grid | None = None,
        shape: tuple[int, int, int] | None = None,
        spacing: Quantity | None = None,
        seed: int = 0,
    ) -> Volume:
        """Sample regions on cubic voxels and return a validated :class:`Volume`.

        Pass a shared Grid as grid. Legacy shape and spacing keywords are
        also supported, but cannot be combined with grid.
        Shape contains three integers from 2 to 32. Spacing is positive, in
        metres or compatible units. Coordinates are ``(i-(N-1)/2)*spacing``.
        Interfaces use centre membership; check refinement at fixed dimensions.
        For a random background, seed selects the realization. Structures replace
        that background at their voxel centres; fluctuations are not added inside
        structures. Uniform backgrounds ignore seed. Keep metadata and seed to
        reproduce a composed sample, whose Volume is a manual field.
        """

        if self.background_refractive_index is None:
            raise ValueError("Set a background with add_background before generating a volume.")

        grid = Grid._resolve(
            grid=grid,
            shape=shape,
            spacing=spacing,
        )

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

        return Volume(
            delta_refractive_index=contrast,
            grid=grid,
            background_refractive_index=self.background_refractive_index,
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
