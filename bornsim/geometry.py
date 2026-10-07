"""Voxel-centre geometry for finite structured dielectric samples.

Lengths use SI metres or compatible quantities. Shapes set absolute index,
not dielectric contrast; Volume retains the solver's linearized constitutive
law. Interfaces are sampled at cell centres without subvoxel averaging.
"""

from dataclasses import asdict, dataclass, field, replace
import numpy as np
from typing import Any
from .units import Quantity, _si
from .media import Medium, RandomMedium
from .volume import Volume
from .grid import Grid
from .rotation import Rotation
from .material import Material
import warnings


def _length(*, value, name, positive=False, vector=False):
    data = _si(
        value=value,
        unit="meter",
        name=name,
        scalar=not vector,
    )
    if vector and np.shape(data) != (3,):
        raise ValueError(f"{name} must contain three coordinates.")
    if np.any(~np.isfinite(data)) or (positive and np.any(np.asarray(data) <= 0)):
        raise ValueError(f"{name} must be finite" + (" and positive." if positive else "."))
    return tuple(float(x) for x in data) if vector else float(data)


def _index(*, value):
    index = _si(
        value=value,
        unit="dimensionless",
        name="index",
        scalar=True,
    )
    if not np.isfinite(index) or index <= 0:
        raise ValueError("index must be finite and positive.")
    return float(index)


def _axis(*, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2 (x, y, z).")
    return int(value)


@dataclass(frozen=True, kw_only=True)
class _Transformable:
    """Return transformed immutable shapes in global SI coordinates."""

    centre: tuple = (0.0, 0.0, 0.0)
    index: Quantity | float | None = None
    material: Material | None = None

    def translated(self, *, offset):
        """Return a translated shape; offset uses metres or length quantities."""
        displacement = _length(value=offset, name="offset", vector=True)
        centre = tuple(np.asarray(self.centre) + displacement)
        return replace(self, centre=centre, index=None)

    def rotated(self, *, rotation, about=None):
        """Return a rotated shape around its centre or an explicit global point.

        Rotation is active: R maps local column coordinates to the global
        frame. Successive world rotations compose as R_new @ R_existing.
        Membership uses local row coordinates (positions-centre) @ R.
        """
        matrix = Rotation._matrix(rotation=rotation)
        centre = np.asarray(self.centre)
        if about is not None:
            point = np.asarray(_length(value=about, name="about", vector=True))
            centre = point + matrix @ (centre - point)
        settings: dict[str, Any] = {"centre": tuple(centre), "index": None}
        if hasattr(self, "rotation"):
            settings["rotation"] = tuple(tuple(row) for row in matrix @ np.asarray(self.rotation))
        return replace(self, **settings)


@dataclass(frozen=True, kw_only=True)
class Layer(_Transformable):
    """A slab with absolute index inside ``lower <= local[axis] < upper``.

    Bounds are SI lengths in the slab local frame, offset by centre and
    oriented by rotation. The default axis is
    z (2); x and y are 0 and 1. The half-open interval gives adjacent slabs an
    unambiguous interface. The slab spans the voxel box in transverse axes;
    it represents a finite sample, not an infinite planar background.
    """

    lower: Quantity | float
    upper: Quantity | float
    index: Quantity | float | None = None
    material: Material | None = None
    axis: int = 2
    centre: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self):
        object.__setattr__(
            self,
            "lower",
            _length(
                value=self.lower,
                name="lower",
            ),
        )
        object.__setattr__(
            self,
            "upper",
            _length(
                value=self.upper,
                name="upper",
            ),
        )
        material = Material._resolve(material=self.material, index=self.index)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "index", material.index)
        object.__setattr__(self, "axis", _axis(value=self.axis))
        object.__setattr__(self, "centre", _length(value=self.centre, name="centre", vector=True))
        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))
        if self.lower >= self.upper:
            raise ValueError("lower must be smaller than upper.")

    def mask(self, *, positions):
        """Return membership at voxel positions in metres, shape (..., 3)."""
        coordinate = ((positions - self.centre) @ np.asarray(self.rotation))[..., self.axis]
        return (coordinate >= self.lower) & (coordinate < self.upper)


@dataclass(frozen=True, kw_only=True)
class Sphere(_Transformable):
    """Set absolute index where ``sum((r-centre)**2) <= radius**2``.

    Radius is positive; centre is a three-component SI length or quantity.
    The closed surface is sampled at voxel centres, giving a staircase boundary.
    """

    radius: Quantity | float
    index: Quantity | float | None = None
    material: Material | None = None
    centre: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(
            self,
            "radius",
            _length(
                value=self.radius,
                name="radius",
                positive=True,
            ),
        )
        material = Material._resolve(material=self.material, index=self.index)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "index", material.index)
        object.__setattr__(
            self,
            "centre",
            _length(
                value=self.centre,
                name="centre",
                vector=True,
            ),
        )

    def mask(self, *, positions):
        """Return membership at voxel positions in metres, shape (..., 3)."""
        return np.sum((positions - self.centre) ** 2, axis=-1) <= self.radius**2


@dataclass(frozen=True, kw_only=True)
class Ellipsoid(_Transformable):
    """Set absolute index where ``sum((local/radii)**2) <= 1``.

    Three positive semi-axis radii align with local x, y, z. All lengths use SI or
    compatible quantities. rotation maps local semi-axes into global coordinates.
    """

    radii: tuple
    index: Quantity | float | None = None
    material: Material | None = None
    centre: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self):
        object.__setattr__(
            self,
            "radii",
            _length(
                value=self.radii,
                name="radii",
                positive=True,
                vector=True,
            ),
        )
        material = Material._resolve(material=self.material, index=self.index)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "index", material.index)
        object.__setattr__(
            self,
            "centre",
            _length(
                value=self.centre,
                name="centre",
                vector=True,
            ),
        )

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

    def mask(self, *, positions):
        """Return membership at voxel positions in metres, shape (..., 3)."""
        local = (positions - self.centre) @ np.asarray(self.rotation)
        return np.sum((local / self.radii) ** 2, axis=-1) <= 1


@dataclass(frozen=True, kw_only=True)
class Box(_Transformable):
    """Set absolute index where ``abs(local) <= size/2`` in every axis.

    Size contains three positive full side lengths along the local axes.
    All lengths use SI or compatible quantities; the boundary is closed.
    """

    size: tuple
    index: Quantity | float | None = None
    material: Material | None = None
    centre: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self):
        object.__setattr__(
            self,
            "size",
            _length(
                value=self.size,
                name="size",
                positive=True,
                vector=True,
            ),
        )
        material = Material._resolve(material=self.material, index=self.index)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "index", material.index)
        object.__setattr__(
            self,
            "centre",
            _length(
                value=self.centre,
                name="centre",
                vector=True,
            ),
        )

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

    def mask(self, *, positions):
        """Return membership at voxel positions in metres, shape (..., 3)."""
        local = (positions - self.centre) @ np.asarray(self.rotation)
        return np.all(np.abs(local) <= np.asarray(self.size) / 2, axis=-1)


@dataclass(frozen=True, kw_only=True)
class Cylinder(_Transformable):
    """A finite circular cylinder with absolute index and closed boundary.

    Membership requires ``abs(local[axis]) <= height/2`` and squared
    transverse distance <= radius**2. Radius and full height are positive SI
    lengths or quantities; axis is 0, 1, or 2, default z. Centre has three lengths.
    """

    radius: Quantity | float
    height: Quantity | float
    index: Quantity | float | None = None
    material: Material | None = None
    centre: tuple = (0.0, 0.0, 0.0)
    axis: int = 2
    rotation: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self):
        object.__setattr__(
            self,
            "radius",
            _length(
                value=self.radius,
                name="radius",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "height",
            _length(
                value=self.height,
                name="height",
                positive=True,
            ),
        )
        material = Material._resolve(material=self.material, index=self.index)
        object.__setattr__(self, "material", material)
        object.__setattr__(self, "index", material.index)
        object.__setattr__(
            self,
            "centre",
            _length(
                value=self.centre,
                name="centre",
                vector=True,
            ),
        )
        object.__setattr__(self, "axis", _axis(value=self.axis))

        object.__setattr__(self, "rotation", tuple(tuple(row) for row in Rotation._matrix(rotation=self.rotation)))

    def mask(self, *, positions):
        """Return membership at voxel positions in metres, shape (..., 3)."""
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
    background_index : float or Quantity
        Positive uniform background index, default 1.33. It fills uncovered
        voxels and extends outside the finite voxel box.

    overlap : {"replace", "preserve", "error"}
        Later regions win, earlier regions win, or reject shared voxel centres.
    warn_on_clipping : bool
        Warn when geometry extends beyond the voxel box; default False.

    Notes
    -----
    Configure this mutable builder with ``add_background`` and ``add_structures``.
    Medium remains abstract; random statistics and geometry shapes remain frozen.
    Calls update the builder and return None. Previously generated volumes do
    not change. Constructor regions are still accepted for existing callers.
    Voxelization returns ``delta_index(r) = n(r) - background_index``. The
    numerical solver uses ``epsilon_r = n0**2 + 2*n0*delta_index``, omitting
    ``delta_index**2`` even at higher Born orders. Strong index differences
    can invalidate this constitutive approximation or Born iteration.
    Regions crossing the box are clipped. Layers therefore have finite lateral
    extent; this is not a transfer-matrix or layered-background Green solver.
    """

    regions: tuple = ()
    background_index: Quantity | float = 1.33
    overlap: str = "replace"
    warn_on_clipping: bool = False
    _background: RandomMedium | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        regions = tuple(self.regions)
        if any(not isinstance(region, (Layer, Sphere, Ellipsoid, Box, Cylinder)) for region in regions):
            raise TypeError("regions must contain Layer, Sphere, Ellipsoid, Box, or Cylinder objects.")
        if self.overlap not in ("replace", "preserve", "error"):
            raise ValueError("overlap must be replace, preserve, or error.")
        if not isinstance(self.warn_on_clipping, bool):
            raise ValueError("warn_on_clipping must be a bool.")
        self.regions = regions
        self.background_index = _index(value=self.background_index)

    def add_background(self, *, index=None, medium=None, material=None):
        """Set the background in place, retaining all existing structures.

        Supply exactly one of ``index`` (a positive uniform refractive index)
        or ``medium`` (RandomMedium statistics). A random background is sampled
        with to_volume(seed=...) and fills only voxels outside structures.
        The exterior scattering background remains uniform at background_index.
        Replacing a background changes the reference n0, not structures'
        absolute refractive indices. Validation precedes any state change.
        Returns None. Random statistics are copied when added.
        """
        if material is not None:
            resolved = Material._resolve(material=material, index=index)
            index = resolved.index
        if (index is None) == (medium is None):
            raise ValueError("Supply exactly one of index or medium.")
        if medium is not None:
            if not isinstance(medium, RandomMedium):
                raise TypeError("medium must be a RandomMedium.")
            background = replace(medium)
            background_index = background.background_index
        else:
            background = None
            background_index = _index(value=index)
        self.background_index = background_index
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

        Region type, absolute index, axis where applicable, and dimensions are
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
                if isinstance(region, Layer) and name == "centre" and np.array_equal(value, [0, 0, 0]):
                    continue
                key = f"{name}_m" if name in length_fields else name
                description[key] = list(value) if isinstance(value, tuple) else value
            regions.append(description)
        metadata: dict[str, object] = {**super().metadata, "regions": regions}
        if self.overlap != "replace":
            metadata["overlap"] = self.overlap
        if self._background is not None:
            metadata["background"] = self._background.metadata
        return metadata

    def to_volume(self, *, grid=None, shape=None, spacing=None, seed=0):
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
            contrast = background.delta_index.copy()
        occupied = np.zeros(grid.shape, dtype=bool)
        for index, region in enumerate(self.regions):
            if self.warn_on_clipping and self._is_clipped(region=region, grid=grid):
                warnings.warn(
                    f"Region {index} ({type(region).__name__}) crosses the voxel box and is clipped.",
                    UserWarning,
                    stacklevel=2,
                )
            mask = region.mask(positions=positions)
            if self.overlap == "error" and np.any(mask & occupied):
                raise ValueError(f"Region {index} overlaps existing structures at voxel centres.")
            selected = mask & ~occupied if self.overlap == "preserve" else mask
            contrast[selected] = region.index - self.background_index
            occupied |= mask
        return Volume(
            delta_index=contrast,
            grid=grid,
            background_index=self.background_index,
        )

    @staticmethod
    def _is_clipped(*, region, grid):
        half_box = np.asarray(grid.shape) * grid.spacing / 2
        centre = np.asarray(region.centre)
        if isinstance(region, Sphere):
            half_extent = np.full(3, region.radius)
        else:
            matrix = np.asarray(region.rotation)
            if isinstance(region, Layer):
                normal = matrix[:, region.axis]
                reach = np.sum(np.abs(normal) * half_box)
                shift = np.dot(centre, normal)
                return region.lower + shift < -reach or region.upper + shift > reach
            if isinstance(region, Box):
                half_extent = np.abs(matrix) @ (np.asarray(region.size) / 2)
            elif isinstance(region, Ellipsoid):
                half_extent = np.sqrt(matrix**2 @ np.asarray(region.radii) ** 2)
            else:
                normal = matrix[:, region.axis]
                half_extent = np.abs(normal) * region.height / 2 + region.radius * np.sqrt(np.maximum(0, 1 - normal**2))
        return bool(np.any(np.abs(centre) + half_extent > half_box * (1 + 1e-12)))
