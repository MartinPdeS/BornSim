"""Composition of finite dielectric regions."""

from dataclasses import asdict, dataclass, field, replace
from copy import deepcopy
import warnings
import numpy as np
from ..base import Medium
from ..random.fields import RandomFieldMedium
from ...geometry import Layer, Sphere, Ellipsoid, Box, Cylinder, _refractive_index
from ...material import Material
from ...grid import Grid
from ...volume import Volume


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
