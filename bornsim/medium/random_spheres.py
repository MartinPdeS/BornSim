"""Seeded collections of identical, nonoverlapping spheres."""

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from .base import Medium
from ..grid import Grid
from ..units import Quantity, validate_units, _refractive_index_values
from .._validation import _integer

if TYPE_CHECKING:
    from ..volume import Volume


@dataclass(frozen=True, kw_only=True)
class RandomSphereMedium(Medium):
    """Place a fixed number of identical spheres by sequential rejection sampling.

    Centres are proposed uniformly inside the grid box, inset by the radius so
    entire spheres remain inside. Proposals overlapping an accepted sphere are
    rejected. This defines a sequential placement distribution, rather than an
    equilibrium hard-sphere distribution. Dense configurations may fail to fit
    within ``max_placement_attempts``; failure raises instead of returning fewer
    spheres. Voxelization samples material at voxel centres, so an unresolved
    sphere can occupy no voxels.

    Both refractive indices are required plain numbers. Radius requires scalar
    length units. Each realization records its medium and generation seed.
    """

    background_refractive_index: float
    sphere_refractive_index: float
    radius: Quantity
    sphere_count: int
    max_placement_attempts: int = 10000

    def __post_init__(self) -> None:
        validate_units(
            self.radius,
            unit="meter",
            name="radius",
            scalar=True,
        )

        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be finite and positive.")

        for name in ("background_refractive_index", "sphere_refractive_index"):
            refractive_index = _refractive_index_values(
                value=getattr(self, name),
                name=name,
                scalar=True,
            )

            if not np.isfinite(refractive_index) or refractive_index <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        if 2 * self.sphere_refractive_index <= self.background_refractive_index:
            raise ValueError("sphere_refractive_index gives nonpositive linearized permittivity.")

        _integer(
            value=self.sphere_count,
            name="sphere_count",
            low=1,
            high=2**31 - 1,
        )

        _integer(
            value=self.max_placement_attempts,
            name="max_placement_attempts",
            low=self.sphere_count,
            high=2**31 - 1,
        )

    @property
    def is_random(self) -> bool:
        return True

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata,
            "model": "random_spheres",
            "sphere_refractive_index": self.sphere_refractive_index,
            "radius_m": float(self.radius.to("meter").magnitude),
            "sphere_count": self.sphere_count,
            "max_placement_attempts": self.max_placement_attempts,
            "placement": "sequential_nonoverlapping_contained",
        }

    def sphere_centers(self, *, grid: Grid, seed: int = 0) -> Quantity:
        """Return reproducible sphere centres with shape (sphere_count, 3)."""

        grid = Grid._resolve(grid=grid)

        seed = _integer(
            value=seed,
            name="seed",
            low=0,
            high=2**32 - 1,
        )

        available_half_width = np.asarray(grid.shape) * grid.spacing / 2 - self.radius

        if np.any(available_half_width < 0):
            raise ValueError("The sphere diameter must fit inside every grid dimension.")

        sphere_volume = self.sphere_count * (4 * np.pi / 3) * self.radius**3

        if sphere_volume > grid.volume:
            raise ValueError("Nonoverlapping spheres cannot fit: their total volume exceeds the grid volume.")

        generator = np.random.default_rng(seed)

        accepted_centers = np.zeros((self.sphere_count, 3)) * grid.spacing

        accepted_count = 0

        for _ in range(self.max_placement_attempts):
            proposed_center = generator.uniform(-1, 1, size=3) * available_half_width

            squared_center_distances = np.sum((accepted_centers[:accepted_count] - proposed_center) ** 2, axis=1)

            if np.any(squared_center_distances < (2 * self.radius) ** 2):
                continue

            accepted_centers[accepted_count] = proposed_center

            accepted_count += 1

            if accepted_count == self.sphere_count:
                return accepted_centers

        raise ValueError(
            f"Could only place {accepted_count} of {self.sphere_count} nonoverlapping spheres "
            f"after {self.max_placement_attempts} proposals; reduce sphere_count or radius, "
            "or increase max_placement_attempts."
        )

    def to_volume(self, *, grid: Grid, seed: int = 0) -> "Volume":
        """Sample the sphere material at voxel centres on an explicit grid."""

        from ..volume import Volume

        sphere_centers = self.sphere_centers(
            grid=grid,
            seed=seed,
        )

        positions = grid.positions

        delta_refractive_index = np.zeros(grid.shape)

        for sphere_center in sphere_centers:
            sphere_mask = np.sum((positions - sphere_center) ** 2, axis=-1) <= self.radius**2

            delta_refractive_index[sphere_mask] = self.sphere_refractive_index - self.background_refractive_index

        return Volume(
            delta_refractive_index=delta_refractive_index,
            background_refractive_index=self.background_refractive_index,
            grid=grid,
            medium=self,
            seed=seed,
        )
