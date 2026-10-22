"""Finite concentric core–shell spheres."""

from dataclasses import dataclass
from ..base import Medium
from ...geometry import Sphere
from ...units import Quantity


@dataclass(frozen=True, kw_only=True)
class CoreShellMedium(Medium):
    """A concentric spherical core and shell in a uniform background.

    Parameters
    ----------
    background_refractive_index, core_refractive_index, shell_refractive_index : float
        Absolute real refractive indices.
    core_radius, outer_radius : Quantity
        Positive length scalars with core_radius strictly below outer_radius.
    centre : Quantity
        Three-component particle position. Interfaces use voxel-centre sampling.
    """

    background_refractive_index: float
    core_refractive_index: float
    shell_refractive_index: float
    core_radius: Quantity
    outer_radius: Quantity
    centre: Quantity

    def __post_init__(self):
        self._composite()

        if self.core_radius >= self.outer_radius:
            raise ValueError("core_radius must be smaller than outer_radius.")

    def _composite(self):
        from ..structured.composite import StructuredMedium

        shell = Sphere(
            radius=self.outer_radius,
            refractive_index=self.shell_refractive_index,
            centre=self.centre,
        )

        core = Sphere(
            radius=self.core_radius,
            refractive_index=self.core_refractive_index,
            centre=self.centre,
        )

        medium = StructuredMedium()

        medium.add_background(refractive_index=self.background_refractive_index)

        medium.add_structures(shell, core)

        return medium

    @property
    def metadata(self):
        """Return ordered shell and core geometry in SI units."""
        return {**self._composite().metadata, "model": "core_shell"}

    def to_volume(self, *, grid, seed=0):
        """Sample the shell first and replace its interior with the core."""
        from ...volume import Volume

        medium = self._composite()

        volume = medium.to_volume(grid=grid, seed=seed)

        return Volume(
            delta_refractive_index=volume.delta_refractive_index,
            background_refractive_index=self.background_refractive_index,
            grid=grid,
            medium=self,
            seed=seed,
        )
