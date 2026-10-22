"""Finite oriented ellipsoidal particles."""

from dataclasses import dataclass
from ..base import Medium
from ...units import Quantity


@dataclass(frozen=True, kw_only=True)
class EllipsoidMedium(Medium):
    """A finite ellipsoidal particle with explicit semi-axes and orientation.

    Parameters
    ----------
    background_refractive_index, particle_refractive_index : float
        Absolute real refractive indices.
    radii : Quantity
        Three positive semi-axis lengths.
    centre : Quantity
        Three-component particle position.
    rotation : Rotation or array
        Rotation mapping local semi-axes into global coordinates.
    """

    background_refractive_index: float
    particle_refractive_index: float
    radii: Quantity
    centre: Quantity
    rotation: object = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def __post_init__(self):
        self._composite()

    def _composite(self):
        from ...geometry import Ellipsoid
        from ..structured.composite import StructuredMedium

        particle = Ellipsoid(
            radii=self.radii,
            centre=self.centre,
            rotation=self.rotation,
            refractive_index=self.particle_refractive_index,
        )

        medium = StructuredMedium()

        medium.add_background(refractive_index=self.background_refractive_index)

        medium.add_structures(particle)

        return medium

    @property
    def metadata(self):
        """Return particle geometry in SI units."""
        return {**self._composite().metadata, "model": "ellipsoid"}

    def to_volume(self, *, grid, seed=0):
        """Voxelize the oriented particle, clipping it to the supplied grid."""
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
