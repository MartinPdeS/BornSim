"""Finite cylindrical particles for rods and antenna-like scatterers."""

from dataclasses import dataclass
from ..base import Medium
from ...units import Quantity


@dataclass(frozen=True, kw_only=True)
class CylinderMedium(Medium):
    """A finite homogeneous cylinder in a uniform background.

    Parameters
    ----------
    background_refractive_index, particle_refractive_index : float
        Absolute real refractive indices.
    radius, height : Quantity
        Radius and full axial length of the cylinder.
    centre : Quantity
        Three-component particle position.
    axis : int
        Local cylinder axis: 0, 1, or 2 for x, y, or z.
    rotation : Rotation or array
        Rotation mapping local cylinder coordinates into global coordinates.

    Notes
    -----
    This is a finite dielectric rod. It can approximate an antenna-like
    scatterer when the refractive-index model is appropriate. It does not
    include conducting boundary conditions, complex permittivity, or currents.
    """

    background_refractive_index: float
    particle_refractive_index: float
    radius: Quantity
    height: Quantity
    centre: Quantity
    axis: int = 2
    rotation: object = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def _composite(self):
        from ...geometry import Cylinder
        from ..structured.composite import StructuredMedium

        particle = Cylinder(
            radius=self.radius,
            height=self.height,
            centre=self.centre,
            axis=self.axis,
            rotation=self.rotation,
            refractive_index=self.particle_refractive_index,
        )

        medium = StructuredMedium()

        medium.add_background(refractive_index=self.background_refractive_index)

        medium.add_structures(particle)

        return medium

    @property
    def metadata(self):
        """Return cylinder geometry in SI units."""
        return {**self._composite().metadata, "model": "cylinder"}

    def to_volume(self, *, grid, seed=0):
        """Voxelize the finite cylinder on the supplied grid."""
        from ...volume import Volume

        volume = self._composite().to_volume(grid=grid, seed=seed)

        return Volume(
            delta_refractive_index=volume.delta_refractive_index,
            background_refractive_index=self.background_refractive_index,
            grid=grid,
            medium=self,
            seed=seed,
        )
