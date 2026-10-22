"""Finite lattice arrays of existing material shapes."""

from dataclasses import dataclass, replace
from itertools import product
import numpy as np
from .composite import StructuredMedium
from ..base import Medium
from ...units import Quantity, validate_units, _integer


@dataclass(frozen=True, kw_only=True)
class PeriodicMedium(Medium):
    """Repeat a motif on a centred finite lattice.

    Parameters
    ----------
    background_refractive_index : float
        Uniform exterior refractive index.
    motif : tuple of geometry objects
        Shapes relative to each lattice site; ordering defines overlap precedence.
    lattice_vectors : Quantity
        Three independent lattice vectors as rows of a (3, 3) length array.
    repetitions : tuple of int
        Positive finite repetition counts along each lattice vector.
    overlap : str
        StructuredMedium overlap policy. Geometry is clipped to the voxel box.

    Notes
    -----
    This generates finite samples, without periodic solver boundary conditions.
    All copies retain their motif orientations. Even repetition counts place
    the array centre halfway between lattice sites.
    """

    background_refractive_index: float
    motif: tuple
    lattice_vectors: Quantity
    repetitions: tuple[int, int, int]
    overlap: str = "replace"

    def __post_init__(self):
        validate_units(self.lattice_vectors, unit="meter", name="lattice_vectors")

        vectors = np.asarray(self.lattice_vectors.to("meter").magnitude)

        if vectors.shape != (3, 3) or not np.all(np.isfinite(vectors)) or np.linalg.matrix_rank(vectors) != 3:
            raise ValueError("lattice_vectors must contain three finite independent length vectors.")

        if len(self.repetitions) != 3:
            raise ValueError("repetitions must contain three counts.")

        counts = tuple(_integer(value=n, name="repetitions", low=1, high=10000) for n in self.repetitions)

        object.__setattr__(self, "repetitions", counts)

        object.__setattr__(self, "motif", tuple(self.motif))

        self._motif_medium()

    def _motif_medium(self):
        return StructuredMedium(
            background_refractive_index=self.background_refractive_index,
            regions=self.motif,
            overlap=self.overlap,
        )

    @property
    def metadata(self):
        """Return motif and finite lattice settings in SI units."""
        return {
            **self._motif_medium().metadata,
            "model": "periodic",
            "lattice_vectors_m": self.lattice_vectors.to("meter").magnitude.tolist(),
            "repetitions": list(self.repetitions),
        }

    def to_volume(self, *, grid, seed=0):
        """Voxelize centred copies, clipping geometry to the supplied grid."""
        from ...volume import Volume

        regions = []

        for site in product(*(range(n) for n in self.repetitions)):
            coordinates = np.asarray(site) - (np.asarray(self.repetitions) - 1) / 2

            offset = coordinates @ self.lattice_vectors

            for shape in self.motif:
                regions.append(shape.translated(offset=offset))

        composed = self._motif_medium()

        composed.regions = tuple(regions)

        volume = composed.to_volume(grid=grid, seed=seed)

        return Volume(
            delta_refractive_index=volume.delta_refractive_index,
            background_refractive_index=self.background_refractive_index,
            grid=grid,
            medium=replace(self),
            seed=seed,
        )
