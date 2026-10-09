"""Numerical random-medium statistics, independent of analytical solutions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from ..grid import Grid

if TYPE_CHECKING:
    from ..volume import Volume


class Medium(ABC):
    """Abstract interface for media that can be sampled into a finite volume.

    Instantiate :class:`bornsim.GaussianMedium` or :class:`bornsim.StructuredMedium`,
    not this base class. Concrete media specify a uniform ``background_refractive_index``
    and implement :meth:`to_volume` using cubic voxels, SI spacing, and centred
    coordinates. The resulting field stores ``delta_refractive_index = n(r) - n0``;
    propagation retains ``epsilon_r = n0**2 + 2*n0*delta_refractive_index``.
    """

    background_refractive_index: float | None

    @property
    def is_random(self) -> bool:
        """Whether independent generation seeds represent random realizations."""

        return False

    @property
    def metadata(self) -> dict[str, object]:
        """Return a fresh JSON-compatible description with numeric SI values.

        Concrete media extend this dictionary with their own statistics or
        geometry. Generation seeds and grids belong to individual volumes.
        """

        return {"background_refractive_index": self.background_refractive_index}

    def add_background(self, *, refractive_index=None, medium=None, material=None):
        """Configure a composable medium's background in place.

        StructuredMedium accepts exactly one of a positive uniform ``refractive_index``
        or a RandomFieldMedium supplied as ``medium``. Statistical media describe
        homogeneous distributions; use StructuredMedium to compose material.
        """

        raise TypeError("Use StructuredMedium to compose backgrounds and structures.")

    def add_structures(self, *structures):
        """Append positional material shapes to a composable medium in place.

        StructuredMedium supports Layer, Sphere, Ellipsoid, Box, and Cylinder.
        Later structures replace earlier material wherever their masks overlap.
        """

        raise TypeError("Use StructuredMedium to compose backgrounds and structures.")

    @abstractmethod
    def to_volume(self, *, grid: Grid, seed: int = 0) -> "Volume":
        """Voxelize or sample this medium on an explicit Grid."""

        raise NotImplementedError
