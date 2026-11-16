"""Numerical random-medium statistics, independent of analytical solutions."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
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
        """Whether independent generation seeds represent random realizations.

        Returns
        -------
        bool
            False for deterministic media; random media override this property.
        """

        return False

    @property
    def metadata(self) -> dict[str, object]:
        """Return a fresh JSON-compatible description with numeric SI values.

        Concrete media extend this dictionary with their own statistics or
        geometry. Generation seeds and grids belong to individual volumes.

        Returns
        -------
        dict of str to object
            Fresh JSON-compatible metadata for this medium.
        """

        return {"background_refractive_index": self.background_refractive_index}

    def add_background(
        self,
        *,
        refractive_index: float | None = None,
        medium: Any = None,
        material: Any = None,
    ) -> None:
        """Configure a composable medium's background in place.

        StructuredMedium accepts exactly one of a positive uniform ``refractive_index``
        or a RandomFieldMedium supplied as ``medium``. Statistical media describe
        homogeneous distributions; use StructuredMedium to compose material.

        Parameters
        ----------
        refractive_index : float or None, optional
            Plain positive dimensionless background refractive index.
        medium : RandomFieldMedium or None, optional
            Statistical medium that defines the random background.
        material : Material or None, optional
            Material supplying refractive index and extinction coefficient.

        Returns
        -------
        None
            The abstract base does not configure state.

        Raises
        ------
        TypeError
            This abstract interface cannot configure a background.
        """

        raise TypeError("Use StructuredMedium to compose backgrounds and structures.")

    def add_structures(self, *structures: Any) -> None:
        """Append positional material shapes to a composable medium in place.

        StructuredMedium supports Layer, Sphere, Ellipsoid, Box, and Cylinder.
        Later structures replace earlier material wherever their masks overlap.

        Parameters
        ----------
        *structures
            Positional geometry objects to append.

        Returns
        -------
        None
            The abstract base does not append structures.

        Raises
        ------
        TypeError
            This abstract interface cannot compose structures.
        """

        raise TypeError("Use StructuredMedium to compose backgrounds and structures.")

    @abstractmethod
    def to_volume(self, *, grid: Grid, seed: int = 0) -> "Volume":
        """Voxelize or sample this medium on an explicit Grid.

        Parameters
        ----------
        grid : Grid
            Explicit spatial discretization.
        seed : int, default=0
            Seed controlling random realization generation.

        Returns
        -------
        Volume
            Sampled refractive-index contrast and its metadata.

        Raises
        ------
        NotImplementedError
            This abstract method must be implemented by concrete media.
        """

        raise NotImplementedError
