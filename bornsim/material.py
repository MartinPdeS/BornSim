"""Immutable, reusable homogeneous optical materials."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from .units import NoUnit, Scalar, validate_annotation


@dataclass(frozen=True, kw_only=True)
class Material:
    """A nondispersive real refractive index shared by any number of shapes.

    ``refractive_index`` is an absolute positive dimensionless refractive index, not a
    contrast. Supply plain numbers; quantities are rejected. The nonnegative
    ``extinction_coefficient`` is kappa in n+i*kappa, using exp(-i omega t).
    Voxelization retains absorption within the finite sample; the exterior
    host remains lossless. Wavelength dispersion is not implemented.
    """

    refractive_index: float
    extinction_coefficient: float = 0.0

    def __post_init__(self) -> None:
        """Validate both dimensionless optical constants.

        Raises
        ------
        ValueError
            If either value is non-finite, negative where disallowed, or has units.
        """
        validate_annotation(
            self.refractive_index,
            annotation=NoUnit[Scalar],
            name="refractive_index",
        )

        refractive_index = float(self.refractive_index)

        if not np.isfinite(refractive_index) or refractive_index <= 0:
            raise ValueError("refractive_index must be finite and positive.")

        object.__setattr__(self, "refractive_index", refractive_index)

        validate_annotation(
            self.extinction_coefficient,
            annotation=NoUnit[Scalar],
            name="extinction_coefficient",
        )

        extinction = float(self.extinction_coefficient)
        if not np.isfinite(extinction) or extinction < 0:
            raise ValueError("extinction_coefficient must be finite and nonnegative.")
        object.__setattr__(self, "extinction_coefficient", extinction)

    @property
    def metadata(self) -> dict[str, float]:
        """Return a fresh JSON-compatible material description.

        Returns
        -------
        dict of str to float
            Refractive index and, when nonzero, extinction coefficient.
        """

        return {
            "refractive_index": self.refractive_index,
            **({"extinction_coefficient": self.extinction_coefficient} if self.extinction_coefficient else {}),
        }

    @classmethod
    def _resolve(cls, *, material: "Material | None" = None, refractive_index: Any = None) -> "Material":
        """Resolve a material object or plain refractive-index value.

        Parameters
        ----------
        material : Material or None
            Existing optical material.
        refractive_index : scalar or None
            Plain, positive dimensionless refractive index used to construct a
            material when ``material`` is omitted.

        Returns
        -------
        Material
            The supplied material or a newly validated instance.

        Raises
        ------
        TypeError
            If ``material`` is not a Material.
        ValueError
            If both or neither input is supplied.
        """
        if material is not None:
            if not isinstance(material, cls):
                raise TypeError("material must be a Material.")

            if refractive_index is not None:
                raise ValueError("Supply material or refractive_index, not both.")

            return material

        if refractive_index is None:
            raise ValueError("Supply material or refractive_index.")

        return cls(refractive_index=refractive_index)
