"""Immutable, reusable homogeneous optical materials."""

from dataclasses import dataclass
import numpy as np
from .units import _refractive_index_values


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
        refractive_index = _refractive_index_values(value=self.refractive_index, name="refractive_index", scalar=True)

        if not np.isfinite(refractive_index) or refractive_index <= 0:
            raise ValueError("refractive_index must be finite and positive.")

        object.__setattr__(self, "refractive_index", refractive_index)

        extinction = _refractive_index_values(
            value=self.extinction_coefficient, name="extinction_coefficient", scalar=True
        )
        if not np.isfinite(extinction) or extinction < 0:
            raise ValueError("extinction_coefficient must be finite and nonnegative.")
        object.__setattr__(self, "extinction_coefficient", extinction)

    @property
    def metadata(self):
        """Fresh JSON-compatible material description."""

        return {
            "refractive_index": self.refractive_index,
            **({"extinction_coefficient": self.extinction_coefficient} if self.extinction_coefficient else {}),
        }

    @classmethod
    def _resolve(cls, *, material=None, refractive_index=None):
        if material is not None:
            if not isinstance(material, cls):
                raise TypeError("material must be a Material.")

            if refractive_index is not None:
                raise ValueError("Supply material or refractive_index, not both.")

            return material

        if refractive_index is None:
            raise ValueError("Supply material or refractive_index.")

        return cls(refractive_index=refractive_index)
