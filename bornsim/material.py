"""Immutable, reusable homogeneous optical materials."""

from dataclasses import dataclass
import numpy as np
from .units import _refractive_index_values


@dataclass(frozen=True, kw_only=True)
class Material:
    """A nondispersive real refractive index shared by any number of shapes.

    ``refractive_index`` is an absolute positive dimensionless refractive index, not a
    contrast. Supply a plain number; quantities are rejected. Absorption and wavelength dispersion are not implemented.
    """

    refractive_index: float

    def __post_init__(self) -> None:
        refractive_index = _refractive_index_values(value=self.refractive_index, name="refractive_index", scalar=True)

        if not np.isfinite(refractive_index) or refractive_index <= 0:
            raise ValueError("refractive_index must be finite and positive.")

        object.__setattr__(self, "refractive_index", refractive_index)

    @property
    def metadata(self):
        """Fresh JSON-compatible material description."""

        return {"refractive_index": self.refractive_index}

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
