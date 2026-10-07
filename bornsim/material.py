"""Immutable, reusable homogeneous optical materials."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, _si


@dataclass(frozen=True, kw_only=True)
class Material:
    """A nondispersive real refractive index shared by any number of shapes.

    index is an absolute positive dimensionless refractive index, not a
    contrast. Absorption and wavelength dispersion are not implemented.
    """

    index: Quantity | float

    def __post_init__(self):
        index = _si(value=self.index, unit="dimensionless", name="index", scalar=True)
        if not np.isfinite(index) or index <= 0:
            raise ValueError("index must be finite and positive.")
        object.__setattr__(self, "index", index)

    @property
    def metadata(self):
        """Fresh JSON-compatible material description."""
        return {"index": self.index}

    @classmethod
    def _resolve(cls, *, material=None, index=None):
        if material is not None:
            if not isinstance(material, cls):
                raise TypeError("material must be a Material.")
            if index is not None:
                raise ValueError("Supply material or index, not both.")
            return material
        if index is None:
            raise ValueError("Supply material or index.")
        return cls(index=index)
