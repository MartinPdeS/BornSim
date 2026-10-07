"""Incident illumination for BornSim scattering calculations."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, _quantity, _si


@dataclass(frozen=True, kw_only=True)
class Source:
    """Define an unpolarized plane wave propagating along the positive z axis.

    Parameters
    ----------
    wavelength : float or Quantity, optional
        Positive, finite vacuum wavelength. Bare numbers mean metres;
        length quantities are converted to metres. Default is 633 nm.

    Attributes
    ----------
    wavelength : Quantity
        Scalar vacuum wavelength stored in metres using TypedUnit's registry.

    Raises
    ------
    ValueError
        If the wavelength is nonpositive, nonfinite, nonscalar, or has units
        incompatible with length.

    Notes
    -----
    The transverse x and y incident polarizations have equal weights and are
    averaged incoherently. Arbitrary incidence directions and polarized
    illumination are not currently supported.

    Examples
    --------
    >>> from bornsim import Source
    >>> from bornsim.units import ureg
    >>> source = Source(wavelength=633 * ureg.nanometer)
    >>> source.wavelength.check("[length]")
    True
    """

    wavelength: Quantity | float = 633e-9

    def __post_init__(self):
        wavelength = _si(
            value=self.wavelength,
            unit="meter",
            name="wavelength",
            scalar=True,
        )
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be finite and positive.")
        object.__setattr__(
            self,
            "wavelength",
            _quantity(
                value=wavelength,
                unit="meter",
                name="wavelength",
            ),
        )
