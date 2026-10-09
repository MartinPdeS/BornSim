"""Incident illumination for BornSim scattering calculations."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, validate_units


@dataclass(frozen=True, kw_only=True)
class Source:
    """Define an unpolarized plane wave propagating along the positive z axis.

    Parameters
    ----------
    wavelength : Quantity
        Positive, finite vacuum wavelength. Explicit length units are required;
        the supplied units are preserved. A wavelength must be supplied.

    Attributes
    ----------
    wavelength : Quantity
        Scalar vacuum wavelength with its supplied length units.

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
    ...
    ...
    >>> source = Source(wavelength=633 * ureg.nanometer)
    >>> source.wavelength.check("[length]")
    True
    """

    wavelength: Quantity

    def __post_init__(self) -> None:
        validate_units(
            self.wavelength,
            unit="meter",
            name="wavelength",
            scalar=True,
        )

        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive.")
