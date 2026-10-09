"""Infinite-medium first-order references, used only to validate numerical calculations."""

from bornsim.medium.random_medium import RandomFieldMedium

import numpy as np
from bornsim.units import Quantity, validate_units, ureg


def infinite_medium_scattering(*, medium: RandomFieldMedium, wavelength: Quantity, theta: Quantity) -> Quantity:
    if not isinstance(medium, RandomFieldMedium):
        raise TypeError("Analytical scattering requires an RandomFieldMedium.")

    validate_units(
        wavelength,
        unit="meter",
        name="wavelength",
        scalar=True,
    )

    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be finite and positive.")

    validate_units(
        theta,
        unit="radian",
        name="theta",
    )

    if np.any(~np.isfinite(theta)) or np.any((theta < 0) | (theta > np.pi * ureg.radian)):
        raise ValueError("theta must be finite and between 0 and pi.")

    if medium.correlation not in ("gaussian", "exponential"):
        raise ValueError("Reference covariance must be gaussian or exponential.")

    k0 = 2 * np.pi / wavelength

    q = 2 * k0 * medium.background_refractive_index * np.sin(theta / 2)

    ell = medium.correlation_length

    variance = (2 * medium.background_refractive_index * medium.refractive_index_std) ** 2

    if medium.correlation == "gaussian":
        spectrum = variance * (2 * np.pi) ** 1.5 * ell**3 * np.exp(-0.5 * (q * ell) ** 2)
    else:
        spectrum = variance * 8 * np.pi * ell**3 / (1 + (q * ell) ** 2) ** 2

    return k0**4 / (16 * np.pi**2) * spectrum * (1 + np.cos(theta) ** 2) / 2 / ureg.steradian


def infinite_medium_properties(
    *, medium: RandomFieldMedium, wavelength: Quantity, quadrature_order: int = 256
) -> dict[str, Quantity | float | None]:
    if not isinstance(quadrature_order, int) or quadrature_order < 16:
        raise ValueError("quadrature_order must be an integer of at least 16.")

    cosine, weights = np.polynomial.legendre.leggauss(quadrature_order)

    differential = infinite_medium_scattering(
        medium=medium,
        wavelength=wavelength,
        theta=np.arccos(cosine) * ureg.radian,
    )

    mu_s = 2 * np.pi * ureg.steradian * np.sum(weights * differential)

    mu_s_prime = 2 * np.pi * ureg.steradian * np.sum(weights * (1 - cosine) * differential)

    g = float(2 * np.pi * ureg.steradian * np.sum(weights * cosine * differential) / mu_s) if mu_s else None

    return {"mu_s": mu_s, "g": g, "mu_s_prime": mu_s_prime}
