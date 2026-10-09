"""Vector first-order Born model, with SI lengths and unpolarized light.

The dielectric covariance uses the weak-fluctuation linearization
C_epsilon(r) = 4 n_background**2 C_n(r). The spectral convention is
Phi(q) = integral C(r) exp(-i q.r) d^3 r (no Fourier prefactor).
"""

from dataclasses import dataclass
from typing import TypedDict
import numpy as np
from .units import Quantity, validate_units, ureg
from .media import RandomMedium


class OpticalProperties(TypedDict):
    """Analytical transport coefficients and dimensionless anisotropy."""

    mu_s: Quantity
    g: float | None
    mu_s_prime: Quantity


@dataclass(frozen=True, kw_only=True)
class AnalyticalMedium(RandomMedium):
    """Define random-medium statistics for the existing analytical first-order model.

    Parameters
    ----------
    background_refractive_index : float
        Positive, finite background refractive index. Plain numbers are required; quantities are rejected.
    refractive_index_std : float
        Nonnegative, finite standard deviation of refractive index fluctuations, not their
        variance. Dimensionless;
    correlation_length : Quantity
        Positive, finite covariance length. Explicit length units are required;
        the supplied units are preserved.
    correlation : {'gaussian', 'exponential'}
        Spatial covariance model.

    Attributes
    ----------
    background_refractive_index, refractive_index_std : float
        Dimensionless statistics stored as numeric SI values.
    correlation_length : Quantity
        Covariance length with its supplied units.
    correlation : str
        Selected spatial covariance model.

    Raises
    ------
    ValueError
        If a statistic is nonfinite, outside its allowed range, nonscalar,
        has incompatible units, or the covariance model is unsupported.

    Notes
    -----
    This concrete subclass replaces the former directly instantiated Medium.
    It retains the existing first-order Gaussian and exponential formulas,
    and inherits numerical voxel generation from RandomMedium.
    The refractive index covariance is ``refractive_index_std**2 * exp(-r**2 / (2 * ell**2))``
    for Gaussian correlation and ``refractive_index_std**2 * exp(-r / ell)`` for
    exponential correlation, where ``ell = correlation_length``. Equal length
    parameters therefore do not imply identical correlation profiles.
    Dielectric contrast is linearized as ``delta_epsilon = 2 * n0 * delta_n``.
    The Gaussian probability distribution used by :func:`random_volume` is
    independent of this choice of spatial covariance.

    Examples
    --------
    >>> from bornsim import AnalyticalMedium
    >>> from bornsim.units import ureg
    ...
    ...
    >>> medium = AnalyticalMedium(
    ...     correlation_length=100 * ureg.nanometer,
    ...     background_refractive_index=1.33,
    ...     refractive_index_std=0.01,
    ...     correlation="gaussian",
    ... )
    >>> medium.correlation
    'gaussian'
    """

    background_refractive_index: float
    refractive_index_std: float
    correlation_length: Quantity
    correlation: str

    def __post_init__(self):
        super().__post_init__()

        if self.correlation not in ("gaussian", "exponential"):
            raise ValueError("AnalyticalMedium correlation must be gaussian or exponential.")


def angular_scattering(*, medium: AnalyticalMedium, wavelength: Quantity, theta: Quantity) -> Quantity:
    """Evaluate the unpolarized first-order differential scattering coefficient.

    Parameters
    ----------
    medium : AnalyticalMedium
        Background refractive index and isotropic fluctuation statistics.
    wavelength : Quantity
        Positive, finite vacuum wavelength. Explicit length units are required.
    theta : array_like or Quantity
        Finite polar scattering angles in [0, pi]. Explicit angular units are required;
        angular quantities may use degrees or radians. Scalars are accepted.

    Returns
    -------
    differential : Quantity
        Differential scattering coefficient in m^-1 sr^-1, with the same shape
        as ``theta``. The result retains physical units.

    Raises
    ------
    ValueError
        If the wavelength is nonpositive, nonfinite, nonscalar, or has
        incompatible units, or angles are invalid or have incompatible units.

    See Also
    --------
    optical_properties : Integrate the coefficient and its angular moments.
    bornsim.solver.Solver.solve : Return analytical curves as unitful results.

    Notes
    -----
    With ``k0 = 2*pi/wavelength`` and ``q = 2*n0*k0*sin(theta/2)``, the
    coefficient is ``k0**4 * Phi_epsilon(q) * (1 + cos(theta)**2) / (32*pi**2)``.
    The spectral convention is the three-dimensional Fourier transform without
    an additional normalization prefactor. The dielectric covariance follows
    ``C_epsilon = 4 * n0**2 * C_n`` under weak-fluctuation linearization.

    For nonzero total scattering, the phase function per steradian is this
    coefficient divided by ``optical_properties(...)["mu_s"]``. Its integral
    over solid angle is one; it is not a probability density per polar angle.

    Examples
    --------
    >>> from bornsim import AnalyticalMedium, angular_scattering
    >>> from bornsim.units import ureg
    ...
    ...
    >>> curve = angular_scattering(
    ...     medium=AnalyticalMedium(
    ...         background_refractive_index=1.33,
    ...         refractive_index_std=0.01,
    ...         correlation_length=100e-9 * ureg.meter,
    ...         correlation="gaussian",
    ...     ),
    ...     wavelength=633 * ureg.nanometer,
    ...     theta=[0, 90, 180] * ureg.degree,
    ... )
    >>> curve.shape
    (3,)
    """

    if not isinstance(medium, AnalyticalMedium):
        raise TypeError("Analytical scattering requires an AnalyticalMedium.")

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

    k0 = 2 * np.pi / wavelength

    q = 2 * k0 * medium.background_refractive_index * np.sin(theta / 2)

    ell = medium.correlation_length

    variance = (2 * medium.background_refractive_index * medium.refractive_index_std) ** 2

    if medium.correlation == "gaussian":
        spectrum = variance * (2 * np.pi) ** 1.5 * ell**3 * np.exp(-0.5 * (q * ell) ** 2)
    else:
        spectrum = variance * 8 * np.pi * ell**3 / (1 + (q * ell) ** 2) ** 2

    return k0**4 / (16 * np.pi**2) * spectrum * (1 + np.cos(theta) ** 2) / 2 / ureg.steradian


def optical_properties(
    *, medium: AnalyticalMedium, wavelength: Quantity, quadrature_order: int = 256
) -> OpticalProperties:
    """Integrate analytical scattering and its moments over solid angle.

    Parameters
    ----------
    medium : AnalyticalMedium
        Background refractive index and isotropic fluctuation statistics.
    wavelength : Quantity
        Positive, finite vacuum wavelength; explicit length units are required.
    quadrature_order : int, optional
        Number of Gauss-Legendre nodes in the cosine of the scattering angle.
        Must be at least 16; default is 256.

    Returns
    -------
    properties : dict
        Unit-bearing coefficients under the following keys:

        * ``mu_s`` : total scattering coefficient in m^-1.
        * ``g`` : dimensionless mean cosine of the scattering angle, or
          ``None`` if ``mu_s`` is zero.
        * ``mu_s_prime`` : reduced scattering coefficient in m^-1, equal to
          ``mu_s * (1 - g)`` when ``g`` is defined.

    Raises
    ------
    ValueError
        If the quadrature order, wavelength, or wavelength units are invalid.

    See Also
    --------
    angular_scattering : Evaluate the integrand at arbitrary polar angles.

    Notes
    -----
    Axial symmetry gives ``mu_s = 2*pi*integral(beta(theta)*sin(theta), theta)``
    over [0, pi], where ``beta`` is the differential coefficient. The
    anisotropy uses the same integral with an additional ``cos(theta)`` factor.
    These coefficients describe the analytical infinite-medium first-order
    model, unlike the finite-sample coefficients from numerical ensembles.
    Increase ``quadrature_order`` to check strongly forward-peaked cases.

    Examples
    --------
    >>> from bornsim import AnalyticalMedium, optical_properties
    ...
    >>> from bornsim.units import ureg
    ...
    >>> properties = optical_properties(
    ...     medium=AnalyticalMedium(
    ...         refractive_index_std=0,
    ...         background_refractive_index=1.33,
    ...         correlation_length=100e-9 * ureg.meter,
    ...         correlation="gaussian",
    ...     ),
    ...     wavelength=633 * ureg.nanometer,
    ... )
    >>> float(properties["mu_s"].to("1 / meter").magnitude)
    0.0
    """

    if not isinstance(quadrature_order, int) or quadrature_order < 16:
        raise ValueError("quadrature_order must be an integer of at least 16.")

    cosine, weights = np.polynomial.legendre.leggauss(quadrature_order)

    differential = angular_scattering(
        medium=medium,
        wavelength=wavelength,
        theta=np.arccos(cosine) * ureg.radian,
    )

    mu_s = 2 * np.pi * ureg.steradian * np.sum(weights * differential)

    mu_s_prime = 2 * np.pi * ureg.steradian * np.sum(weights * (1 - cosine) * differential)

    g = float(2 * np.pi * ureg.steradian * np.sum(weights * cosine * differential) / mu_s) if mu_s else None

    return {"mu_s": mu_s, "g": g, "mu_s_prime": mu_s_prime}
