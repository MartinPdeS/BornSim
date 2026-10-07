"""Outgoing vector Green tensor with open-boundary FFT convolution.

Cubic voxels use an equal-volume spherical self cell, including the
longitudinal contact term, with time convention exp(-i omega t).
"""

import numpy as np
from .units import _si


class GreenOperator:
    """Apply the outgoing background dyadic Green tensor by FFT convolution.

    Parameters
    ----------
    shape : tuple of int
        Three voxel-grid dimensions, normally supplied by a validated Volume.
    spacing : float or Quantity
        Cubic voxel spacing; bare values mean metres. Must be positive.
    wavelength : float or Quantity
        Positive, finite vacuum wavelength; bare values mean metres.
    background_index : float or Quantity
        Positive background index; must be dimensionless.

    Attributes
    ----------
    shape : tuple of int
        Unpadded sample dimensions.
    padded : tuple of int
        Dimensions doubled along each axis for open-boundary convolution.
    spectrum : numpy.ndarray
        Fourier-domain dyadic kernel, shape ``(*padded, 3, 3)``.

    Raises
    ------
    ValueError
        If the wavelength is invalid or a supplied quantity has incompatible units.

    Notes
    -----
    This low-level operator assumes a valid grid and background. It evaluates
    ``k0**2 * integral(G_background(r-r') * source(r'), dV')`` with time
    convention ``exp(-i*omega*t)``. Off-diagonal cells use midpoint quadrature.
    The self cell is an equal-volume sphere including the longitudinal contact
    term. Zero padding prevents periodic propagation across the sample.
    """

    def __init__(self, *, shape, spacing, wavelength, background_index):
        spacing = _si(
            value=spacing,
            unit="meter",
            name="spacing",
            scalar=True,
        )
        wavelength = _si(
            value=wavelength,
            unit="meter",
            name="wavelength",
            scalar=True,
        )
        background_index = _si(
            value=background_index,
            unit="dimensionless",
            name="background_index",
            scalar=True,
        )
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be finite and positive.")
        self.shape = tuple(shape)
        self.padded = tuple(2 * n for n in shape)
        k0 = 2 * np.pi / wavelength
        k = k0 * background_index
        axes = [np.where(np.arange(2 * n) < n, np.arange(2 * n), np.arange(2 * n) - 2 * n) * spacing for n in shape]
        lag = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        radius = np.linalg.norm(lag, axis=-1)
        safe = np.where(radius == 0, spacing, radius)
        unit = lag / safe[..., None]
        t = k * safe
        scalar = np.exp(1j * t) / (4 * np.pi * safe)
        isotropic = 1 + 1j / t - 1 / t**2
        longitudinal = -1 - 3j / t + 3 / t**2
        kernel = scalar[..., None, None] * (
            isotropic[..., None, None] * np.eye(3)
            + longitudinal[..., None, None] * unit[..., :, None] * unit[..., None, :]
        )
        kernel *= k0**2 * spacing**3
        # Integrated Green tensor over an equal-volume sphere, including
        # -I delta(r)/(3 k^2). Use a series to avoid cancellation at small ka.
        a = (3 * spacing**3 / (4 * np.pi)) ** (1 / 3)
        ka = k * a
        if abs(ka) < 1e-3:
            radial = ka**2 / 2 + 1j * ka**3 / 3 - ka**4 / 8 - 1j * ka**5 / 30
        else:
            radial = np.exp(1j * ka) * (1 - 1j * ka) - 1
        kernel[0, 0, 0] = np.eye(3) * k0**2 * (2 * radial - 1) / (3 * k**2)
        self.spectrum = np.fft.fftn(kernel, axes=(0, 1, 2))

    def apply(self, *, source):
        """Propagate a voxel source field through the background Green tensor.

        Parameters
        ----------
        source : numpy.ndarray
            Numeric source values, shape (nx, ny, nz, polarization, 3).
            Born iteration supplies linearized dielectric contrast times field.

        Returns
        -------
        field : numpy.ndarray
            Complex propagated field with the same shape and field units as source.
            The padded convolution is cropped to the original sample grid.
        """
        # source shape: (nx, ny, nz, polarization, vector component)
        transformed = np.fft.fftn(source, s=self.padded, axes=(0, 1, 2))
        propagated = np.einsum("...ij,...pj->...pi", self.spectrum, transformed)
        result = np.fft.ifftn(propagated, axes=(0, 1, 2))
        return result[tuple(slice(0, n) for n in self.shape)]
