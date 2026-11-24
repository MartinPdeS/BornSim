"""Independent small-size Mie reference; no BornSim solver dependencies.

Riccati-Bessel coefficient formulas: Bohren & Huffman, chapter 4.
https://miepython.readthedocs.io/en/latest/07_algorithm.html
Positive imaginary index is passive with exp(-i omega t).
Series evaluation of j_n avoids unstable upward recursion at small x.
This reference is restricted to size parameters below 2, not a public solver.
"""

import math
import numpy as np


def sphere_reference(*, radius_m, wavelength_m, host, refractive_index, cosine, multipole_terms=10):
    """Evaluate sphere cross sections using independent Mie coefficients.

    Parameters
    ----------
    radius_m, wavelength_m : float
        Nominal radius and vacuum wavelength in metres.
    host : float
        Real, positive background refractive index.
    refractive_index : complex
        Sphere refractive index, with positive imaginary part for passive loss.
    cosine : numpy.ndarray
        Observation scattering cosines, shape (observations,).
    multipole_terms : int, optional
        Truncation order, default 10. This reference requires size parameter
        below two; increasing the truncation checks its numerical stability.

    Returns
    -------
    scattering, absorption : float
        Cross sections in square metres.
    angular : numpy.ndarray
        Unpolarized differential cross section in square metres per steradian.
    """

    x = 2 * np.pi * host * radius_m / wavelength_m
    assert 0 < x < 2
    m = refractive_index / host
    count = multipole_terms

    def bessel(z):
        values = []
        for n in range(count + 1):
            factor = math.prod(range(1, 2 * n + 2, 2))
            term = z**n / factor
            value = term
            for k in range(1, 50):
                term *= -(z**2) / (2 * k * (2 * n + 2 * k + 1))
                value += term
            values.append(value)
        return np.array(values)

    j, jm = bessel(x), bessel(m * x)
    y = [-np.cos(x) / x, -np.cos(x) / x**2 - np.sin(x) / x]
    for n in range(1, count):
        y.append((2 * n + 1) / x * y[-1] - y[-2])
    psi, psim = x * j, m * x * jm
    xi = psi + 1j * x * np.array(y)
    a, b = [], []
    for n in range(1, count + 1):
        d = psi[n - 1] - n / x * psi[n]
        dm = psim[n - 1] - n / (m * x) * psim[n]
        dx = xi[n - 1] - n / x * xi[n]
        a.append((m * psim[n] * d - psi[n] * dm) / (m * psim[n] * dx - xi[n] * dm))
        b.append((psim[n] * d - m * psi[n] * dm) / (psim[n] * dx - m * xi[n] * dm))
    a, b = np.array(a), np.array(b)
    weights = 2 * np.arange(1, count + 1) + 1
    k = 2 * np.pi * host / wavelength_m
    scattering = 2 * np.pi / k**2 * np.sum(weights * (abs(a) ** 2 + abs(b) ** 2))
    extinction = 2 * np.pi / k**2 * np.sum(weights * (a + b).real)
    previous, current = np.zeros_like(cosine), np.ones_like(cosine)
    s1, s2 = np.zeros_like(cosine, dtype=complex), np.zeros_like(cosine, dtype=complex)
    for n in range(1, count + 1):
        tau = n * cosine * current - (n + 1) * previous
        factor = (2 * n + 1) / (n * (n + 1))
        s1 += factor * (a[n - 1] * current + b[n - 1] * tau)
        s2 += factor * (a[n - 1] * tau + b[n - 1] * current)
        previous, current = current, ((2 * n + 1) * cosine * current - (n + 1) * previous) / n
    return scattering, extinction - scattering, (abs(s1) ** 2 + abs(s2) ** 2) / (2 * k**2)
