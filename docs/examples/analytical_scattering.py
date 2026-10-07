"""
Analytical scattering with physical units
=========================================

Compute first-order infinite-medium optical properties using TypedUnit quantities.
"""

import matplotlib.pyplot as plt

from bornsim import AnalyticalMedium, Solver, Source
from bornsim.units import ureg

medium = AnalyticalMedium(
    background_index=1.33,
    index_std=0.01,
    correlation_length=100 * ureg.nanometer,
)
solver = Solver(source=Source(wavelength=633 * ureg.nanometer))

result = solver.solve(target=medium)
print({"mu_s": result.mu_s[0], "g": result.g[0], "mu_s_prime": result.mu_s_prime[0]})
result.plot()
plt.show()
