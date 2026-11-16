"""
Coherent and diffuse scattering in an absorbing ensemble
================================================================

Absorption belongs to the random-medium configuration, so every realization
has a reproducible kappa field. Compare total intensity, the intensity of the
mean scattered amplitude, and their difference through third Born order.
"""

import matplotlib.pyplot as plt

from bornsim import ConvergenceReport, EnsembleSampling, GaussianMedium, Grid, Solver, Source
from bornsim.units import ureg

medium = GaussianMedium(
    background_refractive_index=1.33,
    refractive_index_std=0.002,
    correlation_length=60 * ureg.nanometer,
    extinction_coefficient=0.0005,
)

grid = Grid(
    shape=(8, 8, 8),
    spacing=30 * ureg.nanometer,
)

source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(source=source, order=3)

ensemble_sampling = EnsembleSampling(realizations=8, seed=42)

result = solver.ensemble(
    medium=medium,
    grid=grid,
    ensemble_sampling=ensemble_sampling,
)

averaged = result.azimuth_average()

figure, axis = plt.subplots(layout="constrained")

for label, data in (
    ("Total", averaged.differential),
    ("Coherent", averaged.coherent_differential),
    ("Diffuse", averaged.diffuse_differential),
):
    axis.plot(result.angles.to("degree").magnitude, data.magnitude[-1], label=label)

axis.set(xlabel="Scattering angle (degrees)", ylabel="Differential coefficient (m⁻¹ sr⁻¹)")

axis.legend()

plt.show()

absorption_figure, absorption_axis = plt.subplots(layout="constrained")

absorption_axis.errorbar(
    [1, 2, 3],
    result.mu_a.to("1 / meter").magnitude,
    yerr=result.mu_a_stderr.to("1 / meter").magnitude,
    fmt="o-",
)

absorption_axis.set(xlabel="Cumulative scattering order", ylabel="Absorption coefficient (m⁻¹)", xticks=[1, 2, 3])

plt.show()

# %%
# Statistical interpretation
# ----------------------------
# Coherent means squared sample-mean scattered amplitude, not the unscattered
# laser. Diffuse is the population-normalized sample variance of amplitudes.
# For N realizations its expectation is (N-1)/N times the true variance; with
# one realization it is zero, not evidence of no diffuse scattering. Increase
# ensemble size to assess these effects. Error bars here describe sampling
# uncertainty in absorption, not mesh errors or model validity.

# %%
# Ensemble-size stability
# -------------------------
# Shared initial seeds correlate the estimates. These are stability comparisons,
# not independent estimates of an error bar on the difference.
size_results = []

for count in (4, 8, 16):
    seeds = EnsembleSampling(realizations=count, seed=42)

    size_results.append(solver.ensemble(medium=medium, grid=grid, ensemble_sampling=seeds))

report = ConvergenceReport(results=tuple(size_results), labels=("4 seeds", "8 seeds", "16 seeds"))

print(report.summary())
