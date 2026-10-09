Research uses
=============

BornSim helps investigate how a finite refractive-index field produces vector
scattering under the Born approximation. Use it to connect a proposed medium
structure to directional scattering, compare controlled model changes, and
check numerical sensitivity before interpreting a result physically.

Research questions
------------------

Spatial covariance and scattering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare GaussianMedium, ExponentialMedium, and WhittleMaternMedium while holding
the background refractive index and fluctuation standard deviation fixed.
This separates the effect of spatial covariance from the one-point probability
distribution, which is Gaussian for all three random-field models. Vary the
correlation length or Matérn smoothness to study changes in angular scattering,
anisotropy, and finite-sample coefficients.

See :doc:`auto_examples/random_media/covariance_models` and
:doc:`auto_examples/random_media/phase_function` for numerical comparisons.
Equal correlation-length parameters across covariance families do not imply
identical real-space covariance profiles.

Geometry and coherent interference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use StructuredMedium to build layers and shaped inclusions, or
RandomSphereMedium to generate seeded collections of nonoverlapping spheres.
Compare positions, shapes, or arrangements to investigate directional
interference. BornSim combines complex scattering amplitudes coherently within
each incident polarization and retains interference between Born orders.
Adding isolated intensities cannot reconstruct those cumulative results.

See :doc:`auto_examples/structured_media/theory_directional_interference` and
:doc:`auto_examples/born_orders/born_interference`. Random sphere collections use
sequential rejection sampling; they do not represent an equilibrium hard-sphere
ensemble.

Wavelength and approximation sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sweep explicitly supplied wavelengths to explore spectral trends for a chosen
medium. The :doc:`auto_examples/random_media/wavelength_dependence` example
holds refractive indices fixed. Material dispersion requires you to supply
appropriate refractive indices at each wavelength.

Compare successive Born orders to assess sensitivity to the truncation.
Decreasing field terms alone do not establish convergence or accuracy. Compare
against an independent reference or another numerical method when making
quantitative claims.

A reproducible starting point
-----------------------------

This configuration is an illustrative finite-sample study, rather than a
validated description of a particular material. Choose physical parameters
from your research model or measurements, and refine the numerical settings
for the observables you intend to report.

.. code-block:: python

   from bornsim import AngularSampling, EnsembleSampling, Grid, Solver, Source
   from bornsim.medium.random_medium import GaussianMedium
   from bornsim.units import ureg

   grid = Grid(
       shape=(8, 8, 8),
       spacing=25 * ureg.nanometer,
   )

   medium = GaussianMedium(
       background_refractive_index=1.33,
       refractive_index_std=0.01,
       correlation_length=50 * ureg.nanometer,
   )

   source = Source(wavelength=633 * ureg.nanometer)

   sampling = AngularSampling(
       start=0 * ureg.degree,
       end=180 * ureg.degree,
       n_points=91,
       polar_samples=32,
       azimuth_samples=16,
   )

   ensemble_sampling = EnsembleSampling(seeds=(42, 43, 44, 45))

   solver = Solver(
       source=source,
       sampling=sampling,
       order=2,
   )

   result = solver.ensemble(
       medium=medium,
       grid=grid,
       ensemble_sampling=ensemble_sampling,
   )

   result.save(path="research-scattering.npz")

   print(f"Finite-sample scattering coefficient: {result.mu_s}")

Inspect one realization from that ensemble in a separate plotting cell.
The three-dimensional view shows the finite input sample, rather than an
ensemble-averaged medium.

.. code-block:: python

   import matplotlib.pyplot as plt

   preview_volume = medium.to_volume(
       grid=grid,
       seed=ensemble_sampling.seeds[0],
   )

   medium_figure = preview_volume.plot_3d(
       backend="matplotlib",
       mode="slices",
       field="refractive_index",
   )

   plt.show()

Controls for quantitative studies
---------------------------------

Check voxel spacing at fixed physical sample size, then sample size at fixed
spacing. Refine the polar integration and azimuth sampling independently;
adding more plotted output angles does not replace integration refinement.
Increase the number of independent realizations to assess ensemble sampling
uncertainty. Reuse the same seed set across parameter comparisons to keep the
sampling protocol reproducible; the resulting comparisons are correlated.

BornSim currently accepts 2–32 voxels per axis. This limits the combination of
sample size and spatial resolution you can investigate. Boundary effects,
spectral truncation, and the synthesis box can affect random-field statistics.
The :doc:`examples` validation gallery demonstrates separate grid, angular,
ensemble, and finite-size checks.

Physical scope and reporting
----------------------------

The implemented model uses real scalar refractive indices, an unpolarized plane
wave along +z, a uniform exterior background, and the linearized dielectric
contrast ``2 * background_refractive_index * delta_refractive_index``. Higher Born
orders do not restore the omitted quadratic refractive-index term. Strong
contrast, absorption, or more general illumination requires a model beyond
these assumptions. See :doc:`theory` for the equations and Green-tensor self-cell
convention.

Report the wavelength, refractive indices, medium statistics or geometry,
physical sample size, voxel spacing, angular settings, Born order, seed set,
and software version. Saved results retain units, amplitudes, and provenance;
they do not contain the original voxel field. Preserve manually constructed
fields and their generation procedure separately.

The integrated coefficients describe finite-sample cross sections divided by
sample volume. Do not present them as infinite-medium transport coefficients
without a separate justification of that limit. Ensemble standard errors
quantify realization sampling uncertainty, rather than discretization error
or uncertainty in the physical model. Normalized phase-function uncertainty
cannot be inferred simply by dividing differential-scattering error bars by
the mean integrated coefficient.
