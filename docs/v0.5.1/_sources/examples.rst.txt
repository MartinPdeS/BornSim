Examples
========

Browse runnable examples by topic: random media, structured media, Born orders
and interference, plotting and saving results, and numerical validation.
Each gallery example includes a three-dimensional view of its input medium.
Ensemble examples show a labelled seeded realization on the calculation grid;
that view is one finite sample, rather than the ensemble average. Comparisons
of covariance models, correlation lengths, or sphere arrangements show each
medium separately. Each figure has its own code cell and output, including
individual three-dimensional views in comparisons.
The :doc:`medium_visualization` guide explains how to inspect input voxel
fields and choose spatial views.

.. toctree::
   :maxdepth: 2
   :hidden:

   medium_visualization
   auto_examples/index

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Random media
      :link: auto_examples/random_media/index
      :link-type: doc

      Seeded fields, covariance models, and wavelength dependence.

   .. grid-item-card:: Structured media
      :link: auto_examples/structured_media/index
      :link-type: doc

      Spheres, layers, and coherent directional scattering.

   .. grid-item-card:: Born orders
      :link: auto_examples/born_orders/index
      :link-type: doc

      Cumulative scattering and interference between complex amplitudes.

   .. grid-item-card:: Results
      :link: auto_examples/results/index
      :link-type: doc

      Plot, save, and reload unitful scattering results.

   .. grid-item-card:: Validation
      :link: auto_examples/validation/index
      :link-type: doc

      Grid, angular, ensemble, and finite-size comparisons.

   .. grid-item-card:: Medium visualization
      :link: medium_visualization
      :link-type: doc

      Interactive 3D volumes, opacity, slices, and voxel geometry.

Random media
------------

:doc:`auto_examples/random_media/random_spheres` generates seeded, fully contained
nonoverlapping spheres with an explicit count, radius, and refractive indices.

Start with :doc:`auto_examples/random_media/random_medium` for seeded samples,
and :doc:`auto_examples/random_media/wavelength_dependence` for numerical
finite-sample ensemble coefficients across wavelengths.

:doc:`auto_examples/random_media/covariance_models` compares Gaussian and exponential
spatial covariance at the same length parameter, showing both covariance
profiles and their numerical ensemble phase functions. These length parameters have
different definitions; equal values do not describe identical statistics.

:doc:`auto_examples/random_media/random_medium` leads with interactive Gaussian
and exponential volumes whose higher refractive index regions are more opaque. Central
slices and voxel histograms follow as diagnostic checks. It distinguishes the
Gaussian one-point probability distribution from the choice of spatial
covariance, and explains why a finite correlated sample need not have
exactly the ensemble mean or variance.

Structured media
----------------

``Medium`` is the abstract interface. ``GaussianMedium`` describes statistically
homogeneous fluctuations; ``StructuredMedium`` composes geometric materials. ``Volume`` accepts an explicit
three-dimensional field and can therefore represent spatially structured
media. Its background refractive index is uniform, while its fluctuation array may vary
from voxel to voxel.

:doc:`auto_examples/structured_media/dielectric_sphere` builds two constant refractive index spheres in a
homogeneous background and plots their voxelized geometry, full directional
3D phase density and a selected physical meridian. The curved
interface is voxelized and the dielectric contrast remains linearized;
the example does not replace an exact Mie calculation.

:doc:`auto_examples/structured_media/structured_media` combines finite layers,
a sphere, and additive seeded random fluctuations in a custom voxel field.

Normalized phase function
-------------------------

This example compares analytical first-order scattering for three Gaussian
covariance lengths at a vacuum wavelength of 633 nm. The phase function is
the differential scattering coefficient divided by the total scattering
coefficient:

.. math::

   p(\theta) = \frac{\beta(\theta)}{\mu_s}, \qquad
   2\pi \int_0^\pi p(\theta)\sin\theta\,\mathrm{d}\theta = 1.

Here ``p`` is a probability density **per steradian**, not per polar angle.
For this axially symmetric model, the polar-angle density is
``2*pi*p(theta)*sin(theta)`` and integrates to one with respect to angle in
radians. Its values retain units of inverse radians even when the plot's
horizontal axis displays degrees. The mean cosine of the distribution is
the anisotropy ``g``.

See :doc:`auto_examples/random_media/phase_function` for the plot and its top-level code.

A zero scattering coefficient has no normalized phase function. This example
uses the analytical infinite-medium model. A single random volume's angular
cut is insufficient to normalize a phase function over solid angle.

Result plots and interactive 3D directions
------------------------------------------

The medium occupies three spatial dimensions. Scattering directions lie on
the unit sphere and are described by polar angle ``theta`` and azimuth
``phi``. For the isotropic analytical medium under unpolarized illumination,
the incident beam defines the +z axis and rotational symmetry gives
``p(theta, phi) = p(theta)``. This symmetry does not mean scattering is equally
likely in all directions: the phase function can still be strongly forward
peaked. The one-dimensional angular curve describes the full axisymmetric
distribution on the sphere.

``Result.phase_function`` provides normalized per-steradian quantities.
``Result.plot_phase_function`` plots angular curves, a polar meridian cut, or
a 3D surface whose radius and color are the phase-function value. The surface
is a visualization in direction space; its coordinates are not positions
inside the medium. The 3D view requires angles spanning 0 to pi and defaults
to the highest available cumulative order.

For full-volume solves and numerical ensembles, the 3D surface retains
both theta and phi without
azimuth averaging. A fixed realization or a structured medium can therefore
show rotational asymmetry. Multiple realizations retain their intensity
average. Angular and polar plots select sampled meridians; call
``result.azimuth_average()`` explicitly for averaged curves.
Only explicit single-volume angular cuts lack the integrated coefficient
needed for phase normalization. Their differential scattering and field-term
norms can still be plotted directly.

.. code-block:: python

   import matplotlib.pyplot as plt

   result.plot()

   result.plot_phase_function()

   result.plot_phase_function(view='polar')

   phase_figure = result.plot_phase_function(view='3d')

   phase_figure.show()

   numerical.plot_field_norms()

   plt.show()

Ensemble ``plot()`` retains differential-scattering standard-error bars.
Normalized phase plots do not infer errors by dividing those bars by the
mean total coefficient: uncertainty in a ratio also depends on numerator-
denominator covariance, which is not retained in the result.

See :doc:`auto_examples/results/result_plots` for direct result plotting calls and
Matplotlib angular figures and an interactive 3D phase-function surface.
Sphinx Gallery executes the examples and embeds interactive Plotly volumes
and surfaces alongside their cells when the documentation is built.

Coherent terms and reusable results
-----------------------------------

:doc:`auto_examples/born_orders/compare_orders` plots ensemble scattering through
successive Born orders with sampling-error bars and per-realization field
norms. :doc:`auto_examples/born_orders/born_interference` shows why cumulative intensities
cannot be reconstructed by summing isolated-term intensities: complex Born
amplitudes interfere within each incident polarization. Its second panel
shows the signed interference contribution for a fixed volume.

:doc:`auto_examples/results/save_load_results` saves and restores a unitful numerical
result, inspects its provenance, and plots the restored data. It uses a
temporary directory and retains complex amplitudes. Archives preserve result
data and settings; they do not store the original voxel field.

Numerical validation
--------------------

The validation gallery separates four questions. None is answered by
decreasing successive Born terms alone.

* :doc:`auto_examples/validation/grid_refinement` refines a fixed physical cube and
  compares its first-order amplitude with an exact volume integral. It
  isolates midpoint-quadrature error without changing the input field.
* :doc:`auto_examples/validation/angular_convergence` increases polar and azimuth
  integration independently on identical seeded volumes. Plot sampling is
  separate from solid-angle quadrature.
* :doc:`auto_examples/validation/ensemble_sampling` increases independent realization
  count at fixed geometry. Standard errors describe sampling uncertainty;
  they do not include discretization or finite-size error.
* :doc:`auto_examples/validation/finite_size_comparison` compares first-order finite
  ensembles of increasing size with the analytical infinite-medium model.
  Finite windows, spectral synthesis, and limited resolution can produce
  differences that remain outside the sampling error bars.

Theory illustrations
--------------------

The :doc:`theory` section embeds simulated figures with the associated equations.
:doc:`auto_examples/random_media/theory_random_fields` compares seeded field texture and
empirical covariance with the discrete spectral and continuum predictions.
:doc:`auto_examples/structured_media/theory_directional_interference` verifies first-order
translation invariance and shows how relative sphere positions introduce
azimuth-dependent interference. Both examples expose their grids, materials,
seeds and numerical settings.
