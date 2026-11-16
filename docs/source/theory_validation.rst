Theory & validation
===================

Understand the model before interpreting a result, then check the numerical
resolution and sampling relevant to your calculation.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Theory and numerical methods
      :link: theory
      :link-type: doc

      Born equations, Green tensors, coherent amplitudes, and finite volumes.

   .. grid-item-card:: Reliability and benchmarks
      :link: reliability
      :link-type: doc

      Independent references, tested regimes, numerical errors, and limitations.

   .. grid-item-card:: Research workflows
      :link: research
      :link-type: doc

      Design reproducible studies and choose relevant refinement checks.

   .. grid-item-card:: Validation examples
      :link: auto_examples/validation/index
      :link-type: doc

      Run grid, angular, ensemble, absorption, and finite-size comparisons.

Reproducibility
---------------

Record the BornSim version, vacuum wavelength, background refractive index,
fluctuation statistics, material geometry, grid, angular quadrature, Born order,
and random seeds. Keep ensemble sample counts and standard errors alongside
coefficients. Retain the original voxel field or its seeded construction.
Decreasing Born terms alone do not establish convergence.

.. toctree::
   :hidden:
   :maxdepth: 2

   theory
   reliability
   research
