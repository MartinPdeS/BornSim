Resources
=========

Use reproducible inputs and independent numerical checks to assess a study.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Development and releases
      :link: development
      :link-type: doc

      Set up a development environment, check packages, and understand
      documentation versions.

   .. grid-item-card:: Numerical validation
      :link: auto_examples/validation/index
      :link-type: doc

      Compare spatial resolution, angular integration, ensemble sampling,
      and finite-size effects.

Reproducibility
---------------

Record the BornSim version, vacuum wavelength, background refractive index, fluctuation
statistics, material geometry, grid, angular quadrature, Born order, and
random seeds. Retain ensemble sample counts and standard errors alongside
the coefficients. Result archives retain settings and amplitudes, but do
not contain the original voxel field; keep that field or its seeded
construction separately.

Decreasing Born terms alone do not establish convergence. Use the
:doc:`auto_examples/validation/index` examples to separate numerical errors
and sampling uncertainty, and consult :doc:`theory` for model limitations.

.. toctree::
   :hidden:
   :maxdepth: 1

   development
