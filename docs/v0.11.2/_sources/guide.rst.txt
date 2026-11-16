User guide
==========

Follow the calculation workflow from a physical sample to saved scattering
results. Start with :doc:`getting_started` for a complete runnable calculation;
use :doc:`examples` for comparisons and :doc:`api` for parameter details.

Build a medium
--------------

Choose a statistical medium such as ``GaussianMedium``, or compose explicit
shapes with ``StructuredMedium``. Set the background refractive index, geometry,
and fluctuation statistics explicitly. Voxelize on a ``Grid`` with physical
spacing; keep the random seed to reproduce a realization. See the
:doc:`random-medium examples <auto_examples/random_media/index>` and
:doc:`structured-medium examples <auto_examples/structured_media/index>`.

Configure illumination
----------------------

Create a ``Source`` with a unit-bearing vacuum wavelength. Set its incident
direction and polarization when needed; the default is unpolarized illumination
along laboratory +z. The :ref:`illumination-configuration` reference explains
Jones states and coordinate conventions.

Solve and sample directions
---------------------------

Create a ``Solver`` with the source and requested Born order. Use
``AngularSampling`` for full solid-angle integration, or ``Directions`` for
explicit observation directions. Call ``solver.solve(target=volume)`` for a
fixed sample. Use ``solver.ensemble(...)`` with explicit ensemble sampling for
independent random realizations and sampling uncertainty. Angular cuts alone
cannot provide integrated coefficients or phase-function normalization.

Inspect and save results
------------------------

Use ``volume.plot_3d()`` to inspect spatial media with Plotly and
``result.plot_phase_function()`` for a directional 3D surface. Matplotlib
provides angular curves and statistical diagnostics. The
:doc:`medium_visualization` guide describes interactive controls and export.

Save results with ``result.save(...)`` and reload them with ``Result.load(...)``.
The :doc:`result examples <auto_examples/results/index>` demonstrate archives,
normalization, and derived quantities. Preserve the original voxel field or its
seeded construction separately; result archives do not contain it.

Check accuracy
--------------

Before interpreting a calculation, assess spatial, angular, ensemble, and Born
order sensitivity. Finite-sample coefficients are not automatically intrinsic
infinite-medium coefficients. See :doc:`theory_validation` for assumptions and
independent validation.

.. toctree::
   :hidden:
   :maxdepth: 2

   Package overview <overview>
   Visualization and export <medium_visualization>
   Absorption <absorption>
