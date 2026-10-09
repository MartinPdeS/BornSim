Getting started
===============

Install
-------

Use Python 3.11 or newer. From a checkout of BornSim, create a virtual
environment and install the package::

   python -m venv .venv
   .venv/bin/python -m pip install -e .

First scattering calculation
----------------------------

Define a source, material, and grid, then average independent seeded random
volumes. Quantity inputs use the BornSim unit registry; dimensional inputs require explicit
units. The wavelength is the vacuum wavelength.

.. code-block:: python

   from bornsim import EnsembleSampling, Grid, Solver, Source
   from bornsim.medium.random_medium import GaussianMedium
   from bornsim.units import ureg

   source = Source(wavelength=633 * ureg.nanometer)

   grid = Grid(
       shape=(8, 8, 8),
       spacing=50 * ureg.nanometer,
   )

   medium = GaussianMedium(
       background_refractive_index=1.33,
       refractive_index_std=0.01,
       correlation_length=100 * ureg.nanometer,
   )

   ensemble_sampling = EnsembleSampling(
       realizations=4,
       seed=42,
   )

   solver = Solver(
       source=source,
       order=3,
   )

   result = solver.ensemble(
       medium=medium,
       grid=grid,
       ensemble_sampling=ensemble_sampling,
   )

   print(f"mu_s = {result.mu_s.to('1 / millimeter')}, g = {result.g}, mu_s_prime = {result.mu_s_prime.to('1 / millimeter')}")

The returned coefficients describe finite samples. The ensemble standard
errors describe sampling uncertainty, not discretization or finite-size
error. See :doc:`theory` before interpreting higher Born orders.

Interactive three-dimensional views
-----------------------------------

Plotly is included with BornSim and is the default for 3D views. Stronger
refractive-index regions can be made more opaque:

.. code-block:: python

   volume = medium.to_volume(
       grid=grid,
       seed=42,
   )

   medium_figure = volume.plot_3d(
       field='refractive_index',
       opacity_scale='increasing',
   )

   medium_figure.show()

   phase_figure = result.plot_phase_function(view='3d')

   phase_figure.show()

The medium view shows physical positions. The phase surface shows scattering
directions, with radius and color representing probability density per
steradian. Angular and polar plots use Matplotlib.

Next steps
----------

* :doc:`guide` for physical conventions and the broader package overview.
* :doc:`examples` for runnable material, plotting, and validation examples.
* :doc:`api` for keyword-only constructor and method arguments.
* :doc:`resources` for reproducibility and development checks.
