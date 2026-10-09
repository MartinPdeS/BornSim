Public Python API
=================

BornSim computes scattering from finite voxel samples. Choose physical units,
spatial and angular configurations explicitly. GaussianMedium, ExponentialMedium,
and WhittleMaternMedium generate random fields; RandomSphereMedium places identical
nonoverlapping spheres. StructuredMedium voxelizes backgrounds
and shapes. Solver solves a fixed Volume or an ensemble of generated volumes.
Integrated coefficients describe finite-sample cross sections divided by volume.
Infinite-medium analytical formulas are test references, not public solvers.

Numerical calculation
---------------------

.. code-block:: python

   from bornsim import AngularSampling, Grid, Solver, Source
   from bornsim.medium.random_medium import GaussianMedium
   from bornsim.units import ureg

   grid = Grid(
       shape=(4, 4, 4),
       spacing=40 * ureg.nanometer,
   )

   medium = GaussianMedium(
       background_refractive_index=1.33,
       refractive_index_std=0.01,
       correlation_length=80 * ureg.nanometer,
   )

   volume = medium.to_volume(
       grid=grid,
       seed=42,
   )

   sampling = AngularSampling(
       start=0 * ureg.degree,
       end=180 * ureg.degree,
       n_points=19,
   )

   source = Source(wavelength=633 * ureg.nanometer)

   solver = Solver(
       source=source,
       sampling=sampling,
       order=2,
   )

   result = solver.solve(target=volume)

   print(f'mu_s = {result.mu_s}, g = {result.g}')

Configuration
-------------

.. automodule:: bornsim.source
   :members:

.. automodule:: bornsim.grid
   :members:

.. automodule:: bornsim.directions
   :members:

.. automodule:: bornsim.sampling
   :members:

.. automodule:: bornsim.ensemble_sampling
   :members:

Media and geometry
------------------

.. automodule:: bornsim.medium.base
   :members:

.. automodule:: bornsim.medium.random_spheres
   :members:

.. automodule:: bornsim.medium.random_medium
   :members:

.. automodule:: bornsim.geometry
   :members:

.. automodule:: bornsim.material
   :members:

.. automodule:: bornsim.rotation
   :members:

.. automodule:: bornsim.volume
   :members:

Solver and results
------------------

.. automodule:: bornsim.solver
   :members:

.. automodule:: bornsim.results
   :members:

.. automodule:: bornsim.angular_data
   :members:

Numerical engine
----------------

.. automodule:: bornsim.series
   :members:

.. automodule:: bornsim.ensemble
   :members:

Units
-----

.. automodule:: bornsim.units
   :members: validate_units

Archives
--------

Results save and load schema-2 NPZ archives with explicit SI units and no pickle.
Unsupported schemas are rejected. Arrays and provenance are copied when results
are constructed. Result.angular holds immutable directional data; .vectors on
Directions exposes the observation configuration's immutable Cartesian array.
