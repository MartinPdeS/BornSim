Public Python API
=================

BornSim computes scattering from finite voxel samples. Choose physical units,
spatial and angular configurations explicitly. Grid.from_extent derives an
isotropic voxel spacing from explicit axis extents and shape. GaussianMedium,
ExponentialMedium, and WhittleMaternMedium generate random fields;
RandomSphereMedium places identical nonoverlapping spheres. StructuredMedium
voxelizes backgrounds and shapes and can add a seeded fluctuation field over
the full structure. Solver accepts either AngularSampling or Directions for
fixed volumes, or an ensemble of generated volumes. Integrated coefficients
describe finite-sample cross sections divided by volume. Infinite-medium
analytical formulas are test references, not public solvers.

Numerical calculation
---------------------

.. code-block:: python

   from bornsim import AngularSampling, Directions, Grid, Solver, Source
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

   directions = Directions(
       vectors=[
           [0, 0, 1],
           [1, 0, 0],
       ],
   )

   directional_result = solver.solve(
       target=volume,
       sampling=directions,
   )

   cut = solver.solve_cut(
       target=volume,
       directions=directions,
   )

   np.testing.assert_allclose(
       directional_result.differential.magnitude,
       cut.differential.magnitude,
   )

Field construction
------------------

.. code-block:: python

   from bornsim import Grid, Layer, StructuredMedium, Volume
   from bornsim.medium.random_medium import WhittleMaternMedium
   from bornsim.units import ureg

   grid = Grid.from_extent(
       shape=(16, 16, 16),
       extent=(400, 400, 400) * ureg.nanometer,
   )

   layer = Layer.from_center_and_thickness(
       centre=(0, 0, -100) * ureg.nanometer,
       thickness=200 * ureg.nanometer,
       refractive_index=1.335,
   )

   statistics = WhittleMaternMedium(
       background_refractive_index=1.33,
       refractive_index_std=0.002,
       correlation_length=50 * ureg.nanometer,
       smoothness=1.5,
   )

   medium = StructuredMedium()
   medium.add_background(refractive_index=1.33)
   medium.add_structures(layer)
   medium.add_fluctuations(medium=statistics)

   volume = medium.to_volume(
       grid=grid,
       seed=42,
   )

   absolute_refractive_index = volume.refractive_index

   imported_volume = Volume.from_refractive_index(
       refractive_index=absolute_refractive_index,
       background_refractive_index=1.33,
       grid=grid,
   )

Alternatively, independently generated volumes with matching grids and
backgrounds can be combined with ``volume_a + volume_b``. The result's
``refractive_index`` property exposes its immutable absolute voxel field.
``Volume.from_refractive_index`` also constructs a volume from an absolute
unitless index array, converting it to the linearized contrast representation.

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

Rotation inputs accept a :class:`bornsim.rotation.Rotation`, a nested matrix
sequence, a NumPy array, or a dimensionless quantity. Shapes normalize accepted
inputs to immutable :class:`bornsim.rotation.RotationMatrix` tuples;
:class:`bornsim.rotation.RotationInput` describes the accepted forms.

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
