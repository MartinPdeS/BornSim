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

Incident direction and polarization
------------------------------------

``Source`` accepts ``incident_direction``, a real Cartesian unit vector, and
``polarization``, a ``PolarizationState`` or an explicit complex Cartesian
unit electric-field vector. Omitting polarization selects unpolarized light.
The default incident direction is positive z.

The state constructors follow PyMieSim's interface:

.. code-block:: python

   from bornsim import LeftCircular, PolarizationState, RightCircular, Source
   from bornsim.units import ureg

   linear_state = PolarizationState(angle=45 * ureg.degree)

   elliptical_state = PolarizationState(jones_vector=[1, 0.5j])

   left_circular_state = LeftCircular()

   right_circular_state = RightCircular()

   source = Source(
       wavelength=633 * ureg.nanometer,
       incident_direction=[0.6, 0, 0.8],
       polarization=linear_state,
   )

Supply exactly one of ``angle`` (a finite angular quantity) or
``jones_vector`` (two finite complex components). Nonzero Jones vectors are
normalized to unit incident intensity, preserving relative phase. The stored
Jones array is immutable. For positive-z incidence its components are
laboratory x and y. For oblique incidence, the first basis vector is the
normalized projection of laboratory +x onto the transverse plane; the second
is ``incident_direction cross first``. For incidence parallel to laboratory x,
+y is the reference instead. These conventions also define linear angle zero.

``LeftCircular`` is ``[1, +1j] / sqrt(2)`` and ``RightCircular`` is
``[1, -1j] / sqrt(2)``, matching PyMieSim's signs. The time convention is
:math:`\exp(-i\omega t)` and the transverse basis is right-handed.

Explicit three-component Cartesian vectors remain available for complete
control over the electric field in laboratory coordinates. They must already
have unit norm and be transverse to the incident direction. ``Source`` resolves
Jones states to these immutable Cartesian vectors; archives retain the
resolved physical illumination. No PyMieSim dependency is required.

A pure state has one incident-polarization amplitude channel. Unpolarized light
has two orthogonal linear channels, whose intensities are averaged. Partially
polarized mixtures are not currently configurable.

``AngularSampling`` and ``Directions`` always describe laboratory coordinates:
polar angle is measured from positive z, independently of the incident beam.
For oblique incidence, a laboratory polar angle is not the scattering angle.
``g`` and ``mu_s_prime`` nevertheless use the physical scattering cosine,
``observation_direction @ incident_direction``. Azimuth averages remain
averages about laboratory z. The phase function is still normalized over the
full sphere. See :doc:`auto_examples/structured_media/illumination_polarization`.

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

.. automodule:: bornsim.convergence
   :members:

.. automodule:: bornsim.source
   :members:

.. automodule:: bornsim.polarization
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

Media are grouped by construction: ``bornsim.medium.random`` contains continuous
random fields, ``bornsim.medium.structured`` contains composite and finite
periodic structures, and ``bornsim.medium.particles`` contains particle media.
Existing top-level and historical module imports remain supported.

``CoreShellMedium`` represents a spherical core inside a spherical shell,
with explicit radii, centre, and three absolute refractive indices.
``EllipsoidMedium`` represents an oriented particle with three semi-axis radii.
``CylinderMedium`` represents a finite dielectric rod with explicit radius,
height, axis, centre, and orientation; it is suitable for antenna-like
dielectric scatterers.
Both sample interfaces at voxel centres and clip geometry to the grid.
``PeriodicMedium`` repeats existing geometry objects on a centred finite
lattice; its three length-bearing lattice vectors are rows of a matrix.
It uses finite-volume scattering, without periodic boundary conditions.
These media retain the same linearized dielectric contrast and Born validity
limitations as other BornSim media.

.. automodule:: bornsim.medium.structured.periodic
   :members:

.. automodule:: bornsim.medium.particles.core_shell
   :members:

.. automodule:: bornsim.medium.particles.ellipsoid
   :members:

.. automodule:: bornsim.medium.particles.cylinder
   :members:

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

Results save schema-3 NPZ archives with explicit SI units, illumination
settings, and no pickle. Schema-2 archives remain readable as unpolarized
positive-z illumination.
Unsupported schemas are rejected. Arrays and provenance are copied when results
are constructed. Result.angular holds immutable directional data; .vectors on
Directions exposes the observation configuration's immutable Cartesian array.
