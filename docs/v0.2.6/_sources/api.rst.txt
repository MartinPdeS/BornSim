Python API
==========

Constructor, function, and method arguments require keywords, except
``add_structures(*structures)``, which takes positional shape objects.
For example, use ``Volume(delta_refractive_index=field, spacing=50 * ureg.nanometer)`` and
``solver.solve(target=volume)``. Calls with multiple arguments are formatted
with one argument per line and a trailing comma.

Sources, solvers and results
----------------------------

.. automodule:: bornsim.source
   :members: Source

.. automodule:: bornsim.solver
   :members: Solver

.. automodule:: bornsim.results
   :members: Result

Refractive-index names are explicit throughout the API:
``Material(refractive_index=...)``, ``refractive_index_std``,
``background_refractive_index`` and ``Volume(delta_refractive_index=...)``.
Plot field selectors are ``"refractive_index"`` and ``"delta_refractive_index"``.
These replace the shorter names used in earlier releases. Array positions,
such as the ``index`` argument to ``plot_slice``, retain their usual meaning.
Saved results remain readable and retain their original provenance keys.

Spatial and angular configurations
----------------------------------

.. automodule:: bornsim.grid
   :members: Grid

.. automodule:: bornsim.sampling
   :members: AngularSampling

For evenly spaced output angles, use inclusive endpoints and a point count:

.. code-block:: python

   from bornsim.units import ureg

   sampling = AngularSampling(
       start=0 * ureg.degree,
       end=180 * ureg.degree,
       n_points=181,
   )

Defaults are 0 to pi radians with 121 points. Explicit endpoints require angular units.
For irregular or explicitly ordered angles, use ``angles=...`` instead;
explicit angles cannot be combined with range settings. Output resolution
remains independent of the solid-angle integration quadratures.

Define Grid once, pass it to medium.to_volume or solver.ensemble, and retain
it on Volume.grid. Define AngularSampling once on Solver; solve and ensemble
share it. Per-call sampling overrides are supported. Legacy shape, spacing,
angles, polar_samples and azimuth_samples keywords are deprecated. They remain
available with DeprecationWarning; configuration objects cannot be combined
with their individual settings.

Observation directions
----------------------

.. automodule:: bornsim.directions
   :members: Directions

``Directions`` owns validation and immutable storage for Cartesian observation
vectors. Supply it to ``Solver.solve_cut`` or ``BornSeries``. Angular sampling
also returns a ``Directions`` object through ``sampling.directions``.

.. code-block:: python

   from bornsim import Directions
   from bornsim.units import ureg

   directions = Directions(vectors=[[0, 0, 1], [1, 0, 0]])

   vectors = directions.vectors

   angular_directions = Directions.from_angles(
       polar_angles=[0, 90] * ureg.degree,
       azimuth_angles=[0, 0] * ureg.degree,
   )

Each Cartesian vector must have unit length; the class does not silently
normalize inputs. ``from_angles`` accepts paired or broadcastable coordinates
and requires explicit angle units. ``vectors`` returns a read-only NumPy array.


Units
-----

.. automodule:: bornsim.units

Every dimensional input requires an explicit quantity, including angles,
shape coordinates, wavelengths, voxel spacing, covariance lengths, squared
wavenumbers and physical arrays passed to Result or AngularData. For example,
use ``correlation_length=60 * ureg.nanometer``; ``correlation_length=60e-9``
raises ValueError. Refractive indices, their standard deviations and fluctuation
arrays must be plain numbers; quantities are rejected even with dimensionless
or refractive-index units. Smoothness, unit directions and counts may be plain
numbers.
Physical settings have no arbitrary defaults: supply wavelengths, refractive
indices, fluctuation statistics, covariance choice, grid shape and spacing
explicitly. Matérn covariance also requires smoothness. An empty
``StructuredMedium()`` requires ``add_background(...)`` before voxelization. Physical settings and
coordinates retain quantities with their supplied units. Validation checks
compatible dimensions and scalar shape without converting units. Numerical
kernels, plots and archives explicitly select units when they need magnitudes.

``sampling.angles`` is a read-only quantity array. Use
``sampling.angles.to("degree").magnitude`` to obtain a NumPy array in degrees.

.. code-block:: python

   from bornsim import AngularSampling, Grid, RandomMedium, Solver, Source
   from bornsim.units import ureg

   grid = Grid(
       shape=(4, 4, 4),
       spacing=50e-9 * ureg.meter,
   )

   sampling = AngularSampling(angles=[0, 45, 90, 180] * ureg.degree)

   solver = Solver(
       source=Source(wavelength=633 * ureg.nanometer),
       sampling=sampling,
   )

   result = solver.ensemble(
       medium=RandomMedium(
           correlation="gaussian",
           correlation_length=100 * ureg.nanometer,
           background_refractive_index=1.33,
           refractive_index_std=0.01,
       ),
       grid=grid,
       realizations=3,
       seed=42,
   )

   print(f"mu_s = {result.mu_s.to('1 / millimeter')}")

   print(f"Differential scattering = {result.differential.to('1 / meter / steradian')}")

``Result`` arrays carry units, including ensemble standard errors and complex
Born amplitudes. Dimensional inputs require explicit units. Grid spacing,
geometry lengths and covariance lengths retain quantities. Analytical function
outputs also carry units; low-level Born and ensemble kernels document their
numeric SI arrays. Archives continue to store explicit SI metadata.

Validation and reproducibility
------------------------------

``Result`` validates units, array shapes, finite nonnegative intensities,
observation coordinates, and calculation-specific fields at construction.
Full-volume results include integrated coefficients; explicit angular cuts
cannot supply them. Ensemble
errors are NaN for one realization; anisotropy is NaN at zero scattering.
When all integrated coefficients are present, the result checks
``mu_s_prime = mu_s * (1 - g)``.

``medium.metadata`` returns a fresh JSON-compatible description owned by the
medium class. Random media report SI statistics, including Matérn smoothness;
structured media describe their ordered geometry with SI length keys.
Mutating the returned dictionary does not modify the medium. Seeds and voxel
grids describe individual samples and are recorded separately.

``result.provenance`` records the original BornSim and NumPy versions,
actual Born order, medium statistics, grid dimensions and spacing,
seeds, and applicable quadrature settings. Length metadata uses metres,
indicated by ``_m`` keys. Analytical coefficients are marked as
infinite-medium; numerical coefficients describe finite samples.
``random_volume`` retains its medium and seed on the volume so that
``Solver.solve(target=volume)`` can record them too. Manual volumes have no
inferred generation seed or medium statistics. Their exact fields are
identified by a SHA-256 of the C-ordered little-endian float64 fluctuations.

.. code-block:: python

   from bornsim import Result

   result.save(path="scattering.npz")

   restored = Result.load(path="scattering.npz")

   for name, value in restored.provenance.items():
       print(f"{name}: {value}")

   restored.plot()

The versioned NPZ archive stores SI quantities, complex amplitudes,
sampling errors, warnings, and JSON metadata. Loading validates the
archive and never enables pickle. Save/load preserves result data rather
than the input voxel field: retain manual fields separately. Recorded
seeds can reproduce generated fields with compatible recorded software
versions. Loading preserves the original version metadata.

Analytical models
-----------------

.. automodule:: bornsim.model
   :members:

Numerical Born series
---------------------

.. automodule:: bornsim.volume
   :members: Volume

.. automodule:: bornsim.green
   :members: GreenOperator

.. automodule:: bornsim.series
   :members: BornSeries

.. automodule:: bornsim.ensemble
   :members: ensemble_scattering

Numerical medium construction
-----------------------------

See :doc:`theory` for covariance, geometry, constitutive, and propagation
conventions. Material regions use absolute refractive indices and SI lengths.
``Medium`` is abstract; concrete random and structured subclasses implement
``to_volume()``. Existing analytical first-order calculations require
``AnalyticalMedium``. Direct ``Medium(...)`` construction is no longer supported.
Use ``StructuredMedium()`` followed by ``add_background`` and ``add_structures``
to compose materials in place. Supply either a uniform ``refractive_index`` or random
statistics as ``medium`` to ``add_background``. Statistical RandomMedium and
AnalyticalMedium instances describe homogeneous distributions; composition
belongs to StructuredMedium.
Define shapes first, then pass them positionally to ``add_structures``. This
variadic method is the explicit exception to the API's keyword-only convention.
All arguments are validated before any are appended; the ``overlap`` policy determines
precedence (replace, preserve, or error). An empty call leaves the medium unchanged.

``Solver.solve`` computes the full directional scattering distribution and
integrated coefficients for a fixed Volume by default. Its 3D phase plot
retains each azimuth, preserving asymmetry and interference between
structures. Use ``Solver.solve_cut`` with explicit angles or directions for
unnormalized cuts; numerical ``solve`` rejects these keywords.
``Solver.ensemble`` averages independent samples from random media or
structures with a random background. Deterministic structures use solve.


.. automodule:: bornsim.media
   :members: Medium, RandomMedium, random_volume

.. automodule:: bornsim.geometry
   :members: Layer, Sphere, Ellipsoid, Box, Cylinder, StructuredMedium

Directional data and explicit averaging
----------------------------------------

``result.differential`` and ``result.phase_function`` have shape
``(order, polar angle, azimuth)`` after a full solve or ensemble calculation.
``result.amplitudes`` uses the same observation axes followed by polarization
and Cartesian axes. It is absent for ensembles. ``result.angular`` groups
these quantities with unit-vector coordinates in a read-only ``AngularData``.

.. code-block:: python

   from bornsim.units import ureg

   result = solver.solve(target=volume)

   phase_figure = result.plot_phase_function(view="3d")

   phase_figure.show()

   averaged = result.azimuth_average()

   averaged.plot()

   cut = solver.solve_cut(
       target=volume,
       angles=[0, 90, 180] * ureg.degree,
   )

   result.plot_cross_section(area_unit="nanometer**2")

   print(f"Differential cross section = {result.differential_cross_section}")

Averaging acts on intensities, never coherent amplitudes. Ensemble averaged
errors come from the per-realization averaged intensities, including angular
correlations. A cut has no full-solid-angle normalization. Default curve plots
select azimuth index zero; ``azimuth=...`` selects another sampled meridian.
Stored ``sample_volume`` removes the need to pass the original Volume to a
cross-section plot. Archives use schema 2; schema 1 remains readable.

.. automodule:: bornsim.angular_data
   :members: AngularData

Geometry transforms and overlaps
--------------------------------

``shape.translated(offset=...)`` and ``shape.rotated(rotation=...)`` return
new immutable shapes. Rotations act about the shape centre by default;
``about=...`` supplies a world-space pivot. StructuredMedium remains mutable.

.. code-block:: python

   from bornsim.units import ureg

   from bornsim import Box, Rotation, StructuredMedium

   box = Box(
       size=(100, 200, 300) * ureg.nanometer,
       refractive_index=1.34,
   )

   rotation = Rotation(
       axis=(0, 0, 1),
       angle=30 * ureg.degree,
   )

   tilted = box.rotated(rotation=rotation)

   shifted = tilted.translated(offset=(50, 0, 0) * ureg.nanometer)

   medium = StructuredMedium(
       background_refractive_index=1.33,
       overlap="error",
       warn_on_clipping=True,
   )

   medium.add_structures(shifted)

Overlap policies act on voxel masks: ``replace`` lets later regions win,
``preserve`` lets earlier regions win, and ``error`` rejects overlap.
Optional clipping warnings report geometry extending beyond the sampled box.

.. automodule:: bornsim.rotation
   :members: Rotation

Public imports and inspection
-----------------------------

The package root exports the main configuration, medium, geometry and result
classes. Advanced helpers live in ``bornsim.model``, ``bornsim.media``,
``bornsim.ensemble`` and ``bornsim.series``. Their former root imports remain
available with a DeprecationWarning. Prefer ``medium.to_volume`` and Solver
for routine use. Grid, AngularSampling, Solver, AngularData and Result have
compact representations showing sizes, units or available data without
printing entire arrays.

Immutable results and shared angular data
-----------------------------------------

Result is immutable. Its numerical arrays are read-only; unit conversions use
``.to(...)`` and return new quantities. ``result.angular`` returns the same
AngularData object on every access. Convenient ``result.differential`` and
``result.amplitudes`` properties reference that object's arrays. AngularData
validates directional arrays; Result validates additional integrated moments,
realization counts and diagnostics. ``result.provenance`` returns a fresh copy,
so edits cannot change recorded calculation settings. StructuredMedium remains
a mutable geometry builder.

Reusable materials
------------------

.. automodule:: bornsim.material
   :members: Material

.. code-block:: python

   from bornsim.units import ureg

   from bornsim import Material, Sphere, StructuredMedium

   glass = Material(refractive_index=1.34)

   sphere = Sphere(
       radius=150 * ureg.nanometer,
       material=glass,
   )

   shifted = sphere.translated(offset=(50, 0, 0) * ureg.nanometer)

   medium = StructuredMedium()

   medium.add_background(material=Material(refractive_index=1.33))

   medium.add_structures(sphere, shifted)

Materials are immutable and shared across shapes, including transformed copies.
Supply either ``material`` or the compatible ``refractive_index`` shortcut to a shape.
Current materials use real, nondispersive absolute indices; material separation
does not introduce absorption or dispersion.

Ensemble configurations
-----------------------

.. automodule:: bornsim.ensemble_sampling
   :members: EnsembleSampling

.. code-block:: python

   from bornsim import EnsembleSampling

   ensemble_sampling = EnsembleSampling(seeds=[42, 11, 104])

   result = solver.ensemble(
       medium=random_medium,
       grid=grid,
       ensemble_sampling=ensemble_sampling,
   )

Alternatively specify ``realizations`` and an initial ``seed`` inside
EnsembleSampling for consecutive seeds. Explicit lists retain their order and
must contain distinct seeds to estimate independent-realization uncertainty.
Result provenance records the exact seeds. Legacy solver realization-count and
seed keywords are deprecated; the seed on ``medium.to_volume`` remains valid.

Physical meridian selection
---------------------------

.. code-block:: python

   from bornsim.units import ureg

   meridian = result.meridian(azimuth=90 * ureg.degree)

   meridian.plot_phase_function()

   nearest = result.meridian(
       azimuth=87 * ureg.degree,
       method="nearest",
   )

   print(f"Meridian azimuth = {nearest.meridian_azimuth.to('degree')}")

Selection returns AngularData with amplitudes, intensities and standard errors
for that sampled direction plane. Full-solve normalization is retained.
The default exact method rejects unsampled angles; nearest explicitly selects
the closest sample and records its actual angle. Angles wrap modulo 2*pi;
angular values require explicit units. No interpolation or averaging is performed. Selected
meridians support angular curves; use full data for polar-plane or 3D plots.
