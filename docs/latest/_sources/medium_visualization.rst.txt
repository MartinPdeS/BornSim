Medium visualization
====================

Interactive three-dimensional volumes
-------------------------------------

Start with the interactive volumes below. Drag to rotate, scroll to zoom,
and hover to inspect the refractive index. Plotly is included with BornSim and is the
default for 3D views. Angular and polar result plots use Matplotlib.

Interactive random-medium volumes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For random media, a volume view reveals connected high refractive index regions throughout
the sample. Set ``field="refractive_index"`` and ``opacity_scale="increasing"`` to make
low refractive index contours transparent and higher refractive index contours increasingly opaque.
``opacity`` sets the maximum contour opacity. This emphasizes high absolute
refractive index, rather than large positive and negative fluctuations equally. Opacity
is a display setting; it does not represent material absorption.
The opacity ramp uses each sample's displayed refractive index range, so equal opacity
in two different samples need not represent the same refractive index.

The following views use the same Gaussian and exponential samples as
:doc:`auto_examples/random_media/random_medium`: seed 42, 16 voxels per axis,
25 nm spacing, refractive index standard deviation 0.01, and correlation length 75 nm.
Drag to rotate and scroll to zoom. Each view includes Plotly and works offline.
Interactive views are generated during the documentation build alongside
interactive gallery examples.

Gaussian covariance
^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <iframe src="_static/random-medium-gaussian.html" title="Gaussian random medium with increasing refractive index opacity" width="100%" height="560" loading="lazy" style="border:0;"></iframe>

Exponential covariance
^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <iframe src="_static/random-medium-exponential.html" title="Exponential random medium with increasing refractive index opacity" width="100%" height="560" loading="lazy" style="border:0;"></iframe>

.. code-block:: python

   from bornsim.units import ureg

   from bornsim import Grid, RandomMedium

   grid = Grid(
       shape=(16, 16, 16),
       spacing=2.5e-08 * ureg.meter,
   )

   medium = RandomMedium(
       refractive_index_std=0.01,
       correlation_length=7.5e-08 * ureg.meter,
       correlation="gaussian",
       background_refractive_index=1.33,
   )

   volume = medium.to_volume(
       grid=grid,
       seed=42,
   )

   figure = volume.plot_3d(
       mode="volume",
       field="refractive_index",
       surface_count=16,
       opacity=0.2,
       opacity_scale="increasing",
   )

   figure.write_html(
       file="medium.html",
       include_plotlyjs=True,
   )

Plotly also supports ``mode="isosurface"`` and ``mode="slices"``; its default
mode is ``volume``. Uniform fields fall back to slices. The HTML export includes
Plotly itself and works offline. Call ``figure.show()`` to display a Plotly
figure. See its official `3D isosurface documentation
<https://plotly.com/python/3d-isosurface-plots/>`_ and
`rendering documentation <https://plotly.com/python/renderers/>`_.

Three-dimensional media
-----------------------

``Volume.plot_3d()`` returns a Plotly volume by default for generated random
fields, structured samples, and custom voxel arrays. Call ``figure.show()``
to display it or ``figure.write_html()`` to export an interactive view.
Select ``backend="matplotlib"`` for static figures, then use ``plt.show()``
or ``figure.savefig()``. The gallery embeds interactive 3D views alongside their plotting cells.
Constructing a figure opens no window.

Secondary views: voxel cross-sections
--------------------------------------

``Volume.plot_slice()`` builds a Matplotlib figure with physical axis labels,
an equal spatial aspect ratio, and a colorbar. It handles axis orientation and
voxel-edge extents automatically. The default plane is the middle z voxel;
on an even grid its centre is at positive half-spacing, not at zero.

.. code-block:: python

   from bornsim.units import ureg

   from bornsim import Grid, Sphere, StructuredMedium

   structure = StructuredMedium()

   structure.add_background(refractive_index=1.33)

   sphere = Sphere(
       radius=1.5e-07 * ureg.meter,
       refractive_index=1.34,
   )

   structure.add_structures(
       sphere,
   )

   volume = structure.to_volume(
       grid=Grid(
           shape=(24, 24, 24),
           spacing=2e-08 * ureg.meter,
       ),
   )

   figure = volume.plot_slice(
       normal="z",
       field="refractive_index",
       length_unit="nanometer",
   )

   figure.savefig("sphere-slice.png")

Use ``normal="x"`` or ``normal="y"`` for other orientations and ``index``
to select a voxel plane. Field choices match ``plot_3d()``. Color limits use
the entire volume, so different slices share the same scale.

Finite-sample cross sections
----------------------------

``Result.plot()`` plots differential scattering coefficients. For a finite
sample, ``Result.plot_cross_section()`` multiplies curves and ensemble standard
errors by the stored physical sample volume and converts the display area units. Both return
Matplotlib figures and accept a custom ``title`` and ``log_y=True``.

.. code-block:: python

   from bornsim.units import ureg

   from bornsim import Solver, Source

   solver = Solver(
       source=Source(wavelength=6.33e-07 * ureg.meter),
       order=3,
   )

   result = solver.solve(target=volume)

   figure = result.plot_cross_section(
       area_unit="nanometer**2",
   )

   figure.savefig("sphere-scattering.png")

The original Volume is not required for this plot. Normalization uses the entire
voxel box: :math:`d\sigma/d\Omega=V\,\mathrm{differential}`. Cumulative curves
preserve amplitude interference; ``terms=True`` shows isolated terms instead.
Analytical infinite-medium results have no finite-sample cross section.

Sphere offsets and phase functions
----------------------------------

``Sphere.centre`` specifies a three-component offset from the voxel-box centre,
as a quantity with explicit length units. For example,
``centre=np.array([40, 20, 0]) * ureg.nanometer`` shifts a sphere by 40 nm along x
and 20 nm along y. Keep its radius plus the absolute offset within each box
half-width to avoid clipping. Offsets aligned with the voxel spacing translate
the existing sampled mask; other offsets resample the interface.

The :doc:`auto_examples/structured_media/dielectric_sphere` example also plots the sphere's
directional 3D phase function. It calls ``Solver.solve`` with the
fixed generated volume to integrate a single deterministic sample over
polar Gauss nodes and azimuth. This supplies the solid-angle
normalization that a single meridian cut cannot determine. Refine both angular
quadratures to check the numerical normalization.

The phase surface represents probability density per steradian, with incidence
along +z, rather than spatial medium geometry. It retains both polar angle
and azimuth, so interference between multiple structures remains visible.
No azimuth averaging is applied to the 3D view. Increase ``azimuth_samples``
to resolve finer azimuthal features. One deterministic sample provides no
estimate of random sampling uncertainty.

Random fields and orthogonal slices
-----------------------------------

With ``backend="matplotlib"``, the default ``mode="slices"`` shows one voxel
plane along each axis. Plotly also supports explicit ``mode="slices"``. All selected
cells are retained, with voxel-edge extents and a shared color scale. Select
specific planes with ``slice_indices``; otherwise the middle voxels are used.

.. code-block:: python

   from bornsim.units import ureg

   import matplotlib.pyplot as plt
   from bornsim import Grid, RandomMedium

   medium = RandomMedium(
       correlation="matern",
       smoothness=1.5,
       correlation_length=6e-08 * ureg.meter,
       background_refractive_index=1.33,
       refractive_index_std=0.01,
   )

   volume = medium.to_volume(
       grid=Grid(
           shape=(16, 16, 16),
           spacing=2.5e-08 * ureg.meter,
       ),
       seed=42,
   )

   figure = volume.plot_3d(
       backend="matplotlib",
       field="delta_refractive_index",
       slice_indices=(4, 8, 12),
       length_unit="nanometer",
   )

   plt.show()

Structured samples and voxels
-----------------------------

``mode="voxels"`` shows the actual cells with nonzero refractive index contrast. A sphere's
staircase boundary is therefore visible rather than smoothed by interpolation.
Only exposed faces are drawn: interior inclusions are easier to inspect using
slices. For an entirely zero field, voxels display the full uniform box.

.. code-block:: python

   from bornsim.units import ureg

   import matplotlib.pyplot as plt
   from bornsim import Grid, Sphere, StructuredMedium

   sample = StructuredMedium()

   sample.add_background(refractive_index=1.33)

   sphere = Sphere(
       radius=1.5e-07 * ureg.meter,
       refractive_index=1.34,
   )

   sample.add_structures(
       sphere,
   )

   volume = sample.to_volume(
       grid=Grid(
           shape=(24, 24, 24),
           spacing=2e-08 * ureg.meter,
       ),
   )

   figure = volume.plot_3d(
       backend="matplotlib",
       mode="voxels",
       field="refractive_index",
   )

   figure.savefig("sphere-3d.png")

   plt.show()

Random backgrounds under structures
-----------------------------------

The composition methods update an existing ``StructuredMedium``. A random
background fills uncovered voxels; structures overwrite that background,
including its fluctuations, inside their masks. The exterior reference refractive index
is the random statistics' uniform background refractive index.

.. code-block:: python

   from bornsim.units import ureg

   from bornsim import Grid, RandomMedium, Sphere, StructuredMedium

   statistics = RandomMedium(
       refractive_index_std=0.002,
       correlation_length=5e-08 * ureg.meter,
       background_refractive_index=1.33,
       correlation="matern",
       smoothness=1.5,
   )

   sample = StructuredMedium()

   sample.add_background(medium=statistics)

   sphere = Sphere(
       radius=7e-08 * ureg.meter,
       refractive_index=1.345,
   )

   sample.add_structures(
       sphere,
   )

   volume = sample.to_volume(
       grid=Grid(
           shape=(16, 16, 16),
           spacing=2.5e-08 * ureg.meter,
       ),
       seed=42,
   )

   figure = volume.plot_3d(
       field="refractive_index",
       opacity_scale="increasing",
   )

   figure.show()

Interpretation and performance
------------------------------

``field="delta_refractive_index"`` plots fluctuations about the uniform background;
``field="refractive_index"`` plots :math:`n_0+\delta n`. ``field="permittivity"`` displays
the solver's linearized relative permittivity
:math:`n_0^2+2n_0\delta n`, which omits the quadratic fluctuation term.
These views represent the input material, not a calculated electromagnetic
field. Axis labels state the display unit; stored data remains in SI units.
The spatial aspect ratio follows physical dimensions.

Matplotlib slices and voxel geometry show the sampled material. Plotly level
surfaces interpolate between voxel centres and can make curved interfaces
appear smoother than the solver's staircase mask. Neither visualization
replaces checks of voxel refinement for quantitative calculations.

The current grid limit is 32 voxels per axis (32,768 samples). Interactive
volume views reveal high refractive index regions throughout random fields;
slices remain useful for inspecting exact voxel values. Filled voxel views
reveal only the outer faces.
Dense or transparent Matplotlib scenes can have depth-ordering limitations;
see the `mplot3d FAQ <https://matplotlib.org/stable/api/toolkits/mplot3d/faq.html>`_.
Use slices or opaque voxels when checking the sampled geometry. Plotly
rendering depends on browser graphics support and surface count; changing
surface count changes visualization, not numerical resolution.
