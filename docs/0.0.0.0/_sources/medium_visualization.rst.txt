Medium visualization
====================

Voxel cross-sections
--------------------

``Volume.plot_slice()`` builds a Matplotlib figure with physical axis labels,
an equal spatial aspect ratio, and a colorbar. It handles axis orientation and
voxel-edge extents automatically. The default plane is the middle z voxel;
on an even grid its centre is at positive half-spacing, not at zero.

.. code-block:: python

   from bornsim import Grid, Sphere, StructuredMedium

   structure = StructuredMedium()
   structure.add_background(index=1.33)
   sphere = Sphere(
       radius=150e-9,
       index=1.34,
   )

   structure.add_structures(
       sphere,
   )

   volume = structure.to_volume(
       grid=Grid(
           shape=(24, 24, 24),
           spacing=20e-9,
       ),
   )
   figure = volume.plot_slice(
       normal="z",
       field="index",
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

   from bornsim import Solver, Source

   solver = Solver(
       source=Source(wavelength=633e-9),
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
in SI metres or as a length quantity. For example,
``centre=np.array([40, 20, 0]) * ureg.nanometer`` shifts a sphere by 40 nm along x
and 20 nm along y. Keep its radius plus the absolute offset within each box
half-width to avoid clipping. Offsets aligned with the voxel spacing translate
the existing sampled mask; other offsets resample the interface.

The :doc:`auto_examples/dielectric_sphere` example also plots the sphere's
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

Three-dimensional media
-----------------------

``Volume.plot_3d()`` returns a Matplotlib figure by default. It needs no
additional dependency and works for generated random fields, structured
samples, and custom voxel arrays. Use ``plt.show()`` to display figures or
``figure.savefig()`` to export PNG, SVG, or PDF. Rotation is available with
an interactive Matplotlib backend; the documentation gallery displays static
images. Constructing a figure opens no window.

Random fields and orthogonal slices
-----------------------------------

The default ``mode="slices"`` shows one voxel plane along each axis. All selected
cells are retained, with voxel-edge extents and a shared color scale. Select
specific planes with ``slice_indices``; otherwise the middle voxels are used.

.. code-block:: python

   import matplotlib.pyplot as plt
   from bornsim import Grid, RandomMedium

   medium = RandomMedium(
       correlation="matern",
       smoothness=1.5,
       correlation_length=60e-9,
   )
   volume = medium.to_volume(
       grid=Grid(
           shape=(16, 16, 16),
           spacing=25e-9,
       ),
       seed=42,
   )
   figure = volume.plot_3d(
       field="delta_index",
       slice_indices=(4, 8, 12),
       length_unit="nanometer",
   )
   plt.show()

Structured samples and voxels
-----------------------------

``mode="voxels"`` shows the actual cells with nonzero index contrast. A sphere's
staircase boundary is therefore visible rather than smoothed by interpolation.
Only exposed faces are drawn: interior inclusions are easier to inspect using
slices. For an entirely zero field, voxels display the full uniform box.

.. code-block:: python

   import matplotlib.pyplot as plt
   from bornsim import Grid, Sphere, StructuredMedium

   sample = StructuredMedium()
   sample.add_background(index=1.33)
   sphere = Sphere(
       radius=150e-9,
       index=1.34,
   )
   sample.add_structures(
       sphere,
   )
   volume = sample.to_volume(
       grid=Grid(
           shape=(24, 24, 24),
           spacing=20e-9,
       ),
   )
   figure = volume.plot_3d(
       mode="voxels",
       field="index",
   )
   figure.savefig("sphere-3d.png")
   plt.show()

Random backgrounds under structures
-----------------------------------

The composition methods update an existing ``StructuredMedium``. A random
background fills uncovered voxels; structures overwrite that background,
including its fluctuations, inside their masks. The exterior reference index
is the random statistics' uniform background index.

.. code-block:: python

   from bornsim import Grid, RandomMedium, Sphere, StructuredMedium

   statistics = RandomMedium(
       index_std=0.002,
       correlation_length=50e-9,
   )
   sample = StructuredMedium()
   sample.add_background(medium=statistics)
   sphere = Sphere(
       radius=70e-9,
       index=1.345,
   )

   sample.add_structures(
       sphere,
   )
   volume = sample.to_volume(
       grid=Grid(
           shape=(16, 16, 16),
           spacing=25e-9,
       ),
       seed=42,
   )
   figure = volume.plot_3d(
       mode="slices",
       field="index",
   )
   import matplotlib.pyplot as plt

   plt.show()

Optional Plotly backend
-----------------------

For browser rotation, hover values, volume level surfaces, and interactive HTML
exports, install the optional ``BornSim[visualization]`` extra. It is not needed
for the standard examples, documentation build, or Matplotlib plots.

.. code-block:: console

   .venv/bin/python -m pip install -e ".[visualization]"

.. code-block:: python

   figure = volume.plot_3d(
       backend="plotly",
       mode="volume",
       field="delta_index",
       surface_count=6,
       opacity=0.15,
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

Interpretation and performance
------------------------------

``field="delta_index"`` plots fluctuations about the uniform background;
``field="index"`` plots :math:`n_0+\delta n`. ``field="permittivity"`` displays
the solver's linearized relative permittivity
:math:`n_0^2+2n_0\delta n`, which omits the quadratic fluctuation term.
These views represent the input material, not a calculated electromagnetic
field. Axis labels state the display unit; stored data remains in SI units.
The spatial aspect ratio follows physical dimensions.

Matplotlib slices and voxel geometry show the sampled material. Plotly level
surfaces interpolate between voxel centres and can make curved interfaces
appear smoother than the solver's staircase mask. Neither visualization
replaces checks of voxel refinement for quantitative calculations.

The current grid limit is 32 voxels per axis (32,768 samples). Random fields
are usually clearer as slices; filled voxel views reveal only the outer faces.
Dense or transparent Matplotlib scenes can have depth-ordering limitations;
see the `mplot3d FAQ <https://matplotlib.org/stable/api/toolkits/mplot3d/faq.html>`_.
Use opaque voxels (the default) and slices for routine inspection. Plotly
rendering depends on browser graphics support and surface count; changing
surface count changes visualization, not numerical resolution.
