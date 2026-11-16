Medium visualization
====================

Interactive three-dimensional views
-----------------------------------

BornSim uses Matplotlib for two-dimensional plots, including voxel slices,
angular scattering curves, and diagnostics. The 3D medium and phase-function
views use Plotly exclusively. ``Volume.plot_3d()`` returns an interactive
volume, and ``Result.plot_phase_function()`` defaults to an interactive
directional surface. Plotly is included with BornSim; no backend selection is
needed for 3D plots.

Drag to rotate, scroll to zoom, and hover to inspect values. Constructing a
figure opens no window. Call ``figure.show()`` to display it or
``figure.write_html()`` to export an offline interactive view. The gallery
embeds 3D figures alongside the code that constructs them.

The views use Plotly's white canvas with BornSim's navy-to-cyan-to-gold
sequential palette. Signed refractive-index fluctuations use a symmetric
navy-to-gold diverging scale. BornSim retains its Cartesian orientation rather
than MomentRT's downward depth axis.

Random-medium volumes
---------------------

The previews below use the Gaussian and exponential samples from
:doc:`auto_examples/random_media/random_medium`: seed 42, 16 voxels per axis,
25 nm spacing, refractive index standard deviation 0.01, and correlation
length 75 nm. Each HTML view includes Plotly and works offline.

Gaussian covariance
~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <iframe src="_static/random-medium-gaussian.html" title="Gaussian random medium" width="100%" height="560" loading="lazy" style="border:0;"></iframe>

Exponential covariance
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <iframe src="_static/random-medium-exponential.html" title="Exponential random medium" width="100%" height="560" loading="lazy" style="border:0;"></iframe>

.. code-block:: python

   from bornsim import GaussianMedium, Grid
   from bornsim.units import ureg

   grid = Grid(
       shape=(16, 16, 16),
       spacing=25 * ureg.nanometer,
   )

   medium = GaussianMedium(
       background_refractive_index=1.33,
       refractive_index_std=0.01,
       correlation_length=75 * ureg.nanometer,
   )

   volume = medium.to_volume(
       grid=grid,
       seed=42,
   )

   figure = volume.plot_3d(
       field="refractive_index",
       surface_count=16,
       opacity=0.2,
       opacity_scale="increasing",
   )

   figure.write_html(
       file="medium.html",
       include_plotlyjs=True,
   )

Volume views include **Opacity** and **Iso-surface density** sliders.
``mode="isosurface"`` provides isosurfaces with a density slider. Initial
slider values come from ``opacity`` and ``surface_count``, including custom
values. Adjusting controls preserves the camera and changes only the display,
not the material samples or solver resolution.

``opacity_scale="increasing"`` makes lower contours more transparent and
higher contours more opaque, with ``opacity`` as the maximum. With
``field="refractive_index"``, this emphasizes high absolute refractive index.
The ramp uses each sample's displayed range, so equal opacity across samples
need not represent the same refractive index. Opacity does not represent
material absorption. Uniform fields support volume rendering; an isosurface
requires a nonuniform field.

Exact voxel cross-sections
---------------------------

``Volume.plot_slice()`` uses Matplotlib to display exact sampled values with
physical axis labels, equal spatial aspect, and a shared color scale. Use
``normal="x"``, ``"y"``, or ``"z"`` to choose orientation, and ``index`` to
select a voxel plane. The default is the middle z voxel; on an even grid its
centre is at positive half-spacing, rather than zero.

.. code-block:: python

   slice_figure = volume.plot_slice(
       normal="z",
       field="refractive_index",
       length_unit="nanometer",
   )

   slice_figure.savefig("medium-slice.png")

Plotly contours interpolate between voxel centres. Use slices to inspect
staircase boundaries and exact voxel values; smooth contours do not replace
checks of grid refinement. The 3D spatial aspect follows physical dimensions.
All samples are retained, up to the grid limit of 32 voxels per axis.
Rendering cost also depends on surface count and browser graphics support.

Directional phase surfaces and cross sections
---------------------------------------------

``Result.plot_phase_function(view="3d")`` displays probability density per
steradian in laboratory coordinates, with density encoded by surface radius
and color. It retains sampled polar and azimuth angles, preserving directional
interference. An explicitly azimuth-averaged result instead produces a surface
of revolution labelled as such. A cut alone has no solid-angle normalization.
Increase angular sampling to resolve directional structure.

.. code-block:: python

   from bornsim import Solver, Source

   source = Source(wavelength=633 * ureg.nanometer)

   solver = Solver(
       source=source,
       order=3,
   )

   result = solver.solve(target=volume)

   phase_figure = result.plot_phase_function(view="3d")

   phase_figure.write_html(
       file="phase-function.html",
       include_plotlyjs=True,
   )

``Result.plot_cross_section()`` uses a Plotly 3D scene and restores
finite-sample area units using the recorded sample volume. Ensemble standard errors use the
same conversion. Cumulative curves preserve amplitude interference;
``terms=True`` selects isolated terms, which cannot be summed to reconstruct
cumulative intensities.

``field="permittivity"`` displays the solver's linearized relative
permittivity :math:`n_0^2+2n_0\delta n`, omitting the quadratic fluctuation term.
Medium views display input material rather than a calculated electromagnetic
field. Spatial display units do not change stored quantities. Numerical
coefficients remain finite-sample cross sections divided by the entire voxel
box volume, rather than intrinsic infinite-medium transport coefficients.
