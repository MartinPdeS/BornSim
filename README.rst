.. image:: docs/source/_static/logo.png
   :alt: BornSim logo
   :width: 420

.. documentation-content-start

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Badge
     - Status
   * - Python versions
     - |python|
   * - Documentation
     - |docs|
   * - Continuous integration
     - |tests|
   * - Static quality checks
     - |quality|
   * - Test coverage
     - |coverage|
   * - Latest release tag
     - |release|
   * - Package publication
     - |publication|
   * - License
     - |license|

BornSim
========

A Python package for vector Born-series scattering from continuous, isotropic refractive-index fluctuations, such as idealized tissue models.

Install
-------

Requires Python 3.11 or newer.

.. code-block:: console

    python -m venv .venv
    .venv/bin/python -m pip install -e .


BornSim constructor, function, and method arguments are keyword-only, except
``add_structures(*structures)``, which takes positional shape objects.
Calls with multiple arguments use one argument per line and a trailing comma.

The Python API accepts TypedUnit quantities; dimensional inputs require explicit units. The wavelength is the **vacuum** wavelength. Outputs are angular differential scattering, μs, anisotropy g, and μs′, with scattering coefficients in inverse metres.

.. code-block:: python

    import matplotlib.pyplot as plt
    from bornsim import EnsembleSampling, Grid, RandomMedium, Source, Solver
    from bornsim.units import ureg

    grid = Grid(
        shape=(8, 8, 8),
        spacing=50 * ureg.nanometer,
    )

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100 * ureg.nanometer,
        correlation="gaussian",
    )

    solver = Solver(
        source=Source(wavelength=633 * ureg.nanometer),
        order=3,
    )

    result = solver.ensemble(
        medium=medium,
        grid=grid,
        ensemble_sampling=EnsembleSampling(
            realizations=4,
            seed=42,
        ),
    )

    print(
        f"mu_s = {result.mu_s.to('1 / millimeter')}, "
        f"g = {result.g}, "
        f"mu_s_prime = {result.mu_s_prime.to('1 / millimeter')}"
    )

    result.plot()

    plt.show()

``Source`` describes the supported unpolarized plane wave propagating along +z.
``RandomMedium`` defines numerical random-field statistics.
``Grid`` defines spatial shape and spacing once, and is shared by
``medium.to_volume(grid=grid)`` and ``solver.ensemble(grid=grid, ...)``.
Every ``Volume`` retains its grid. ``AngularSampling`` groups output polar
angles, polar integration nodes and azimuth resolution; pass it as
``Solver(sampling=sampling, ...)`` to share settings across calculations.

``Solver.solve(target=volume)`` computes full directional scattering and
solid-angle integrals by default, providing normalized 3D phase functions
for one fixed sample. Use ``solve_cut`` with explicit ``angles`` for an x-z
angular cut, or explicit unit ``directions`` for arbitrary observations. Neither cut
supplies integrated coefficients. Angle quantities may use degrees or radians.
``Solver.ensemble(medium=medium, grid=grid, ensemble_sampling=...)`` averages
independent seeded random volumes and returns standard errors. It accepts
random media and structures with a random background. Deterministic
structures generate a volume and use ``solve`` directly.

These calculations return a ``Result`` whose ``differential`` array has
shape (order, polar angle, azimuth) for full calculations. Its ``plot()`` returns a Matplotlib figure, with
one-standard-error bars for ensembles; ``plot(terms=True)`` shows isolated
numerical terms without interference or error bars. Call ``plt.show()`` to display the figures or
``figure.savefig("scattering.svg")`` to export a figure. Full-volume and
ensemble coefficients describe finite samples and remain distinct from
infinite-medium analytical ones.
Undefined anisotropy is ``NaN`` in this object API. Existing functions and
their return formats remain available.

``result.phase_function`` normalizes every sampled direction using its full
solid-angle integral. Full ``differential`` data have shape
``(order, polar angle, azimuth)``. Curves select one sampled meridian;
``result.azimuth_average()`` explicitly requests averaged intensities.
``result.plot_phase_function(view="3d")`` preserves directional asymmetry.
For notebook rotation, install the notebook extra and use
``%matplotlib widget``. ``result.amplitudes`` retains complex amplitudes with
the same observation axes followed by polarization and Cartesian axes.
Use ``solver.solve_cut`` for unnormalized angular cuts, and
``result.plot_cross_section()`` for finite-sample area per steradian without
passing the original Volume. Saved results retain the physical sample volume.

``result.plot_field_norms()`` shows numerical field terms per realization.
See ``docs/examples/results/result_plots.py`` for direct plotting examples in the
Sphinx Gallery documentation.

Results validate array shapes and physical ranges at construction.
``result.provenance`` records calculation settings, versions, seeds, grid
and quadrature information. ``result.save(path="scattering.npz")`` and
``Result.load(path="scattering.npz")`` preserve units, complex amplitudes,
uncertainty, and metadata without pickle. Input voxel fields are not stored;
retain manually supplied fields separately. See the numerical validation
gallery for independent refinement and sampling checks.

Units
-----

``bornsim.units`` exposes TypedUnit's shared ``ureg``, ``Quantity``, ``Length``,
``Angle``, ``Dimensionless``, and ``RefractiveIndex``, using the same registry
as PyMieSim. Physical inputs accept compatible units and reject incompatible
dimensions. Lengths and angles require explicit units. Refractive indices,
their standard deviations and fluctuation arrays require plain numbers without
units, including when passed to Material or shape constructors.
Physical configuration has no arbitrary defaults. Supply wavelengths,
refractive indices, fluctuation statistics, covariance choice, grid shape and
voxel spacing explicitly. ``StructuredMedium()`` starts as an empty builder;
call ``add_background(...)`` before generating a volume. Numerical controls
retain defaults, and zero centres and identity shape rotations define the
neutral coordinate conventions.

Physical settings and coordinates retain quantities in the supplied units.
Validation checks units and scalar shape without converting them. FFT and
Fourier kernels extract explicit SI magnitudes where numeric arrays are needed.

``Source.wavelength`` and every numerical array in ``Result`` are quantities.
Angles, amplitudes, scattering coefficients and uncertainties retain physical
units. Saved archives use radians, metres and inverse metres explicitly. Anisotropy, direction vectors, and relative
field norms are dimensionless. Use ``.to("unit")`` to convert and
``.magnitude`` to retrieve an array. Plots convert to degrees and SI
scattering units explicitly, including uncertainty bars.

Analytical functions return unit-bearing scattering coefficients.
``random_volume`` retains unit-bearing voxel spacing and coordinates.
The low-level Born and ensemble kernels document their numeric SI outputs.


Physical model
--------------

Real-valued refractive index fluctuations are linearized as δε ≈ 2 n₀ δn. For an unpolarized incident wave, the differential scattering coefficient is k₀⁴ Φε(q) (1 + cos²θ)/(32π²), where q = 2 n₀ k₀ sin(θ/2), k₀ = 2π/λvac and Φε is the three-dimensional Fourier transform of the dielectric covariance without a Fourier normalization prefactor.

The refractive index covariance is σn² exp(−r²/(2ℓ²)) for the Gaussian model and σn² exp(−r/ℓ) for the exponential model. These definitions matter when comparing correlation lengths between publications. Gauss–Legendre quadrature integrates over solid angle. Increase ``quadrature_order`` to check convergence for strongly forward-peaked scattering.

The analytical solver is a first-order, single-scattering model. The numerical solver includes repeated interactions through the chosen Born order. No slab-transmission observable or particle-packing generator is provided;
fixed particle configurations can be composed and propagated numerically. A dense medium can still have weak fluctuations; density alone does not establish Born validity. Contrast, correlation length, wavelength, and propagation distance matter. Decreasing Born terms do not certify convergence.

Scientific references:

- `Nonscalar elastic light scattering from continuous media in the Born approximation <https://pmc.ncbi.nlm.nih.gov/articles/PMC3839346/>`_.
- `Accuracy of the Born approximation in calculating the scattering coefficient of biological continuous random media <https://opg.optica.org/ol/abstract.cfm?uri=ol-34-17-2679>`_.

Check
-----

.. code-block:: console

    .venv/bin/python -m pytest tests


Tests cover the analytic short-correlation limit, contrast scaling, independent angular integration, and input validation.

Numerical media and geometry
----------------------------

``RandomMedium`` generates numerical Gaussian random fields with Gaussian,
exponential, or Whittle–Matérn spatial covariance. Matérn ``smoothness`` uses
the convention ``x = sqrt(2 * nu) * r / correlation_length``; ``nu = 0.5``
reproduces exponential covariance. ``Medium`` is an abstract base class;
instantiate ``RandomMedium`` or ``StructuredMedium`` and call ``to_volume()``.
The existing analytical statistics class is named ``AnalyticalMedium``.
Old ``Medium(...)`` calls must be replaced; choose ``correlation="gaussian"``
for the analytical Gaussian model. Choose the covariance family explicitly. Matérn models also require smoothness.

``StructuredMedium`` voxelizes ordered ``Layer``, ``Sphere``, ``Ellipsoid``,
``Box``, and ``Cylinder`` regions. Regions specify absolute indices; later
regions replace earlier ones in overlaps. Layers are finite slabs clipped
to the box, with a uniform background outside the sample.

.. code-block:: python

    from bornsim.units import ureg

    from bornsim import Grid, RandomMedium, Layer, Sphere, StructuredMedium
    from bornsim.media import random_volume
    from bornsim import Solver, Source

    random_sample = random_volume(
        medium=RandomMedium(
            correlation="matern",
            smoothness=1.5,
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
        ),
        grid=Grid(
            shape=(12, 12, 12),
            spacing=3e-08 * ureg.meter,
        ),
        seed=42,
    )

    structure = StructuredMedium()

    structure.add_background(refractive_index=1.33)

    layer = Layer(
        lower=-1.8e-07 * ureg.meter,
        upper=0 * ureg.meter,
        refractive_index=1.34,
    )

    sphere = Sphere(
        radius=7e-08 * ureg.meter,
        refractive_index=1.345,
        centre=(0, 0, 4e-08) * ureg.meter,
    )

    structure.add_structures(
        layer,
        sphere,
    )

    structured_sample = structure.to_volume(
        grid=Grid(
            shape=(12, 12, 12),
            spacing=3e-08 * ureg.meter,
        ),
    )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        order=3,
    )

    result = solver.solve(target=structured_sample)

``Solver`` delegates numerical propagation to ``BornSeries``, which reuses
its Green operator across compatible volumes. Ensemble calculations build
one engine for all realizations. Voxel fields, random generation, Green
propagation, Born iteration, and ensemble statistics live in separate modules.

The numerical method uses a vector volume integral on cubic voxels, rather
than finite elements. The documentation's numerical theory section derives
field synthesis, geometry masks, Born orders, the Green self cell, and
scattering normalization.

Three-dimensional media
-----------------------

``Volume.plot_3d()`` defaults to an interactive Plotly volume.
``result.plot_phase_function(view="3d")`` defaults to a Plotly surface.
Select ``backend="matplotlib"`` explicitly for static 3D figures, orthogonal
slices, or sampled voxel geometry such as spheres.

.. code-block:: python

    from bornsim.units import ureg

    from bornsim import Grid, RandomMedium

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    volume = medium.to_volume(
        grid=Grid(
            shape=(16, 16, 16),
            spacing=2.5e-08 * ureg.meter,
        ),
        seed=42,
    )

    figure = volume.plot_3d(
        field="refractive_index",
        opacity_scale="increasing",
        length_unit="nanometer",
    )

    figure.show()

Matplotlib figures rotate with an interactive backend and can be exported
with ``figure.savefig()``. The documentation gallery embeds interactive 3D volumes and phase surfaces.
Plotly is included with BornSim and is the default for 3D views. Call
``figure.show()`` for browser interaction or ``figure.write_html()`` to export
an interactive view. Angular and polar result plots use Matplotlib.
See the medium visualization documentation for units and rendering conventions.

Numerical Born series
---------------------

Generate a seeded random volume and evaluate cumulative Born orders, or average independent realizations with one-standard-error estimates. Isolated term intensities exclude interference and must not be summed to obtain cumulative intensity.

.. code-block:: python

    from bornsim.units import ureg

    import numpy as np
    from bornsim import Directions, EnsembleSampling, Grid, RandomMedium
    from bornsim.series import BornSeries
    from bornsim.media import random_volume
    from bornsim.ensemble import ensemble_scattering

    medium = RandomMedium(
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
        correlation="gaussian",
        background_refractive_index=1.33,
    )

    volume = random_volume(
        medium=medium,
        grid=Grid(
            shape=(12, 12, 12),
            spacing=5e-08 * ureg.meter,
        ),
        seed=42,
    )

    directions = Directions(vectors=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    # directions.vectors is the immutable Cartesian array.

    engine = BornSeries(
        grid=volume.grid,
        background_refractive_index=volume.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=directions,
        order=3,
    )

    result = engine.solve(volume=volume)

    # result.amplitudes[j] is the (j+1)-th term, in metres.
    # result.differential[j] includes amplitude interference through order j+1.
    ensemble = ensemble_scattering(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
        order=3,
        ensemble_sampling=EnsembleSampling(
            realizations=4,
            seed=42,
        ),
        shape=(12, 12, 12),
        spacing=50e-9 * ureg.meter,
    )

The API supports orders 1–12. Work limits bound synchronous calculations. Random fields have a **Gaussian probability distribution**, with Gaussian, exponential or Whittle–Matérn spatial covariance. Those are distinct choices. A spectral generator samples a doubled periodic box and crops it; ensemble point variance is normalized, but individual samples retain their random means and variances. Finite resolution truncates the spectrum, particularly for exponential covariance, and finite synthesis boxes approximate the continuum statistics.

The solver uses the outgoing dyadic background Green tensor, incident propagation along +z and an incoherent average over x/y incident polarizations. Zero-padded FFT convolution evaluates the finite-volume interactions without periodic propagation. The singular self cell uses an equal-volume sphere with the longitudinal contact term; off-diagonal cells use midpoint quadrature. This is an approximate voxel discretization, not a validated full-wave tissue simulator.

All orders use the same **linearized dielectric contrast** δε = 2 n₀ δn as the analytical model. They are perturbation orders in this dielectric contrast; the omitted δn² constitutive term is not restored by increasing Born order. Generated nonpositive dielectric values are rejected.

A first-order scattered amplitude is calculated from the incident field; subsequent amplitudes use successive applications of the Green operator. Amplitudes are summed before squaring. Ensemble averaging then averages intensities, never random amplitudes. The numerical coefficients are finite-sample cross sections divided by volume, and include the total scattered intensity; they are not automatically intrinsic infinite-medium transport coefficients. The mean anisotropy is calculated from averaged angular moments.

Numerical integration uses ``polar_samples`` Gauss nodes (default 32, range 16–128) and an azimuth count set by ``azimuth_samples`` (default 8). Plot angles are separately configurable. Check angular convergence for electrically large samples; increase ``polar_samples`` and ``azimuth_samples`` before using such samples quantitatively. Check grid refinement, sample size, synthesis-box effects and realization count as well. A single realization has unknown standard error (``NaN``). Growing field terms trigger a diagnostic; decreasing terms do not certify convergence. Modified/convergent Born series are not implemented.

Higher-order references:

- `Electromagnetic Born-series formulation and amplitude scaling <https://doi.org/10.1093/ptep/ptae008>`_.
- `Discrete-dipole and volume-integral discretization background <https://collaborate.princeton.edu/en/publications/discrete-dipole-approximation-for-scattering-calculations/>`_.
- `Modified convergent Born series, a distinct future solver <https://arxiv.org/abs/1601.05997>`_.

Regression checks compare open-boundary FFT interactions against an independently assembled dyadic matrix, compare orders 1–3 against a direct linear solve, verify per-order contrast scaling and interference, test first-order grid refinement, check random-field statistics, and verify ensemble means and standard errors.

Repository layout
-----------------

.. code-block:: text

    bornsim/             Public API, numerical solvers, Matplotlib plots
    tests/analytical/    First-order analytical regression tests
    tests/numerical/     Independent Born-series and random-field checks
    tests/packaging/     Metadata and release-tool tests
    docs/source/         Sphinx documentation
    docs/examples/       Runnable examples
    development/         Exploratory work excluded from distributions
    tools/               Release and wheel-verification commands
    conda.recipe/        Conda package recipe
    .github/workflows/   Quality, tests, documentation and publication checks

Development
-----------

.. code-block:: console

    make editable PYTHON=.venv/bin/python
    make check PYTHON=.venv/bin/python
    make docs PYTHON=.venv/bin/python
    make package-check PYTHON=.venv/bin/python
    make release-check PYTHON=.venv/bin/python

See ``CONTRIBUTING.md`` for development and release instructions. The package
uses the MIT license in ``LICENSE``; software citation metadata is provided in
``CITATION.cff``. The repository currently has no configured GitHub remote.

API configuration and data ownership
------------------------------------

Use Grid, AngularSampling and EnsembleSampling to define spatial, angular and
realization settings. Legacy individual configuration keywords are deprecated.
EnsembleSampling accepts consecutive seeds or an explicit list of distinct
seeds. Material objects share a real absolute refractive index across shapes.
Results and their arrays are read-only; StructuredMedium remains mutable.
``result.angular`` owns the directional data and is reused on every access.
``result.meridian(azimuth=90 * ureg.degree)`` selects sampled directions using
physical angles. Exact selection is the default; nearest selection must be
requested explicitly and never interpolates or averages intensities.

Theory and simulated figures
----------------------------

The documentation's theory section derives the vector volume integral, Born
orders, coherent amplitudes, phase normalization and ensemble uncertainty.
Simulated figures connect these equations to random-field covariance,
translation invariance, two-particle interference and voxel refinement.
The figures link to runnable examples and state their numerical limitations.

.. |python| image:: https://img.shields.io/badge/Python-3.11%2B-3776AB.svg
   :alt: Python 3.11 or newer
   :target: https://www.python.org/
.. |docs| image:: https://github.com/MartinPdeS/BornSim/actions/workflows/deploy_documentation.yml/badge.svg
   :alt: Documentation build status
   :target: https://martinpdes.github.io/BornSim/docs/latest/
.. |tests| image:: https://github.com/MartinPdeS/BornSim/actions/workflows/tests.yml/badge.svg
   :alt: Test status
   :target: https://github.com/MartinPdeS/BornSim/actions/workflows/tests.yml
.. |quality| image:: https://github.com/MartinPdeS/BornSim/actions/workflows/quality.yml/badge.svg
   :alt: Static quality check status
   :target: https://github.com/MartinPdeS/BornSim/actions/workflows/quality.yml
.. |coverage| image:: https://raw.githubusercontent.com/MartinPdeS/BornSim/python-coverage-comment-action-data/badge.svg
   :alt: Test coverage
   :target: https://github.com/MartinPdeS/BornSim/actions/workflows/deploy_coverage.yml
.. |release| image:: https://img.shields.io/github/v/tag/MartinPdeS/BornSim.svg
   :alt: Latest release tag
   :target: https://github.com/MartinPdeS/BornSim/tags
.. |publication| image:: https://github.com/MartinPdeS/BornSim/actions/workflows/deploy_release.yml/badge.svg
   :alt: Package publication status
   :target: https://github.com/MartinPdeS/BornSim/actions/workflows/deploy_release.yml
.. |license| image:: https://img.shields.io/github/license/MartinPdeS/BornSim.svg
   :alt: MIT license
   :target: https://github.com/MartinPdeS/BornSim/blob/master/LICENSE
