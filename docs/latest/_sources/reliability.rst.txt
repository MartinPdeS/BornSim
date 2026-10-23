Reliability and statistical interpretation
===========================================

BornSim separates numerical stability, equation consistency, statistical
uncertainty and physical validity. No single diagnostic certifies all four.

Numerical refinement
--------------------

``ConvergenceReport(results=..., labels=..., tolerance=0.01)`` compares the
highest order of each full angular result. It reports relative changes in
``mu_s`` and ``mu_a``, absolute changes in ``g``, and the worst final-order
equation residual across realizations. ``report.rows`` returns fresh numeric
records; ``report.summary()`` returns readable text. A row marked stable means
only that the compared observables changed less than the chosen tolerance.
Reports omit nonfinite g values, such as the zero-scattering limit.

When both results supply nonzero scattering and the full polar output range,
the report also compares azimuth-averaged phase curves with an L1 solid-angle
difference. It linearly interpolates their sampled curves onto the union of
cos(theta) coordinates and integrates with trapezoidal weights. This diagnostic
can detect changes missed by mu_s and g alone. It is omitted for partial polar
outputs and does not test azimuthal structure or replace quadrature refinement.

Change one control at a time: grid spacing at fixed physical extent, angular
quadrature, Born order, sample extent, or ensemble size. Random fields sampled
on different grids with the same seed are not guaranteed to represent the
same continuous realization. Use deterministic geometry for clean grid studies,
or compare random-medium ensembles with their sampling uncertainties.

``result.residual_norms`` evaluates

.. math::

   r_M = \frac{\|E_{inc}+K\chi E_{[M]}-E_{[M]}\|}{\|E_{inc}\|}.

The array has one entry per cumulative internal-field order, or shape
(realizations, order) for ensembles. It uses the same discretized integral
operator as the solver and includes both incident polarizations. Born-series
recursion makes this equal to the norm of the next omitted field term divided
by the incident-field norm. The final order requires one extra Green operation.
A small residual establishes consistency with the discrete equation, not
accuracy of the mesh or linearized constitutive model. Absorption reported at
scattering order M still uses the internal field through M-1.

See :doc:`auto_examples/validation/convergence_reports`.

Absorption and uncertainty
---------------------------

``Material(refractive_index=..., extinction_coefficient=...)`` assigns passive
loss to regions before voxelization. A background Material fills uncovered
sample voxels with that absorption; the exterior propagation host stays real
and lossless. Structures replace both real index and kappa according to the
overlap policy. Additive fluctuations add their real-index and kappa fields.
Gaussian, exponential and Whittle-Matern media accept a uniform nonnegative
``extinction_coefficient``; their ensemble realizations retain it.

Ensembles report ``mu_a`` and ``mu_a_stderr`` (sample standard deviation divided
by sqrt(N)). One realization has unknown absorption uncertainty, represented
by NaN. These errors quantify sampling uncertainty, not systematic model error.
Existing ``Volume.with_absorption`` remains available for arbitrary voxel maps.

See :doc:`absorption` and
:doc:`auto_examples/validation/absorbing_sphere_balance`.

Coherent and diffuse scattering
--------------------------------

At each order and observation direction, let f_j be the cumulative scattered
amplitude of realization j. Ensemble results expose

.. math::

   D_{coh} = \frac{|\overline f|^2}{V},\qquad
   D_{diff} = \frac{\overline{|f|^2}-|\overline f|^2}{V}.

Here squared magnitudes include the average over incident polarizations and
sum over outgoing Cartesian components. ``coherent_differential`` contains
D_coh; ``diffuse_differential`` contains D_diff; ``differential`` is their sum.
Coherent refers to the mean **scattered** field, excluding the direct beam.
Azimuth averaging happens after this separation, not before amplitude averaging.

These are empirical population-normalized estimates: diffuse scattering has
finite-N factor (N-1)/N bias, and coherent intensity contains a positive
finite-sample variance contribution. With N=1, diffuse is zero by construction,
not evidence of a physically coherent medium. They are not independently
normalized phase functions or automatically bulk transport coefficients.
The components, residuals and absorption statistics survive saving/loading
and azimuth averaging. Old archives without them return None.

See :doc:`auto_examples/results/coherent_diffuse`.

Physical validation
--------------------

Independent automated references check weak-contrast lossless and absorbing
spheres against Mie scattering and absorption cross sections, angular patterns,
and the forward optical theorem. The Mie reference uses Riccati-Bessel series
coefficients independently of BornSim's Green operator. See the
`Mie coefficient equations <https://miepython.readthedocs.io/en/latest/07_algorithm.html>`_.
Tests use an equal-occupied-volume reference radius to separate staircase
volume errors from propagation errors. Grid refinement and nominal geometry
validation remain necessary. BornSim still omits the quadratic dielectric
contrast at every order; these weak-contrast tests do not validate high-index
particles or optically thick transport.
