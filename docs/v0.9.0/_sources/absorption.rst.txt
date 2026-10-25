Absorbing samples
=================

BornSim supports passive absorption in a finite sample embedded in a real,
nonabsorbing uniform host. Generate any random or structured volume, then call
``volume.with_absorption(extinction_coefficient=kappa)``. The input is a
nonnegative unitless scalar or an array with the voxel grid's shape. An array
allows absorption to be restricted to particular particles or regions.

.. code-block:: python

   from bornsim import GaussianMedium, Grid, Solver, Source
   from bornsim.units import ureg

   grid = Grid(shape=(8, 8, 8), spacing=30 * ureg.nanometer)
   medium = GaussianMedium(
       background_refractive_index=1.33,
       refractive_index_std=0.002,
       correlation_length=60 * ureg.nanometer,
   )
   volume = medium.to_volume(grid=grid, seed=42)
   absorbing = volume.with_absorption(extinction_coefficient=0.001)
   source = Source(wavelength=633 * ureg.nanometer)
   solver = Solver(source=source, order=3)
   result = solver.solve(target=absorbing)
   print(f"Finite-sample absorption coefficient: {result.mu_a}")
   print(f"Absorption cross section: {result.absorption_cross_section}")

Model and interpretation
------------------------

For a runnable comparison through **fourth Born order**, see
:doc:`auto_examples/born_orders/absorbing_random_medium`. It compares absorbing
and nonabsorbing samples, plots scattering and phase functions, and checks
the first-order absorption limit before examining higher orders.

With time dependence exp(-i omega t), positive kappa represents loss.
The solver retains its linearized constitutive model, now with complex contrast
``chi = 2*n0*(delta_n + i*kappa)``. It does not use the exact squared complex
refractive index. Existing real-index plots display the real index field;
``volume.extinction_coefficient`` stores the absorption field separately.

Absorbed power is evaluated from the local dissipation integral. At reported
scattering order M, ``mu_a`` uses the cumulative internal field through order
M-1, matching the field used to form the scattering source at that order.
It averages the configured incident modes (one pure state or two unpolarized
modes) and divides by the entire voxel-box
volume, including nonabsorbing voxels. With only the incident field retained,
a uniform kappa gives ``mu_a = 4*pi*kappa/lambda_vacuum``.

The scattering phase function remains normalized by scattered power alone.
Absorption changes scattering through the complex source and internal field;
it is not a multiplicative attenuation applied to the final phase function.
``mu_a`` is preserved by result archives and azimuth averaging. Existing
archives without absorption data return ``mu_a=None``.

An absorbing exterior host and gain media are not supported. Random-field
media accept ``extinction_coefficient`` directly and their ensemble solves
return ``mu_a`` and ``mu_a_stderr``. Material objects also accept kappa for
structured regions. See :doc:`reliability` for composition rules and examples.
Do not assume a finite Born truncation satisfies the optical theorem exactly.
Check grid and order convergence before interpreting energy budgets, especially
for optically thick or strongly absorbing samples.
