# Working on BornSim

- The root `bornsim/` package contains the public Python API and Matplotlib plotting.
- Keep `Source`, `Solver`, and `Result` implementations in `source.py`, `solver.py`, and `results.py`; `api.py` is a convenience import module.
- Analytical first-order formulas live only in `tests/analytical/reference.py`; the public solver runs finite-volume Born calculations. Keep voxel fields in `volume.py`, random-field generation in `medium/random_medium.py` and random sphere collections in `medium/random_spheres.py`, propagation in `green.py`, the `BornSeries` engine in `series.py`, and ensemble statistics in `ensemble.py`.
- Use `Directions` for numerical observation configurations; access its immutable Cartesian array through `.vectors`.
- Use `BornSeries` instances for numerical Born calculations; do not add a free-function wrapper for that engine.
- Validate supplied quantities directly with `bornsim.units.validate_units`; do not reassign fields merely to validate them. This validator permits a positional value.
- Construct quantities by multiplying values by units, such as `0 * ureg.meter`; use `.to(...)` to convert existing quantities.
- Require explicit unit-bearing quantities for all dimensional API inputs, including angles and squared wavenumbers. Refractive indices, their standard deviations and fluctuation arrays must be plain numbers without units. Other dimensionless values and counts may be bare numbers; preserve physical quantities and their supplied units through calculations. Extract explicit magnitudes only at numerical kernels, plotting, or serialization boundaries.
- Require explicit physical configuration: wavelengths, refractive indices, fluctuation statistics, covariance choices, grid shapes and voxel spacing have no arbitrary defaults. StructuredMedium may start unconfigured, but requires an explicit background before voxelization. Keep numerical controls and neutral coordinate conventions distinct.
- Use specific physical names in examples, such as `correlation_length`, `refractive_index_std`, `voxel_offsets_x`, `distance_between_points`, and `synthesis_grid_shape`.
- Preserve SI units in the API, amplitude interference, random seeds, and ensemble uncertainty. Distinguish finite-sample coefficients from infinite-medium transport coefficients.
- Keep the linearized dielectric contrast and Green-tensor self-cell convention explicit. Do not claim universal Born convergence from decreasing terms.
- BornSim constructors, functions, and methods require keyword-only arguments, apart from implicit `self` and `cls`. `add_structures(*structures)` is the explicit exception and takes positional shape objects.
- Format BornSim calls supplying multiple arguments over multiple lines, with one argument per line and a trailing comma.
- Assign constructed objects to named variables before calling their methods; keep solver configuration separate from `result = solver.solve(...)`.
- Build structured examples with `StructuredMedium()` and in-place `add_background`/`add_structures` calls. Define shapes in named variables before adding them, and batch multiple shapes in argument order. Keep `Medium` abstract and distinguish background replacement from additive fluctuations.
- Keep `if`, `elif`, and `else` statement headers on one line; use named conditions when needed.
- Use Matplotlib for default medium plotting and documentation examples. Plotly is an optional explicitly selected backend; core plotting and docs builds must work without it.
- Use explicit `refractive_index` names for the optical quantity, including `background_refractive_index`, `refractive_index_std`, and `delta_refractive_index`. Reserve `index` for array positions.
- Print calculated values with labeled f-strings rather than dictionaries or multiple positional print arguments. Preserve units already attached to results.
- Put a blank line between separate statements and calculation steps, including assignments and calls within loops. Keep imports and related dataclass fields grouped, and comments attached to the statement they explain.
- Use four spaces and follow Ruff formatting. Do not use `from __future__ import annotations`.
- Tests are grouped into `analytical`, `numerical`, and `packaging`; keep independent numerical references and validation coverage.
- Reuse `.venv` and run `make check PYTHON=.venv/bin/python`; UI changes also need a real-browser check.
- Run `make docs`, `make package-check`, `make release-check`, and `git diff --check` for packaging/documentation changes.
- `README.rst` and `docs/source` use reStructuredText. Keep documentation examples runnable.
- Keep version metadata in pyproject.toml, bornsim/_version.py, CITATION.cff, .zenodo.json, and conda.recipe/meta.yaml aligned using the release tools.
- Release commands create commits/tags or push only when explicitly requested. Preserve unrelated work.

## Shared Python style preferences

Apply these conventions to new and edited Python code. Leave unrelated existing code unchanged. Preserve each repository's existing public APIs and project-specific scientific or behavioral requirements.

- Use keyword-only parameters by default for new functions and methods, apart from implicit `self` and `cls`. Preserve existing public calling conventions unless an API change is part of the task.
- Use NumPy-style docstrings for classes, functions, methods, and properties. Document parameters, return values, units, array shapes, and limitations where they apply.
- Keep imports at module scope when practical. Use local imports only when needed to avoid circular dependencies or defer optional, expensive, or side-effectful imports.
- Keep all imports in one contiguous block, without blank lines between standard-library, third-party, and project imports.
- Avoid redundant self-alias imports such as `from package.module import Name as Name`; import the name directly unless the alias clarifies its meaning.
- Keep simple, single-purpose UI component calls on one line when practical, even when a text string is long. Use multiline formatting when several arguments or nested components make the call easier to read.
- Prefer descriptive variable and parameter names over abbreviations, including for local values and callback parameters.
- Avoid extracting a helper used only once unless it improves clarity or separates behavior that belongs in another layer.
- Keep callbacks focused on reading inputs, calling a service or workflow, and returning outputs. Move domain decisions and complicated data conversion into the appropriate service or workflow module.
- Write comments to explain non-obvious reasoning, scientific assumptions, or compatibility constraints; do not use them to narrate code that is already clear.
