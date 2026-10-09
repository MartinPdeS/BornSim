Development and releases
========================

Use Python 3.11 or newer and an isolated virtual environment.

.. code-block:: console

    python -m venv .venv
    make editable PYTHON=.venv/bin/python
    make check PYTHON=.venv/bin/python
    make docs PYTHON=.venv/bin/python
    make package-check PYTHON=.venv/bin/python
    make release-check PYTHON=.venv/bin/python

The default test suite includes analytical, numerical and packaging tests
and generates terminal and HTML coverage reports.

Version metadata is explicit and checked across project metadata, package,
citation, Zenodo and Conda files. After committing changes, an initial release
can be created with ``make tag VERSION=v0.1.0``. Later,
``make release patch`` (or ``minor`` or ``major``) aligns release metadata,
commits it, creates an annotated tag and pushes HEAD and that tag.

A repository remote and publishing credentials must be configured separately.
The PyPI workflow requires ``PYPI_API_TOKEN``. Conda verification is available
through a manual workflow; channel upload is not configured. See
``CONTRIBUTING.md`` for further instructions.

Code readability
----------------

Spell out ``refractive_index`` in names for the optical quantity, including
``background_refractive_index``, ``refractive_index_std`` and
``delta_refractive_index``. Reserve ``index`` for array positions.
Separate statements and calculation steps with a blank line, including
assignments and calls inside loops. Keep imports and related dataclass fields
grouped, and keep comments attached to the statements they explain.

Internal rendering and archives
-------------------------------

Keep Volume and Result responsible for physical data, units and validation.
Their public plotting and save/load methods retain their documented keyword
arguments and delegate to internal helpers:

* ``_VolumePlotter`` in ``bornsim/_volume_plotting.py`` owns slices, 3D
  geometry, display units, color scales and backend selection.
* ``_ResultPlotter`` in ``bornsim/_result_plotting.py`` owns scattering,
  phase-function and field-norm figures.
* ``_ResultArchive`` in ``bornsim/_archives.py`` owns the shared SI archive
  schema, NPZ serialization and safe loading without pickle. Results are
  immutable and validated at construction; loading validates the archive.

Import Matplotlib and Plotly only when a rendering method needs
that backend. Rendering must preserve the stored data and SI units. Keep
normalization mathematics on AngularData and angular integration on
AngularSampling; renderers display those quantities without recomputing
scattering or discarding directional asymmetry.

Documentation versions
----------------------

The header version dropdown follows the same layout as PyMieSim. ``latest``
tracks ``master``; tagged releases are published under their own
``docs/vX.Y.Z`` directories. A branch build updates only ``latest``, so
later development changes do not overwrite released documentation.

The deployment retains the five newest release versions plus ``latest``
and regenerates ``version_switcher.json`` on the ``documentation_page``
branch. Tagged builds set the ``tag`` environment variable to the release
tag so the dropdown selects the matching version. Local builds default to
``latest``; to check a release label locally, run::

   tag=v0.2.3 make docs PYTHON=.venv/bin/python

Theory figures and reproducibility
----------------------------------

Theory figures come from runnable scripts in ``docs/examples`` and are captured
by Sphinx Gallery. Keep captions explicit about the model, sampling and limits.
The translation illustration asserts equality of first-order intensities for
an integer-voxel shift. Random-field covariance compares independent-realization
estimates with both the exact discrete synthesis expectation and the continuum
covariance convention. Its errors use realizations as independent samples.

Numerical tests compare FFT Green propagation against an independently assembled
nonperiodic matrix, Born terms against a direct linear solve, and first-order
amplitudes against independent volume integrals. These checks complement the
figures; visually decreasing terms alone cannot validate a simulation.
