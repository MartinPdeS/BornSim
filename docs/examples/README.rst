Example gallery
===============

Choose a topic below. Each runnable script includes its calculation settings,
explanations, and figures alongside the code that produces them. Medium
volumes and 3D phase surfaces use Plotly exclusively. Matplotlib is used for
angular curves and numerical diagnostics; spatial samples are never rendered
as 2D images.
Inputs use SI-compatible quantities. Analytical examples describe
infinite-medium first-order scattering; numerical Born-series examples describe
finite samples.

Keep each separate figure in its own code cell: call ``plt.show()`` after
plotting, then insert a ``# %%`` separator and narrative comments before
creating the next figure. Multiple axes comparing related data may share a
single figure. This keeps each gallery image beside its plotting code and
displays figures one at a time when running the scripts locally.

All 3D calls return Plotly figures. Assign each figure to a named variable;
the gallery embeds it automatically. When running scripts locally, call
``figure.show(renderer="browser")`` unless ``--no-browser`` is supplied.
The documentation builder supplies that argument and exports interactive
figures with a local Plotly library. Controls affect display only, not
numerical resolution.
