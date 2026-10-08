Example gallery
===============

Choose a topic below. Each runnable script includes its calculation settings,
explanations, and figures alongside the code that produces them. Medium
volumes and 3D phase surfaces are interactive: drag to rotate, scroll to zoom,
and hover to inspect values. Angular curves and diagnostic plots use Matplotlib.
Inputs use SI-compatible quantities. Analytical examples describe
infinite-medium first-order scattering; numerical Born-series examples describe
finite samples.

Keep each separate figure in its own code cell: call ``plt.show()`` after
plotting, then insert a ``# %%`` separator and narrative comments before
creating the next figure. Multiple axes comparing related data may share a
single figure. This keeps each gallery image beside its plotting code and
displays figures one at a time when running the scripts locally.

Assign each Plotly figure to a named variable so the gallery can embed it.
When running a script, call ``figure.show(renderer="browser")`` unless
``--no-browser`` is supplied. The documentation builder uses that argument
and exports the figures with a shared local Plotly library, so the downloaded
documentation works offline. Keep static slices as secondary diagnostic views.
