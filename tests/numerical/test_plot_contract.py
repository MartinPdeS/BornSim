import builtins
import numpy as np
import pytest
from bornsim import Result, Source
from bornsim.units import ureg


@pytest.mark.parametrize("wavelength", [0, -1, np.nan, np.inf])
def test_invalid_source(wavelength):
    with pytest.raises(ValueError, match="wavelength"):
        Source(wavelength=wavelength * ureg.meter)


@pytest.mark.parametrize("backend", [None, "plotly"])
def test_3d_uniform_phase_is_a_sphere_and_polar_cut_is_closed(backend):
    radius = 1 / (4 * np.pi)

    result = Result(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="volume",
        differential=np.full((1, 3), 2 * radius) / ureg.meter / ureg.steradian,
        angles=[180, 90, 0] * ureg.degree,
        mu_s=np.array([2]) / ureg.meter,
        azimuth_averaged=True,
    )

    default_surface = result.plot_phase_function()

    assert default_surface.data[0].type == "surface"

    surface = result.plot_phase_function(
        view="3d",
        backend=backend,
    )

    trace = surface.data[0]

    assert trace.type == "surface"

    np.testing.assert_allclose(np.sqrt(trace.x**2 + trace.y**2 + trace.z**2), radius)

    np.testing.assert_allclose(trace.surfacecolor, radius)

    assert trace.colorbar.title.text == "p (sr⁻¹)"

    assert surface.layout.scene.aspectmode == "data"
    for data in (result, result.angular):
        with pytest.raises(ValueError, match="only backend='plotly'"):
            data.plot_phase_function(
                view="3d",
                backend="matplotlib",
            )

    polar = result.plot_phase_function(view="polar").axes[0].lines[0]

    assert polar.get_xdata()[0] == 0

    assert polar.get_xdata()[-1] == 2 * np.pi

    np.testing.assert_allclose(polar.get_ydata(), radius)

    angular = result.plot_phase_function(view="angular", log_y=True).axes[0].lines[0]

    np.testing.assert_allclose(angular.get_xdata(), [0, 90, 180])

    np.testing.assert_allclose(angular.get_ydata(), radius)


def test_phase_rejects_zero_normalization():
    result = Result(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="volume",
        differential=np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1, np.pi] * ureg.radian,
        mu_s=[0] * (1 / ureg.meter),
        azimuth_averaged=True,
    )

    with pytest.raises(ValueError, match="positive, finite"):
        result.phase_function


def test_two_dimensional_rendering_without_importing_plotly(monkeypatch):
    """Render 2D views without importing the 3D backend."""

    original_import = builtins.__import__

    def without_plotly(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("plotly"):
            raise ImportError("Plotly is unavailable")

        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_plotly)

    source = Source(wavelength=633 * ureg.nanometer)

    result = Result(
        source=source,
        kind="volume",
        differential=np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian),
        angles=[0, 90, 180] * ureg.degree,
        mu_s=[4 * np.pi] * (1 / ureg.meter),
        sample_volume=1 * ureg.micrometer**3,
        azimuth_averaged=True,
        field_norms=[0.01],
    )

    for data in (result, result.angular):
        figure = data.plot()

        figure.canvas.draw()

        for view in ("angular", "polar"):
            figure = data.plot_phase_function(view=view)

            figure.canvas.draw()

        with pytest.raises(ImportError):
            data.plot_phase_function(
                view="3d",
                backend="plotly",
            )

    figure = result.plot_cross_section()

    figure.canvas.draw()
