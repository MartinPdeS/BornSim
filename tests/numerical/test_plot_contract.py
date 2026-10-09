import numpy as np
import pytest
from bornsim import Result, Source
from bornsim.units import ureg


@pytest.mark.parametrize("wavelength", [0, -1, np.nan, np.inf])
def test_invalid_source(wavelength):
    with pytest.raises(ValueError, match="wavelength"):
        Source(wavelength=wavelength * ureg.meter)


@pytest.mark.parametrize("backend", [None, "matplotlib"])
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

    surface = result.plot_phase_function(
        view="3d",
        backend=backend,
    )

    if backend is None:
        trace = surface.data[0]

        assert trace.type == "surface"

        np.testing.assert_allclose(np.sqrt(trace.x**2 + trace.y**2 + trace.z**2), radius)

        np.testing.assert_allclose(trace.surfacecolor, radius)

        assert trace.colorbar.title.text == "p (sr⁻¹)"

        assert surface.layout.scene.aspectmode == "data"
    else:
        axis = surface.axes[0]

        assert axis.name == "3d"

        np.testing.assert_allclose(axis.get_box_aspect(), np.repeat(axis.get_box_aspect()[0], 3))

        colors = axis.collections[0].get_facecolors()

        np.testing.assert_allclose(colors, np.broadcast_to(colors[0], colors.shape))

        assert surface.axes[1].get_ylabel() == "p (sr⁻¹)"

    polar = result.plot_phase_function(view="polar").axes[0].lines[0]

    assert polar.get_xdata()[0] == 0

    assert polar.get_xdata()[-1] == 2 * np.pi

    np.testing.assert_allclose(polar.get_ydata(), radius)

    angular = result.plot_phase_function(log_y=True).axes[0].lines[0]

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
