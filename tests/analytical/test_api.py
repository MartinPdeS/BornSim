import builtins

import numpy as np
import pytest

from bornsim import AnalyticalMedium, Result, Solver, Source
from bornsim.model import angular_scattering, optical_properties
from bornsim.units import ureg


def test_analytical_result_and_plot():
    medium = AnalyticalMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="gaussian",
        smoothness=1.5,
    )

    source = Source(wavelength=5.32e-07 * ureg.meter)

    angles = np.array([0.0, 0.5, np.pi])

    result = Solver(source=source).solve(
        target=medium,
        angles=angles * ureg.radian,
    )

    angles[0] = 1

    assert isinstance(result, Result)

    assert result.kind == "analytical"

    assert result.source is source

    np.testing.assert_allclose(
        result.differential.magnitude[0, :, 0],
        angular_scattering(
            medium=medium,
            wavelength=source.wavelength.to("meter"),
            theta=result.angles.to("radian"),
        ),
    )

    expected = optical_properties(
        medium=medium,
        wavelength=source.wavelength.to("meter"),
    )

    for name in ("mu_s", "g", "mu_s_prime"):
        expected_value = expected[name]

        reference = (
            [expected_value] if name == "g" else expected_value.to(getattr(result, name).units).magnitude.reshape(1)
        )

        np.testing.assert_allclose(getattr(result, name).magnitude, reference)

    figure = result.plot(log_y=True)

    np.testing.assert_allclose(figure.axes[0].lines[0].get_xdata(), result.angles.to("degree").magnitude)

    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), result.differential.magnitude[0, :, 0])

    assert figure.axes[0].get_yscale() == "log"

    assert not figure.axes[0].containers

    with pytest.raises(ValueError, match="unavailable"):
        result.plot(terms=True)


def test_zero_scattering_has_undefined_anisotropy():
    result = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    ).solve(
        target=AnalyticalMedium(
            refractive_index_std=0,
            background_refractive_index=1.33,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        )
    )

    np.testing.assert_array_equal(result.differential.magnitude, 0)

    assert np.isnan(result.g[0])


@pytest.mark.parametrize("wavelength", [0, -1, np.nan, np.inf])
def test_invalid_source(wavelength):
    with pytest.raises(ValueError, match="wavelength"):
        Source(wavelength=wavelength * ureg.meter)


@pytest.mark.parametrize("angles", [[], [[0]], [-0.1], [np.pi + 0.1], [np.nan]])
def test_invalid_angles(angles):
    with pytest.raises(ValueError, match="angles"):
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            )
        ).solve(
            target=AnalyticalMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            angles=angles * ureg.radian,
        )


def test_invalid_solver_inputs():
    with pytest.raises(TypeError, match="Source"):
        Solver(source=633e-9)

    with pytest.raises(ValueError, match="order"):
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            ),
            order=13,
        )

    with pytest.raises(ValueError, match="quadrature_order"):
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            ),
            quadrature_order=True,
        )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    )

    with pytest.raises(TypeError, match="target"):
        solver.solve(target=None)

    with pytest.raises(TypeError, match="medium"):
        solver.ensemble(
            medium=None,
            shape=(12, 12, 12),
            spacing=50e-9 * ureg.meter,
        )

    with pytest.raises(ValueError, match="directions"):
        solver.solve(
            target=AnalyticalMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            directions=[[0, 0, 1]],
        )


def test_matplotlib_plotting_does_not_import_plotly(monkeypatch):
    original_import = builtins.__import__

    def without_plotly(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("Plotly is not installed")

        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_plotly)

    result = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    ).solve(
        target=AnalyticalMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        )
    )

    assert result.plot().axes[0].name == "rectilinear"

    assert result.plot_phase_function().axes[0].name == "rectilinear"


def test_phase_function_normalization_and_rayleigh_limit():
    theta = np.linspace(0, np.pi, 4097)

    result = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    ).solve(
        target=AnalyticalMedium(
            correlation_length=1e-12 * ureg.meter,
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation="gaussian",
            smoothness=1.5,
        ),
        angles=theta * ureg.radian,
    )

    phase = result.azimuth_average().phase_function.to("1 / steradian").magnitude[0]

    # Independent short-correlation reference includes unpolarized dipole scattering.
    reference = 3 * (1 + np.cos(theta) ** 2) / (16 * np.pi)

    np.testing.assert_allclose(phase, reference, rtol=1e-8)

    integral = np.sum((phase * np.sin(theta))[1:] + (phase * np.sin(theta))[:-1]) * np.diff(theta)[0] / 2

    assert 2 * np.pi * integral == pytest.approx(1, abs=1e-7)

    result = Result(
        source=result.source,
        kind=result.kind,
        differential=result.differential.to("1 / centimeter / steradian").to("1 / meter / steradian"),
        angles=result.angles.to("radian"),
        azimuths=result.azimuths.to("radian"),
        mu_s=result.mu_s.to("1 / millimeter").to("1 / meter"),
    )

    np.testing.assert_allclose(result.azimuth_average().phase_function.magnitude[0], phase)


@pytest.mark.parametrize("backend", [None, "matplotlib"])
def test_3d_uniform_phase_is_a_sphere_and_polar_cut_is_closed(backend):
    radius = 1 / (4 * np.pi)

    result = Result(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        kind="analytical",
        differential=np.full((1, 3), 2 * radius) / ureg.meter / ureg.steradian,
        angles=[180, 90, 0] * ureg.degree,
        mu_s=np.array([2]) / ureg.meter,
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
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        kind="analytical",
        differential=np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1, np.pi] * ureg.radian,
        mu_s=[0] * (1 / ureg.meter),
    )

    with pytest.raises(ValueError, match="positive, finite"):
        result.phase_function


@pytest.mark.parametrize("view", ["polar", "3d"])
def test_directional_views_require_full_angular_coverage(view):
    result = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    ).solve(
        target=AnalyticalMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        ),
        angles=[0, 0.5, 1] * ureg.radian,
    )

    with pytest.raises(ValueError, match="spanning"):
        result.plot_phase_function(view=view)


def test_plotting_rejects_invalid_views_orders_and_unavailable_norms():
    result = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    ).solve(
        target=AnalyticalMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        )
    )

    with pytest.raises(ValueError, match="view"):
        result.plot_phase_function(view="unknown")

    with pytest.raises(ValueError, match="log_y"):
        result.plot_phase_function(
            backend="matplotlib",
            view="3d",
            log_y=True,
        )

    with pytest.raises(ValueError, match="order"):
        result.plot_phase_function(order=2)

    with pytest.raises(ValueError, match="numerical"):
        result.plot_field_norms()
