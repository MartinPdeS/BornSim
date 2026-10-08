import numpy as np
import pytest

from bornsim import RandomMedium, Solver, Source
from bornsim.series import BornSeries
from bornsim.ensemble import ensemble_scattering
from bornsim.media import random_volume


def test_volume_preserves_amplitudes_and_interference():
    volume = random_volume(
        medium=RandomMedium(correlation="gaussian"),
        shape=(2, 2, 2),
        spacing=30e-9,
        seed=42,
    )
    directions = np.array([[0, 0, 1], [1, 0, 0], [0, 0, -1]])
    solver = Solver(
        source=Source(),
        order=3,
    )
    result = solver.solve_cut(
        target=volume,
        directions=directions,
    )
    expected = BornSeries(
        shape=volume.delta_index.shape,
        spacing=volume.spacing,
        background_index=volume.background_index,
        wavelength=solver.source.wavelength,
        directions=directions,
        order=3,
    ).solve(volume=volume)
    for name in ("directions", "amplitudes", "differential", "term_differential", "field_norms"):
        np.testing.assert_array_equal(getattr(result, name).magnitude, getattr(expected, name))
    assert result.warnings == expected.warnings
    assert result.mu_s is None
    assert result.angles is None
    np.testing.assert_allclose(
        result.differential.magnitude,
        np.sum(np.abs(np.cumsum(result.amplitudes.magnitude, axis=0)) ** 2, axis=(-1, -2)) / (2 * volume.volume),
    )
    figure = result.plot(terms=True)
    assert figure.axes[0].get_xlabel() == "Direction index"
    assert figure.axes[0].lines[0].get_marker() == "o"
    np.testing.assert_array_equal(figure.axes[0].lines[1].get_ydata(), result.term_differential.magnitude[1])
    meridian = solver.solve_cut(
        target=volume,
        angles=[0, np.pi / 2, np.pi],
    )
    np.testing.assert_allclose(meridian.differential.magnitude, result.differential.magnitude)
    with pytest.raises(ValueError, match="directions"):
        solver.solve_cut(
            target=volume,
            angles=[0],
            directions=directions,
        )
    with pytest.raises(ValueError, match="integrated mu_s"):
        meridian.plot_phase_function(
            view="3d",
            backend="matplotlib",
        )
    norms = result.plot_field_norms(log_y=False)
    np.testing.assert_array_equal(norms.axes[0].lines[0].get_ydata(), result.field_norms.magnitude)
    np.testing.assert_array_equal(norms.axes[0].lines[0].get_xdata(), [1, 2, 3])
    assert norms.axes[0].get_yscale() == "linear"


@pytest.mark.parametrize("realizations", [1, 3])
def test_ensemble_preserves_seed_uncertainty_and_coefficients(realizations):
    medium = RandomMedium(correlation="gaussian")
    solver = Solver(
        source=Source(),
        order=2,
    )
    options = dict(
        shape=(2, 2, 2),
        spacing=30e-9,
        realizations=realizations,
        seed=42,
        angles=[0, 0.5, np.pi],
        azimuth_samples=4,
        polar_samples=16,
    )
    result = solver.ensemble(
        medium=medium,
        **options,
    )
    expected = ensemble_scattering(
        medium=medium,
        wavelength=solver.source.wavelength,
        order=2,
        **options,
    )
    for name, key in [
        ("differential", "directional_differential"),
        ("term_differential", "directional_terms"),
        ("stderr", "directional_stderr"),
        ("mu_s", "mu_s"),
        ("g", "g"),
        ("mu_s_prime", "mu_s_prime"),
        ("field_norms", "field_norms"),
    ]:
        np.testing.assert_array_equal(getattr(result, name).magnitude, expected[key])
    assert result.realizations == realizations
    assert result.warnings == expected["warnings"]
    figure = result.plot()
    if realizations == 1:
        assert np.all(np.isnan(result.stderr))
        assert not figure.axes[0].containers
    else:
        segments = figure.axes[0].containers[0].lines[2][0].get_segments()
        errors = [(segment[1, 1] - segment[0, 1]) / 2 for segment in segments]
        np.testing.assert_allclose(errors, result.stderr.magnitude[0, :, 0])
    assert not result.plot(terms=True).axes[0].containers
    phase = result.phase_function.to("1 / steradian").magnitude
    np.testing.assert_allclose(phase, expected["directional_differential"] / expected["mu_s"][:, None, None])
    phase_plot = result.plot_phase_function(order=1)
    assert len(phase_plot.axes[0].lines) == 1
    np.testing.assert_allclose(phase_plot.axes[0].lines[0].get_ydata(), phase[0, :, 0])
    assert not phase_plot.axes[0].containers
    surface = result.plot_phase_function(
        view="3d",
        backend="matplotlib",
    )
    assert surface.axes[0].name == "3d"
    assert "Through order 2" in surface.axes[0].get_title()
    interactive = result.plot_phase_function(view="3d")
    trace = interactive.data[0]
    expected_surface = np.concatenate([phase[-1], phase[-1, :, :1]], axis=-1)
    np.testing.assert_allclose(trace.surfacecolor, expected_surface)
    np.testing.assert_allclose(np.sqrt(trace.x**2 + trace.y**2 + trace.z**2), expected_surface)
    assert trace.type == "surface"
    assert "Through order 2" in interactive.layout.title.text
    angular_surface = result.angular.plot_phase_function(view="3d")
    np.testing.assert_allclose(angular_surface.data[0].surfacecolor, expected_surface)
    norms = result.plot_field_norms()
    assert len(norms.axes[0].lines) == realizations
    for index, trace in enumerate(norms.axes[0].lines):
        np.testing.assert_array_equal(trace.get_ydata(), expected["field_norms"][index])
    assert norms.axes[0].get_yscale() == "log"
