import numpy as np
import pytest

from bornsim import (
    AngularSampling,
    Grid,
    Box,
    Cylinder,
    Ellipsoid,
    Layer,
    Solver,
    Source,
    Sphere,
    StructuredMedium,
    Volume,
)
from bornsim.units import ureg


def test_layers_interfaces_units_and_last_region_wins():
    medium = StructuredMedium(
        regions=[
            Layer(
                lower=-100 * ureg.nanometer,
                upper=0 * ureg.nanometer,
                index=1.34,
            ),
            Layer(
                lower=0,
                upper=100e-9,
                index=1.35,
            ),
            Box(
                size=(40e-9, 40e-9, 40e-9),
                index=1.36,
            ),
        ]
    )
    volume = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50 * ureg.nanometer,
    )
    expected = np.empty((3, 3, 3))
    expected[:, :, 0] = 0.01
    expected[:, :, 1:] = 0.02
    expected[1, 1, 1] = 0.03
    np.testing.assert_allclose(volume.delta_index, expected, atol=1e-15)
    np.testing.assert_allclose(volume.positions[1, 1, :, 2], [-50e-9, 0, 50e-9], rtol=1e-15, atol=0)
    assert volume.medium is None and volume.seed is None


@pytest.mark.parametrize(
    "region, expected",
    [
        (
            Sphere(
                radius=1,
                index=1.34,
            ),
            [True, True, False, False],
        ),
        (
            Ellipsoid(
                radii=(2, 1, 1),
                index=1.34,
            ),
            [True, True, True, False],
        ),
        (
            Box(
                size=(4, 2, 2),
                index=1.34,
            ),
            [True, True, True, True],
        ),
        (
            Cylinder(
                radius=1,
                height=4,
                index=1.34,
                axis=0,
            ),
            [True, True, True, False],
        ),
    ],
)
def test_geometry_membership_against_known_points(region, expected):
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 1]])
    np.testing.assert_array_equal(region.mask(positions=points), expected)
    shifted = region.translated(offset=(3, -2, 4))
    np.testing.assert_array_equal(shifted.mask(positions=points + [3, -2, 4]), expected)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_layer_axes_half_open_boundary(axis):
    region = Layer(
        lower=0,
        upper=1,
        index=1.34,
        axis=axis,
    )
    points = np.zeros((4, 3))
    points[:, axis] = [-1, 0, 0.5, 1]
    np.testing.assert_array_equal(region.mask(positions=points), [False, True, True, False])


def test_composed_scatterers_match_manual_volume_and_scattering():
    medium = StructuredMedium(
        regions=[
            Sphere(
                radius=60e-9,
                index=1.34,
                centre=(-40e-9, 0, 0),
            ),
            Cylinder(
                radius=40e-9,
                height=120e-9,
                index=1.35,
            ),
        ]
    )
    volume = medium.to_volume(
        shape=(4, 4, 4),
        spacing=40e-9,
    )
    manual = np.zeros((4, 4, 4))
    for i, j, k in np.ndindex(manual.shape):
        x, y, z = (np.array([i, j, k]) - 1.5) * 40e-9
        if (x + 40e-9) ** 2 + y**2 + z**2 <= (60e-9) ** 2:
            manual[i, j, k] = 1.34 - 1.33
        if x**2 + y**2 <= (40e-9) ** 2 and abs(z) <= 60e-9:
            manual[i, j, k] = 1.35 - 1.33
    np.testing.assert_array_equal(volume.delta_index, manual)
    solver = Solver(
        source=Source(),
        order=3,
    )
    actual = solver.solve_cut(
        target=volume,
        angles=[0, 1, 2],
    )
    expected = solver.solve_cut(
        target=Volume(
            delta_index=manual,
            spacing=40e-9,
        ),
        angles=[0, 1, 2],
    )
    np.testing.assert_array_equal(actual.amplitudes.magnitude, expected.amplitudes.magnitude)


def test_empty_clipped_and_invalid_constitutive_law():
    assert not StructuredMedium().to_volume(shape=(2, 2, 2)).delta_index.any()
    clipped = StructuredMedium(
        regions=[
            Sphere(
                radius=1,
                index=1.34,
            )
        ]
    ).to_volume(shape=(2, 2, 2))
    np.testing.assert_allclose(clipped.delta_index, 0.01)
    with pytest.raises(ValueError, match="permittivity"):
        StructuredMedium(
            regions=[
                Sphere(
                    radius=1,
                    index=0.1,
                )
            ]
        ).to_volume(shape=(2, 2, 2))
    for shape in ((1, 2, 2), (33, 2, 2), (2, 2)):
        with pytest.raises(ValueError):
            StructuredMedium().to_volume(shape=shape)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Sphere(
            radius=0,
            index=1.34,
        ),
        lambda: Sphere(
            radius=1,
            index=-1,
        ),
        lambda: Sphere(
            radius=1,
            index=1.34,
            centre=(0, 0),
        ),
        lambda: Ellipsoid(
            radii=(1, -1, 1),
            index=1.34,
        ),
        lambda: Box(
            size=(1, 1, np.inf),
            index=1.34,
        ),
        lambda: Cylinder(
            radius=1,
            height=0,
            index=1.34,
        ),
        lambda: Cylinder(
            radius=1,
            height=2,
            index=1.34,
            axis=True,
        ),
        lambda: Layer(
            lower=1,
            upper=0,
            index=1.34,
        ),
        lambda: Layer(
            lower=0,
            upper=1,
            index=1.34,
            axis=3,
        ),
        lambda: Sphere(
            radius=1 * ureg.second,
            index=1.34,
        ),
    ],
)
def test_invalid_geometry(factory):
    with pytest.raises(ValueError):
        factory()
    with pytest.raises(TypeError):
        StructuredMedium(regions=[object()])


@pytest.mark.parametrize("centre", [(0, 0, 0), (50e-9, 0, 0)])
def test_single_structured_sample_phase_matches_unpolarized_dipole(centre):
    medium = StructuredMedium()
    sphere = Sphere(
        radius=20e-9,
        index=1.34,
        centre=centre,
    )
    medium.add_structures(
        sphere,
    )
    solver = Solver(
        source=Source(),
        order=1,
    )
    angles = np.linspace(0, np.pi, 31)
    grid = Grid(
        shape=(3, 3, 3),
        spacing=50e-9,
    )
    sampling = AngularSampling(
        angles=angles,
        polar_samples=16,
        azimuth_samples=4,
    )
    volume = medium.to_volume(grid=grid)
    result = solver.solve(
        target=volume,
        sampling=sampling,
    )
    # One occupied voxel is a discrete dipole: the unpolarized phase function
    # is 3*(1 + cos(theta)**2)/(16*pi), regardless of its translation.
    expected = 3 * (1 + np.cos(angles) ** 2) / (16 * np.pi)
    np.testing.assert_allclose(result.azimuth_average().phase_function.magnitude[0], expected, rtol=1e-12)
    assert result.stderr is None
    assert result.kind == "volume"
    assert result.realizations is None
    figure = result.plot_phase_function(view="3d")
    figure.canvas.draw()
    assert figure.axes[0].name == "3d"


def test_two_sphere_directional_phase_preserves_interference_and_3d_surface(monkeypatch, tmp_path):
    from mpl_toolkits.mplot3d import Axes3D
    from bornsim import Result

    spacing = 50e-9
    medium = StructuredMedium()
    medium.add_background(index=1.0)
    left = Sphere(
        radius=20e-9,
        index=1.01,
        centre=(-spacing, 0, 0),
    )
    right = Sphere(
        radius=20e-9,
        index=1.01,
        centre=(spacing, 0, 0),
    )
    medium.add_structures(left, right)
    wavelength = 633e-9
    solver = Solver(
        source=Source(wavelength=wavelength),
        order=1,
    )
    theta = np.linspace(0, np.pi, 31)
    grid = Grid(
        shape=(3, 3, 3),
        spacing=spacing,
    )
    sampling = AngularSampling(
        angles=theta,
        polar_samples=32,
        azimuth_samples=16,
    )
    volume = medium.to_volume(grid=grid)
    result = solver.solve(
        target=volume,
        sampling=sampling,
    )
    phi = result.azimuths.magnitude
    # Independent two-dipole far-field reference: their relative phase is
    # q_x * separation. Squaring the coherent sum gives 2 + 2*cos(q_x*d).
    k = 2 * np.pi / wavelength
    separation = 2 * spacing
    angular = (1 + np.cos(theta[:, None]) ** 2) * (
        2 + 2 * np.cos(k * separation * np.sin(theta[:, None]) * np.cos(phi))
    )
    # Independently integrate the reference on a finer solid-angle grid.
    cosine, weights = np.polynomial.legendre.leggauss(128)
    reference_phi = np.arange(128) * 2 * np.pi / 128
    quadrature = (1 + cosine[:, None] ** 2) * (
        2 + 2 * np.cos(k * separation * np.sqrt(1 - cosine[:, None] ** 2) * np.cos(reference_phi))
    )
    normalization = np.sum(quadrature * weights[:, None]) * 2 * np.pi / 128
    expected = angular / normalization
    phase = result.directional_phase_function.magnitude[0]
    np.testing.assert_allclose(phase, expected, rtol=1e-12)
    assert phase[15, 0] < 0.8 * phase[15, 4]
    np.testing.assert_allclose(phase.mean(axis=-1), result.azimuth_average().phase_function.magnitude[0], rtol=1e-12)
    restored = Result.load(path=result.save(path=tmp_path / "directional.npz"))
    np.testing.assert_array_equal(restored.azimuths.magnitude, phi)
    np.testing.assert_array_equal(
        restored.directional_phase_function.magnitude, result.directional_phase_function.magnitude
    )

    captured = {}
    original = Axes3D.plot_surface

    def capture_surface(self, x, y, z, **kwargs):
        captured["radius"] = np.sqrt(x**2 + y**2 + z**2)
        return original(self, x, y, z, **kwargs)

    monkeypatch.setattr(Axes3D, "plot_surface", capture_surface)
    figure = restored.plot_phase_function(view="3d")
    figure.canvas.draw()
    np.testing.assert_allclose(captured["radius"][:, :-1], expected, rtol=1e-12)
    np.testing.assert_allclose(captured["radius"][:, -1], expected[:, 0], rtol=1e-12)
    assert "full azimuthal distribution" in figure.axes[0].get_title()
