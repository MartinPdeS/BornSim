import builtins
import unicodedata

import numpy as np
import pytest

from bornsim import RandomMedium, Sphere, StructuredMedium, Volume


@pytest.fixture
def plotly_backend():
    pytest.importorskip("plotly")


def sample():
    return RandomMedium().to_volume(
        shape=(3, 4, 5),
        spacing=40e-9,
        seed=42,
    )


@pytest.mark.parametrize("normal, dimension", [("x", 0), ("y", 1), ("z", 2)])
def test_matplotlib_slice_orientation_extents_and_units(normal, dimension):
    volume = Volume(
        delta_index=np.arange(24).reshape(2, 3, 4) / 1000,
        spacing=40e-9,
    )
    figure = volume.plot_slice(
        normal=normal,
        index=0,
        field="permittivity",
        length_unit="micrometer",
    )
    axis = figure.axes[0]
    image = axis.images[0]
    expected = volume.background_index**2 + 2 * volume.background_index * volume.delta_index
    np.testing.assert_array_equal(image.get_array(), np.take(expected, 0, axis=dimension).T)
    horizontal, vertical = [i for i in range(3) if i != dimension]
    half_width = volume.delta_index.shape[horizontal] * 0.04 / 2
    half_height = volume.delta_index.shape[vertical] * 0.04 / 2
    np.testing.assert_allclose(image.get_extent(), [-half_width, half_width, -half_height, half_height])
    # Pint versions may format micro as U+00B5 or U+03BC; both mean micrometres.
    assert unicodedata.normalize("NFKC", axis.get_xlabel()) == f"{'xyz'[horizontal]} (μm)"
    assert unicodedata.normalize("NFKC", axis.get_ylabel()) == f"{'xyz'[vertical]} (μm)"
    assert image.get_clim() == (expected.min(), expected.max())


def test_even_grid_slice_title_and_symmetric_contrast_scale():
    volume = Volume(
        delta_index=np.arange(24).reshape(2, 3, 4) / 1000,
        spacing=20e-9,
    )
    figure = volume.plot_slice(field="delta_index")
    axis = figure.axes[0]
    assert "z = +10 nm" in axis.get_title()
    assert axis.images[0].get_clim() == (-0.023, 0.023)


@pytest.mark.parametrize(
    "settings",
    [{"normal": "q"}, {"index": -1}, {"index": True}, {"field": "bad"}, {"length_unit": "second"}],
)
def test_invalid_matplotlib_slice_settings(settings):
    with pytest.raises(ValueError):
        sample().plot_slice(**settings)


@pytest.mark.parametrize("mode, trace_type", [("volume", "volume"), ("isosurface", "isosurface")])
def test_plotly_values_coordinates_units_and_levels(mode, trace_type, plotly_backend):
    volume = sample()
    before = volume.delta_index.copy()
    figure = volume.plot_3d(
        backend="plotly",
        mode=mode,
        field="permittivity",
        length_unit="micrometer",
        surface_count=5,
    )
    trace = figure.data[0]
    assert trace.type == trace_type
    np.testing.assert_allclose(trace.x, volume.positions[..., 0].ravel() * 1e6)
    np.testing.assert_allclose(trace.y, volume.positions[..., 1].ravel() * 1e6)
    np.testing.assert_allclose(trace.z, volume.positions[..., 2].ravel() * 1e6)
    expected = volume.background_index**2 + 2 * volume.background_index * volume.delta_index
    np.testing.assert_array_equal(trace.value, expected.ravel())
    assert trace.surface.count == 5
    assert trace.isomin < trace.isomax
    assert figure.layout.scene.xaxis.title.text == "x (micrometer)"
    assert figure.layout.scene.aspectmode == "data"
    np.testing.assert_array_equal(volume.delta_index, before)


def test_slices_use_requested_voxel_planes_and_shared_color_domain(plotly_backend):
    volume = sample()
    figure = volume.plot_3d(
        backend="plotly",
        mode="slices",
        field="index",
        slice_indices=(0, 2, 4),
    )
    expected = volume.background_index + volume.delta_index
    for trace, reference in zip(figure.data, (expected[0], expected[:, 2], expected[:, :, 4])):
        assert trace.type == "surface"
        np.testing.assert_array_equal(trace.surfacecolor, reference)
        np.testing.assert_array_equal(trace.customdata, reference)
        assert trace.cmin == expected.min()
        assert trace.cmax == expected.max()
    assert sum(trace.showscale for trace in figure.data) == 1
    np.testing.assert_allclose(figure.data[0].x, volume.positions[0, :, :, 0] * 1e9)


def test_uniform_medium_falls_back_to_visible_slices(plotly_backend):
    volume = Volume(
        delta_index=np.zeros((2, 3, 4)),
        spacing=50e-9,
    )
    figure = volume.plot_3d(
        backend="plotly",
    )
    assert len(figure.data) == 3
    assert all(trace.type == "surface" for trace in figure.data)
    assert "uniform field" in figure.layout.title.text
    assert figure.data[0].cmin < 0 < figure.data[0].cmax
    assert figure.data[0].colorbar.tickvals == (0.0,)
    assert "δn=0" in figure.data[0].hovertemplate


def test_structured_medium_and_html_export(tmp_path, plotly_backend):
    volume = StructuredMedium(
        regions=[
            Sphere(
                radius=70e-9,
                index=1.34,
            )
        ]
    ).to_volume(
        shape=(8, 8, 8),
        spacing=25e-9,
    )
    figure = volume.plot_3d(backend="plotly", mode="isosurface")
    assert np.max(figure.data[0].value) == pytest.approx(0.01)
    path = tmp_path / "medium.html"
    figure.write_html(path, include_plotlyjs=True)
    assert "Plotly.newPlot" in path.read_text()


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "invalid"},
        {"field": "invalid"},
        {"length_unit": "second"},
        {"length_unit": "not_a_unit"},
        {"surface_count": 0},
        {"surface_count": True},
        {"opacity": 0},
        {"opacity": np.nan},
        {"opacity": 2},
        {"opacity": [0.1]},
        {"mode": "slices", "slice_indices": (3, 0, 0)},
        {"mode": "slices", "slice_indices": (0, 0)},
        {"slice_indices": (0, 0, 0)},
    ],
)
def test_invalid_visualization_settings(settings):
    with pytest.raises(ValueError):
        sample().plot_3d(backend="plotly", **settings)


def test_missing_optional_plotly_has_installation_guidance(monkeypatch):
    original = builtins.__import__

    def without_plotly(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("plotly"):
            raise ImportError("not installed")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_plotly)
    figure = sample().plot_3d()
    figure.canvas.draw()
    assert figure.axes[0].name == "3d"
    with pytest.raises(ImportError, match=r"BornSim\[visualization\]"):
        sample().plot_3d(
            backend="plotly",
        )


def test_matplotlib_3d_slices_preserve_every_cell_and_physical_extents(monkeypatch):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    calls = []
    original = Axes3D.plot_surface

    def record_surface(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(Axes3D, "plot_surface", record_surface)
    volume = sample()
    before = volume.delta_index.copy()
    figure = volume.plot_3d(
        field="index",
        length_unit="micrometer",
        slice_indices=(0, 2, 4),
    )
    figure.canvas.draw()
    axis = figure.axes[0]
    assert axis.name == "3d"
    np.testing.assert_allclose(axis.get_xlim(), [-0.06, 0.06])
    np.testing.assert_allclose(axis.get_ylim(), [-0.08, 0.08])
    np.testing.assert_allclose(axis.get_zlim(), [-0.1, 0.1])
    assert axis.get_xlabel() == "x (micrometer)"
    assert axis.get_zlabel() == "z (micrometer)"
    values = volume.background_index + volume.delta_index
    norm = Normalize(vmin=values.min(), vmax=values.max())
    planes = (values[0], values[:, 2], values[:, :, 4])
    expected_positions = (-0.04, 0.02, 0.08)
    assert len(calls) == 3
    for dimension, (call, plane, position) in enumerate(zip(calls, planes, expected_positions)):
        np.testing.assert_allclose(call["XYZ"[dimension]], position)
        np.testing.assert_allclose(call["facecolors"], plt.get_cmap("viridis")(norm(plane)))
        assert call["X"].shape == (plane.shape[0] + 1, plane.shape[1] + 1)
    np.testing.assert_array_equal(volume.delta_index, before)


def test_matplotlib_voxels_show_the_sampled_sphere_mask(monkeypatch):
    from mpl_toolkits.mplot3d import Axes3D

    calls = []
    original = Axes3D.voxels

    def record_voxels(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes3D, "voxels", record_voxels)
    medium = StructuredMedium()
    sphere = Sphere(
        radius=20e-9,
        index=1.34,
    )
    medium.add_structures(
        sphere,
    )
    volume = medium.to_volume(
        shape=(3, 3, 3),
        spacing=50e-9,
    )
    figure = volume.plot_3d(
        mode="voxels",
        field="index",
    )
    figure.canvas.draw()
    coordinates, options = calls[0]
    expected = np.zeros((3, 3, 3), dtype=bool)
    expected[1, 1, 1] = True
    np.testing.assert_array_equal(options["filled"], expected)
    np.testing.assert_allclose(coordinates[0][:, 0, 0], [-75, -25, 25, 75])
    assert options["alpha"] == 1
    assert len(figure.axes[0].collections) == 1


@pytest.mark.parametrize("mode", ["slices", "voxels"])
def test_matplotlib_uniform_medium_has_visible_geometry_and_single_color_tick(mode):
    volume = Volume(
        delta_index=np.zeros((2, 3, 4)),
        spacing=50e-9,
    )
    figure = volume.plot_3d(mode=mode)
    figure.canvas.draw()
    assert figure.axes[0].collections
    np.testing.assert_array_equal(figure.axes[1].get_yticks(), [0])


@pytest.mark.parametrize(
    "settings",
    [
        {"backend": "bad"},
        {"mode": "volume"},
        {"mode": "isosurface"},
        {"backend": "plotly", "mode": "voxels"},
        {"mode": "voxels", "slice_indices": (0, 0, 0)},
    ],
)
def test_backend_mode_contract_is_explicit(settings):
    with pytest.raises(ValueError):
        sample().plot_3d(**settings)
