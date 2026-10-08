"""Volume rendering behind the public Volume plotting methods.

All physical fields remain on Volume. Display conversions, color scales,
backend selection and figure creation belong to this internal renderer.
"""

import numpy as np
from ._validation import _integer


class _VolumePlotter:
    """Render one volume without changing its field, grid or SI units."""

    def __init__(self, *, volume):
        self._volume = volume

    def _plot_field(self, *, field):
        """Return the selected material field and display label."""
        fields = {
            "delta_index": (self._volume.delta_index, "Index fluctuation δn"),
            "index": (self._volume.background_index + self._volume.delta_index, "Refractive index n"),
            "permittivity": (
                self._volume.background_index**2 + 2 * self._volume.background_index * self._volume.delta_index,
                "Linearized relative permittivity",
            ),
        }
        if field not in fields:
            raise ValueError("field must be delta_index, index, or permittivity.")
        return fields[field]

    def plot_slice(self, *, normal="z", index=None, field="index", length_unit="nanometer", title=None):
        """Render one voxel plane with full-volume color limits."""
        from pint.errors import DimensionalityError, UndefinedUnitError
        import matplotlib.pyplot as plt
        from .units import _quantity

        if normal not in ("x", "y", "z"):
            raise ValueError("normal must be x, y, or z.")
        axis_index = "xyz".index(normal)
        count = self._volume.delta_index.shape[axis_index]
        selected = _integer(
            value=count // 2 if index is None else index,
            name="index",
            low=0,
            high=count - 1,
        )
        values, label = self._plot_field(field=field)
        try:
            spacing = _quantity(
                value=self._volume.spacing,
                unit="meter",
                name="spacing",
            ).to(length_unit)
        except (DimensionalityError, UndefinedUnitError, TypeError, ValueError) as error:
            raise ValueError("length_unit must name a length unit.") from error
        unit = f"{spacing.units:~}"
        width = float(spacing.magnitude)
        horizontal, vertical = [i for i in range(3) if i != axis_index]
        half_width = self._volume.delta_index.shape[horizontal] * width / 2
        half_height = self._volume.delta_index.shape[vertical] * width / 2
        low, high = float(values.min()), float(values.max())
        if field == "delta_index":
            high = float(np.abs(values).max())
            low = -high
        if low == high:
            padding = max(abs(low), 1.0) * 1e-6
            low, high = low - padding, high + padding
        figure, axis = plt.subplots(layout="constrained")
        image = axis.imshow(
            np.take(values, selected, axis=axis_index).T,
            origin="lower",
            extent=(-half_width, half_width, -half_height, half_height),
            interpolation="nearest",
            aspect="equal",
            cmap="RdBu_r" if field == "delta_index" else "viridis",
            vmin=low,
            vmax=high,
        )
        position = (selected - (count - 1) / 2) * width
        axis.set(
            xlabel=f"{'xyz'[horizontal]} ({unit})",
            ylabel=f"{'xyz'[vertical]} ({unit})",
            title=title if title is not None else f"{label} at {normal} = {position:+g} {unit}",
        )
        figure.colorbar(image, ax=axis, label=label)
        return figure

    def plot_3d(
        self,
        *,
        backend="plotly",
        mode=None,
        field="delta_index",
        length_unit="nanometer",
        surface_count=8,
        opacity=None,
        opacity_scale="uniform",
        slice_indices=None,
    ):
        """Validate rendering options and draw the selected 3D backend."""
        if backend not in ("matplotlib", "plotly"):
            raise ValueError("backend must be matplotlib or plotly.")
        if mode is None:
            mode = "slices" if backend == "matplotlib" else "volume"
        modes = ("slices", "voxels") if backend == "matplotlib" else ("volume", "isosurface", "slices")
        if mode not in modes:
            raise ValueError(f"{backend} mode must be one of {modes}.")
        if opacity_scale not in ("uniform", "increasing"):
            raise ValueError("opacity_scale must be uniform or increasing.")
        if opacity_scale != "uniform" and (backend != "plotly" or mode != "volume"):
            raise ValueError("increasing opacity_scale is only supported for Plotly volume mode.")
        values, label = self._plot_field(field=field)
        count = _integer(
            value=surface_count,
            name="surface_count",
            low=1,
            high=32,
        )
        if opacity is None:
            opacity = 1.0 if backend == "matplotlib" else 0.15
        if isinstance(opacity, (bool, np.bool_)) or not isinstance(opacity, (int, float, np.integer, np.floating)):
            raise ValueError("opacity must be finite and in (0, 1].")
        alpha = float(opacity)
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("opacity must be finite and in (0, 1].")
        if slice_indices is not None and mode != "slices":
            raise ValueError("slice_indices are only supported for slices mode.")
        indices = (
            tuple(n // 2 for n in self._volume.delta_index.shape) if slice_indices is None else tuple(slice_indices)
        )
        if len(indices) != 3:
            raise ValueError("slice_indices must contain three voxel indices.")
        indices = tuple(
            _integer(
                value=value,
                name="slice index",
                low=0,
                high=n - 1,
            )
            for value, n in zip(indices, self._volume.delta_index.shape)
        )
        from pint.errors import DimensionalityError, UndefinedUnitError
        from .units import _quantity

        try:
            positions = (
                _quantity(
                    value=self._volume.positions,
                    unit="meter",
                    name="positions",
                )
                .to(length_unit)
                .magnitude
            )
        except (DimensionalityError, UndefinedUnitError, TypeError, ValueError) as error:
            raise ValueError("length_unit must name a length unit.") from error
        low, high = float(values.min()), float(values.max())
        uniform = low == high
        if uniform and backend == "plotly":
            mode = "slices"
        # Use a shared color domain for all slices; preserve zero for contrasts.
        if field == "delta_index" and not uniform:
            color_low, color_high = -float(np.abs(values).max()), float(np.abs(values).max())
        else:
            padding = max(abs(low), 1.0) * 1e-6 if uniform else 0
            color_low, color_high = low - padding, high + padding
        colorscale = "RdBu_r" if field == "delta_index" else "Viridis"
        if backend == "matplotlib":
            return self._plot_3d_matplotlib(
                mode=mode,
                values=values,
                label=label,
                positions=positions,
                length_unit=length_unit,
                indices=indices,
                color_low=color_low,
                color_high=color_high,
                colormap="RdBu_r" if field == "delta_index" else "viridis",
                opacity=alpha,
                uniform=uniform,
            )
        import plotly.graph_objects as go

        colorbar = {"title": label, **({"tickvals": [low]} if uniform else {})}
        hover = (
            f"x=%{{x:.4g}} {length_unit}<br>y=%{{y:.4g}} {length_unit}"
            f"<br>z=%{{z:.4g}} {length_unit}<br>{label}=%{{value:.5g}}<extra></extra>"
        )
        if mode == "slices":
            traces = []
            for axis, index in enumerate(indices):
                plane: list[slice | int] = [slice(None)] * 3
                plane[axis] = index
                coordinate = positions[tuple(plane)]
                traces.append(
                    go.Surface(
                        x=coordinate[..., 0],
                        y=coordinate[..., 1],
                        z=coordinate[..., 2],
                        surfacecolor=values[tuple(plane)],
                        customdata=values[tuple(plane)],
                        text=[[f"{value:.5g}" for value in row] for row in values[tuple(plane)]],
                        cmin=color_low,
                        cmax=color_high,
                        colorscale=colorscale,
                        showscale=axis == 0,
                        colorbar=colorbar,
                        hovertemplate=hover.replace("%{value:.5g}", f"{low:.5g}" if uniform else "%{text}"),
                        name=f"{'xyz'[axis]} slice",
                    )
                )
        else:
            trace = go.Volume if mode == "volume" else go.Isosurface
            opacity_options = {"opacityscale": [[0, 0], [1, 1]]} if opacity_scale == "increasing" else {}
            traces = [
                trace(
                    x=positions[..., 0].ravel(),
                    y=positions[..., 1].ravel(),
                    z=positions[..., 2].ravel(),
                    value=values.ravel(),
                    isomin=low + (high - low) * 0.05,
                    isomax=high - (high - low) * 0.05,
                    cmin=color_low,
                    cmax=color_high,
                    surface_count=count,
                    opacity=alpha,
                    colorscale=colorscale,
                    colorbar=colorbar,
                    caps={"x_show": False, "y_show": False, "z_show": False},
                    hovertemplate=hover,
                    **opacity_options,
                )
            ]
        figure = go.Figure(data=traces)
        figure.update_layout(
            title=f"{label}: {mode}" + (" (uniform field)" if uniform else ""),
            scene={
                "xaxis_title": f"x ({length_unit})",
                "yaxis_title": f"y ({length_unit})",
                "zaxis_title": f"z ({length_unit})",
                "aspectmode": "data",
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 45},
        )
        return figure

    def _plot_3d_matplotlib(
        self,
        *,
        mode,
        values,
        label,
        positions,
        length_unit,
        indices,
        color_low,
        color_high,
        colormap,
        opacity,
        uniform,
    ):
        """Render sampled cells with physical extents and a shared color scale."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        shape = self._volume.delta_index.shape
        width = positions[1, 0, 0, 0] - positions[0, 0, 0, 0]
        edges = [(np.arange(n + 1) - n / 2) * width for n in shape]
        norm = Normalize(vmin=color_low, vmax=color_high)
        cmap = plt.get_cmap(colormap)
        figure = plt.figure(figsize=(8, 6), layout="constrained")
        axis = figure.add_subplot(projection="3d")
        if mode == "voxels":
            filled = self._volume.delta_index != 0
            if not np.any(filled):
                filled = np.ones(shape, dtype=bool)
            voxel_coordinates = np.meshgrid(*edges, indexing="ij")
            axis.voxels(
                *voxel_coordinates,
                filled=filled,
                facecolors=cmap(norm(values)),
                edgecolors=(0.1, 0.1, 0.1, 0.25),
                linewidth=0.3,
                shade=False,
                alpha=opacity,
            )
        else:
            for normal, index in enumerate(indices):
                horizontal, vertical = [i for i in range(3) if i != normal]
                first, second = np.meshgrid(edges[horizontal], edges[vertical], indexing="ij")
                position = (index - (shape[normal] - 1) / 2) * width
                coordinates = [np.full_like(first, position) for _ in range(3)]
                coordinates[horizontal], coordinates[vertical] = first, second
                axis.plot_surface(
                    X=coordinates[0],
                    Y=coordinates[1],
                    Z=coordinates[2],
                    facecolors=cmap(norm(np.take(values, index, axis=normal))),
                    rstride=1,
                    cstride=1,
                    shade=False,
                    linewidth=0,
                    antialiased=False,
                )
        axis.set(
            xlabel=f"x ({length_unit})",
            ylabel=f"y ({length_unit})",
            zlabel=f"z ({length_unit})",
            xlim=(edges[0][0], edges[0][-1]),
            ylim=(edges[1][0], edges[1][-1]),
            zlim=(edges[2][0], edges[2][-1]),
            title=f"{label}: {mode}" + (" (uniform field)" if uniform else ""),
        )
        axis.set_box_aspect(shape)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        colorbar = figure.colorbar(mappable, ax=axis, label=label, shrink=0.65, pad=0.1)
        if uniform:
            colorbar.set_ticks([float(values.flat[0])])
        return figure
