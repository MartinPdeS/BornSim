"""Finite refractive-index fields on centred cubic voxel grids."""

from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import ArrayLike
from types import NotImplementedType
from typing import TYPE_CHECKING, Literal, Self
from .grid import Grid
from .medium import Medium
from .units import _integer, _refractive_index_values, Quantity

if TYPE_CHECKING:
    from matplotlib.figure import Figure as MatplotlibFigure
    from plotly.graph_objects import Figure as PlotlyFigure


@dataclass(frozen=True, kw_only=True)
class Volume:
    """Store a finite voxel field and its explicit spatial Grid.

    The copied, read-only delta_refractive_index array has shape grid.shape.
    ``refractive_index`` returns the read-only absolute field
    ``background_refractive_index + delta_refractive_index``. Addition combines
    contrasts from volumes only when their grids and backgrounds match.
    Background refractive index is a plain number; grid spacing and positions
    carry physical units. Relative permittivity is linearized as
    background_refractive_index**2 + 2*background_refractive_index*delta_refractive_index.
    The quadratic fluctuation term is omitted at every Born order.
    Medium.to_volume records generation statistics and seed. ``medium_metadata``
    returns the metadata snapshot captured when the volume was created. Manually
    constructed volumes have only their explicitly supplied provenance.

    Use :meth:`from_refractive_index` when the input array contains absolute
    refractive indices rather than fluctuations from the background.
    """

    delta_refractive_index: np.ndarray
    background_refractive_index: float
    grid: Grid
    spacing: Quantity = field(init=False)
    medium: Medium | None = None
    seed: int | None = None
    _medium_metadata: dict[str, object] | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        data = np.array(
            _refractive_index_values(
                value=self.delta_refractive_index,
                name="delta_refractive_index",
            ),
            copy=True,
        )

        if data.ndim != 3 or any(n < 2 or n > 32 for n in data.shape) or not np.all(np.isfinite(data)):
            raise ValueError("delta_refractive_index must be a finite 3D array with 2–32 cells per axis.")

        grid = Grid._resolve(grid=self.grid)

        if grid.shape != data.shape:
            raise ValueError("grid shape must match delta_refractive_index.")

        object.__setattr__(self, "grid", grid)

        object.__setattr__(self, "spacing", grid.spacing)

        object.__setattr__(
            self,
            "background_refractive_index",
            _refractive_index_values(
                value=self.background_refractive_index,
                name="background_refractive_index",
                scalar=True,
            ),
        )

        if not np.isfinite(self.background_refractive_index) or self.background_refractive_index <= 0:
            raise ValueError("background_refractive_index must be finite and positive.")

        if np.any(self.background_refractive_index**2 + 2 * self.background_refractive_index * data <= 0):
            raise ValueError("Linearized relative permittivity must stay positive.")

        if (self.medium is None) != (self.seed is None):
            raise ValueError("medium and seed must be supplied together.")

        if self.medium is not None:
            if not isinstance(self.medium, Medium):
                raise TypeError("medium must be a Medium.")

            if self.medium.background_refractive_index != self.background_refractive_index:
                raise ValueError("medium background_refractive_index must match the volume.")

            object.__setattr__(
                self,
                "seed",
                _integer(
                    value=self.seed,
                    name="seed",
                    low=0,
                    high=2**32 - 1,
                ),
            )

        object.__setattr__(
            self,
            "_medium_metadata",
            deepcopy(self.medium.metadata) if self.medium is not None else None,
        )

        data.setflags(write=False)

        object.__setattr__(self, "delta_refractive_index", data)

    @classmethod
    def from_refractive_index(
        cls,
        *,
        refractive_index: ArrayLike,
        background_refractive_index: float,
        grid: Grid,
        medium: Medium | None = None,
        seed: int | None = None,
    ) -> Self:
        """Construct a volume from an absolute refractive-index field.

        Parameters
        ----------
        refractive_index : array_like
            Plain numerical absolute refractive indices, with shape
            ``grid.shape``. Values must not carry units.
        background_refractive_index : float
            Positive, unitless reference refractive index.
        grid : Grid
            Explicit spatial grid matching the field shape.
        medium : Medium, optional
            Medium metadata associated with generation. Must be supplied
            together with ``seed``.
        seed : int, optional
            Generation seed, supplied together with ``medium``.

        Returns
        -------
        Volume
            Immutable volume storing the field as
            ``refractive_index - background_refractive_index``.

        Raises
        ------
        ValueError
            If the field, background, grid shape, or resulting linearized
            permittivity is invalid.
        TypeError
            If ``medium`` is not a Medium instance.
        """

        absolute_refractive_index = _refractive_index_values(
            value=refractive_index,
            name="refractive_index",
        )

        background = _refractive_index_values(
            value=background_refractive_index,
            name="background_refractive_index",
            scalar=True,
        )

        return cls(
            delta_refractive_index=absolute_refractive_index - background,
            background_refractive_index=background,
            grid=grid,
            medium=medium,
            seed=seed,
        )

    @property
    def positions(self) -> Quantity:
        """Return voxel-centre coordinates relative to the sample centre.

        Returns
        -------
        positions : Quantity
            Unit-bearing coordinates, shape (nx, ny, nz, 3), with the final
            axis ordered as x, y, z.
        """

        return getattr(self, "grid").positions

    @property
    def volume(self) -> Quantity:
        """Return the total physical volume of all voxels.

        Returns
        -------
        volume : Quantity
            Unit-bearing sample volume, equal to ``delta_refractive_index.size*spacing**3``.
        """

        return getattr(self, "grid").volume

    @property
    def refractive_index(self) -> np.ndarray:
        """Return the read-only absolute refractive-index field.

        Returns
        -------
        numpy.ndarray
            Absolute refractive indices, equal to the background plus
            ``delta_refractive_index``.
        """

        data = self.background_refractive_index + self.delta_refractive_index

        data.setflags(write=False)

        return data

    @property
    def medium_metadata(self) -> dict[str, object] | None:
        """Return a copy of the generation metadata captured at construction.

        Returns
        -------
        dict[str, object] or None
            A fresh copy of the source medium's metadata, or None when the
            volume has no associated medium.
        """

        return deepcopy(self._medium_metadata)

    def __add__(self, other: object) -> "Volume | NotImplementedType":
        """Add refractive-index contrasts from volumes on the same grid.

        Both volumes must have the same grid and background refractive index.
        The resulting volume contains no generation provenance because it is
        a manual composition of its inputs.
        """

        if not isinstance(other, Volume):
            return NotImplemented

        if self.grid != other.grid:
            raise ValueError("Volumes must use the same grid to be added.")

        if self.background_refractive_index != other.background_refractive_index:
            raise ValueError("Volumes must use the same background_refractive_index to be added.")

        return Volume(
            delta_refractive_index=self.delta_refractive_index + other.delta_refractive_index,
            background_refractive_index=self.background_refractive_index,
            grid=self.grid,
        )

    def plot_slice(
        self,
        *,
        normal: Literal["x", "y", "z"] = "z",
        index: int | None = None,
        field: Literal["refractive_index", "delta_refractive_index", "permittivity"] = "refractive_index",
        length_unit: str = "nanometer",
        title: str | None = None,
    ) -> "MatplotlibFigure":
        """Build a Matplotlib figure of one voxel plane.

        Parameters
        ----------
        normal : {'x', 'y', 'z'}, optional
            Axis perpendicular to the plane; default 'z'.
        index : int, optional
            Voxel index along the normal. Defaults to n//2, the positive
            central plane for an even grid. The title gives its actual position.
        field : {'refractive_index', 'delta_refractive_index', 'permittivity'}, optional
            Display n0 + delta_refractive_index, fluctuations, or linearized relative
            permittivity n0**2 + 2*n0*delta_refractive_index. Default 'refractive_index'.
        length_unit : str, optional
            Spatial display unit; default 'nanometer'. Stored SI data is unchanged.
        title : str, optional
            Override the title describing the field and plane position.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Equal-aspect image with voxel-edge extents and a labelled colorbar.
            Call pyplot.show() to display or figure.savefig() to export.
            Colors use the full volume's range to compare different planes;
            fluctuation colors are symmetric about zero.

        Raises
        ------
        ValueError
            If the plane, index, field, or length unit is invalid.
        """

        import matplotlib.pyplot as plt

        if normal not in ("x", "y", "z"):
            raise ValueError("normal must be x, y, or z.")

        axis_index = "xyz".index(normal)

        count = self.delta_refractive_index.shape[axis_index]

        selected = _integer(
            value=count // 2 if index is None else index,
            name="index",
            low=0,
            high=count - 1,
        )

        values, label = self._plot_field(field=field)

        spacing = self._plot_spacing(length_unit=length_unit)

        unit = f"{spacing.units:~}"

        width = float(spacing.magnitude)

        horizontal, vertical = [i for i in range(3) if i != axis_index]

        half_width = self.delta_refractive_index.shape[horizontal] * width / 2

        half_height = self.delta_refractive_index.shape[vertical] * width / 2

        color_low, color_high, _ = self._plot_color_limits(values=values, field=field)

        figure, axis = plt.subplots(layout="constrained")

        image = axis.imshow(
            np.take(values, selected, axis=axis_index).T,
            origin="lower",
            extent=(-half_width, half_width, -half_height, half_height),
            interpolation="nearest",
            aspect="equal",
            cmap="RdBu_r" if field == "delta_refractive_index" else "viridis",
            vmin=color_low,
            vmax=color_high,
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
        backend: Literal["matplotlib", "plotly"] = "plotly",
        mode: Literal["voxels", "volume", "isosurface"] | None = None,
        field: Literal["delta_refractive_index", "refractive_index", "permittivity"] = "delta_refractive_index",
        length_unit: str = "nanometer",
        surface_count: int = 8,
        opacity: float | int | None = None,
        opacity_scale: Literal["uniform", "increasing"] = "uniform",
    ) -> "MatplotlibFigure | PlotlyFigure":
        """Build a three-dimensional figure of the finite voxel medium.

        Parameters
        ----------
        backend : {'matplotlib', 'plotly'}, optional
            Default is Plotly for browser interaction. Select 'matplotlib'
            explicitly for a Matplotlib figure.
        mode : str, optional
            Matplotlib supports 'voxels' (default). Plotly supports 'volume'
            (default) and 'isosurface'. Uniform Plotly fields are rendered as
            volume data; an isosurface cannot be generated from a uniform field.
        field : {'delta_refractive_index', 'refractive_index', 'permittivity'}, optional
            Scalar field to display, default refractive index fluctuation. 'refractive_index' displays
            n0 + delta_refractive_index; 'permittivity' displays the solver's linearized
            relative permittivity n0**2 + 2*n0*delta_refractive_index, not (n0+delta_refractive_index)**2.
        length_unit : str, optional
            Spatial display unit, default 'nanometer'. Stored SI data is unchanged.
        surface_count : int, optional
            Number of contours, from 1 to 32; default 8. More surfaces increase
            browser rendering cost. Used only by Plotly volume and isosurface modes.
        opacity : float, optional
            Surface opacity in (0, 1]. Default is 1 for Matplotlib and 0.15
            for Plotly.
        opacity_scale : {'uniform', 'increasing'}, optional
            Default 'uniform' gives all contours the same opacity. 'increasing'
            is available only for Plotly volume mode and scales opacity from
            zero at the lowest displayed contour to ``opacity`` at the highest.
            Use field='refractive_index' to emphasize high refractive index, rather than
            the magnitude of positive and negative fluctuations.

        Returns
        -------
        figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Matplotlib: call pyplot.show() to display or figure.savefig() to
            export. Rotation needs an interactive Matplotlib backend; gallery
            images are static. Plotly: call figure.show() or figure.write_html().
            Constructing a figure opens no window or browser.

        Raises
        ------
        ValueError
            If a mode, scalar field, display unit, or rendering setting is invalid.

        Notes
        -----
        Matplotlib voxels preserve staircase interfaces. Dense transparent
        scenes can have depth-ordering limitations in Matplotlib. Plotly
        contours interpolate between voxel centres and are visualization aids,
        not exact curved interfaces or a replacement for grid refinement.
        Aspect ratio follows physical dimensions. All voxels are retained, with
        at most 32**3 samples under Volume's grid limit; browser performance
        also depends on surface count and graphics support. This plots the
        input material, not an electromagnetic field.
        """

        if backend not in ("matplotlib", "plotly"):
            raise ValueError("backend must be matplotlib or plotly.")

        if mode is None:
            mode = "voxels" if backend == "matplotlib" else "volume"

        if backend == "matplotlib" and mode != "voxels":
            raise ValueError("matplotlib mode must be voxels.")

        if backend == "plotly" and mode not in ("volume", "isosurface"):
            raise ValueError("plotly mode must be volume or isosurface.")

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

        spacing = self._plot_spacing(length_unit=length_unit)

        positions = self.positions.to(spacing.units).magnitude

        low, high = float(values.min()), float(values.max())

        color_low, color_high, uniform = self._plot_color_limits(values=values, field=field)

        if uniform and backend == "plotly" and mode == "isosurface":
            raise ValueError("An isosurface cannot be generated from a uniform field; use mode='volume'.")

        if backend == "matplotlib":
            return self._plot_3d_matplotlib(
                values=values,
                label=label,
                positions=positions,
                length_unit=length_unit,
                color_low=color_low,
                color_high=color_high,
                colormap="RdBu_r" if field == "delta_refractive_index" else "viridis",
                opacity=alpha,
                uniform=uniform,
            )

        if mode == "volume":
            plotly_mode: Literal["volume", "isosurface"] = "volume"
        elif mode == "isosurface":
            plotly_mode = "isosurface"
        else:
            raise ValueError("plotly mode must be volume or isosurface.")

        return self._plot_3d_plotly(
            mode=plotly_mode,
            field=field,
            values=values,
            label=label,
            positions=positions,
            length_unit=length_unit,
            surface_count=count,
            opacity=alpha,
            opacity_scale=opacity_scale,
            low=low,
            high=high,
            color_low=color_low,
            color_high=color_high,
            uniform=uniform,
        )

    def _plot_spacing(self, *, length_unit: str) -> Quantity:
        """Convert voxel spacing to a display unit or raise a domain error."""

        from pint.errors import DimensionalityError, UndefinedUnitError

        try:
            return self.spacing.to(length_unit)
        except (DimensionalityError, UndefinedUnitError, TypeError, ValueError) as error:
            raise ValueError("length_unit must name a length unit.") from error

    @staticmethod
    def _plot_color_limits(
        *,
        values: np.ndarray,
        field: Literal["delta_refractive_index", "refractive_index", "permittivity"],
    ) -> tuple[float, float, bool]:
        """Return a shared color range and whether the field is uniform."""

        low, high = float(values.min()), float(values.max())

        uniform = low == high

        if field == "delta_refractive_index":
            extent = float(np.abs(values).max())

            low, high = -extent, extent

        if low == high:
            padding = max(abs(low), 1.0) * 1e-6

            low, high = low - padding, high + padding

        return low, high, uniform

    def _plot_3d_plotly(
        self,
        *,
        mode: Literal["volume", "isosurface"],
        field: Literal["delta_refractive_index", "refractive_index", "permittivity"],
        values: np.ndarray,
        label: str,
        positions: np.ndarray,
        length_unit: str,
        surface_count: int,
        opacity: float,
        opacity_scale: Literal["uniform", "increasing"],
        low: float,
        high: float,
        color_low: float,
        color_high: float,
        uniform: bool,
    ) -> "PlotlyFigure":
        """Render volume or isosurface traces using Plotly."""

        import plotly.graph_objects as go

        colorbar = {"title": label, **({"tickvals": [low]} if uniform else {})}

        hover = (
            f"x=%{{x:.4g}} {length_unit}<br>y=%{{y:.4g}} {length_unit}"
            f"<br>z=%{{z:.4g}} {length_unit}<br>{label}=%{{value:.5g}}<extra></extra>"
        )

        trace = go.Volume if mode == "volume" else go.Isosurface

        opacity_options = {"opacityscale": [[0, 0], [1, 1]]} if opacity_scale == "increasing" else {}

        traces = [
            trace(
                x=positions[..., 0].ravel(),
                y=positions[..., 1].ravel(),
                z=positions[..., 2].ravel(),
                value=values.ravel(),
                isomin=color_low if uniform else low + (high - low) * 0.05,
                isomax=color_high if uniform else high - (high - low) * 0.05,
                cmin=color_low,
                cmax=color_high,
                surface_count=surface_count,
                opacity=opacity,
                colorscale="RdBu_r" if field == "delta_refractive_index" else "Viridis",
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

    def _plot_field(
        self,
        *,
        field: Literal["delta_refractive_index", "refractive_index", "permittivity"],
    ) -> tuple[np.ndarray, str]:
        """Return the selected material field and display label."""

        fields = {
            "delta_refractive_index": (self.delta_refractive_index, "Refractive-index fluctuation δn"),
            "refractive_index": (self.refractive_index, "Refractive index n"),
            "permittivity": (
                self.background_refractive_index**2
                + 2 * self.background_refractive_index * self.delta_refractive_index,
                "Linearized relative permittivity",
            ),
        }

        if field not in fields:
            raise ValueError("field must be delta_refractive_index, refractive_index, or permittivity.")

        return fields[field]

    def _plot_3d_matplotlib(
        self,
        *,
        values: np.ndarray,
        label: str,
        positions: np.ndarray,
        length_unit: str,
        color_low: float,
        color_high: float,
        colormap: str,
        opacity: float,
        uniform: bool,
    ) -> "MatplotlibFigure":
        """Render sampled cells with physical extents and a shared color scale."""

        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        shape = self.delta_refractive_index.shape

        width = positions[1, 0, 0, 0] - positions[0, 0, 0, 0]

        edges = [(np.arange(n + 1) - n / 2) * width for n in shape]

        norm = Normalize(vmin=color_low, vmax=color_high)

        cmap = plt.get_cmap(colormap)

        figure = plt.figure(figsize=(8, 6), layout="constrained")

        axis = figure.add_subplot(projection="3d")

        filled = self.delta_refractive_index != 0

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

        axis.set(
            xlabel=f"x ({length_unit})",
            ylabel=f"y ({length_unit})",
            zlabel=f"z ({length_unit})",
            xlim=(edges[0][0], edges[0][-1]),
            ylim=(edges[1][0], edges[1][-1]),
            zlim=(edges[2][0], edges[2][-1]),
            title=f"{label}: voxels" + (" (uniform field)" if uniform else ""),
        )

        axis.set_box_aspect(shape)

        mappable = ScalarMappable(norm=norm, cmap=cmap)

        colorbar = figure.colorbar(mappable, ax=axis, label=label, shrink=0.65, pad=0.1)

        if uniform:
            colorbar.set_ticks([float(values.flat[0])])

        return figure
