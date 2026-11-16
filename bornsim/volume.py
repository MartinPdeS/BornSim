"""Finite refractive-index fields on centred cubic voxel grids."""

from copy import deepcopy
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from types import NotImplementedType
from typing import Any, TYPE_CHECKING, Literal, Self
from .grid import Grid
from .medium import Medium
from .plot_style import DIVERGING_COLORS, DIVERGING_CMAP, GRID, NAVY, SEQUENTIAL_COLORS, SEQUENTIAL_CMAP, TEXT
from .units import NoUnit, Quantity, Scalar, validate_annotation

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
    extinction_coefficient: np.ndarray | float = 0.0
    _medium_metadata: dict[str, object] | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_annotation(
            self.delta_refractive_index,
            annotation=NoUnit,
            name="delta_refractive_index",
        )

        data = np.array(self.delta_refractive_index, dtype=float, copy=True)

        if data.ndim != 3 or any(n < 2 or n > 32 for n in data.shape) or not np.all(np.isfinite(data)):
            raise ValueError("delta_refractive_index must be a finite 3D array with 2–32 cells per axis.")

        grid = Grid._resolve(grid=self.grid)

        if grid.shape != data.shape:
            raise ValueError("grid shape must match delta_refractive_index.")

        object.__setattr__(self, "grid", grid)

        object.__setattr__(self, "spacing", grid.spacing)

        validate_annotation(
            self.background_refractive_index,
            annotation=NoUnit[Scalar],
            name="background_refractive_index",
        )

        object.__setattr__(self, "background_refractive_index", float(self.background_refractive_index))

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

            if (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, (int, np.integer))
                or not 0 <= self.seed <= 2**32 - 1
            ):
                raise ValueError("seed must be an integer between 0 and 4294967295.")

            object.__setattr__(
                self,
                "seed",
                int(self.seed),
            )

        object.__setattr__(
            self,
            "_medium_metadata",
            deepcopy(self.medium.metadata) if self.medium is not None else None,
        )

        data.setflags(write=False)

        object.__setattr__(self, "delta_refractive_index", data)

        validate_annotation(
            self.extinction_coefficient,
            annotation=NoUnit,
            name="extinction_coefficient",
        )

        absorption = np.array(self.extinction_coefficient, dtype=float, copy=True)
        if absorption.ndim == 0:
            absorption = np.full(grid.shape, float(absorption))
        if absorption.shape != grid.shape or not np.all(np.isfinite(absorption)) or np.any(absorption < 0):
            raise ValueError("extinction_coefficient must be finite, nonnegative and scalar or grid-shaped.")
        absorption.setflags(write=False)
        object.__setattr__(self, "extinction_coefficient", absorption)

    def with_absorption(self, *, extinction_coefficient: Any) -> Self:
        """Return a sample with passive absorption in a real, lossless host.

        Parameters
        ----------
        extinction_coefficient : float or array_like
            Unitless nonnegative kappa, scalar or matching the voxel grid.
            Uses exp(-i omega t) and linearized contrast 2*n0*(delta_n+i*kappa).

        Returns
        -------
        Volume
            Immutable sample retaining its real index field and generation metadata.
        """
        from dataclasses import replace

        return replace(self, extinction_coefficient=extinction_coefficient)

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

        validate_annotation(
            refractive_index,
            annotation=NoUnit,
            name="refractive_index",
        )

        absolute_refractive_index = np.asarray(refractive_index, dtype=float)

        validate_annotation(
            background_refractive_index,
            annotation=NoUnit[Scalar],
            name="background_refractive_index",
        )

        background = float(background_refractive_index)

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
            extinction_coefficient=self.extinction_coefficient + other.extinction_coefficient,
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
            Equal-aspect image of the selected voxel plane. Call pyplot.show()
            to display or figure.savefig() to export.

        Raises
        ------
        ValueError
            If the plane, index, field, or length unit is invalid.
        """

        if normal not in ("x", "y", "z"):
            raise ValueError("normal must be x, y, or z.")

        axis_index = "xyz".index(normal)

        count = self.delta_refractive_index.shape[axis_index]

        selected = count // 2 if index is None else index

        if isinstance(selected, bool) or not isinstance(selected, (int, np.integer)) or not 0 <= selected < count:
            raise ValueError(f"index must be an integer between 0 and {count - 1}.")

        selected = int(selected)

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
            cmap=DIVERGING_CMAP if field == "delta_refractive_index" else SEQUENTIAL_CMAP,
            vmin=color_low,
            vmax=color_high,
        )

        position = (selected - (count - 1) / 2) * width

        axis.set(
            xlabel=f"{'xyz'[horizontal]} ({unit})",
            ylabel=f"{'xyz'[vertical]} ({unit})",
            title=title if title is not None else f"{label} at {normal} = {position:+g} {unit}",
        )

        axis.set_facecolor("white")
        axis.set_axisbelow(True)
        axis.tick_params(colors=TEXT, labelsize=9)
        axis.title.set_color(NAVY)
        axis.title.set_fontweight("semibold")
        axis.grid(color=GRID, linewidth=0.8)

        for spine in axis.spines.values():
            spine.set_color("#cbd5df")

        colorbar = figure.colorbar(image, ax=axis, label=label)
        colorbar.ax.yaxis.label.set_color(NAVY)
        colorbar.ax.tick_params(colors=TEXT, labelsize=9)

        return figure

    def plot_3d(
        self,
        *,
        backend: Literal["plotly"] = "plotly",
        mode: Literal["volume", "isosurface"] | None = None,
        field: Literal["delta_refractive_index", "refractive_index", "permittivity"] = "delta_refractive_index",
        length_unit: str = "nanometer",
        surface_count: int = 8,
        opacity: float | int | None = None,
        opacity_scale: Literal["uniform", "increasing"] = "uniform",
    ) -> "PlotlyFigure":
        """Build a three-dimensional figure of the finite voxel medium.

        Parameters
        ----------
        backend : {'plotly'}, optional
            Plotly is the only supported 3D backend. Default is 'plotly'.
        mode : str, optional
            Choose 'volume' (default) or 'isosurface'. Uniform fields are rendered as
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
            An interactive density slider adjusts this display setting.
        opacity : float, optional
            Initial surface opacity in (0, 1]. Default is 0.12 for volumes
            and 0.5 for isosurfaces. Plotly volumes
            include an opacity slider, with zero available to hide the field.
        opacity_scale : {'uniform', 'increasing'}, optional
            Default 'uniform' gives all contours the same opacity. 'increasing'
            is available only for Plotly volume mode and scales opacity from
            zero at the lowest displayed contour to ``opacity`` at the highest.
            Use field='refractive_index' to emphasize high refractive index, rather than
            the magnitude of positive and negative fluctuations.

        Returns
        -------
        figure : plotly.graph_objects.Figure
            Call figure.show() to display or figure.write_html() to export.
            Constructing a figure opens no window or browser.

        Raises
        ------
        ValueError
            If a mode, scalar field, display unit, or rendering setting is invalid.

        Notes
        -----
        Plotly contours interpolate between voxel centres and are visualization
        aids, not exact curved interfaces or a replacement for grid refinement.
        Aspect ratio follows physical dimensions. All voxels are retained, with
        at most 32**3 samples under Volume's grid limit; browser performance
        also depends on surface count and graphics support. This plots the
        input material, not an electromagnetic field.
        """

        if backend != "plotly":
            raise ValueError("BornSim supports Plotly 3D rendering only.")

        if mode is None:
            mode = "volume"

        if mode not in ("volume", "isosurface"):
            raise ValueError("mode must be volume or isosurface.")

        if opacity_scale not in ("uniform", "increasing"):
            raise ValueError("opacity_scale must be uniform or increasing.")

        if opacity_scale != "uniform" and mode != "volume":
            raise ValueError("increasing opacity_scale is only supported for Plotly volume mode.")

        values, label = self._plot_field(field=field)

        if (
            isinstance(surface_count, bool)
            or not isinstance(surface_count, (int, np.integer))
            or not 1 <= surface_count <= 32
        ):
            raise ValueError("surface_count must be an integer between 1 and 32.")

        count = int(surface_count)

        if opacity is None:
            opacity = 0.12 if mode == "volume" else 0.5

        if isinstance(opacity, (bool, np.bool_)) or not isinstance(opacity, (int, float, np.integer, np.floating)):
            raise ValueError("opacity must be finite and in (0, 1].")

        alpha = float(opacity)

        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("opacity must be finite and in (0, 1].")

        spacing = self._plot_spacing(length_unit=length_unit)

        positions = self.positions.to(spacing.units).magnitude

        low, high = float(values.min()), float(values.max())

        color_low, color_high, uniform = self._plot_color_limits(values=values, field=field)

        if uniform and mode == "isosurface":
            raise ValueError("An isosurface cannot be generated from a uniform field; use mode='volume'.")

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

        colorbar = {
            "title": {"text": label},
            "thickness": 14,
            "len": 0.68,
            "outlinewidth": 0,
            "tickfont": {"color": TEXT, "size": 10},
            **({"tickvals": [low]} if uniform else {}),
        }

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
                isomin=color_low if uniform else low,
                isomax=color_high if uniform else high,
                cmin=color_low,
                cmax=color_high,
                surface_count=surface_count,
                opacity=opacity,
                colorscale=DIVERGING_COLORS if field == "delta_refractive_index" else SEQUENTIAL_COLORS,
                colorbar=colorbar,
                caps={"x_show": False, "y_show": False, "z_show": False},
                hovertemplate=hover,
                **opacity_options,
            )
        ]

        figure = go.Figure(data=traces)

        density_values = sorted({1, 2, 4, 6, 8, 12, 16, 24, 32, surface_count})

        sliders = [
            {
                "active": density_values.index(surface_count),
                "currentvalue": {"font": {"size": 12, "color": NAVY}, "prefix": "Iso-surface density: "},
                "font": {"size": 9, "color": TEXT},
                "len": 0.38,
                "minorticklen": 0,
                "pad": {"t": 14},
                "ticklen": 2,
                "x": 0.58,
                "y": 0,
                "bgcolor": "white",
                "bordercolor": "#dce5ed",
                "borderwidth": 0,
                "steps": [
                    {"label": str(value), "method": "restyle", "args": [{"surface.count": value}, [0]]}
                    for value in density_values
                ],
            }
        ]

        if mode == "volume":
            opacity_values = sorted({step / 20 for step in range(21)} | {opacity})

            sliders.append(
                {
                    "active": opacity_values.index(opacity),
                    "currentvalue": {"font": {"size": 12, "color": NAVY}, "prefix": "Opacity: "},
                    "font": {"size": 9, "color": TEXT},
                    "len": 0.38,
                    "minorticklen": 0,
                    "pad": {"t": 14},
                    "ticklen": 2,
                    "x": 0.08,
                    "y": 0,
                    "bgcolor": "white",
                    "bordercolor": "#dce5ed",
                    "borderwidth": 0,
                    "steps": [
                        {
                            "label": f"{value:.2f}"
                            if value == opacity or np.isclose(value * 4, round(value * 4))
                            else "",
                            "method": "restyle",
                            "args": [{"opacity": value}, [0]],
                        }
                        for value in opacity_values
                    ],
                }
            )

        figure.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            title={
                "text": f"{label}: {mode}" + (" · uniform field" if uniform else ""),
                "font": {"color": NAVY, "size": 18},
                "x": 0.03,
                "xanchor": "left",
            },
            scene={
                "xaxis": {
                    "title": {"text": f"x ({length_unit})", "font": {"color": NAVY, "size": 12}},
                    "showbackground": False,
                    "showgrid": True,
                    "gridcolor": "#e3eaf0",
                    "zerolinecolor": "#cbd5df",
                    "tickfont": {"color": TEXT, "size": 10},
                },
                "yaxis": {
                    "title": {"text": f"y ({length_unit})", "font": {"color": NAVY, "size": 12}},
                    "showbackground": False,
                    "showgrid": True,
                    "gridcolor": "#e3eaf0",
                    "zerolinecolor": "#cbd5df",
                    "tickfont": {"color": TEXT, "size": 10},
                },
                "zaxis": {
                    "title": {"text": f"z ({length_unit})", "font": {"color": NAVY, "size": 12}},
                    "showbackground": False,
                    "showgrid": True,
                    "gridcolor": "#e3eaf0",
                    "zerolinecolor": "#cbd5df",
                    "tickfont": {"color": TEXT, "size": 10},
                },
                "bgcolor": "white",
                "aspectmode": "data",
                "camera": {
                    "eye": {"x": 1.3, "y": 1.3, "z": 1.05},
                    "projection": {"type": "orthographic"},
                },
            },
            margin={"l": 0, "r": 10, "b": 58, "t": 70},
            sliders=sliders,
            uirevision="bornsim-volume",
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
