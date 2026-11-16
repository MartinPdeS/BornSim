"""Render result data without owning normalization, persistence, or solver state.

Two-dimensional plots use Matplotlib. All three-dimensional plots use
Plotly. Renderers accept Result or AngularData where applicable and leave
physical arrays and their units unchanged.
"""

from typing import Any, cast
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.ticker import MaxNLocator
import numpy as np
import plotly.graph_objects as go

from .plot_style import GRID, NAVY, SEQUENTIAL_COLORS, SERIES_COLORS, TEXT
from .units import Dimensionless, NoUnit, Quantity, ureg, validate_annotation


def _style_axis(*, axis) -> None:
    """Apply BornSim's navy and cyan style to Matplotlib axes."""

    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.tick_params(colors=TEXT, labelsize=9)
    axis.title.set_color(NAVY)
    axis.title.set_fontweight("semibold")
    axis.grid(color=GRID, linewidth=0.8)

    if axis.name == "polar":
        axis.spines["polar"].set_color("#cbd5df")
    else:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        for spine in axis.spines.values():
            spine.set_color("#cbd5df")


def _select_azimuth(*, result, azimuth):
    """Validate a meridian position against the available azimuth samples."""

    count = 1 if result.azimuths is None else len(result.azimuths)

    if isinstance(azimuth, bool) or not isinstance(azimuth, (int, np.integer)) or not 0 <= azimuth < count:
        raise ValueError(f"azimuth must be an integer between 0 and {count - 1}.")

    return int(azimuth)


def _angular_label(*, result, azimuth):
    """Describe the selected meridian or the explicit intensity average."""

    if getattr(result, "meridian_azimuth", None) is not None:
        degrees = float(np.rad2deg(result.meridian_azimuth.magnitude))

        return f" · meridian phi = {degrees:g} degrees"

    if result.azimuth_averaged:
        return " · azimuth average"

    if result.azimuths is not None:
        degrees = float(np.rad2deg(result.azimuths.magnitude[azimuth]))

        return f" · meridian phi = {degrees:g} degrees"

    return ""


def plot_differential(
    *,
    result,
    terms,
    log_y,
    title,
    azimuth=0,
    scale=1.0,
    ylabel="Differential scattering (m⁻¹ sr⁻¹)",
):
    """Render curves and uncertainty using one shared display scale.

    Parameters
    ----------
    result : Result or AngularData
        Unitful differential data with order and observation axes.
    terms : bool
        Select isolated terms instead of cumulative intensities.
    log_y : bool
        Use a logarithmic vertical axis.
    title : str or None
        Custom title, or None for the finite-sample label.
    azimuth : int, optional
        Sampled meridian position for full directional data.
    scale : float, optional
        Numeric display factor applied equally to intensities and errors.
    ylabel : str, optional
        Display label including the units after applying scale.

    Returns
    -------
    matplotlib.figure.Figure
        Curves in degrees or direction positions, without displaying a window.
    """

    values = result.term_differential if terms else result.differential

    if values is None:
        raise ValueError("Isolated term curves are unavailable for this result.")

    values = scale * values.to("1 / meter / steradian").magnitude

    selected = _select_azimuth(result=result, azimuth=azimuth)

    if values.ndim == 3:
        values = values[..., selected]

    x = np.arange(values.shape[1]) if result.angles is None else np.rad2deg(result.angles.to("radian").magnitude)

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

    for index, curve in enumerate(values):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]

        errors = None

        if not terms and result.stderr is not None:
            candidate = result.stderr.to("1 / meter / steradian").magnitude[index]

            if candidate.ndim == 2:
                candidate = candidate[:, selected]

            if np.any(np.isfinite(candidate)):
                errors = scale * np.ma.masked_invalid(candidate)

        label = f"{'Term' if terms else 'Through order'} {index + 1}"

        style: dict[str, Any] = (
            {"color": color, "marker": "o", "linestyle": "none", "markersize": 5}
            if result.angles is None
            else {"color": color, "linewidth": 2.4}
        )

        if errors is None:
            axis.plot(
                x,
                curve,
                label=label,
                **style,
            )
        else:
            axis.errorbar(x, curve, yerr=errors, capsize=2, label=label, **style)

    default_title = "Finite-sample scattering"

    default_title += _angular_label(result=result, azimuth=selected)

    axis.set(
        title=default_title if title is None else title,
        xlabel="Direction index" if result.angles is None else "Laboratory polar angle (degrees)",
        ylabel=ylabel,
        yscale="log" if log_y else "linear",
    )

    _style_axis(axis=axis)

    axis.legend(frameon=False, labelcolor=TEXT, fontsize=9)

    return figure


def plot_cross_section(
    *,
    result,
    volume=None,
    area_unit="nanometer**2",
    terms=False,
    log_y=False,
    title=None,
    azimuth=0,
):
    """Scale finite-sample curves and uncertainties by their box volume.

    Parameters
    ----------
    result : Result
        Finite-sample data, volume, and optional grid provenance.
    volume : Volume or None, optional
        Explicit sample, which must match any recorded grid and volume.
    area_unit : str, optional
        Area display unit, default square nanometres.
    terms, log_y : bool, optional
        Select isolated terms or a logarithmic vertical axis.
    title : str or None, optional
        Custom title or the finite-sample cross-section label.
    azimuth : int, optional
        Sampled meridian position.

    Returns
    -------
    matplotlib.figure.Figure
        Cross sections per steradian, with consistently scaled uncertainties.
    """

    from .volume import Volume
    from pint.errors import DimensionalityError, UndefinedUnitError

    if volume is not None:
        if not isinstance(volume, Volume):
            raise TypeError("volume must be a Volume.")

        grid = result.provenance.get("grid")

        if grid is not None:
            recorded_background_refractive_index = grid.get(
                "background_refractive_index",
                volume.background_refractive_index,
            )

            incompatible_grid = (
                tuple(grid["shape"]) != volume.delta_refractive_index.shape
                or grid["spacing_m"] != float(volume.spacing.to("meter").magnitude)
                or recorded_background_refractive_index != volume.background_refractive_index
            )

            if incompatible_grid:
                raise ValueError("volume must match the recorded sample grid and background_refractive_index.")

        physical_volume = float(volume.volume.to("meter**3").magnitude)

        recorded = result.sample_volume

        invalid_condition = recorded is not None and not np.isclose(
            recorded.to("meter**3").magnitude, physical_volume, rtol=1e-10, atol=0
        )

        if invalid_condition:
            raise ValueError("volume must match the recorded sample volume.")
    else:
        if result.sample_volume is None:
            raise ValueError(
                "sample_volume is unavailable; supply the sample Volume for a result without recorded volume."
            )

        physical_volume = float(result.sample_volume.to("meter**3").magnitude)

    try:
        area = (1.0 * ureg.meter**2).to(area_unit)
    except (DimensionalityError, UndefinedUnitError, TypeError, ValueError) as error:
        raise ValueError("area_unit must name an area unit.") from error

    return plot_differential(
        result=result,
        terms=terms,
        log_y=log_y,
        title="Finite-sample differential cross section" if title is None else title,
        scale=physical_volume * float(area.magnitude),
        azimuth=azimuth,
        ylabel=f"Differential cross section ({area.units:~P} sr⁻¹)",
    )


def plot_phase_function(*, result, view="3d", order=None, log_y=False, azimuth=0, backend=None):
    """Render normalized densities while retaining directional 3D asymmetry.

    Parameters
    ----------
    result : Result or AngularData
        Normalized angular data in inverse steradians. Surface views require
        at least three polar angles spanning zero to pi and full azimuth data
        or an explicitly averaged curve.
    view : {'angular', 'polar', '3d'}, optional
        Display geometry; 3D radii are densities, not spatial coordinates.
    order : int or None, optional
        One-based cumulative order. None selects all curves or the last surface.
    log_y : bool, optional
        Logarithmic vertical axis, available only for angular curves.
    azimuth : int, optional
        Sampled meridian position for angular and polar views.
    backend : {'matplotlib', 'plotly'} or None, optional
        None selects Plotly for 3D surfaces and Matplotlib for 2D views.
        Other backend/view combinations are rejected.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        Figure without opening a window or changing stored physical data.
    """

    if view != "angular" and getattr(result, "meridian_azimuth", None) is not None:
        raise ValueError("Use full angular data for polar or 3D plots; a selected meridian has only one side.")

    if view not in ("angular", "polar", "3d"):
        raise ValueError("view must be angular, polar, or 3d.")

    if backend is None:
        backend = "plotly" if view == "3d" else "matplotlib"

    if backend not in ("matplotlib", "plotly"):
        raise ValueError("backend must be matplotlib or plotly.")

    if view == "3d" and backend != "plotly":
        raise ValueError("3D phase plots support only backend='plotly'.")

    if backend == "plotly" and view != "3d":
        raise ValueError("The Plotly backend is only supported for the 3D phase view.")

    if log_y and view != "angular":
        raise ValueError("log_y is only supported for the angular view.")

    phase = result.phase_function.magnitude

    theta = result.angles.to("radian").magnitude

    indices = np.argsort(theta)

    theta, phase = theta[indices], phase[:, indices]

    if order is not None and (
        isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 1 <= order <= len(phase)
    ):
        raise ValueError(f"order must be an integer between 1 and {len(phase)}.")

    orders = range(len(phase)) if order is None else [int(order) - 1]

    insufficient_angular_coverage = view in ("polar", "3d") and (
        len(np.unique(theta)) < 3
        or not np.isclose(theta[0], 0, atol=1e-10, rtol=0)
        or not np.isclose(theta[-1], np.pi, atol=1e-10, rtol=0)
    )

    if insufficient_angular_coverage:
        raise ValueError("Polar and 3D views require at least three distinct angles spanning 0 to pi.")

    title = "Directional finite-sample phase function"

    meridian = _select_azimuth(result=result, azimuth=azimuth)

    degrees = np.rad2deg(theta)

    if view == "3d":
        selected = len(phase) - 1 if order is None else orders[0]

        if phase.ndim == 3:
            phi = np.concatenate([result.azimuths.magnitude, [2 * np.pi]])

            directional = phase[selected]

            radius = np.concatenate([directional, directional[:, :1]], axis=-1)

            surface_label = "full azimuthal distribution"
        elif result.azimuth_averaged:
            phi = np.linspace(0, 2 * np.pi, 97)

            radius = np.broadcast_to(phase[selected, :, None], (len(theta), len(phi)))

            surface_label = "explicit azimuth-average surface"

            title = "Azimuth-averaged finite-sample phase function"
        else:
            raise ValueError(
                "Directional phase data are unavailable; recompute with Solver.solve using AngularSampling."
            )

        sine = np.sin(theta)[:, None]

        x = radius * sine * np.cos(phi)

        y = radius * sine * np.sin(phi)

        z = radius * np.cos(theta)[:, None]

        figure = go.Figure(
            data=[
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=radius,
                    customdata=radius,
                    colorscale=SEQUENTIAL_COLORS,
                    cmin=0,
                    cmax=float(radius.max()),
                    colorbar={"title": {"text": "p (sr⁻¹)"}, "thickness": 14, "outlinewidth": 0},
                    hovertemplate="Phase density p=%{customdata:.5g} sr⁻¹<extra></extra>",
                )
            ],
        )

        figure.update_layout(
            template="plotly_white",
            title={
                "text": f"Phase function<br><sup>Through order {selected + 1} · {surface_label}</sup>",
                "font": {"color": NAVY, "size": 18},
                "x": 0.03,
                "xanchor": "left",
            },
            scene={
                "xaxis": {"title": "p ŝx (sr⁻¹)", "showbackground": False, "gridcolor": "#e3eaf0"},
                "yaxis": {"title": "p ŝy (sr⁻¹)", "showbackground": False, "gridcolor": "#e3eaf0"},
                "zaxis": {"title": "p ŝz (sr⁻¹)", "showbackground": False, "gridcolor": "#e3eaf0"},
                "aspectmode": "data",
                "bgcolor": "white",
                "camera": {"eye": {"x": 1.3, "y": 1.3, "z": 1.05}, "projection": {"type": "orthographic"}},
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 65},
            uirevision="bornsim-phase",
        )

        return figure

    else:
        opposite = None

        if phase.ndim == 3:
            if view == "polar":
                count = len(result.azimuths)

                if count % 2:
                    raise ValueError("A full polar plane requires an even azimuth sample count.")

                opposite = phase[..., (meridian + count // 2) % count]

            phase = phase[..., meridian]

        title += _angular_label(result=result, azimuth=meridian)

        projection = "polar" if view == "polar" else None

        figure, axis = plt.subplots(figsize=(8, 5), subplot_kw={"projection": projection}, layout="constrained")

        for index in orders:
            if view == "polar":
                axis.plot(
                    np.concatenate([theta, 2 * np.pi - theta[-2::-1]]),
                    np.concatenate([phase[index], (phase if opposite is None else opposite)[index, -2::-1]]),
                    label=f"Through order {index + 1}",
                    color=SERIES_COLORS[index % len(SERIES_COLORS)],
                    linewidth=2,
                )
            else:
                axis.plot(
                    degrees,
                    phase[index],
                    label=f"Through order {index + 1}",
                    color=SERIES_COLORS[index % len(SERIES_COLORS)],
                    linewidth=2,
                )

        if view == "polar":
            polar_axis = cast(PolarAxes, axis)

            polar_axis.set_theta_zero_location("N")

            polar_axis.set_theta_direction(-1)

            axis.set_ylim(bottom=0)

            axis.set_title(f"{title}\nMeridian cut · incidence at 0° · p (sr⁻¹)", pad=20)
        else:
            axis.set(
                title=title,
                xlabel="Laboratory polar angle θ (degrees)",
                ylabel="Phase function p (sr⁻¹)",
                yscale="log" if log_y else "linear",
            )

        _style_axis(axis=axis)

        axis.legend(frameon=False, labelcolor=TEXT, fontsize=9)

    return figure


def plot_field_norms(*, result, log_y=True):
    """Draw relative Born field norms for each available realization.

    Parameters
    ----------
    result : Result
        Dimensionless field-term norms with an order axis and optional
        realization axis. Decreasing terms do not certify Born convergence.
    log_y : bool, optional
        Use a logarithmic vertical axis.

    Returns
    -------
    matplotlib.figure.Figure
        Norms relative to the incident field, without displaying a window.
    """

    if result.field_norms is None:
        raise ValueError("Field norms are only available for numerical results.")

    validate_annotation(
        result.field_norms,
        annotation=Dimensionless | NoUnit,
        name="field_norms",
    )

    norms = result.field_norms

    if isinstance(norms, Quantity):
        norms = norms.to("dimensionless").magnitude

    norms = np.atleast_2d(np.asarray(norms, dtype=float))

    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

    for index, curve in enumerate(norms):
        axis.plot(
            np.arange(1, len(curve) + 1),
            curve,
            "o-",
            color=SERIES_COLORS[index % len(SERIES_COLORS)],
            label=f"Realization {index + 1}",
        )

    axis.set(
        title="Born field terms\nDecreasing terms do not certify convergence",
        xlabel="Field-term order j",
        ylabel="‖Ej‖ / ‖Einc‖ (dimensionless)",
        yscale="log" if log_y else "linear",
    )

    axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    _style_axis(axis=axis)

    axis.legend(frameon=False, labelcolor=TEXT, fontsize=9)

    return figure
