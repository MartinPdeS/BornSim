"""Result plotting behind the public Result visualization methods."""

import numpy as np
from ._validation import _integer
from .units import _dimensionless, ureg


class _ResultPlotter:
    """Display one result without changing its data, normalization or units."""

    def __init__(self, *, result):
        self._result = result

    @staticmethod
    def _pyplot():
        import matplotlib.pyplot as plt

        return plt

    def plot(self, *, terms=False, log_y=False, title=None, azimuth=0):
        """Render coherent or isolated differential scattering curves."""

        return self._plot_differential(
            terms=terms,
            log_y=log_y,
            title=title,
            azimuth=azimuth,
        )

    def plot_cross_section(
        self, *, volume=None, area_unit="nanometer**2", terms=False, log_y=False, title=None, azimuth=0
    ):
        """Scale finite-sample curves and uncertainties by their box volume."""

        from .volume import Volume
        from pint.errors import DimensionalityError, UndefinedUnitError

        if self._result.kind == "analytical":
            raise ValueError("Cross sections require a finite-sample result.")

        if volume is not None:
            if not isinstance(volume, Volume):
                raise TypeError("volume must be a Volume.")

            grid = self._result.provenance.get("grid")

            if grid is not None:
                # Saved results retain their original provenance keys.
                recorded_background_refractive_index = grid.get(
                    "background_refractive_index",
                    grid.get("background_index", volume.background_refractive_index),
                )

                incompatible_grid = (
                    tuple(grid["shape"]) != volume.delta_refractive_index.shape
                    or grid["spacing_m"] != float(volume.spacing.to("meter").magnitude)
                    or recorded_background_refractive_index != volume.background_refractive_index
                )

                if incompatible_grid:
                    raise ValueError("volume must match the recorded sample grid and background_refractive_index.")

            physical_volume = float(volume.volume.to("meter**3").magnitude)

            recorded = self._result.sample_volume

            invalid_condition = recorded is not None and not np.isclose(
                recorded.to("meter**3").magnitude, physical_volume, rtol=1e-10, atol=0
            )

            if invalid_condition:
                raise ValueError("volume must match the recorded sample volume.")
        else:
            if self._result.sample_volume is None:
                raise ValueError("sample_volume is unavailable; supply the sample Volume for a legacy result.")

            physical_volume = float(self._result.sample_volume.to("meter**3").magnitude)

        try:
            area = (1.0 * ureg.meter**2).to(area_unit)
        except (DimensionalityError, UndefinedUnitError, TypeError, ValueError) as error:
            raise ValueError("area_unit must name an area unit.") from error

        return self._plot_differential(
            terms=terms,
            log_y=log_y,
            title="Finite-sample differential cross section" if title is None else title,
            scale=physical_volume * float(area.magnitude),
            azimuth=azimuth,
            ylabel=f"Differential cross section ({area.units:~P} sr⁻¹)",
        )

    def _plot_differential(
        self, *, terms, log_y, title, azimuth=0, scale=1.0, ylabel="Differential scattering (m⁻¹ sr⁻¹)"
    ):
        """Render curves and uncertainty using one shared display scale."""

        values = self._result.term_differential if terms else self._result.differential

        if values is None:
            raise ValueError("Isolated term curves are unavailable for this result.")

        values = scale * values.to("1 / meter / steradian").magnitude

        selected = self._select_azimuth(azimuth=azimuth)

        if values.ndim == 3:
            values = values[..., selected]

        plt = self._pyplot()

        x = (
            np.arange(values.shape[1])
            if self._result.angles is None
            else np.rad2deg(self._result.angles.to("radian").magnitude)
        )

        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

        for index, curve in enumerate(values):
            errors = None

            if not terms and self._result.stderr is not None:
                candidate = self._result.stderr.to("1 / meter / steradian").magnitude[index]

                if candidate.ndim == 2:
                    candidate = candidate[:, selected]

                if np.any(np.isfinite(candidate)):
                    errors = scale * np.ma.masked_invalid(candidate)

            label = f"{'Term' if terms else 'Through order'} {index + 1}"

            style = {"marker": "o", "linestyle": "none"} if self._result.angles is None else {"linewidth": 2}

            if errors is None:
                axis.plot(
                    x,
                    curve,
                    label=label,
                    **style,
                )
            else:
                axis.errorbar(x, curve, yerr=errors, capsize=2, label=label, **style)

        default_title = "Analytical scattering" if self._result.kind == "analytical" else "Finite-sample scattering"

        default_title += self._angular_label(azimuth=selected)

        axis.set(
            title=default_title if title is None else title,
            xlabel="Direction index" if self._result.angles is None else "Scattering angle (degrees)",
            ylabel=ylabel,
            yscale="log" if log_y else "linear",
        )

        axis.grid(alpha=0.25)

        axis.legend(frameon=False)

        return figure

    def _select_azimuth(self, *, azimuth):
        count = 1 if self._result.azimuths is None else len(self._result.azimuths)

        return _integer(value=azimuth, name="azimuth", low=0, high=count - 1)

    def _angular_label(self, *, azimuth):
        if getattr(self._result, "meridian_azimuth", None) is not None:
            degrees = float(np.rad2deg(self._result.meridian_azimuth.magnitude))

            return f" · meridian phi = {degrees:g} degrees"

        if self._result.azimuth_averaged:
            return " · azimuth average"

        if self._result.azimuths is not None:
            degrees = float(np.rad2deg(self._result.azimuths.magnitude[azimuth]))

            return f" · meridian phi = {degrees:g} degrees"

        return ""

    def plot_phase_function(self, *, view="angular", order=None, log_y=False, azimuth=0, backend=None):
        """Render normalized densities while retaining directional 3D asymmetry."""

        if view != "angular" and getattr(self._result, "meridian_azimuth", None) is not None:
            raise ValueError("Use full angular data for polar or 3D plots; a selected meridian has only one side.")

        if view not in ("angular", "polar", "3d"):
            raise ValueError("view must be angular, polar, or 3d.")

        if backend is None:
            backend = "plotly" if view == "3d" else "matplotlib"

        if backend not in ("matplotlib", "plotly"):
            raise ValueError("backend must be matplotlib or plotly.")

        if backend == "plotly" and view != "3d":
            raise ValueError("The Plotly backend is only supported for the 3D phase view.")

        if log_y and view != "angular":
            raise ValueError("log_y is only supported for the angular view.")

        phase = self._result.phase_function.magnitude

        theta = self._result.angles.to("radian").magnitude

        indices = np.argsort(theta)

        theta, phase = theta[indices], phase[:, indices]

        orders = (
            range(len(phase))
            if order is None
            else [
                _integer(
                    value=order,
                    name="order",
                    low=1,
                    high=len(phase),
                )
                - 1
            ]
        )

        insufficient_angular_coverage = view in ("polar", "3d") and (
            len(np.unique(theta)) < 3
            or not np.isclose(theta[0], 0, atol=1e-10, rtol=0)
            or not np.isclose(theta[-1], np.pi, atol=1e-10, rtol=0)
        )

        if insufficient_angular_coverage:
            raise ValueError("Polar and 3D views require at least three distinct angles spanning 0 to pi.")

        title = (
            "Analytical phase function"
            if self._result.kind == "analytical"
            else "Directional finite-sample phase function"
        )

        meridian = self._select_azimuth(azimuth=azimuth)

        degrees = np.rad2deg(theta)

        if view == "3d":
            selected = len(phase) - 1 if order is None else orders[0]

            if phase.ndim == 3:
                phi = np.concatenate([self._result.azimuths.magnitude, [2 * np.pi]])

                directional = phase[selected]

                radius = np.concatenate([directional, directional[:, :1]], axis=-1)

                surface_label = "full azimuthal distribution"
            elif self._result.kind == "analytical" or self._result.azimuth_averaged:
                phi = np.linspace(0, 2 * np.pi, 97)

                radius = np.broadcast_to(phase[selected, :, None], (len(theta), len(phi)))

                surface_label = (
                    "axisymmetric analytical surface"
                    if self._result.kind == "analytical"
                    else "explicit azimuth-average surface"
                )

                if self._result.azimuth_averaged:
                    title = "Azimuth-averaged finite-sample phase function"
            else:
                raise ValueError(
                    "Directional phase data are unavailable; recompute with Solver.solve using AngularSampling."
                )

            sine = np.sin(theta)[:, None]

            x = radius * sine * np.cos(phi)

            y = radius * sine * np.sin(phi)

            z = radius * np.cos(theta)[:, None]

            if backend == "plotly":
                import plotly.graph_objects as go

                figure = go.Figure(
                    data=[
                        go.Surface(
                            x=x,
                            y=y,
                            z=z,
                            surfacecolor=radius,
                            customdata=radius,
                            colorscale="Viridis",
                            cmin=0,
                            cmax=float(radius.max()),
                            colorbar={"title": "p (sr⁻¹)"},
                            hovertemplate="Phase density p=%{customdata:.5g} sr⁻¹<extra></extra>",
                        )
                    ],
                )

                figure.update_layout(
                    title=f"{title}<br>Through order {selected + 1} · {surface_label}",
                    scene={
                        "xaxis_title": "p ŝx (sr⁻¹)",
                        "yaxis_title": "p ŝy (sr⁻¹)",
                        "zaxis_title": "p ŝz (sr⁻¹); incidence +z",
                        "aspectmode": "data",
                    },
                )

                return figure

            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            plt = self._pyplot()

            norm = Normalize(vmin=0, vmax=float(radius.max()))

            cmap = plt.get_cmap("viridis")

            figure = plt.figure(figsize=(9, 6), layout="constrained")

            axis = figure.add_subplot(projection="3d")

            axis.plot_surface(
                x,
                y,
                z,
                facecolors=cmap(norm(radius)),
                rcount=len(theta),
                ccount=len(phi),
                linewidth=0,
                shade=False,
            )

            extent = np.array([np.ptp(x), np.ptp(y), np.ptp(z)])

            axis.set_box_aspect(extent)

            axis.set(
                xlabel="p ŝx (sr⁻¹)",
                ylabel="p ŝy (sr⁻¹)",
                zlabel="p ŝz (sr⁻¹); incidence +z",
            )

            axis.set_title(f"{title}\nThrough order {selected + 1} · {surface_label}")

            axis.view_init(elev=20, azim=45)

            figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, label="p (sr⁻¹)", shrink=0.75, pad=0.12)
        else:
            plt = self._pyplot()

            opposite = None

            if phase.ndim == 3:
                if view == "polar":
                    count = len(self._result.azimuths)

                    if count % 2:
                        raise ValueError("A full polar plane requires an even azimuth sample count.")

                    opposite = phase[..., (meridian + count // 2) % count]

                phase = phase[..., meridian]

            title += self._angular_label(azimuth=meridian)

            projection = "polar" if view == "polar" else None

            figure, axis = plt.subplots(figsize=(8, 5), subplot_kw={"projection": projection}, layout="constrained")

            for index in orders:
                if view == "polar":
                    axis.plot(
                        np.concatenate([theta, 2 * np.pi - theta[-2::-1]]),
                        np.concatenate([phase[index], (phase if opposite is None else opposite)[index, -2::-1]]),
                        label=f"Through order {index + 1}",
                        linewidth=2,
                    )
                else:
                    axis.plot(
                        degrees,
                        phase[index],
                        label=f"Through order {index + 1}",
                        linewidth=2,
                    )

            if view == "polar":
                axis.set_theta_zero_location("N")

                axis.set_theta_direction(-1)

                axis.set_ylim(bottom=0)

                axis.set_title(f"{title}\nMeridian cut · incidence at 0° · p (sr⁻¹)", pad=20)
            else:
                axis.set(
                    title=title,
                    xlabel="Scattering angle θ (degrees)",
                    ylabel="Phase function p (sr⁻¹)",
                    yscale="log" if log_y else "linear",
                )

            axis.grid(alpha=0.25)

            axis.legend(frameon=False)

        return figure

    def plot_field_norms(self, *, log_y=True):
        """Draw relative Born field norms for each available realization."""

        if self._result.field_norms is None:
            raise ValueError("Field norms are only available for numerical results.")

        norms = np.atleast_2d(
            _dimensionless(
                value=self._result.field_norms,
                name="field_norms",
            )
        )

        plt = self._pyplot()

        from matplotlib.ticker import MaxNLocator

        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

        for index, curve in enumerate(norms):
            axis.plot(
                np.arange(1, len(curve) + 1),
                curve,
                "o-",
                label=f"Realization {index + 1}",
            )

        axis.set(
            title="Born field terms\nDecreasing terms do not certify convergence",
            xlabel="Field-term order j",
            ylabel="‖Ej‖ / ‖Einc‖ (dimensionless)",
            yscale="log" if log_y else "linear",
        )

        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

        axis.grid(alpha=0.25)

        axis.legend(frameon=False)

        return figure
