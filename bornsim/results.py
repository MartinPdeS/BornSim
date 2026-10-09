"""Unitful scattering results, validated archives, and Matplotlib plots."""

from dataclasses import dataclass
from types import SimpleNamespace
import json
import numpy as np
from .source import Source
from ._archives import _RESULT_UNITS
from .angular_data import AngularData
from .units import Quantity, validate_units, ureg

_ANGULAR_FIELDS = (
    "differential",
    "directions",
    "angles",
    "azimuths",
    "amplitudes",
    "term_differential",
    "stderr",
    "azimuth_stderr",
    "mu_s",
    "sample_volume",
)


@dataclass(frozen=True, kw_only=True, repr=False, init=False)
class Result:
    """Store unitful scattering, diagnostics and reproducible settings.

    Full solves retain ``differential`` and ``term_differential`` with shape
    (order, polar angle, azimuth). Cuts and explicitly averaged results use
    (order, observation). ``amplitudes`` always adds polarization and Cartesian
    axes of sizes two and three to the differential shape. Ensembles retain
    intensities and standard errors, without coherent amplitudes.

    ``angular`` groups read-only coordinates, intensities, amplitudes and phase
    densities. ``azimuth_average()`` explicitly averages intensities and retains
    the covariance-correct standard error of per-realization averaged curves.
    Plots of full results select one sampled meridian by default.

    Angles use radians, amplitudes metres, differential data m^-1 sr^-1,
    integrated coefficients m^-1, and sample_volume m^3. Dimensional arrays require explicit units.
    Every full phase density uses its order's solid-angle integral. Cuts have
    no inferred integrated coefficients or phase normalization.

    Numerical coefficients are finite-sample cross sections divided by the
    entire voxel-box volume. ``differential_cross_section`` restores area per
    steradian using stored sample_volume. Analytical coefficients describe an
    infinite-medium model and have no finite-sample cross section.

    Source and kind are required. Supply angular or individual differential
    data, never both. Results are immutable; provenance returns a fresh copy. Other fields describe sampling,
    isolated terms, integrated moments, uncertainty, diagnostics and provenance.
    NaN errors represent one realization; NaN anisotropy represents zero
    scattering. Arrays and JSON-compatible provenance are copied and validated.
    The directional_differential and directional_amplitudes fields are legacy
    aliases; use differential and amplitudes for both cuts and full solves.

    Save writes schema 2 archives; load also promotes schema 1 directional data.
    Numerical cumulative intensities retain coherent amplitude interference.
    Isolated intensities must not be summed to reconstruct cumulative results.
    """

    source: Source
    kind: str
    angular: AngularData
    g: Quantity | None
    mu_s_prime: Quantity | None
    field_norms: Quantity | None
    warnings: tuple
    realizations: int | None
    _provenance_json: str

    def __init__(
        self,
        *,
        source,
        kind,
        angular=None,
        differential=None,
        sample_volume=None,
        azimuth_averaged=False,
        azimuth_stderr=None,
        angles=None,
        directions=None,
        azimuths=None,
        directional_differential=None,
        mu_s=None,
        g=None,
        mu_s_prime=None,
        amplitudes=None,
        directional_amplitudes=None,
        term_differential=None,
        stderr=None,
        field_norms=None,
        warnings=(),
        realizations=None,
        provenance=None,
    ):
        if not isinstance(source, Source):
            raise TypeError("source must be a Source.")

        if kind not in ("analytical", "volume", "ensemble"):
            raise ValueError("kind must be analytical, volume, or ensemble.")

        values = {name: value for name, value in locals().items() if name in _RESULT_UNITS}

        if angular is not None:
            if not isinstance(angular, AngularData):
                raise TypeError("angular must be an AngularData.")

            conflicts = any(
                value is not None for name, value in values.items() if name not in ("g", "mu_s_prime", "field_norms")
            )

            if conflicts or azimuth_averaged:
                raise ValueError("Supply angular or individual angular fields, not both.")

            if angular.kind != kind:
                raise ValueError("angular.kind must match result kind.")

            for name in _ANGULAR_FIELDS:
                values[name] = getattr(angular, name)

            azimuth_averaged = angular.azimuth_averaged

            # Result stores explicit cut directions only; full vectors live in AngularData.
            if angular.azimuths is not None or kind != "volume":
                values["directions"] = None

        for name, value in values.items():
            if value is not None and (angular is None or name not in _ANGULAR_FIELDS):
                unit = _RESULT_UNITS[name]

                if unit == "dimensionless" and not isinstance(value, Quantity):
                    value = np.array(value, copy=True) * ureg.dimensionless

                validate_units(
                    value,
                    unit=unit,
                    name=name,
                )

                if not np.issubdtype(np.asarray(value.magnitude).dtype, np.number):
                    raise ValueError(f"{name} must contain numeric values.")

                values[name] = np.array(value.magnitude, copy=True) * value.units

        provenance = {} if provenance is None else provenance

        if not isinstance(provenance, dict):
            raise ValueError("provenance must be a JSON-compatible dictionary.")

        try:
            encoded = json.dumps(provenance, allow_nan=False)

            provenance = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must be a JSON-compatible dictionary with finite values.") from error

        if not isinstance(warnings, (tuple, list)) or any(not isinstance(item, str) for item in warnings):
            raise ValueError("warnings must be a sequence of strings.")

        state = SimpleNamespace(**values, kind=kind, realizations=realizations, azimuth_averaged=azimuth_averaged)

        from ._archives import _ResultArchive

        _ResultArchive._promote_legacy_directional_data(result=state)

        if state.sample_volume is None and kind != "analytical":
            grid = provenance.get("grid")

            if isinstance(grid, dict) and "shape" in grid and "spacing_m" in grid:
                state.sample_volume = float(np.prod(grid["shape"]) * grid["spacing_m"] ** 3) * ureg.meter**3

        from ._result_validation import _ResultValidator

        if angular is None:
            angular = AngularData(
                **{name: getattr(state, name) for name in _ANGULAR_FIELDS}, kind=kind, azimuth_averaged=azimuth_averaged
            )

        _ResultValidator.validate(result=state)

        for name, value in {
            "source": source,
            "kind": kind,
            "angular": angular,
            "warnings": tuple(warnings),
            "realizations": state.realizations,
            "_provenance_json": encoded,
        }.items():
            object.__setattr__(self, name, value)

        for name in ("g", "mu_s_prime", "field_norms"):
            value = getattr(state, name)

            if value is not None:
                value.magnitude.setflags(write=False)

            object.__setattr__(self, name, value)

    @property
    def differential(self):
        """Read-only angular data owned by angular."""

        return self.angular.differential

    @property
    def angles(self):
        """Read-only angular data owned by angular."""

        return self.angular.angles

    @property
    def azimuths(self):
        """Read-only angular data owned by angular."""

        return self.angular.azimuths

    @property
    def amplitudes(self):
        """Read-only angular data owned by angular."""

        return self.angular.amplitudes

    @property
    def mu_s(self):
        """Read-only angular data owned by angular."""

        return self.angular.mu_s

    @property
    def sample_volume(self):
        """Read-only angular data owned by angular."""

        return self.angular.sample_volume

    @property
    def stderr(self):
        """Read-only angular data owned by angular."""

        return self.angular.stderr

    @property
    def azimuth_stderr(self):
        """Read-only angular data owned by angular."""

        return self.angular.azimuth_stderr

    @property
    def term_differential(self):
        """Read-only angular data owned by angular."""

        return self.angular.term_differential

    @property
    def azimuth_averaged(self):
        """Read-only angular data owned by angular."""

        return self.angular.azimuth_averaged

    @property
    def directions(self):
        """Explicit cut coordinates; full-grid vectors are angular.directions."""

        return self.angular.directions if self.angular.azimuths is None and self.kind == "volume" else None

    @property
    def directional_differential(self):
        """Legacy alias for full directional intensities."""

        return self.differential if self.differential.ndim == 3 else None

    @property
    def directional_amplitudes(self):
        """Legacy alias for full coherent amplitudes."""

        return self.amplitudes if self.differential.ndim == 3 else None

    @property
    def provenance(self):
        """Return a fresh metadata copy; edits cannot change this result."""

        return json.loads(self._provenance_json)

    def meridian(self, *, azimuth, method="exact"):
        """Select a physical azimuth from full data without interpolation."""

        return self.angular.meridian(
            azimuth=azimuth,
            method=method,
        )

    def azimuth_average(self):
        """Return explicit averaged intensities with covariance-correct errors."""

        angular = self.angular.azimuth_average()

        return Result(
            source=self.source,
            kind=self.kind,
            angular=angular,
            g=self.g,
            mu_s_prime=self.mu_s_prime,
            field_norms=self.field_norms,
            warnings=self.warnings,
            realizations=self.realizations,
            provenance=self.provenance,
        )

    @property
    def differential_cross_section(self):
        """Finite-sample d-sigma/d-Omega in square metres per steradian."""

        if self.sample_volume is None:
            raise ValueError("sample_volume is unavailable; this result has no finite-sample cross section.")

        return (getattr(self, "differential") * self.sample_volume).to("meter**2 / steradian")

    def __repr__(self):
        shape = self.differential.shape

        available = [
            name for name in ("amplitudes", "mu_s", "stderr", "sample_volume") if getattr(self, name) is not None
        ]

        return f"Result(kind={self.kind!r}, orders={shape[0]}, angular_shape={shape[1:]}, available={available}, azimuth_averaged={self.azimuth_averaged})"

    def save(self, *, path):
        """Save data, units, warnings, and provenance in a versioned NPZ archive.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file, typically ending in ``.npz``. The exact path is
            used without appending an extension. An existing file is replaced.

        Returns
        -------
        path : pathlib.Path
            Destination of the compressed NumPy archive.

        Raises
        ------
        ValueError
            If result fields were modified into an invalid state.
        OSError
            If the destination cannot be written.

        Notes
        -----
        Arrays are stored in SI units, including complex Born amplitudes.
        Metadata is JSON, and loading never enables pickle. The archive stores
        result data rather than the original voxel field; retain that field
        separately for manual volumes. A field SHA-256 identifies the input.
        Generated volumes also record their medium and seed. Exact seeded
        reproduction depends on the recorded implementation versions.
        """

        from ._archives import _ResultArchive

        return _ResultArchive.save(
            result=self,
            path=path,
        )

    @classmethod
    def load(cls, *, path):
        """Load and validate a BornSim result archive without using pickle.

        Parameters
        ----------
        path : str or pathlib.Path
            Archive created by ``Result.save``.

        Returns
        -------
        result : Result
            Restored quantities, complex amplitudes, diagnostics, and original
            provenance. Loading does not change the recorded package version.

        Raises
        ------
        ValueError
            If the schema is unsupported, metadata or arrays are missing or
            inconsistent, or result validation fails.
        OSError
            If the archive cannot be read.
        """

        from ._archives import _ResultArchive

        return _ResultArchive.load(
            result_type=cls,
            path=path,
        )

    @property
    def phase_function(self):
        """Return directional phase density per steradian.

        Full solves retain (order, polar angle, azimuth). Explicitly averaged
        results retain (order, polar angle). Cuts have no normalization.
        Every order uses its own full solid-angle scattering coefficient.
        """

        return self.angular.phase_function

    def plot(self, *, terms=False, log_y=False, title=None, azimuth=0):
        """Build a Matplotlib figure of differential scattering curves.

        Parameters
        ----------
        terms : bool, optional
            Plot isolated numerical terms instead of cumulative curves. Default is
            False. Isolated terms omit interference and ensemble error bars.
        log_y : bool, optional
            Use a logarithmic scattering axis. Default is False.
        azimuth : int, optional
            Index of the sampled meridian; default zero. Use azimuth_average()
            explicitly to plot averaged intensities.
        title : str, optional
            Override the default title.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with angles in degrees and scattering in m^-1 sr^-1. Explicit
            direction samples use input indices and markers. Ensemble cumulative
            curves show one-standard-error bars where sampling error is known.

        Raises
        ------
        ValueError
            If ``terms=True`` and isolated-term curves are unavailable.

        Notes
        -----
        The figure is registered with pyplot and returned without displaying
        it. Call ``matplotlib.pyplot.show()`` to display open figures or
        ``figure.savefig(path)`` to export a PNG, SVG, or PDF. Zero values cannot be displayed on a log axis.
        A one-realization ensemble has unknown sampling error, so no error bars
        are shown. This method plots differential scattering, not a normalized
        phase function.

        Examples
        --------
        >>> from bornsim import AnalyticalMedium, Solver, Source
        ...
        >>> from bornsim.units import ureg
        ...
        >>> solver = Solver(
        ...     source=Source(
        ...         wavelength=633e-9 * ureg.meter,
        ...     )
        ... )

        ...
        >>> result = solver.solve(
        ...     target=AnalyticalMedium(
        ...         background_refractive_index=1.33,
        ...         refractive_index_std=0.01,
        ...         correlation_length=100e-9 * ureg.meter,
        ...         correlation="gaussian",
        ...     )
        ... )
        >>> figure = result.plot(log_y=True)
        >>> len(figure.axes[0].lines)
        1
        """

        from ._result_plotting import _ResultPlotter

        return _ResultPlotter(
            result=self,
        ).plot(
            terms=terms,
            log_y=log_y,
            azimuth=azimuth,
            title=title,
        )

    def plot_cross_section(
        self, *, volume=None, area_unit="nanometer**2", terms=False, log_y=False, title=None, azimuth=0
    ):
        """Build differential cross-section curves for a finite sample.

        Parameters
        ----------
        volume : Volume, optional
            Optional legacy sample. Normally the recorded sample_volume
            supplies the conversion without retaining the input voxel field.
        area_unit : str, optional
            Display area unit; default 'nanometer**2'.
        terms : bool, optional
            Show isolated terms, which omit interference and sampling error
            bars, instead of cumulative coherent curves. Default False.
        log_y : bool, optional
            Use a logarithmic scattering axis. Default False.
        azimuth : int, optional
            Index of the sampled meridian; default zero. Use azimuth_average()
            explicitly to plot averaged intensities.
        title : str, optional
            Override the default title.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Angle or direction curves, with ensemble standard errors scaled
            by the same sample volume. Returned without displaying it.

        Raises
        ------
        TypeError
            If volume is not a Volume.
        ValueError
            If the result is analytical, recorded grid dimensions differ,
            area_unit is invalid, or isolated curves are unavailable.

        Notes
        -----
        Uses dσ/dΩ = V * differential. These are finite-sample cross sections,
        not intrinsic infinite-medium transport coefficients. An optional supplied
        volume must agree with the recorded physical volume and grid.
        Stored scattering data and its SI units remain unchanged.
        """

        from ._result_plotting import _ResultPlotter

        return _ResultPlotter(
            result=self,
        ).plot_cross_section(
            volume=volume,
            area_unit=area_unit,
            terms=terms,
            log_y=log_y,
            azimuth=azimuth,
            title=title,
        )

    @property
    def directional_phase_function(self):
        """Compatibility alias for the primary full phase_function."""

        if self.differential.ndim != 3:
            raise ValueError(
                "Directional phase data are unavailable; recompute with Solver.solve using AngularSampling."
            )

        return self.phase_function

    def plot_phase_function(self, *, view="angular", order=None, log_y=False, azimuth=0, backend=None):
        """Plot normalized phase functions as angular curves, polar cuts, or a surface.

        Parameters
        ----------
        view : {'angular', 'polar', '3d'}, optional
            Angular curves select a sampled meridian by default. Polar cuts
            use its opposite azimuth for the other half of the plane. The 3D surface uses p(theta, phi), with
            radius and color in sr^-1, retaining directional asymmetry.
            Analytical distributions are independent of phi.
        order : int, optional
            One-based cumulative order. Angular and polar views show all
            orders by default; the 3D view shows only the highest order.
        log_y : bool, optional
            Logarithmic vertical scale for angular curves. Default is False;
            only supported with ``view='angular'``.
        backend : {'plotly', 'matplotlib'}, optional
            The 3D view defaults to Plotly. Angular and polar views use
            Matplotlib. Select 'matplotlib' explicitly for a static 3D figure.

        Returns
        -------
        figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
            Returned without displaying it. Call ``figure.show()`` for Plotly,
            or ``matplotlib.pyplot.show()`` for Matplotlib. The 3D coordinates
            are probability-density radii, not spatial positions.

        Raises
        ------
        ValueError
            If phase normalization is unavailable, the view or order is
            invalid, directional data are unavailable for a numerical 3D view,
            log scaling is requested for another view, or polar/3D data do not
            span the full [0, pi] range with at least three angles.

        Notes
        -----
        Numerical 3D surfaces retain every sampled azimuth; no azimuth
        averaging is applied. For one realization this displays the fixed
        sample's directional distribution; multiple realizations display
        the ensemble intensity average. Older archives lacking directional
        data must be recomputed for a directional 3D view. Angular and polar
        views select sampled meridians; averaging requires azimuth_average(). A flat phase function produces a spherical
        surface; forward scattering extends towards +z.
        Samples are sorted by angle for rendering. Error bars for normalized
        ratios are omitted because the required covariance is not stored.

        Examples
        --------
        >>> from bornsim import AnalyticalMedium, Solver, Source
        ...
        >>> from bornsim.units import ureg
        ...
        >>> solver = Solver(
        ...     source=Source(
        ...         wavelength=633e-9 * ureg.meter,
        ...     )
        ... )

        ...
        >>> result = solver.solve(
        ...     target=AnalyticalMedium(
        ...         background_refractive_index=1.33,
        ...         refractive_index_std=0.01,
        ...         correlation_length=100e-9 * ureg.meter,
        ...         correlation="gaussian",
        ...     )
        ... )
        >>> figure = result.plot_phase_function(view="3d")
        >>> figure.data[0].type
        'surface'
        """

        from ._result_plotting import _ResultPlotter

        return _ResultPlotter(
            result=self,
        ).plot_phase_function(
            view=view,
            order=order,
            log_y=log_y,
            azimuth=azimuth,
            backend=backend,
        )

    def plot_field_norms(self, *, log_y=True):
        """Plot relative Born field-term norms for each available realization.

        Parameters
        ----------
        log_y : bool, optional
            Use a logarithmic vertical axis. Default is True. Zero norms
            cannot be displayed on a logarithmic axis.

        Returns
        -------
        figure : matplotlib.figure.Figure
            One trace per realization versus one-based field-term order.
            Norms are dimensionless relative to the incident field.

        Raises
        ------
        ValueError
            If no numerical field norms are available.

        Notes
        -----
        The figure preserves per-realization diagnostics. Decreasing field
        terms do not certify Born convergence or remove discretization error.
        """

        from ._result_plotting import _ResultPlotter

        return _ResultPlotter(
            result=self,
        ).plot_field_norms(
            log_y=log_y,
        )


@dataclass(kw_only=True)
class BornResult:
    """Store numeric SI results for one finite-volume Born calculation.

    Parameters
    ----------
    directions : numpy.ndarray
        Dimensionless observation vectors, shape (observation, 3).
    amplitudes : numpy.ndarray
        Complex isolated amplitudes in metres, shape
        (order, observation, incident polarization, vector component).
    differential : numpy.ndarray
        Cumulative scattering in m^-1 sr^-1, shape (order, observation).
    term_differential : numpy.ndarray
        Isolated-term scattering in m^-1 sr^-1, same shape as differential.
    field_norms : numpy.ndarray
        Dimensionless relative field norms, shape (order,).
    warnings : tuple of str
        Numerical-resolution and Born-term diagnostics.

    Attributes
    ----------
    directions, amplitudes, differential, term_differential, field_norms : numpy.ndarray
        Arrays described above, without attached units.
    warnings : tuple of str
        Numerical diagnostics.

    See Also
    --------
    bornsim.series.BornSeries.solve : Produce this numeric result.
    bornsim.results.Result : Unitful result with plotting support.

    Notes
    -----
    Row zero denotes first order. Cumulative intensities include interference
    between amplitudes; isolated intensities must not be summed to reconstruct
    them. Integrated coefficients are not inferred from observation samples.
    """

    directions: np.ndarray
    amplitudes: np.ndarray  # (order, direction, incident polarization, vector)
    differential: np.ndarray  # cumulative orders; m^-1 sr^-1
    term_differential: np.ndarray  # each term alone, excludes interference
    field_norms: np.ndarray  # ||E_j|| / ||E_inc||
    warnings: tuple[str, ...]
