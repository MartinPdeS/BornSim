"""Unitful scattering results, validated archives, and Matplotlib plots."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar
import json
import numpy as np
from .source import Source
from .angular_data import AngularData
from .units import Quantity, _dimensionless, _integer, validate_units, ureg

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

_DIAGNOSTIC_FIELDS = (
    "g",
    "mu_s_prime",
    "mu_a",
    "mu_a_stderr",
    "residual_norms",
    "coherent_differential",
    "field_norms",
)


@dataclass(frozen=True, kw_only=True, repr=False, init=False)
class Result:
    """Store unitful scattering, diagnostics and reproducible settings.

    Full solves retain ``differential`` and ``term_differential`` with shape
    (order, polar angle, azimuth). Cuts and explicitly averaged results use
    (order, observation). ``amplitudes`` always adds polarization and Cartesian
    axes of sizes one or two and three to the differential shape. Ensembles retain
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
    steradian using stored sample_volume.

    ``mu_a`` is the finite-sample absorption coefficient, if available. At
    scattering order M it uses the cumulative internal field through M-1.
    ``absorption_cross_section`` restores absorbed-power cross sections.

    Source and kind are required. Supply angular or individual differential
    data, never both. Results are immutable; provenance returns a fresh copy. Other fields describe sampling,
    isolated terms, integrated moments, uncertainty, diagnostics and provenance.
    NaN errors represent one realization; NaN anisotropy represents zero
    scattering. Arrays and JSON-compatible provenance are copied and validated.
    Use differential and amplitudes for both cuts and full solves.

    Save and load use schema 3 archives (schema 2 remains readable); unsupported schemas are rejected.
    Numerical cumulative intensities retain coherent amplitude interference.
    Isolated intensities must not be summed to reconstruct cumulative results.
    """

    source: Source
    kind: str
    angular: AngularData
    g: Quantity | None
    mu_a: Quantity | None
    residual_norms: Quantity | None
    mu_a_stderr: Quantity | None
    coherent_differential: Quantity | None
    mu_s_prime: Quantity | None
    field_norms: Quantity | None
    warnings: tuple
    realizations: int | None
    _provenance_json: str

    _RESULT_UNITS: ClassVar[dict[str, str]] = {
        "differential": "1 / meter / steradian",
        "term_differential": "1 / meter / steradian",
        "stderr": "1 / meter / steradian",
        "angles": "radian",
        "azimuths": "radian",
        "amplitudes": "meter",
        "mu_s": "1 / meter",
        "mu_s_prime": "1 / meter",
        "mu_a": "1 / meter",
        "mu_a_stderr": "1 / meter",
        "residual_norms": "dimensionless",
        "coherent_differential": "1 / meter / steradian",
        "g": "dimensionless",
        "directions": "dimensionless",
        "field_norms": "dimensionless",
        "sample_volume": "meter**3",
        "azimuth_stderr": "1 / meter / steradian",
    }

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
        mu_s=None,
        g=None,
        mu_s_prime=None,
        mu_a=None,
        mu_a_stderr=None,
        residual_norms=None,
        coherent_differential=None,
        amplitudes=None,
        term_differential=None,
        stderr=None,
        field_norms=None,
        warnings=(),
        realizations=None,
        provenance=None,
    ):
        if not isinstance(source, Source):
            raise TypeError("source must be a Source.")

        if kind not in ("volume", "ensemble"):
            raise ValueError("kind must be volume or ensemble.")

        if angular is None and differential is None:
            raise ValueError("differential is required.")

        values = {name: value for name, value in locals().items() if name in self._RESULT_UNITS}

        if angular is not None:
            if not isinstance(angular, AngularData):
                raise TypeError("angular must be an AngularData.")

            conflicts = any(value is not None for name, value in values.items() if name not in _DIAGNOSTIC_FIELDS)

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
                unit = self._RESULT_UNITS[name]

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

        if angular is None:
            angular = AngularData(
                **{name: getattr(state, name) for name in _ANGULAR_FIELDS}, kind=kind, azimuth_averaged=azimuth_averaged
            )

        if angular.amplitudes is not None and angular.amplitudes.shape[-2] != len(source.polarization_vectors):
            raise ValueError("amplitudes polarization channels must match the source.")

        self._validate_state(result=state)

        for name, value in {
            "source": source,
            "kind": kind,
            "angular": angular,
            "warnings": tuple(warnings),
            "realizations": state.realizations,
            "_provenance_json": encoded,
        }.items():
            object.__setattr__(self, name, value)

        for name in _DIAGNOSTIC_FIELDS:
            value = getattr(state, name)

            if value is not None:
                value.magnitude.setflags(write=False)

            object.__setattr__(self, name, value)

    @staticmethod
    def _validate_state(*, result: SimpleNamespace) -> None:
        """Validate result axes, physical ranges and realization uncertainty."""

        orders, observations = result.differential.shape[:2]

        if result.kind == "ensemble":
            result.realizations = _integer(value=result.realizations, name="realizations", low=1, high=32)
        elif result.realizations is not None:
            raise ValueError("realizations is only available for an ensemble.")

        if result.kind != "volume" and result.directions is not None:
            raise ValueError("ensemble results require angles and no directions.")

        if result.differential.ndim == 3 and result.directions is not None:
            raise ValueError("Full data require angles and azimuths without cut directions.")

        shapes = {
            "g": (orders,),
            "mu_s_prime": (orders,),
            "mu_a": (orders,),
            "mu_a_stderr": (orders,),
            "residual_norms": (result.realizations, orders) if result.kind == "ensemble" else (orders,),
            "coherent_differential": result.differential.shape,
            "field_norms": (result.realizations, orders) if result.kind == "ensemble" else (orders,),
        }

        if result.coherent_differential is not None:
            if result.kind != "ensemble":
                raise ValueError("coherent_differential requires an ensemble.")
            if result.coherent_differential.shape != result.differential.shape:
                raise ValueError("coherent_differential must match differential.")
            total = result.differential.to("1 / meter / steradian").magnitude
            coherent = result.coherent_differential.to("1 / meter / steradian").magnitude
            if np.any(coherent > total + 1e-12 * np.maximum(total, np.finfo(float).tiny)):
                raise ValueError("Coherent intensity cannot exceed total ensemble intensity.")

        for name, shape in shapes.items():
            quantity = getattr(result, name)

            if quantity is None:
                continue

            values = quantity.magnitude

            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}; got {values.shape}.")

            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"{name} must contain numeric real values.")

            if np.any(np.isinf(values)) or (name not in ("g", "mu_a_stderr") and np.any(~np.isfinite(values))):
                raise ValueError(f"{name} must be finite, except for documented unknown values.")

            if name != "g" and np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative.")

        if result.g is not None:
            g = result.g.magnitude

            if np.any(np.abs(g) > 1 + 1e-12):
                raise ValueError("g must lie between -1 and 1.")

            if result.mu_s is None and np.any(np.isnan(g)):
                raise ValueError("NaN g requires a zero mu_s coefficient.")

            if result.mu_s is not None and np.any(np.isnan(g) != (result.mu_s.magnitude == 0)):
                raise ValueError("g must be NaN exactly where mu_s is zero.")

        if all(value is not None for value in (result.mu_s, result.g, result.mu_s_prime)):
            expected = result.mu_s * (1 - np.nan_to_num(result.g.to("dimensionless").magnitude))

            if not np.allclose(result.mu_s_prime, expected, rtol=1e-10, atol=0):
                raise ValueError("mu_s_prime must equal mu_s * (1 - g), with zero at zero scattering.")

        if result.mu_a_stderr is not None and result.kind != "ensemble":
            raise ValueError("mu_a_stderr is only available for ensembles.")

        for name in ("stderr", "azimuth_stderr", "mu_a_stderr"):
            errors = getattr(result, name)

            if result.realizations == 1 and errors is not None and not np.all(np.isnan(errors.magnitude)):
                raise ValueError(f"{name} must be NaN for one realization.")

            if result.realizations != 1 and errors is not None and np.any(~np.isfinite(errors.magnitude)):
                raise ValueError(f"{name} must be finite for multiple realizations.")

        incomplete_integrals = (
            result.kind == "volume"
            and result.differential.ndim == 2
            and not result.azimuth_averaged
            and any(value is not None for value in (result.mu_s, result.g, result.mu_s_prime))
        )

        if incomplete_integrals:
            raise ValueError("integrated coefficients are unavailable for a single-volume angular cut.")

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
            mu_a=self.mu_a,
            mu_a_stderr=self.mu_a_stderr,
            residual_norms=self.residual_norms,
            coherent_differential=(
                self.coherent_differential.mean(axis=-1)
                if self.coherent_differential is not None and self.differential.ndim == 3
                else self.coherent_differential
            ),
            field_norms=self.field_norms,
            warnings=self.warnings,
            realizations=self.realizations,
            provenance=self.provenance,
        )

    @property
    def diffuse_differential(self):
        """Finite-ensemble mean intensity minus squared mean amplitude (m⁻¹ sr⁻¹).

        The coherent estimate includes the ensemble-mean scattered field, not
        the direct incident beam. With N samples, this uncorrected diffuse
        estimate has factor (N-1)/N finite-sample bias; it is zero for N=1.
        """
        if self.coherent_differential is None:
            return None

        values = np.maximum(self.differential - self.coherent_differential, 0 * self.differential.units)
        values.magnitude.setflags(write=False)
        return values

    @property
    def absorption_cross_section(self):
        """Absorbed power divided by incident irradiance, in area units."""
        if self.mu_a is None or self.sample_volume is None:
            return None

        return self.mu_a * self.sample_volume

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

        from pathlib import Path

        arrays = {}

        units = {}

        for name, unit in self._RESULT_UNITS.items():
            value = getattr(self, name)

            if value is not None:
                arrays[name] = value.to(unit).magnitude

                units[name] = unit

        metadata = {
            "format": "bornsim-result",
            "schema_version": 3,
            "source": self.source.metadata,
            "azimuth_averaged": self.azimuth_averaged,
            "wavelength_m": float(self.source.wavelength.to("meter").magnitude),
            "kind": self.kind,
            "realizations": self.realizations,
            "warnings": self.warnings,
            "provenance": self.provenance,
            "units": units,
        }

        encoded_metadata = np.array(json.dumps(metadata, allow_nan=False))

        path = Path(path)

        with path.open("wb") as stream:
            np.savez_compressed(stream, metadata=encoded_metadata, **arrays)

        return path

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

        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))

                unsupported_schema = (
                    metadata["format"] != "bornsim-result"
                    or type(metadata["schema_version"]) is not int
                    or metadata["schema_version"] not in (2, 3)
                )

                if unsupported_schema:
                    raise ValueError("Unsupported result archive schema.")

                units = metadata["units"]

                if not isinstance(units, dict) or not set(units) <= set(cls._RESULT_UNITS):
                    raise ValueError("Invalid result archive fields.")

                if any(unit != cls._RESULT_UNITS[name] for name, unit in units.items()):
                    raise ValueError("Invalid result archive units; archives require SI units.")

                if set(archive.files) != set(units) | {"metadata"} or "differential" not in units:
                    raise ValueError("Result archive arrays do not match metadata.")

                arrays = {name: archive[name] * ureg.Unit(unit) for name, unit in units.items()}

                return cls(
                    source=(
                        Source._from_metadata(metadata=metadata["source"])
                        if metadata["schema_version"] == 3
                        else Source(wavelength=metadata["wavelength_m"] * ureg.meter)
                    ),
                    kind=metadata["kind"],
                    azimuth_averaged=metadata.get("azimuth_averaged", False),
                    realizations=metadata["realizations"],
                    warnings=metadata["warnings"],
                    provenance=metadata["provenance"],
                    **arrays,
                )
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Invalid or incomplete result archive metadata.") from error

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
        phase function."""

        return self._plot_differential(
            result=self,
            terms=terms,
            log_y=log_y,
            title=title,
            azimuth=azimuth,
        )

    def plot_cross_section(
        self, *, volume=None, area_unit="nanometer**2", terms=False, log_y=False, title=None, azimuth=0
    ):
        """Build differential cross-section curves for a finite sample.

        Parameters
        ----------
        volume : Volume, optional
            Optional sample. Normally the recorded sample_volume
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
            If recorded grid dimensions differ,
            area_unit is invalid, or isolated curves are unavailable.

        Notes
        -----
        Uses dσ/dΩ = V * differential. These are finite-sample cross sections,
        not intrinsic infinite-medium transport coefficients. An optional supplied
        volume must agree with the recorded physical volume and grid.
        Stored scattering data and its SI units remain unchanged.
        """

        return self._plot_cross_section(
            result=self,
            volume=volume,
            area_unit=area_unit,
            terms=terms,
            log_y=log_y,
            azimuth=azimuth,
            title=title,
        )

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
        surface in laboratory coordinates; forward scattering follows the source direction.
        Samples are sorted by angle for rendering. Error bars for normalized
        ratios are omitted because the required covariance is not stored."""

        return self._plot_phase_function(
            result=self,
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

        return self._plot_field_norms(
            result=self,
            log_y=log_y,
        )

    @staticmethod
    def _pyplot():
        import matplotlib.pyplot as plt

        return plt

    @staticmethod
    def _select_azimuth(*, result, azimuth):
        count = 1 if result.azimuths is None else len(result.azimuths)

        return _integer(value=azimuth, name="azimuth", low=0, high=count - 1)

    @staticmethod
    def _angular_label(*, result, azimuth):
        if getattr(result, "meridian_azimuth", None) is not None:
            degrees = float(np.rad2deg(result.meridian_azimuth.magnitude))

            return f" · meridian phi = {degrees:g} degrees"

        if result.azimuth_averaged:
            return " · azimuth average"

        if result.azimuths is not None:
            degrees = float(np.rad2deg(result.azimuths.magnitude[azimuth]))

            return f" · meridian phi = {degrees:g} degrees"

        return ""

    @staticmethod
    def _plot_differential(
        *,
        result,
        terms,
        log_y,
        title,
        azimuth=0,
        scale=1.0,
        ylabel="Differential scattering (m⁻¹ sr⁻¹)",
    ):
        """Render curves and uncertainty using one shared display scale."""

        values = result.term_differential if terms else result.differential

        if values is None:
            raise ValueError("Isolated term curves are unavailable for this result.")

        values = scale * values.to("1 / meter / steradian").magnitude

        selected = Result._select_azimuth(result=result, azimuth=azimuth)

        if values.ndim == 3:
            values = values[..., selected]

        plt = Result._pyplot()

        x = np.arange(values.shape[1]) if result.angles is None else np.rad2deg(result.angles.to("radian").magnitude)

        figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")

        for index, curve in enumerate(values):
            errors = None

            if not terms and result.stderr is not None:
                candidate = result.stderr.to("1 / meter / steradian").magnitude[index]

                if candidate.ndim == 2:
                    candidate = candidate[:, selected]

                if np.any(np.isfinite(candidate)):
                    errors = scale * np.ma.masked_invalid(candidate)

            label = f"{'Term' if terms else 'Through order'} {index + 1}"

            style = {"marker": "o", "linestyle": "none"} if result.angles is None else {"linewidth": 2}

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

        default_title += Result._angular_label(result=result, azimuth=selected)

        axis.set(
            title=default_title if title is None else title,
            xlabel="Direction index" if result.angles is None else "Laboratory polar angle (degrees)",
            ylabel=ylabel,
            yscale="log" if log_y else "linear",
        )

        axis.grid(alpha=0.25)

        axis.legend(frameon=False)

        return figure

    @staticmethod
    def _plot_cross_section(
        *,
        result,
        volume=None,
        area_unit="nanometer**2",
        terms=False,
        log_y=False,
        title=None,
        azimuth=0,
    ):
        """Scale finite-sample curves and uncertainties by their box volume."""

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

        return Result._plot_differential(
            result=result,
            terms=terms,
            log_y=log_y,
            title="Finite-sample differential cross section" if title is None else title,
            scale=physical_volume * float(area.magnitude),
            azimuth=azimuth,
            ylabel=f"Differential cross section ({area.units:~P} sr⁻¹)",
        )

    @staticmethod
    def _plot_phase_function(*, result, view="angular", order=None, log_y=False, azimuth=0, backend=None):
        """Render normalized densities while retaining directional 3D asymmetry."""

        if view != "angular" and getattr(result, "meridian_azimuth", None) is not None:
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

        phase = result.phase_function.magnitude

        theta = result.angles.to("radian").magnitude

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

        title = "Directional finite-sample phase function"

        meridian = Result._select_azimuth(result=result, azimuth=azimuth)

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
                        "zaxis_title": "p ŝz (sr⁻¹); laboratory z",
                        "aspectmode": "data",
                    },
                )

                return figure

            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize

            plt = Result._pyplot()

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
                zlabel="p ŝz (sr⁻¹); laboratory z",
            )

            axis.set_title(f"{title}\nThrough order {selected + 1} · {surface_label}")

            axis.view_init(elev=20, azim=45)

            figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, label="p (sr⁻¹)", shrink=0.75, pad=0.12)
        else:
            plt = Result._pyplot()

            opposite = None

            if phase.ndim == 3:
                if view == "polar":
                    count = len(result.azimuths)

                    if count % 2:
                        raise ValueError("A full polar plane requires an even azimuth sample count.")

                    opposite = phase[..., (meridian + count // 2) % count]

                phase = phase[..., meridian]

            title += Result._angular_label(result=result, azimuth=meridian)

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
                    xlabel="Laboratory polar angle θ (degrees)",
                    ylabel="Phase function p (sr⁻¹)",
                    yscale="log" if log_y else "linear",
                )

            axis.grid(alpha=0.25)

            axis.legend(frameon=False)

        return figure

    @staticmethod
    def _plot_field_norms(*, result, log_y=True):
        """Draw relative Born field norms for each available realization."""

        if result.field_norms is None:
            raise ValueError("Field norms are only available for numerical results.")

        norms = np.atleast_2d(
            _dimensionless(
                value=result.field_norms,
                name="field_norms",
            )
        )

        plt = Result._pyplot()

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
    mu_a: np.ndarray | None = None
    residual_norms: np.ndarray | None = None
