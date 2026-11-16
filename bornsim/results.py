"""Unitful scattering results, validated archives, and Matplotlib plots."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar, Literal
import json
from pathlib import Path
import numpy as np
from .source import Source
from .angular_data import AngularData
from .units import Quantity, validate_quantity, ureg
from . import plotting

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

                validate_quantity(
                    value,
                    unit=unit,
                    name=name,
                    finite=False,
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
            if (
                isinstance(result.realizations, bool)
                or not isinstance(result.realizations, (int, np.integer))
                or not 1 <= result.realizations <= 32
            ):
                raise ValueError("realizations must be an integer between 1 and 32.")

            result.realizations = int(result.realizations)
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
    def differential(self) -> Quantity:
        """Return read-only differential scattering data.

        Returns
        -------
        Quantity
            Differential intensity in inverse metres per steradian, shaped
            ``(order, observation)`` or ``(order, polar, azimuth)``.
        """

        return self.angular.differential

    @property
    def angles(self) -> Quantity | None:
        """Return read-only polar-angle coordinates in radians.

        Returns
        -------
        Quantity or None
            Polar angles for the observation axis, when available.
        """

        return self.angular.angles

    @property
    def azimuths(self) -> Quantity | None:
        """Return read-only sampled azimuth coordinates in radians.

        Returns
        -------
        Quantity or None
            Azimuth coordinates for full angular data.
        """

        return self.angular.azimuths

    @property
    def amplitudes(self) -> Quantity | None:
        """Return read-only complex scattering amplitudes.

        Returns
        -------
        Quantity or None
            Amplitudes in metres, with polarization and Cartesian vector axes.
        """

        return self.angular.amplitudes

    @property
    def mu_s(self) -> Quantity | None:
        """Return integrated scattering coefficients.

        Returns
        -------
        Quantity or None
            Finite-sample coefficients in inverse metres, one per Born order.
        """

        return self.angular.mu_s

    @property
    def sample_volume(self) -> Quantity | None:
        """Return the finite-sample voxel-box volume.

        Returns
        -------
        Quantity or None
            Sample volume in cubic metres.
        """

        return self.angular.sample_volume

    @property
    def stderr(self) -> Quantity | None:
        """Return pointwise ensemble standard errors.

        Returns
        -------
        Quantity or None
            Standard errors in inverse metres per steradian.
        """

        return self.angular.stderr

    @property
    def azimuth_stderr(self) -> Quantity | None:
        """Return standard errors for azimuth-averaged intensities.

        Returns
        -------
        Quantity or None
            Errors in inverse metres per steradian, shaped by order and polar angle.
        """

        return self.angular.azimuth_stderr

    @property
    def term_differential(self) -> Quantity | None:
        """Return read-only isolated Born-term intensities.

        Returns
        -------
        Quantity or None
            Isolated term intensities in inverse metres per steradian.
        """

        return self.angular.term_differential

    @property
    def azimuth_averaged(self) -> bool:
        """Whether intensities were explicitly averaged over azimuth.

        Returns
        -------
        bool
            True for azimuth-averaged angular data.
        """

        return self.angular.azimuth_averaged

    @property
    def directions(self) -> Quantity | None:
        """Return explicit Cartesian observation directions when stored.

        Returns
        -------
        Quantity or None
            Unit vectors for cut data. Full-grid vectors are available through
            ``angular.directions``.
        """

        return self.angular.directions if self.angular.azimuths is None and self.kind == "volume" else None

    @property
    def provenance(self) -> dict[str, Any]:
        """Return a fresh copy of provenance metadata.

        Returns
        -------
        dict of str to object
            JSON-compatible provenance. Mutating the returned object does not
            change this result.
        """

        return json.loads(self._provenance_json)

    def meridian(
        self,
        *,
        azimuth: Quantity,
        method: Literal["exact", "nearest"] = "exact",
    ) -> AngularData:
        """Select one sampled azimuth from full data without interpolation.

        Parameters
        ----------
        azimuth : Quantity
            Requested scalar azimuth with explicit angular units.
        method : {'exact', 'nearest'}, default='exact'
            Require a matching sample or select the closest sample.

        Returns
        -------
        AngularData
            Selected meridian data, available independently of the Result wrapper.

        Raises
        ------
        ValueError
            If angular sampling is incomplete or the requested sample is absent.
        """

        return self.angular.meridian(
            azimuth=azimuth,
            method=method,
        )

    def azimuth_average(self) -> "Result":
        """Average intensities and their realization-level errors over azimuth.

        Returns
        -------
        Result
            New result marked as azimuth-averaged. Coherent amplitudes are not
            averaged; covariance-correct ensemble errors are retained.

        Raises
        ------
        ValueError
            If the result has no full azimuth grid.
        """

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
    def diffuse_differential(self) -> Quantity | None:
        """Finite-ensemble mean intensity minus squared mean amplitude (m⁻¹ sr⁻¹).

        The coherent estimate includes the ensemble-mean scattered field, not
        the direct incident beam. With N samples, this uncorrected diffuse
        estimate has factor (N-1)/N finite-sample bias; it is zero for N=1.

        Returns
        -------
        Quantity or None
            Nonnegative diffuse differential intensity in inverse metres per
            steradian, or None when the coherent estimate is unavailable.
        """
        if self.coherent_differential is None:
            return None

        values = np.maximum(self.differential - self.coherent_differential, 0 * self.differential.units)
        values.magnitude.setflags(write=False)
        return values

    @property
    def absorption_cross_section(self) -> Quantity | None:
        """Return absorbed-power cross sections for the finite sample.

        Returns
        -------
        Quantity or None
            Absorption cross section in square metres, or None if absorption
            coefficients or sample volume are unavailable.
        """
        if self.mu_a is None or self.sample_volume is None:
            return None

        return self.mu_a * self.sample_volume

    @property
    def differential_cross_section(self) -> Quantity:
        """Return finite-sample differential cross sections.

        Returns
        -------
        Quantity
            ``dσ/dΩ`` in square metres per steradian, with the shape of
            ``differential``.

        Raises
        ------
        ValueError
            If the result does not include a finite sample volume.
        """

        if self.sample_volume is None:
            raise ValueError("sample_volume is unavailable; this result has no finite-sample cross section.")

        return (getattr(self, "differential") * self.sample_volume).to("meter**2 / steradian")

    def __repr__(self) -> str:
        """Return a concise summary of result kind, orders, and available data."""
        shape = self.differential.shape

        available = [
            name for name in ("amplitudes", "mu_s", "stderr", "sample_volume") if getattr(self, name) is not None
        ]

        return f"Result(kind={self.kind!r}, orders={shape[0]}, angular_shape={shape[1:]}, available={available}, azimuth_averaged={self.azimuth_averaged})"

    def save(self, *, path: str | Path) -> Path:
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
    def load(cls, *, path: str | Path) -> "Result":
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
    def phase_function(self) -> Quantity:
        """Return directional phase density per steradian.

        Full solves retain (order, polar angle, azimuth). Explicitly averaged
        results retain (order, polar angle). Cuts have no normalization.
        Every order uses its own full solid-angle scattering coefficient.

        Returns
        -------
        Quantity
            Phase density in inverse steradians. Shape matches ``differential``.

        Raises
        ------
        ValueError
            If normalization is unavailable, as for a directional cut.
        """

        return self.angular.phase_function

    def plot(
        self,
        *,
        terms: bool = False,
        log_y: bool = False,
        title: str | None = None,
        azimuth: int = 0,
    ) -> Any:
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

        return plotting.plot_differential(
            result=self,
            terms=terms,
            log_y=log_y,
            title=title,
            azimuth=azimuth,
        )

    def plot_cross_section(
        self,
        *,
        volume: Any = None,
        area_unit: str = "nanometer**2",
        terms: bool = False,
        log_y: bool = False,
        title: str | None = None,
        azimuth: int = 0,
    ) -> Any:
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

        return plotting.plot_cross_section(
            result=self,
            volume=volume,
            area_unit=area_unit,
            terms=terms,
            log_y=log_y,
            azimuth=azimuth,
            title=title,
        )

    def plot_phase_function(
        self,
        *,
        view: Literal["angular", "polar", "3d"] = "angular",
        order: int | None = None,
        log_y: bool = False,
        azimuth: int = 0,
        backend: Literal["matplotlib", "plotly"] | None = None,
    ) -> Any:
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
            3D views use Plotly exclusively; angular and polar views use
            Matplotlib. Omit backend to select the renderer for the view.

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

        return plotting.plot_phase_function(
            result=self,
            view=view,
            order=order,
            log_y=log_y,
            azimuth=azimuth,
            backend=backend,
        )

    def plot_field_norms(self, *, log_y: bool = True) -> Any:
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

        return plotting.plot_field_norms(
            result=self,
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
    mu_a: np.ndarray | None = None
    residual_norms: np.ndarray | None = None
