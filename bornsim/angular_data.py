"""Consistent unitful directional scattering data for solves and cuts."""

from dataclasses import dataclass
import numpy as np
from .units import Quantity, _quantity, _si


@dataclass(frozen=True, kw_only=True, repr=False)
class AngularData:
    """Group coordinates, amplitudes, intensities and normalization.

    Full angular data have shape (order, polar angle, azimuth). Cuts and
    explicitly averaged curves have shape (order, observation). Directions
    have the same observation axes and a final Cartesian axis of size three.
    Amplitudes add incident-polarization and vector axes of sizes two and
    three. Ensemble amplitudes are absent: intensities, not amplitudes, are
    averaged across realizations. Arrays are copied, unitful and read-only.
    """

    differential: Quantity
    directions: Quantity | None = None
    amplitudes: Quantity | None = None
    angles: Quantity | None = None
    azimuths: Quantity | None = None
    term_differential: Quantity | None = None
    stderr: Quantity | None = None
    azimuth_stderr: Quantity | None = None
    mu_s: Quantity | None = None
    sample_volume: Quantity | None = None
    kind: str = "volume"
    azimuth_averaged: bool = False
    meridian_azimuth: Quantity | None = None

    def __post_init__(self):
        units = {
            "differential": "1 / meter / steradian",
            "directions": "dimensionless",
            "amplitudes": "meter",
            "angles": "radian",
            "azimuths": "radian",
            "term_differential": "1 / meter / steradian",
            "stderr": "1 / meter / steradian",
            "azimuth_stderr": "1 / meter / steradian",
            "mu_s": "1 / meter",
            "sample_volume": "meter**3",
            "meridian_azimuth": "radian",
        }
        for name, unit in units.items():
            value = getattr(self, name)
            if value is not None:
                quantity = _quantity(value=value, unit=unit, name=name)
                quantity.magnitude.setflags(write=False)
                object.__setattr__(self, name, quantity)
        shape = self.differential.shape
        if len(shape) not in (2, 3) or 0 in shape:
            raise ValueError("differential must have nonempty order and observation axes.")
        if self.directions is not None and self.directions.shape != (*shape[1:], 3):
            raise ValueError("directions must match the observation axes.")
        if self.amplitudes is not None and self.amplitudes.shape != (*shape, 2, 3):
            raise ValueError("amplitudes must have shape matching the order and observation axes.")
        self._validate_values()
        if self.directions is None and not self.azimuth_averaged:
            theta = getattr(self, "angles").magnitude
            if self.azimuths is not None:
                theta, phi = np.meshgrid(theta, self.azimuths.magnitude, indexing="ij")
            else:
                phi = np.zeros_like(theta)
            directions = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
            quantity = _quantity(value=directions, unit="dimensionless", name="directions")
            quantity.magnitude.setflags(write=False)
            object.__setattr__(self, "directions", quantity)

    def _validate_values(self):
        shape = self.differential.shape
        shapes = {
            "angles": (shape[1],),
            "azimuths": (shape[2],) if len(shape) == 3 else (),
            "term_differential": shape,
            "stderr": shape,
            "azimuth_stderr": shape[:2],
            "mu_s": (shape[0],),
            "sample_volume": (),
            "meridian_azimuth": (),
        }
        if shape[0] > 12:
            raise ValueError("differential must contain at most 12 Born orders.")
        if self.kind == "analytical" and shape[0] != 1:
            raise ValueError("analytical differential must contain exactly one order.")
        if self.kind not in ("volume", "ensemble", "analytical") or not isinstance(self.azimuth_averaged, bool):
            raise ValueError("kind or azimuth_averaged is invalid.")
        if len(shape) == 3 and (self.angles is None or self.azimuths is None or self.azimuth_averaged):
            raise ValueError("Full angular data require angles and azimuths without averaging.")
        if len(shape) == 2 and self.azimuths is not None:
            raise ValueError("azimuths require full angular data.")
        if self.angles is None and self.directions is None:
            raise ValueError("angles or directions must identify observations.")
        for name in ("differential", "directions", "amplitudes", *shapes):
            quantity = getattr(self, name)
            if quantity is None:
                continue
            values = quantity.magnitude
            if name in shapes and values.shape != shapes[name]:
                raise ValueError(f"{name} must have shape {shapes[name]}.")
            real_required = name != "amplitudes"
            if not np.issubdtype(values.dtype, np.number) or (real_required and np.iscomplexobj(values)):
                raise ValueError(
                    f"{name} must contain numeric {'real' if real_required else 'complex or real'} values."
                )
            unknown_error = (
                name in ("stderr", "azimuth_stderr") and self.kind == "ensemble" and np.all(np.isnan(values))
            )
            if not unknown_error and np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must contain finite values.")
            nonnegative = name in ("differential", "term_differential", "stderr", "azimuth_stderr", "mu_s")
            if nonnegative and np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative.")
        invalid_directions = self.directions is not None and not np.allclose(
            np.linalg.norm(self.directions.magnitude, axis=-1), 1, rtol=0, atol=1e-10
        )
        if invalid_directions:
            raise ValueError("directions must contain unit vectors.")
        if self.kind != "volume" and self.angles is None:
            raise ValueError("analytical and ensemble results require angles.")
        if self.directions is not None and self.angles is not None:
            expected_cosine = np.cos(self.angles.magnitude)
            if len(shape) == 3:
                expected_cosine = expected_cosine[:, None]
            if not np.allclose(self.directions.magnitude[..., 2], expected_cosine, rtol=0, atol=1e-10):
                raise ValueError("angles must match the polar angles of directions.")
        if self.angles is not None and np.any((self.angles.magnitude < 0) | (self.angles.magnitude > np.pi)):
            raise ValueError("angles must lie between zero and pi.")
        if self.azimuths is not None:
            phi = self.azimuths.magnitude
            invalid_phi = not 4 <= len(phi) <= 32 or not np.allclose(
                phi, np.arange(len(phi)) * 2 * np.pi / len(phi), atol=1e-12, rtol=0
            )
            if invalid_phi:
                raise ValueError("azimuths must uniformly cover [0, 2*pi), starting at zero.")
        if self.sample_volume is not None and (self.kind == "analytical" or self.sample_volume.magnitude <= 0):
            raise ValueError("sample_volume must describe a positive finite-sample volume.")
        unnormalized_cut = (
            self.kind == "volume" and len(shape) == 2 and not self.azimuth_averaged and self.meridian_azimuth is None
        )
        if unnormalized_cut and self.mu_s is not None:
            raise ValueError("integrated coefficients are unavailable for a single-volume angular cut.")
        if self.azimuth_averaged and (self.directions is not None or self.amplitudes is not None):
            raise ValueError("Azimuth-averaged data have no single directions or coherent amplitudes.")
        if self.kind != "volume" and self.amplitudes is not None:
            raise ValueError("amplitudes are only available for a single volume.")
        if self.kind != "ensemble" and (self.stderr is not None or self.azimuth_stderr is not None):
            raise ValueError("stderr and azimuth_stderr are only available for an ensemble.")

    @property
    def phase_function(self):
        """Density per steradian, normalized by the full solid-angle integral."""
        if self.mu_s is None or np.any(~np.isfinite(self.mu_s.magnitude)) or np.any(self.mu_s.magnitude <= 0):
            raise ValueError("Phase normalization requires positive, finite integrated mu_s.")
        axes = (len(self.mu_s),) + (1,) * (self.differential.ndim - 1)
        return (self.differential / self.mu_s.reshape(axes)).to("1 / steradian")

    @property
    def directional_phase_function(self):
        """Compatibility alias for full directional phase densities."""
        if self.differential.ndim != 3:
            raise ValueError("Full directional phase data are unavailable.")
        return self.phase_function

    @property
    def differential_cross_section(self):
        """Finite-sample differential cross section in square metres per sr."""
        if self.sample_volume is None:
            raise ValueError("sample_volume is unavailable for this angular data.")
        return (self.differential * self.sample_volume).to("meter**2 / steradian")

    def azimuth_average(self):
        """Average intensities explicitly; never average coherent amplitudes."""
        if self.azimuths is None:
            raise ValueError("Azimuth averaging requires a full angular grid.")
        return AngularData(
            differential=self.differential.mean(axis=-1),
            angles=self.angles,
            term_differential=None if self.term_differential is None else self.term_differential.mean(axis=-1),
            stderr=self.azimuth_stderr,
            mu_s=self.mu_s,
            sample_volume=self.sample_volume,
            kind=self.kind,
            azimuth_averaged=True,
        )

    def meridian(self, *, azimuth, method="exact"):
        """Select a sampled meridian using an angle, preserving normalization.

        Bare azimuth means radians and is periodic modulo 2*pi. exact requires
        a sampled azimuth within 1e-10 radians; nearest explicitly selects the
        closest sample. No interpolation or azimuth averaging is performed.
        meridian_azimuth records the actual selected angle.
        """
        if self.azimuths is None:
            raise ValueError("Meridian selection requires a full angular grid.")
        if method not in ("exact", "nearest"):
            raise ValueError("method must be exact or nearest.")
        requested = _si(value=azimuth, unit="radian", name="azimuth", scalar=True)
        if not np.isfinite(requested):
            raise ValueError("azimuth must be finite.")
        phi = self.azimuths.magnitude
        distance = np.abs((phi - requested + np.pi) % (2 * np.pi) - np.pi)
        index = int(np.argmin(distance))
        if method == "exact" and distance[index] > 1e-10:
            raise ValueError("azimuth is not sampled; use method='nearest' to select the closest meridian.")
        return AngularData(
            differential=self.differential[..., index],
            angles=self.angles,
            directions=getattr(self, "directions")[:, index],
            amplitudes=None if self.amplitudes is None else self.amplitudes[:, :, index],
            term_differential=None if self.term_differential is None else self.term_differential[..., index],
            stderr=None if self.stderr is None else self.stderr[..., index],
            mu_s=self.mu_s,
            sample_volume=self.sample_volume,
            kind=self.kind,
            meridian_azimuth=phi[index],
        )

    def plot(self, *, terms=False, log_y=False, title=None, azimuth=0):
        """Plot one sampled meridian, or an explicitly averaged curve."""
        from ._result_plotting import _ResultPlotter

        return _ResultPlotter(result=self).plot(
            terms=terms,
            log_y=log_y,
            title=title,
            azimuth=azimuth,
        )

    def plot_phase_function(self, *, view="angular", order=None, log_y=False, azimuth=0):
        """Plot normalized directional densities with the same Result interface."""
        from ._result_plotting import _ResultPlotter

        return _ResultPlotter(result=self).plot_phase_function(
            view=view,
            order=order,
            log_y=log_y,
            azimuth=azimuth,
        )

    def __repr__(self):
        return (
            f"AngularData(shape={self.differential.shape}, intensity_unit=m^-1 sr^-1, "
            f"amplitudes={'available' if self.amplitudes is not None else 'absent'}, "
            f"normalized={self.mu_s is not None}, azimuth_averaged={self.azimuth_averaged})"
        )
