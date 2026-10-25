"""Incident illumination for BornSim scattering calculations."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from .units import Quantity, validate_units
from .polarization import PolarizationState


def _transverse_basis(*, incident_direction: np.ndarray) -> np.ndarray:
    """Build a right-handed basis with projected laboratory x as reference."""

    direction = incident_direction / np.linalg.norm(incident_direction)

    reference = np.array([1.0, 0.0, 0.0])

    first = reference - np.dot(reference, direction) * direction

    if np.linalg.norm(first) < 1e-10:
        reference = np.array([0.0, 1.0, 0.0])

        first = reference - np.dot(reference, direction) * direction

    first /= np.linalg.norm(first)

    second = np.cross(direction, first)

    return np.stack([first, second]).astype(complex)


@dataclass(frozen=True, kw_only=True, eq=False)
class Source:
    """Define a plane wave with arbitrary propagation and polarization.

    Parameters
    ----------
    wavelength : Quantity
        Positive, finite vacuum wavelength with explicit length units.
    incident_direction : array_like, optional
        Real Cartesian unit 3-vector in laboratory coordinates. Defaults to
        positive z. Values must be unitless; no automatic normalization occurs.
    polarization : PolarizationState or array_like, optional
        A Jones state in the local transverse basis, or a complex Cartesian
        unit electric-field 3-vector perpendicular to
        incident_direction. Real vectors describe linear polarization; complex
        vectors allow circular and elliptical polarization. Omit for an equal
        incoherent mixture of two orthogonal transverse linear polarizations.
        Vectors are copied into immutable storage. Use exp(-i omega t).

    Notes
    -----
    Observation angles remain laboratory polar and azimuth angles. They are
    scattering angles relative to the beam only for positive-z incidence.
    A pure polarization has one amplitude channel; unpolarized light has two.
    Both configurations have the same total incident intensity normalization.
    """

    wavelength: Quantity
    incident_direction: ArrayLike = (0.0, 0.0, 1.0)
    polarization: PolarizationState | ArrayLike | None = None

    def __post_init__(self) -> None:
        validate_units(
            self.wavelength,
            unit="meter",
            name="wavelength",
            scalar=True,
        )

        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("wavelength must be finite and positive.")

        for name in ("incident_direction", "polarization"):
            value = getattr(self, name)

            if value is None and name == "polarization":
                continue

            if name == "polarization" and isinstance(value, PolarizationState):
                basis = _transverse_basis(incident_direction=np.asarray(self.incident_direction))

                value = value.jones_vector @ basis

            if isinstance(value, Quantity):
                raise TypeError(f"{name} must be a plain unitless vector.")

            if name == "incident_direction" and np.iscomplexobj(value):
                raise ValueError("incident_direction must be real.")

            dtype = np.float64 if name == "incident_direction" else np.complex128

            vector = np.asarray(value, dtype=dtype)

            valid_vector = (
                vector.shape == (3,)
                and np.all(np.isfinite(vector))
                and np.isclose(np.linalg.norm(vector), 1, atol=1e-10, rtol=0)
            )

            if not valid_vector:
                raise ValueError(f"{name} must be a finite unit 3-vector.")

            immutable = np.frombuffer(vector.tobytes(), dtype=dtype)

            object.__setattr__(self, name, immutable)

        if self.polarization is not None and not np.isclose(
            np.dot(np.asarray(self.incident_direction), np.asarray(self.polarization)), 0, atol=1e-10, rtol=0
        ):
            raise ValueError("polarization must be transverse to incident_direction.")

    @property
    def polarization_vectors(self) -> np.ndarray:
        """Return unit electric-field modes, mixed incoherently with equal weight."""

        if self.polarization is not None:
            return np.asarray(self.polarization)[None, :].copy()

        return _transverse_basis(incident_direction=np.asarray(self.incident_direction))

    @property
    def metadata(self) -> dict:
        """Return fresh JSON-compatible illumination settings in SI units."""

        polarization = None if self.polarization is None else np.asarray(self.polarization)

        return {
            "wavelength_m": float(self.wavelength.to("meter").magnitude),
            "incident_direction": np.asarray(self.incident_direction).tolist(),
            "polarization_real": None if polarization is None else polarization.real.tolist(),
            "polarization_imag": None if polarization is None else polarization.imag.tolist(),
        }

    @classmethod
    def _from_metadata(cls, *, metadata: dict) -> "Source":
        from .units import ureg

        real = metadata["polarization_real"]

        imaginary = metadata["polarization_imag"]

        if (real is None) != (imaginary is None):
            raise ValueError("Incomplete source polarization metadata.")

        polarization = None if real is None else np.asarray(real) + 1j * np.asarray(imaginary)

        return cls(
            wavelength=metadata["wavelength_m"] * ureg.meter,
            incident_direction=metadata["incident_direction"],
            polarization=polarization,
        )
