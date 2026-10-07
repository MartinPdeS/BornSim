"""Versioned numeric result archives with explicit SI schema and no pickle."""

import json
from pathlib import Path
import numpy as np
from .source import Source
from .units import _quantity, _si


_RESULT_UNITS = {
    "differential": "1 / meter / steradian",
    "term_differential": "1 / meter / steradian",
    "stderr": "1 / meter / steradian",
    "angles": "radian",
    "azimuths": "radian",
    "directional_differential": "1 / meter / steradian",
    "amplitudes": "meter",
    "directional_amplitudes": "meter",
    "mu_s": "1 / meter",
    "mu_s_prime": "1 / meter",
    "g": "dimensionless",
    "directions": "dimensionless",
    "field_norms": "dimensionless",
    "sample_volume": "meter**3",
    "azimuth_stderr": "1 / meter / steradian",
}


class _ResultArchive:
    """Encode and restore validated result data without storing voxel fields."""

    @staticmethod
    def save(*, result, path):
        """Revalidate and encode result arrays with explicit SI metadata."""
        # Immutable results retain their construction-time validation.
        validated = result
        arrays = {}
        units = {}
        for name, unit in _RESULT_UNITS.items():
            if name in ("directional_differential", "directional_amplitudes"):
                continue
            value = getattr(validated, name)
            if value is not None:
                arrays[name] = value.to(unit).magnitude
                units[name] = unit
        metadata = {
            "format": "bornsim-result",
            "schema_version": 2,
            "azimuth_averaged": validated.azimuth_averaged,
            "wavelength_m": _si(
                value=validated.source.wavelength,
                unit="meter",
                name="wavelength",
                scalar=True,
            ),
            "kind": validated.kind,
            "realizations": validated.realizations,
            "warnings": validated.warnings,
            "provenance": validated.provenance,
            "units": units,
        }
        encoded_metadata = np.array(json.dumps(metadata, allow_nan=False))
        path = Path(path)
        with path.open("wb") as stream:
            np.savez_compressed(stream, metadata=encoded_metadata, **arrays)
        return path

    @staticmethod
    def load(*, result_type, path):
        """Check the archive schema and construct the requested result class."""
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                unsupported_schema = (
                    metadata["format"] != "bornsim-result"
                    or type(metadata["schema_version"]) is not int
                    or metadata["schema_version"] not in (1, 2)
                )
                if unsupported_schema:
                    raise ValueError("Unsupported result archive schema.")
                units = metadata["units"]
                if not isinstance(units, dict) or not set(units) <= set(_RESULT_UNITS):
                    raise ValueError("Invalid result archive fields.")
                if any(unit != _RESULT_UNITS[name] for name, unit in units.items()):
                    raise ValueError("Invalid result archive units; archives require SI units.")
                if set(archive.files) != set(units) | {"metadata"} or "differential" not in units:
                    raise ValueError("Result archive arrays do not match metadata.")
                arrays = {
                    name: _quantity(
                        value=archive[name],
                        unit=unit,
                        name=name,
                    )
                    for name, unit in units.items()
                }
                return result_type(
                    source=Source(wavelength=metadata["wavelength_m"]),
                    kind=metadata["kind"],
                    azimuth_averaged=metadata.get("azimuth_averaged", False),
                    realizations=metadata["realizations"],
                    warnings=metadata["warnings"],
                    provenance=metadata["provenance"],
                    **arrays,
                )
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Invalid or incomplete result archive metadata.") from error

    @staticmethod
    def _promote_legacy_directional_data(*, result):
        """Restore schema-1 averages without losing their directional arrays."""
        if result.differential is None:
            raise ValueError("differential is required.")
        data = getattr(result, "differential").magnitude
        legacy = getattr(result, "directional_differential")
        if data.ndim == 2 and legacy is not None:
            if result.azimuths is None:
                raise ValueError("azimuths and directional_differential must be supplied together.")
            expected = (*data.shape, len(getattr(result, "azimuths").magnitude))
            if legacy.shape != expected:
                raise ValueError(f"directional_differential must have shape {expected}.")
            if not np.allclose(legacy.magnitude.mean(axis=-1), data, rtol=1e-10, atol=0):
                raise ValueError("differential must equal the azimuth average of directional_differential.")
            result.differential = legacy
            if result.stderr is not None:
                result.azimuth_stderr, result.stderr = result.stderr, None
            # Legacy isolated terms were stored only as an average. Do not
            # invent per-direction terms from that curve.
            result.term_differential = None
            data = legacy.magnitude
        if result.directional_amplitudes is not None:
            invalid_condition = result.amplitudes is not None and not np.allclose(
                getattr(result, "amplitudes").magnitude,
                getattr(result, "directional_amplitudes").magnitude,
                rtol=1e-12,
                atol=0,
            )
            if invalid_condition:
                raise ValueError("amplitudes and directional_amplitudes must agree.")
            result.amplitudes = result.directional_amplitudes
        if data.ndim == 3:
            if legacy is not None and not np.allclose(legacy.magnitude, data, rtol=1e-12, atol=0):
                raise ValueError("directional_differential must agree with differential.")
            result.directional_differential = result.differential
            result.directional_amplitudes = result.amplitudes
        elif result.directional_amplitudes is not None:
            raise ValueError("directional_amplitudes require a full angular single-volume result.")
