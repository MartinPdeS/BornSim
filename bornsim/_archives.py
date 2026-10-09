"""Versioned numeric result archives with explicit SI schema and no pickle."""

import json
from pathlib import Path
import numpy as np
from .source import Source
from .units import ureg


_RESULT_UNITS = {
    "differential": "1 / meter / steradian",
    "term_differential": "1 / meter / steradian",
    "stderr": "1 / meter / steradian",
    "angles": "radian",
    "azimuths": "radian",
    "amplitudes": "meter",
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
            value = getattr(validated, name)

            if value is not None:
                arrays[name] = value.to(unit).magnitude

                units[name] = unit

        metadata = {
            "format": "bornsim-result",
            "schema_version": 2,
            "azimuth_averaged": validated.azimuth_averaged,
            "wavelength_m": float(validated.source.wavelength.to("meter").magnitude),
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
                    or metadata["schema_version"] != 2
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

                arrays = {name: archive[name] * ureg.Unit(unit) for name, unit in units.items()}

                return result_type(
                    source=Source(wavelength=metadata["wavelength_m"] * ureg.meter),
                    kind=metadata["kind"],
                    azimuth_averaged=metadata.get("azimuth_averaged", False),
                    realizations=metadata["realizations"],
                    warnings=metadata["warnings"],
                    provenance=metadata["provenance"],
                    **arrays,
                )
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("Invalid or incomplete result archive metadata.") from error
