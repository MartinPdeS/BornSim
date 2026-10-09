"""Shape, coordinate, uncertainty and physical-range validation for Result."""

import numpy as np
from ._validation import _integer


class _ResultValidator:
    """Validate primary directional data and explicitly averaged curves."""

    @staticmethod
    def validate(*, result):
        orders, observations = result.differential.shape[:2]

        if result.kind == "ensemble":
            result.realizations = _integer(value=result.realizations, name="realizations", low=1, high=32)
        elif result.realizations is not None:
            raise ValueError("realizations is only available for an ensemble.")

        if result.kind != "volume" and result.directions is not None:
            raise ValueError("analytical and ensemble results require angles and no directions.")

        if result.differential.ndim == 3 and result.directions is not None:
            raise ValueError("Full data require angles and azimuths without cut directions.")

        shapes = {
            "g": (orders,),
            "mu_s_prime": (orders,),
            "field_norms": (result.realizations, orders) if result.kind == "ensemble" else (orders,),
        }

        for name, shape in shapes.items():
            quantity = getattr(result, name)

            if quantity is None:
                continue

            values = quantity.magnitude

            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}; got {values.shape}.")

            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"{name} must contain numeric real values.")

            if np.any(np.isinf(values)) or (name != "g" and np.any(~np.isfinite(values))):
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

        for name in ("stderr", "azimuth_stderr"):
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

        if result.kind == "analytical" and (result.field_norms is not None or result.term_differential is not None):
            raise ValueError("field_norms and term_differential are only available for numerical results.")
