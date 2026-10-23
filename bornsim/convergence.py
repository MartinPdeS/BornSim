"""Observable stability comparisons, distinct from physical validity claims."""

from dataclasses import dataclass
from typing import Any
import numpy as np

from .results import Result


@dataclass(frozen=True, kw_only=True)
class ConvergenceReport:
    """Compare final-order observables across an explicit refinement sequence.

    Parameters
    ----------
    results : tuple of Result
        At least two full angular results ordered from coarse to fine. Change
        one control at a time: grid, quadrature, sample extent, ensemble size,
        or solver order. The user controls physical equivalence of the cases.
    labels : tuple of str
        One descriptive label per calculation.
    tolerance : float, optional
        Relative tolerance for mu_s and mu_a, absolute tolerance for g.
        Default 0.01. Stability does not certify the Born model's validity.

    Notes
    -----
    Each row uses the highest order in its result. Relative differences use
    max(abs(previous), abs(current)); two zeros have zero difference. Unknown
    observables are omitted, and unknown residuals remain None. Compare
    ensemble uncertainties separately from discretization errors.
    """

    results: tuple[Result, ...]
    labels: tuple[str, ...]
    tolerance: float = 0.01

    def __post_init__(self):
        if len(self.results) < 2 or len(self.labels) != len(self.results):
            raise ValueError("Supply at least two results and one label per result.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive.")
        if any(not isinstance(result, Result) or result.mu_s is None for result in self.results):
            raise ValueError("Convergence comparisons require full angular Result instances.")
        if any(not isinstance(label, str) for label in self.labels):
            raise ValueError("labels must be strings.")
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "labels", tuple(self.labels))

    @property
    def rows(self) -> tuple[dict, ...]:
        """Fresh numeric rows: coefficients in m⁻¹, g and residual dimensionless."""
        rows = []
        previous: dict[str, Any] | None = None
        previous_result = None
        for label, result in zip(self.labels, self.results):
            row: dict[str, Any] = {"label": label}
            for name in ("mu_s", "mu_a", "g"):
                value = getattr(result, name)
                row[name] = (
                    None
                    if value is None
                    else float(value.to("dimensionless" if name == "g" else "1 / meter").magnitude[-1])
                )
            row["residual"] = (
                None if result.residual_norms is None else float(np.max(result.residual_norms.magnitude[..., -1]))
            )
            changes: dict[str, float] = {}
            if previous is not None:
                for name in ("mu_s", "mu_a", "g"):
                    a, b = previous[name], row[name]
                    if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                        denominator = max(abs(a), abs(b)) if name != "g" else 1.0
                        changes[name] = abs(b - a) / denominator if denominator else 0.0
                if previous_result is not None:
                    phase_change = self._phase_change(previous=previous_result, current=result)
                    if phase_change is not None:
                        changes["phase_l1"] = phase_change
            row["changes"] = changes
            row["stable"] = bool(changes) and all(value <= self.tolerance for value in changes.values())
            rows.append(row)
            previous = row
            previous_result = result
        return tuple(rows)

    @staticmethod
    def _phase_change(*, previous: Result, current: Result) -> float | None:
        """Compare full-polar-range azimuth-averaged phase curves on shared cos(theta)."""
        curves = []
        for result in (previous, current):
            if result.angles is None or result.mu_s.magnitude[-1] == 0:
                return None
            theta = result.angles.to("radian").magnitude
            if len(theta) < 2 or not np.isclose(theta.min(), 0) or not np.isclose(theta.max(), np.pi):
                return None
            phase = result.phase_function.to("1 / steradian").magnitude[-1]
            if phase.ndim == 2:
                phase = phase.mean(axis=-1)
            cosine = np.cos(theta)
            indices = np.argsort(cosine)
            curves.append((cosine[indices], phase[indices]))
        nodes = np.union1d(curves[0][0], curves[1][0])
        difference = np.abs(np.interp(nodes, *curves[0]) - np.interp(nodes, *curves[1]))
        return float(2 * np.pi * np.sum((difference[1:] + difference[:-1]) / 2 * np.diff(nodes)))

    def summary(self) -> str:
        """Return a compact report without claiming physical convergence."""
        lines = ["Observable stability (not a physical-validity certificate):"]
        for row in self.rows:
            lines.append(
                f"{row['label']}: mu_s={row['mu_s']:.6g} m^-1, "
                f"mu_a={row['mu_a']}, residual={row['residual']}, stable={row['stable']}"
            )
            if row["changes"]:
                lines.append(f"  Changes: {row['changes']}")
        return "\n".join(lines)
