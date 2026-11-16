"""Reproducible independent-realization sampling configurations."""

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np


@dataclass(frozen=True, kw_only=True, init=False)
class EnsembleSampling:
    """Choose consecutive seeds or an explicit ordered set of distinct seeds.

    realizations defaults to four and seed to zero when seeds is omitted.
    Explicit seeds cannot be combined with realizations or seed. Seeds range
    from zero to 2**32-1; there are 1 to 32 independent realizations. Duplicate
    seeds are rejected because they do not provide independent uncertainty.
    """

    realizations: int
    seed: int
    seeds: tuple[int, ...]

    def __init__(
        self,
        *,
        realizations: int | None = None,
        seed: int | None = None,
        seeds: Sequence[int] | None = None,
    ) -> None:
        if seeds is not None:
            if realizations is not None or seed is not None:
                raise ValueError("Supply seeds or realizations/seed, not both.")

            values = tuple(seeds)

            if not 1 <= len(values) <= 32:
                raise ValueError("realizations must be an integer between 1 and 32.")

            normalized_seeds = []

            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= value <= 2**32 - 1:
                    raise ValueError("seed must be an integer between 0 and 4294967295.")

                normalized_seeds.append(int(value))

            seeds = tuple(normalized_seeds)

            if len(set(seeds)) != len(seeds):
                raise ValueError("seeds must be distinct for independent realizations.")
        else:
            count = 4 if realizations is None else realizations

            if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 32:
                raise ValueError("realizations must be an integer between 1 and 32.")

            count = int(count)

            first = 0 if seed is None else seed

            if isinstance(first, bool) or not isinstance(first, (int, np.integer)) or not 0 <= first <= 2**32 - 1:
                raise ValueError("seed must be an integer between 0 and 4294967295.")

            first = int(first)

            if first + count - 1 > 2**32 - 1:
                raise ValueError("seed range exceeds the supported maximum.")

            seeds = tuple(range(first, first + count))

        object.__setattr__(self, "seeds", seeds)

        object.__setattr__(self, "realizations", len(seeds))

        object.__setattr__(self, "seed", seeds[0])

    @property
    def metadata(self):
        """Return exact ordered seeds for reproducibility."""

        return {"realizations": self.realizations, "seeds": list(getattr(self, "seeds"))}

    @classmethod
    def _resolve(cls, *, ensemble_sampling: "EnsembleSampling | None" = None) -> "EnsembleSampling":
        if ensemble_sampling is None:
            return cls()

        if not isinstance(ensemble_sampling, cls):
            raise TypeError("ensemble_sampling must be an EnsembleSampling.")

        return ensemble_sampling
