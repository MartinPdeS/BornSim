"""Reproducible independent-realization sampling configurations."""

from dataclasses import dataclass
import warnings
from ._validation import _integer


@dataclass(frozen=True, kw_only=True)
class EnsembleSampling:
    """Choose consecutive seeds or an explicit ordered set of distinct seeds.

    realizations defaults to four and seed to zero when seeds is omitted.
    Explicit seeds cannot be combined with realizations or seed. Seeds range
    from zero to 2**32-1; there are 1 to 32 independent realizations. Duplicate
    seeds are rejected because they do not provide independent uncertainty.
    """

    realizations: int | None = None
    seed: int | None = None
    seeds: tuple | None = None

    def __post_init__(self):
        if self.seeds is not None:
            if self.realizations is not None or self.seed is not None:
                raise ValueError("Supply seeds or realizations/seed, not both.")
            values = tuple(self.seeds)
            _integer(value=len(values), name="realizations", low=1, high=32)
            seeds = tuple(_integer(value=value, name="seed", low=0, high=2**32 - 1) for value in values)
            if len(set(seeds)) != len(seeds):
                raise ValueError("seeds must be distinct for independent realizations.")
        else:
            count = _integer(
                value=4 if self.realizations is None else self.realizations, name="realizations", low=1, high=32
            )
            first = _integer(value=0 if self.seed is None else self.seed, name="seed", low=0, high=2**32 - 1)
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
    def _resolve(cls, *, ensemble_sampling=None, realizations=None, seed=None):
        if ensemble_sampling is not None:
            if not isinstance(ensemble_sampling, cls):
                raise TypeError("ensemble_sampling must be an EnsembleSampling.")
            if realizations is not None or seed is not None:
                raise ValueError("Supply ensemble_sampling or realizations/seed, not both.")
            return ensemble_sampling
        if realizations is not None or seed is not None:
            warnings.warn(
                "Use EnsembleSampling instead of realizations/seed keywords.", DeprecationWarning, stacklevel=3
            )
        return cls(realizations=realizations, seed=seed)
