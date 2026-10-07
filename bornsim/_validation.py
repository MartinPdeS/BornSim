"""Shared validation for bounded integer settings."""

import numpy as np


def _integer(*, value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")
    return int(value)
