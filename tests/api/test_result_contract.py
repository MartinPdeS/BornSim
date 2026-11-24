"""Validate public result construction and archive contracts."""

import numpy as np
import pytest
from bornsim import Result, Source
from bornsim.units import ureg


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"kind": "unknown"}, "kind"),
        ({"differential": None}, "differential.*required"),
        ({"differential": [1, 2] * (1 / ureg.meter / ureg.steradian)}, "differential"),
        ({"differential": np.ones((1, 0)) * (1 / ureg.meter / ureg.steradian)}, "differential"),
        ({"differential": [[1, -1, 1]] * (1 / ureg.meter / ureg.steradian)}, "nonnegative"),
        ({"differential": [[1, np.nan, 1]] * (1 / ureg.meter / ureg.steradian)}, "finite"),
        ({"differential": [[1j, 1, 1]] * (1 / ureg.meter / ureg.steradian)}, "real"),
        ({"differential": (np.full((1, 3), np.nan) * (1 / ureg.meter / ureg.steradian)).astype(str)}, "numeric"),
        ({"angles": [0, 1] * ureg.radian}, "angles.*shape"),
        ({"angles": [0, 1, np.inf] * ureg.radian}, "angles.*finite"),
        ({"angles": [-1, 1, 2] * ureg.radian}, "angles.*between"),
        ({"angles": None}, "angles or directions"),
        ({"mu_s": [[1]] * (1 / ureg.meter)}, "mu_s.*shape"),
        ({"mu_s": [-1] * (1 / ureg.meter)}, "mu_s.*nonnegative"),
        ({"mu_s": [np.nan] * (1 / ureg.meter)}, "mu_s.*finite"),
        ({"mu_s": [np.inf] * (1 / ureg.meter)}, "mu_s.*finite"),
        ({"g": [1.1]}, "g.*between"),
        ({"g": [np.nan], "mu_s": [1] * (1 / ureg.meter)}, "g.*NaN"),
        ({"g": [0], "mu_s": [0] * (1 / ureg.meter)}, "g.*NaN"),
        ({"g": [0.5], "mu_s": [2] * (1 / ureg.meter), "mu_s_prime": [2] * (1 / ureg.meter)}, "mu_s_prime.*equal"),
        ({"stderr": np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian)}, "stderr.*ensemble"),
        ({"realizations": 2}, "realizations.*ensemble"),
        ({"warnings": "warning"}, "warnings"),
        ({"provenance": []}, "provenance"),
        ({"provenance": {"seed": np.array([42])}}, "provenance"),
        ({"provenance": {"spacing_m": np.nan}}, "provenance"),
    ],
)
def test_invalid_results_fail_at_construction(changes, message):
    options = dict(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="volume",
        differential=np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1, 2] * ureg.radian,
        azimuth_averaged=True,
    )

    options.update(changes)

    with pytest.raises(ValueError, match=message):
        Result(**options)


def test_result_requires_source():
    with pytest.raises(TypeError, match="Source"):
        Result(
            source=6.33e-07,
            kind="volume",
            differential=[[1]] * (1 / ureg.meter / ureg.steradian),
            angles=[0] * ureg.radian,
            azimuth_averaged=True,
        )


def test_result_copies_inputs_and_preserves_unknown_zero_anisotropy():
    provenance = {"grid": {"shape": [2, 2, 2]}}

    differential = np.zeros((1, 3))

    result = Result(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        kind="volume",
        differential=differential * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1, 2] * ureg.radian,
        mu_s=[0] * (1 / ureg.meter),
        g=[np.nan],
        mu_s_prime=[0] * (1 / ureg.meter),
        provenance=provenance,
        azimuth_averaged=True,
    )

    differential[:] = 42

    provenance["grid"]["shape"][0] = 32

    np.testing.assert_array_equal(result.differential.magnitude, 0)

    assert result.provenance["grid"]["shape"] == [2, 2, 2]
