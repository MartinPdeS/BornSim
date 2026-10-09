"""Validate public result construction and archive contracts."""

import json

import numpy as np
import pytest

from bornsim import AnalyticalMedium, Result, Solver, Source
from bornsim.units import ureg


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"kind": "unknown"}, "kind"),
        ({"differential": None}, "differential.*required"),
        ({"differential": [1, 2] * (1 / ureg.meter / ureg.steradian)}, "differential"),
        ({"differential": np.ones((1, 0)) * (1 / ureg.meter / ureg.steradian)}, "differential"),
        ({"differential": np.ones((2, 3)) * (1 / ureg.meter / ureg.steradian)}, "one order"),
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
        (
            {"g": [0.5], "mu_s": [2] * (1 / ureg.meter), "mu_s_prime": [2] * (1 / ureg.meter)},
            "mu_s_prime.*equal",
        ),
        ({"stderr": np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian)}, "stderr.*ensemble"),
        ({"realizations": 2}, "realizations.*ensemble"),
        ({"amplitudes": np.ones((1, 3, 2, 3)) * ureg.meter}, "amplitudes.*volume"),
        ({"term_differential": np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian)}, "numerical"),
        ({"field_norms": [1]}, "numerical"),
        ({"warnings": "warning"}, "warnings"),
        ({"provenance": []}, "provenance"),
        ({"provenance": {"seed": np.array([42])}}, "provenance"),
        ({"provenance": {"spacing_m": np.nan}}, "provenance"),
    ],
)
def test_invalid_results_fail_at_construction(changes, message):
    options = dict(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        kind="analytical",
        differential=np.ones((1, 3)) * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1, 2] * ureg.radian,
    )

    options.update(changes)

    with pytest.raises(ValueError, match=message):
        Result(**options)


def test_result_requires_source():
    with pytest.raises(TypeError, match="Source"):
        Result(
            source=633e-9,
            kind="analytical",
            differential=[[1]] * (1 / ureg.meter / ureg.steradian),
            angles=[0] * ureg.radian,
        )


def test_result_copies_inputs_and_preserves_unknown_zero_anisotropy():
    provenance = {"grid": {"shape": [2, 2, 2]}}

    differential = np.zeros((1, 3))

    result = Result(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        kind="analytical",
        differential=differential * (1 / ureg.meter / ureg.steradian),
        angles=[0, 1, 2] * ureg.radian,
        mu_s=[0] * (1 / ureg.meter),
        g=[np.nan],
        mu_s_prime=[0] * (1 / ureg.meter),
        provenance=provenance,
    )

    differential[:] = 42

    provenance["grid"]["shape"][0] = 32

    np.testing.assert_array_equal(result.differential.magnitude, 0)

    assert result.provenance["grid"]["shape"] == [2, 2, 2]


def test_analytical_archive_preserves_units_settings_and_original_version(tmp_path):
    result = Solver(
        source=Source(wavelength=532 * ureg.nanometer),
        quadrature_order=64,
    ).solve(
        target=AnalyticalMedium(
            correlation_length=80 * ureg.nanometer,
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation="gaussian",
            smoothness=1.5,
        ),
        angles=[0, 90, 180] * ureg.degree,
    )

    assert result.provenance["order"] == 1

    assert result.provenance["quadrature_order"] == 64

    assert result.provenance["medium"]["correlation_length_m"] == pytest.approx(80e-9)

    assert result.provenance["coefficient_scope"] == "infinite-medium"

    provenance = result.provenance

    provenance["bornsim_version"] = "recorded-version"

    result = Result(
        source=result.source,
        kind=result.kind,
        differential=result.differential.to("1 / centimeter / steradian").to("1 / meter / steradian"),
        angles=result.angles.to("radian"),
        azimuths=result.azimuths.to("radian"),
        mu_s=result.mu_s.to("1 / meter"),
        g=result.g,
        mu_s_prime=result.mu_s_prime.to("1 / meter"),
        provenance=provenance,
    )

    original = result.differential

    path = result.save(path=tmp_path / "result.data")

    assert result.differential is original

    assert path.name == "result.data"

    restored = Result.load(path=path)

    assert restored.provenance == result.provenance

    assert restored.source.wavelength.to("nanometer").magnitude == pytest.approx(532)

    np.testing.assert_allclose(
        restored.differential.magnitude, result.differential.to(restored.differential.units).magnitude
    )

    np.testing.assert_array_equal(restored.angles.magnitude, result.angles.magnitude)

    np.testing.assert_allclose(restored.phase_function.magnitude, result.phase_function.magnitude)

    assert restored.plot().axes[0].get_ylabel() == "Differential scattering (m⁻¹ sr⁻¹)"

    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)

        assert json.loads(str(archive["metadata"].item()))["schema_version"] == 2


@pytest.mark.parametrize("damage", ["version", "missing", "shape", "units", "metadata", "pickle"])
def test_load_rejects_invalid_archives(tmp_path, damage):
    path = (
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            )
        )
        .solve(
            target=AnalyticalMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            )
        )
        .save(path=tmp_path / "result.npz")
    )

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    metadata = json.loads(str(arrays["metadata"].item()))

    if damage == "version":
        metadata["schema_version"] = 999
    elif damage == "missing":
        del arrays["differential"]
    elif damage == "shape":
        arrays["angles"] = np.array([0.0])
    elif damage == "units":
        metadata["units"]["differential"] = "second"
    elif damage == "metadata":
        del metadata["provenance"]
    elif damage == "pickle":
        arrays["differential"] = np.array([{"object": True}], dtype=object)

    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez(path, **arrays)

    with pytest.raises(ValueError):
        Result.load(path=path)


def test_results_reject_mutation(tmp_path):
    result = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    ).solve(
        target=AnalyticalMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        )
    )

    destination = tmp_path / "keep.txt"

    destination.write_text("Keep this file")

    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        result.angles = np.array([0])

    assert destination.read_text() == "Keep this file"
