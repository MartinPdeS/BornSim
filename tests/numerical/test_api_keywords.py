"""The public keyword-only contract prevents ambiguous physical parameters."""

from bornsim.medium.random_medium import GaussianMedium

import inspect
import pytest
import bornsim
from bornsim import Result, Solver, Volume
from bornsim.green import GreenOperator
from bornsim.series import BornSeries


@pytest.mark.parametrize(
    "factory",
    [getattr(bornsim, name) for name in bornsim.__all__ if callable(getattr(bornsim, name))],
    ids=lambda factory: factory.__name__,
)
def test_public_constructors_and_functions_require_keywords(factory):
    assert all(
        (
            parameter.kind == inspect.Parameter.KEYWORD_ONLY
            for parameter in inspect.signature(factory).parameters.values()
        )
    )


@pytest.mark.parametrize(
    "method",
    [
        bornsim.Medium.to_volume,
        bornsim.Medium.add_background,
        Volume.plot_3d,
        GaussianMedium.to_volume,
        GaussianMedium.spectral_weight,
        bornsim.StructuredMedium.to_volume,
        bornsim.StructuredMedium.add_background,
        bornsim.Layer.mask,
        bornsim.Sphere.mask,
        bornsim.Ellipsoid.mask,
        bornsim.Box.mask,
        bornsim.Cylinder.mask,
        BornSeries.solve,
        Solver.solve,
        Solver.solve_cut,
        Result.azimuth_average,
        bornsim.AngularData.azimuth_average,
        bornsim.Box.translated,
        bornsim.Box.rotated,
        Solver.ensemble,
        Result.save,
        Result.load,
        Result.plot,
        Result.plot_phase_function,
        Result.plot_field_norms,
        GreenOperator.__init__,
        GreenOperator.apply,
    ],
    ids=lambda method: method.__qualname__,
)
def test_public_methods_require_keywords(method):
    assert all(
        (
            parameter.kind == inspect.Parameter.KEYWORD_ONLY
            for name, parameter in inspect.signature(method).parameters.items()
            if name not in ("self", "cls")
        )
    )


@pytest.mark.parametrize("method", [bornsim.Medium.add_structures, bornsim.StructuredMedium.add_structures])
def test_structure_composition_is_the_explicit_positional_exception(method):
    parameters = list(inspect.signature(method).parameters.values())

    assert [parameter.name for parameter in parameters] == ["self", "structures"]

    assert parameters[1].kind == inspect.Parameter.VAR_POSITIONAL
