"""The public keyword-only contract prevents ambiguous physical parameters."""

import inspect

import numpy as np
import pytest

import bornsim
from bornsim import RandomMedium, Result, Solver, Source, Volume
from bornsim.green import GreenOperator
from bornsim.series import BornSeries
from bornsim.media import random_volume
from bornsim.units import ureg


@pytest.mark.parametrize(
    "factory",
    [getattr(bornsim, name) for name in bornsim.__all__ if callable(getattr(bornsim, name))],
    ids=lambda factory: factory.__name__,
)
def test_public_constructors_and_functions_require_keywords(factory):
    assert all(
        parameter.kind == inspect.Parameter.KEYWORD_ONLY for parameter in inspect.signature(factory).parameters.values()
    )


@pytest.mark.parametrize(
    "method",
    [
        bornsim.Medium.to_volume,
        bornsim.Medium.add_background,
        Volume.plot_3d,
        RandomMedium.to_volume,
        RandomMedium.spectral_weight,
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
        parameter.kind == inspect.Parameter.KEYWORD_ONLY
        for name, parameter in inspect.signature(method).parameters.items()
        if name not in ("self", "cls")
    )


@pytest.mark.parametrize("method", [bornsim.Medium.add_structures, bornsim.StructuredMedium.add_structures])
def test_structure_composition_is_the_explicit_positional_exception(method):
    parameters = list(inspect.signature(method).parameters.values())

    assert [parameter.name for parameter in parameters] == ["self", "structures"]

    assert parameters[1].kind == inspect.Parameter.VAR_POSITIONAL


def test_positional_calls_are_rejected_and_named_calls_work():
    field = np.zeros((2, 2, 2))

    with pytest.raises(TypeError):
        Source(633e-9)

    with pytest.raises(TypeError):
        Volume(
            field,
            spacing=5e-08 * ureg.meter,
        )

    with pytest.raises(TypeError):
        random_volume(
            RandomMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="matern",
                smoothness=1.5,
            )
        )

    volume = Volume(
        delta_refractive_index=field,
        spacing=5e-08 * ureg.meter,
        background_refractive_index=1.33,
    )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    )

    with pytest.raises(TypeError):
        solver.solve(volume)

    result = solver.solve_cut(
        target=volume,
        angles=[0, 1] * ureg.radian,
    )

    assert not result.differential.magnitude.any()
