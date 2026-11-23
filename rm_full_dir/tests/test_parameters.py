import numpy as np
import pytest

from dmri_models import Parameter, ParameterSpace, Simplex


def example_space() -> ParameterSpace:
    return ParameterSpace(
        [
            Parameter("S0", 1.0, (0.0, 2.0)),
            Parameter("fixed_d", 3.0, (0.1, 4.0), vary=False),
        ],
        simplexes=[
            Simplex(
                ("soma_fraction", "neurite_fraction", "extra_fraction"),
                (0.3, 0.4, 0.3),
            )
        ],
    )


def test_pack_unpack_round_trip_preserves_physical_values() -> None:
    space = example_space()
    expected = {
        "S0": 1.2,
        "fixed_d": 3.0,
        "soma_fraction": 0.2,
        "neurite_fraction": 0.5,
        "extra_fraction": 0.3,
    }
    actual = space.unpack(space.pack(expected))
    assert actual == pytest.approx(expected)


def test_simplex_coordinates_always_produce_valid_fractions() -> None:
    space = example_space()
    rng = np.random.default_rng(4)
    for vector in rng.normal(0, 100, size=(100, space.dimension)):
        vector[0] = 1.0
        values = space.unpack(vector)
        fractions = np.array(
            [
                values["soma_fraction"],
                values["neurite_fraction"],
                values["extra_fraction"],
            ]
        )
        assert np.all(fractions >= 0)
        assert fractions.sum() == pytest.approx(1.0, abs=1e-15)


def test_fixed_parameter_is_not_in_fitting_vector() -> None:
    space = example_space()
    assert "fixed_d" not in space.varying_names
    assert space.unpack(space.initial_vector())["fixed_d"] == 3.0


def test_invalid_simplex_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        Simplex(("a", "b"), (0.4, 0.5))


def test_parameter_initial_value_must_be_inside_bounds() -> None:
    with pytest.raises(ValueError, match="outside"):
        Parameter("d", 4.0, (0.0, 3.0))


def test_random_vectors_are_reproducible_and_valid() -> None:
    space = example_space()
    first = space.random_vectors(5, rng=np.random.default_rng(9))
    second = space.random_vectors(5, rng=np.random.default_rng(9))
    np.testing.assert_allclose(first, second)
    for vector in first:
        space.unpack(vector)

