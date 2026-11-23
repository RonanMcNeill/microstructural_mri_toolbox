import numpy as np
import pytest

from dmri_models import Acquisition, Stick, Tensor, Zeppelin


def fibonacci_sphere(count: int) -> np.ndarray:
    index = np.arange(count)
    z = 1.0 - 2.0 * (index + 0.5) / count
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius = np.sqrt(1.0 - z**2)
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


@pytest.mark.parametrize("beta", [-0.5, 0.0, 0.4, 1.0])
@pytest.mark.parametrize(
    ("d_parallel", "d_perpendicular"),
    [(1.7, 0.0), (1.7, 0.6), (0.5, 1.2), (1.0, 1.0)],
)
def test_zeppelin_analytical_mean_matches_dense_directional_average(
    beta,
    d_parallel,
    d_perpendicular,
) -> None:
    directions = fibonacci_sphere(20000)
    b = 2.0
    directional = Acquisition.from_b_values(
        np.full(directions.shape[0], b),
        b_unit="ms/um^2",
        gradients=directions,
        b_tensor_shape=beta,
    )
    shell = Acquisition.from_b_values(
        [b],
        b_unit="ms/um^2",
        b_tensor_shape=beta,
    )
    compartment = Zeppelin("d_parallel", "d_perpendicular", "axis")
    parameters = {
        "d_parallel": d_parallel,
        "d_perpendicular": d_perpendicular,
        "axis": np.array([0, 0, 1]),
    }
    numerical = np.mean(compartment.signal(directional, parameters))
    analytical = compartment.spherical_mean(shell, parameters)[0]
    assert analytical == pytest.approx(numerical, rel=2e-5, abs=2e-7)


def test_zeppelin_diso_ddelta_form_matches_codivide_expression() -> None:
    b = np.array([0.0, 0.5, 1.0, 2.0])
    beta = np.array([-0.5, 0.0, 0.4, 1.0])
    d_iso = 1.1
    d_delta = 0.45
    d_parallel = d_iso * (1.0 + 2.0 * d_delta)
    d_perpendicular = d_iso * (1.0 - d_delta)
    acquisition = Acquisition.from_b_values(
        b,
        b_unit="ms/um^2",
        b_tensor_shape=beta,
    )
    actual = Zeppelin("d_parallel", "d_perpendicular", "axis").spherical_mean(
        acquisition,
        {
            "d_parallel": d_parallel,
            "d_perpendicular": d_perpendicular,
            "axis": np.array([0, 0, 1]),
        },
    )

    alpha = 3.0 * b * d_iso * beta * d_delta
    expected_factor = np.ones_like(alpha)
    positive = alpha > 1e-12
    negative = alpha < -1e-12
    from scipy.special import erf, erfi

    root = np.sqrt(alpha[positive])
    expected_factor[positive] = np.sqrt(np.pi) * erf(root) / (2 * root)
    root = np.sqrt(-alpha[negative])
    expected_factor[negative] = np.sqrt(np.pi) * erfi(root) / (2 * root)
    expected = np.exp(-b * d_iso + alpha / 3.0) * expected_factor
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_general_tensor_mean_matches_dense_directional_average() -> None:
    directions = fibonacci_sphere(30000)
    directional = Acquisition.from_b_values(
        np.full(directions.shape[0], 2.0),
        b_unit="ms/um^2",
        gradients=directions,
    )
    shell = Acquisition.from_b_values([2.0], b_unit="ms/um^2")
    tensor = Tensor(("d_x", "d_y", "d_z"))
    parameters = {"d_x": 0.3, "d_y": 0.8, "d_z": 1.7}
    numerical = np.mean(tensor.signal(directional, parameters))
    quadrature = tensor.spherical_mean(shell, parameters)[0]
    assert quadrature == pytest.approx(numerical, rel=2e-5)


def test_general_tensor_mean_reduces_to_zeppelin_analytical_mean() -> None:
    acquisition = Acquisition.from_b_values(
        np.linspace(0, 4, 20),
        b_unit="ms/um^2",
    )
    tensor = Tensor(("d_perpendicular", "d_perpendicular", "d_parallel"))
    zeppelin = Zeppelin("d_parallel", "d_perpendicular", "axis")
    parameters = {
        "d_parallel": 1.7,
        "d_perpendicular": 0.5,
        "axis": np.array([0, 0, 1]),
    }
    np.testing.assert_allclose(
        tensor.spherical_mean(acquisition, parameters),
        zeppelin.spherical_mean(acquisition, parameters),
        rtol=2e-13,
        atol=1e-15,
    )


def test_stick_spherical_mean_is_zeppelin_zero_radial_limit() -> None:
    acquisition = Acquisition.from_b_values(
        [0, 1, 2],
        b_unit="ms/um^2",
        b_tensor_shape=[1, 0, -0.5],
    )
    stick = Stick("d", powder_average=True)
    zeppelin = Zeppelin("d", "zero", "axis")
    np.testing.assert_allclose(
        stick.signal(acquisition, {"d": 1.7}),
        zeppelin.spherical_mean(
            acquisition,
            {"d": 1.7, "zero": 0.0, "axis": np.array([0, 0, 1])},
        ),
    )
