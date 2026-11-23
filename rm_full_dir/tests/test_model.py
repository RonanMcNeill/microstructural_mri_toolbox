import numpy as np
import pytest

from dmri_models import (
    Acquisition,
    Ball,
    Component,
    Model,
    Sphere,
    Stick,
    T2Relaxation,
    Tensor,
    Zeppelin,
)
from dmri_models.compartments import BinghamFOD, FODConvolved, WatsonFOD
from dmri_models.compartments import (
    BinghamStickHypergeometric,
    WatsonStickHypergeometric,
    WatsonZeppelinHypergeometric,
)


def test_ball_matches_analytical_signal() -> None:
    acquisition = Acquisition.from_b_values([0, 1, 2], b_unit="ms/um^2")
    ball = Ball(diffusivity="d")
    signal = ball.signal(acquisition, {"d": 0.8})
    np.testing.assert_allclose(signal, np.exp(-acquisition.b_values * 0.8))


def test_arbitrary_mixture_of_repeated_compartment_type() -> None:
    acquisition = Acquisition.from_b_values([0, 1, 2], b_unit="ms/um^2")
    model = Model(
        [
            Component("slow", Ball("d_slow")),
            Component("fast", Ball("d_fast")),
        ]
    )
    parameters = {
        "S0": 2.0,
        "slow_fraction": 0.7,
        "fast_fraction": 0.3,
        "d_slow": 0.5,
        "d_fast": 2.0,
    }
    expected = 2.0 * (
        0.7 * np.exp(-acquisition.b_values * 0.5)
        + 0.3 * np.exp(-acquisition.b_values * 2.0)
    )
    np.testing.assert_allclose(model.signal(acquisition, parameters), expected)


def test_model_rejects_fractions_that_do_not_sum_to_one() -> None:
    acquisition = Acquisition.from_b_values([0, 1], b_unit="ms/um^2")
    model = Model([Component("a", Ball("d")), Component("b", Ball("d"))])
    with pytest.raises(ValueError, match="sum to one"):
        model.signal(
            acquisition,
            {"S0": 1, "a_fraction": 0.7, "b_fraction": 0.4, "d": 1},
        )


def test_model_level_spherical_mean_matches_powder_stick_formula() -> None:
    acquisition = Acquisition.from_b_values(
        [0.0, 0.5, 1.0, 2.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
    )
    model = Model([Component("neurite", Stick("d_parallel", orientation="axis"))])
    parameters = {
        "S0": 1.0,
        "neurite_fraction": 1.0,
        "d_parallel": 1.5,
        "axis": np.array([0.3, 0.7, 0.2]),
    }

    expected = Stick("d_parallel", powder_average=True).spherical_mean(
        acquisition,
        {"d_parallel": 1.5},
    )
    actual = model.spherical_mean_signal(parameters=parameters, acquisition=acquisition)
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)


def test_model_level_spherical_mean_uses_component_analytic_implementations() -> None:
    acquisition = Acquisition.from_pgse(
        [0.0, 0.5, 1.0],
        b_unit="ms/um^2",
        pulse_duration=30.0,
        diffusion_time=50.0,
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    model = Model(
        [
            Component("extra", Ball("d_ex")),
            Component("soma", Sphere("r", "d_sphere")),
        ]
    )
    parameters = {
        "S0": 1.2,
        "extra_fraction": 0.4,
        "soma_fraction": 0.6,
        "d_ex": 1.8,
        "r": 8.0,
        "d_sphere": 3.0,
    }
    expected = model.signal(acquisition, parameters)
    actual = model.spherical_mean_signal(acquisition, parameters)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_powder_averaged_stick_has_unit_b0_signal() -> None:
    acquisition = Acquisition.from_b_values([0, 1], b_unit="ms/um^2")
    stick = Stick("d_parallel", powder_average=True)
    signal = stick.signal(acquisition, {"d_parallel": 1.7})
    assert signal[0] == pytest.approx(1.0)
    assert 0 < signal[1] < 1


def test_powder_stick_regression_against_sandi_formula() -> None:
    acquisition = Acquisition.from_b_values(
        [0, 0.5, 1.0, 2.0],
        b_unit="ms/um^2",
    )
    signal = Stick("d_parallel", powder_average=True).signal(
        acquisition,
        {"d_parallel": 1.5},
    )
    np.testing.assert_allclose(
        signal,
        [1.0, 0.7975078096, 0.6633509458, 0.5043435602],
        rtol=1e-9,
    )


def test_oriented_stick_uses_gradient_projection() -> None:
    acquisition = Acquisition.from_b_values(
        [1, 1],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0]],
    )
    stick = Stick("d_parallel", orientation="axis")
    signal = stick.signal(
        acquisition,
        {"d_parallel": 2.0, "axis": np.array([1, 0, 0])},
    )
    np.testing.assert_allclose(signal, [np.exp(-2.0), 1.0])


def test_stick_linear_encoding_matches_sandi_formula() -> None:
    acquisition = Acquisition.from_b_values(
        [0.5, 1.0, 2.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        b_tensor_shape=1,
    )
    orientation = np.array([1.0, 2.0, 3.0])
    signal = Stick("d_parallel", orientation="axis").signal(
        acquisition,
        {"d_parallel": 1.7, "axis": orientation},
    )
    unit = orientation / np.linalg.norm(orientation)
    expected = np.exp(
        -acquisition.b_values
        * 1.7
        * (acquisition.gradients @ unit) ** 2
    )
    np.testing.assert_allclose(signal, expected, rtol=1e-14)


def test_powder_stick_supports_spherical_b_tensor_encoding() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0],
        b_unit="ms/um^2",
        b_tensor_shape=0,
    )
    signal = Stick("d", powder_average=True).signal(acquisition, {"d": 1.7})
    np.testing.assert_allclose(signal, np.exp(-1.7 / 3.0))


def test_watson_kappa_zero_matches_powder_average_stick() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0, 1.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0]],
    )
    compartment = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=WatsonFOD(kappa="kappa", theta="theta", phi="phi", n_theta=18, n_phi=36),
        orientation_parameter="axis",
    )
    parameters = {"d": 1.7, "kappa": 0.0, "theta": np.pi / 2.0, "phi": 0.0}
    signal = compartment.signal(acquisition, parameters)
    expected = Stick("d", powder_average=True).signal(
        Acquisition.from_b_values([1.0], b_unit="ms/um^2"),
        {"d": 1.7},
    )[0]
    np.testing.assert_allclose(signal, [expected, expected], rtol=1e-3, atol=1e-3)


def test_watson_large_kappa_approaches_single_orientation_stick() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0, 1.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0]],
    )
    compartment = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=WatsonFOD(kappa="kappa", theta="theta", phi="phi", n_theta=24, n_phi=48),
        orientation_parameter="axis",
    )
    parameters = {"d": 1.5, "kappa": 100.0, "theta": np.pi / 2.0, "phi": 0.0}
    signal = compartment.signal(acquisition, parameters)
    expected = Stick("d", orientation="axis").signal(
        acquisition,
        {"d": 1.5, "axis": np.array([1.0, 0.0, 0.0])},
    )
    np.testing.assert_allclose(signal, expected, rtol=2e-2, atol=2e-2)


def test_bingham_zero_concentration_matches_powder_average_stick() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0, 1.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0]],
    )
    compartment = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=BinghamFOD(
            kappa1="kappa1",
            kappa2="kappa2",
            theta="theta",
            phi="phi",
            psi="psi",
            n_theta=18,
            n_phi=36,
        ),
        orientation_parameter="axis",
    )
    parameters = {
        "d": 1.7,
        "kappa1": 0.0,
        "kappa2": 0.0,
        "theta": np.pi / 2.0,
        "phi": 0.0,
        "psi": 0.0,
    }
    signal = compartment.signal(acquisition, parameters)
    expected = Stick("d", powder_average=True).signal(
        Acquisition.from_b_values([1.0], b_unit="ms/um^2"),
        {"d": 1.7},
    )[0]
    np.testing.assert_allclose(signal, [expected, expected], rtol=1e-3, atol=1e-3)


def test_watson_hypergeometric_matches_quadrature_convolution() -> None:
    acquisition = Acquisition.from_b_values(
        [0.5, 1.0, 2.0, 3.0],
        b_unit="ms/um^2",
        gradients=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        b_tensor_shape=0.6,
    )
    parameters = {
        "d": 1.6,
        "kappa": 3.5,
        "theta": 1.0,
        "phi": -0.6,
    }

    quadrature_model = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=WatsonFOD(kappa="kappa", theta="theta", phi="phi", n_theta=28, n_phi=56),
        orientation_parameter="axis",
    )
    hypergeometric_model = WatsonStickHypergeometric(
        diffusivity="d",
        kappa="kappa",
        theta="theta",
        phi="phi",
        n_theta=28,
        n_phi=56,
    )

    expected = quadrature_model.signal(acquisition, parameters)
    actual = hypergeometric_model.signal(acquisition, parameters)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)


def test_bingham_hypergeometric_matches_quadrature_convolution() -> None:
    acquisition = Acquisition.from_b_values(
        [0.5, 1.0, 2.0, 3.0],
        b_unit="ms/um^2",
        gradients=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, -1, 1],
        ],
        b_tensor_shape=0.8,
    )
    parameters = {
        "d": 1.7,
        "kappa1": -8.0,
        "kappa2": -2.0,
        "theta": 1.2,
        "phi": 0.7,
        "psi": -0.4,
    }

    quadrature_model = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=BinghamFOD(
            kappa1="kappa1",
            kappa2="kappa2",
            theta="theta",
            phi="phi",
            psi="psi",
            n_theta=28,
            n_phi=56,
        ),
        orientation_parameter="axis",
    )
    hypergeometric_model = BinghamStickHypergeometric(
        diffusivity="d",
        kappa1="kappa1",
        kappa2="kappa2",
        theta="theta",
        phi="phi",
        psi="psi",
        n_theta=28,
        n_phi=56,
    )

    expected = quadrature_model.signal(acquisition, parameters)
    actual = hypergeometric_model.signal(acquisition, parameters)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)


def test_watson_zeppelin_hypergeometric_matches_quadrature_convolution() -> None:
    acquisition = Acquisition.from_b_values(
        [0.5, 1.0, 2.0, 3.0],
        b_unit="ms/um^2",
        gradients=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, -1],
        ],
        b_tensor_shape=0.7,
    )
    parameters = {
        "d_parallel": 1.8,
        "d_perpendicular": 0.5,
        "kappa": 4.2,
        "theta": 0.9,
        "phi": -0.5,
    }

    quadrature_model = FODConvolved(
        response=Zeppelin(
            parallel_diffusivity="d_parallel",
            perpendicular_diffusivity="d_perpendicular",
            orientation="axis",
        ),
        fod=WatsonFOD(kappa="kappa", theta="theta", phi="phi", n_theta=28, n_phi=56),
        orientation_parameter="axis",
    )
    hypergeometric_model = WatsonZeppelinHypergeometric(
        parallel_diffusivity="d_parallel",
        perpendicular_diffusivity="d_perpendicular",
        kappa="kappa",
        theta="theta",
        phi="phi",
        n_theta=28,
        n_phi=56,
    )

    expected = quadrature_model.signal(acquisition, parameters)
    actual = hypergeometric_model.signal(acquisition, parameters)
    np.testing.assert_allclose(actual, expected, rtol=4e-4, atol=4e-4)


def test_tensor_linear_encoding_matches_sandi_formula() -> None:
    acquisition = Acquisition.from_b_values(
        [0.5, 1.0, 2.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        b_tensor_shape=1,
    )
    orientation = np.array([1.0, 2.0, 3.0])
    signal = Zeppelin("d_parallel", "d_perpendicular", "axis").signal(
        acquisition,
        {
            "d_parallel": 1.8,
            "d_perpendicular": 0.4,
            "axis": orientation,
        },
    )
    unit = orientation / np.linalg.norm(orientation)
    projection = acquisition.gradients @ unit
    adc = 0.4 + (1.8 - 0.4) * projection**2
    np.testing.assert_allclose(
        signal,
        np.exp(-acquisition.b_values * adc),
        rtol=1e-14,
    )


def test_spherical_b_tensor_signal_is_orientation_independent() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0, 1.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0]],
        b_tensor_shape=0,
    )
    tensor = Zeppelin("d_parallel", "d_perpendicular", "axis")
    parameters = {
        "d_parallel": 1.8,
        "d_perpendicular": 0.3,
        "axis": np.array([1, 0, 0]),
    }
    signal = tensor.signal(acquisition, parameters)
    expected_md = (1.8 + 2 * 0.3) / 3
    np.testing.assert_allclose(signal, np.exp(-expected_md))


def test_oblate_zeppelin_is_supported() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0]],
    )
    signal = Zeppelin("d_parallel", "d_perpendicular", "axis").signal(
        acquisition,
        {
            "d_parallel": 0.5,
            "d_perpendicular": 1.0,
            "axis": np.array([1, 0, 0]),
        },
    )
    np.testing.assert_allclose(signal, np.exp(-0.5))


def test_general_tensor_matches_direct_quadratic_form() -> None:
    gradients = np.eye(3)
    acquisition = Acquisition.from_b_values(
        [1.0, 1.0, 1.0],
        b_unit="ms/um^2",
        gradients=gradients,
    )
    tensor = Tensor(("d_x", "d_y", "d_z"))
    signal = tensor.signal(
        acquisition,
        {"d_x": 0.3, "d_y": 0.8, "d_z": 1.7},
    )
    np.testing.assert_allclose(signal, np.exp(-np.array([0.3, 0.8, 1.7])))


def test_general_tensor_rotation_rotates_principal_axes() -> None:
    acquisition = Acquisition.from_b_values(
        [1.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0]],
    )
    rotation = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    signal = Tensor(("d_x", "d_y", "d_z"), "rotation").signal(
        acquisition,
        {
            "d_x": 0.3,
            "d_y": 0.8,
            "d_z": 1.7,
            "rotation": rotation,
        },
    )
    np.testing.assert_allclose(signal, np.exp(-1.7))


def test_general_tensor_spherical_encoding_uses_mean_diffusivity() -> None:
    acquisition = Acquisition.from_b_values(
        [2.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0]],
        b_tensor_shape=0,
    )
    signal = Tensor(("d_x", "d_y", "d_z")).signal(
        acquisition,
        {"d_x": 0.3, "d_y": 0.8, "d_z": 1.9},
    )
    np.testing.assert_allclose(signal, np.exp(-2.0))


def test_t2_relaxation_weights_underlying_compartment() -> None:
    acquisition = Acquisition.from_b_values(
        [0, 1],
        b_unit="ms/um^2",
        echo_time=[40, 80],
    )
    relaxed_ball = T2Relaxation(Ball("d"), t2="T2")
    signal = relaxed_ball.signal(acquisition, {"d": 1.0, "T2": 80.0})
    expected = np.exp(-acquisition.b_values) * np.exp(
        -acquisition.echo_time / 80.0
    )
    np.testing.assert_allclose(signal, expected)


def test_t2_relaxation_requires_echo_time() -> None:
    acquisition = Acquisition.from_b_values([0], b_unit="ms/um^2")
    relaxed_ball = T2Relaxation(Ball("d"), t2="T2")
    with pytest.raises(ValueError, match="requires echo_time"):
        relaxed_ball.signal(acquisition, {"d": 1.0, "T2": 80.0})


def test_sphere_regression_against_multid_and_sandi_gpa_references() -> None:
    acquisition = Acquisition.from_pgse(
        [0, 0.5, 1.0, 2.0],
        b_unit="ms/um^2",
        pulse_duration=30,
        diffusion_time=50,
    )
    signal = Sphere(radius="radius", diffusivity="d_sphere").signal(
        acquisition,
        {"radius": 10.0, "d_sphere": 3.0},
    )
    # Independently generated by both valid source implementations. They agree
    # to machine precision once G is supplied in their respective units.
    np.testing.assert_allclose(
        signal,
        [
            1.0,
            0.9103254459130931,
            0.8286924174768718,
            0.6867311227836619,
        ],
        rtol=1e-13,
        atol=1e-15,
    )


def test_sphere_zero_gradient_has_unit_signal() -> None:
    acquisition = Acquisition.from_pgse(
        [0],
        b_unit="ms/um^2",
        pulse_duration=30,
        diffusion_time=50,
    )
    signal = Sphere("radius", "d_sphere").signal(
        acquisition,
        {"radius": 10.0, "d_sphere": 3.0},
    )
    assert signal[0] == pytest.approx(1.0, abs=1e-15)


def test_sphere_attenuation_decreases_with_b_for_fixed_timing() -> None:
    acquisition = Acquisition.from_pgse(
        [0, 0.5, 1.0, 2.0],
        b_unit="ms/um^2",
        pulse_duration=30,
        diffusion_time=50,
    )
    signal = Sphere("radius", "d_sphere").signal(
        acquisition,
        {"radius": 10.0, "d_sphere": 3.0},
    )
    assert np.all(np.diff(signal) < 0)


@pytest.mark.parametrize(
    "missing_field",
    ["pulse_duration", "diffusion_time", "gradient_strength"],
)
def test_sphere_requires_every_pgse_field(missing_field) -> None:
    fields = {
        "pulse_duration": [30.0],
        "diffusion_time": [50.0],
        "gradient_strength": [0.02],
    }
    fields[missing_field] = None
    acquisition = Acquisition.from_b_values(
        [1.0],
        b_unit="ms/um^2",
        **fields,
    )
    with pytest.raises(ValueError, match=missing_field):
        Sphere("radius", "d_sphere").signal(
            acquisition,
            {"radius": 10.0, "d_sphere": 3.0},
        )
