import numpy as np

from dmri_models import Acquisition, Stick
from dmri_models.compartments import (
    FODConvolved,
    SMIKernelConvolved,
    SphericalHarmonicFOD,
)


def _random_unit_directions(n: int, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _positive_sh_coefficients(l_max: int) -> np.ndarray:
    # Keep anisotropic coefficients deliberately small so FOD density remains positive.
    count = SphericalHarmonicFOD.coefficient_count(l_max, include_l0=False)
    coeffs = np.zeros(count, dtype=float)
    coeffs[2] = 0.06
    coeffs[7] = 0.02
    coeffs[8] = -0.01
    return coeffs


def test_smi_kernel_signal_matches_quadrature_for_sh_fod() -> None:
    l_max = 8
    gradients = _random_unit_directions(48, seed=11)
    b_values = np.linspace(0.2, 2.8, gradients.shape[0])
    acquisition = Acquisition.from_b_values(
        b_values,
        b_unit="ms/um^2",
        gradients=gradients,
        b_tensor_shape=np.ones_like(b_values),
    )

    fod = SphericalHarmonicFOD(
        coefficients="coeffs",
        l_max=l_max,
        include_l0=False,
        rectify=False,
        n_theta=30,
        n_phi=60,
    )
    params = {"d": 1.7, "coeffs": _positive_sh_coefficients(l_max)}

    quadrature = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
    )
    smi_like = SMIKernelConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
        n_projections=200,
    )

    expected = quadrature.signal(acquisition, params)
    actual = smi_like.signal(acquisition, params)
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


def test_smi_like_fit_recovers_anisotropic_coefficients_from_self_simulation() -> None:
    l_max = 6
    gradients = _random_unit_directions(80, seed=17)
    b_values = np.linspace(0.4, 2.5, gradients.shape[0])
    acquisition = Acquisition.from_b_values(
        b_values,
        b_unit="ms/um^2",
        gradients=gradients,
        b_tensor_shape=np.ones_like(b_values),
    )

    fod = SphericalHarmonicFOD(
        coefficients="coeffs",
        l_max=l_max,
        include_l0=False,
        rectify=False,
    )
    model = SMIKernelConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
        n_projections=180,
    )

    true_coeffs = _positive_sh_coefficients(l_max)[: fod.coefficient_count(l_max, include_l0=False)]
    params = {"d": 1.5, "coeffs": true_coeffs}
    simulated = model.signal(acquisition, params)

    estimated = model.fit_coefficients(
        acquisition,
        simulated,
        {"d": 1.5},
        include_l0=False,
        ridge=1e-8,
    )
    np.testing.assert_allclose(estimated, true_coeffs, rtol=2e-3, atol=2e-3)


def test_smi_like_fit_reconstructs_quadrature_signal() -> None:
    l_max = 6
    gradients = _random_unit_directions(96, seed=22)
    b_values = np.linspace(0.3, 3.0, gradients.shape[0])
    acquisition = Acquisition.from_b_values(
        b_values,
        b_unit="ms/um^2",
        gradients=gradients,
        b_tensor_shape=np.ones_like(b_values),
    )

    fod = SphericalHarmonicFOD(
        coefficients="coeffs",
        l_max=l_max,
        include_l0=False,
        rectify=False,
        n_theta=32,
        n_phi=64,
    )
    true_coeffs = _positive_sh_coefficients(l_max)[: fod.coefficient_count(l_max, include_l0=False)]

    quadrature = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
    )
    smi_like = SMIKernelConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
        n_projections=220,
    )

    observed = quadrature.signal(acquisition, {"d": 1.6, "coeffs": true_coeffs})
    estimated = smi_like.fit_coefficients(
        acquisition,
        observed,
        {"d": 1.6},
        include_l0=False,
        ridge=1e-8,
    )
    reconstructed = smi_like.signal(acquisition, {"d": 1.6, "coeffs": estimated})

    np.testing.assert_allclose(reconstructed, observed, rtol=3e-2, atol=3e-2)


def test_smi_like_fit_with_pinv_matches_lstsq_for_well_conditioned_case() -> None:
    l_max = 6
    gradients = _random_unit_directions(120, seed=31)
    b_values = np.linspace(0.2, 2.4, gradients.shape[0])
    acquisition = Acquisition.from_b_values(
        b_values,
        b_unit="ms/um^2",
        gradients=gradients,
        b_tensor_shape=np.ones_like(b_values),
    )

    fod = SphericalHarmonicFOD(
        coefficients="coeffs",
        l_max=l_max,
        include_l0=False,
        rectify=False,
    )
    model = SMIKernelConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
        n_projections=180,
    )

    true_coeffs = _positive_sh_coefficients(l_max)[: fod.coefficient_count(l_max, include_l0=False)]
    observed = model.signal(acquisition, {"d": 1.6, "coeffs": true_coeffs})

    estimated_lstsq = model.fit_coefficients(
        acquisition,
        observed,
        {"d": 1.6},
        include_l0=False,
        ridge=0.0,
        solver="lstsq",
    )
    estimated_pinv = model.fit_coefficients(
        acquisition,
        observed,
        {"d": 1.6},
        include_l0=False,
        ridge=0.0,
        solver="pinv",
    )

    np.testing.assert_allclose(estimated_pinv, estimated_lstsq, rtol=2e-6, atol=2e-8)


def test_smi_like_fit_rejects_unknown_solver() -> None:
    l_max = 4
    gradients = _random_unit_directions(40, seed=7)
    acquisition = Acquisition.from_b_values(
        np.linspace(0.2, 1.8, gradients.shape[0]),
        b_unit="ms/um^2",
        gradients=gradients,
        b_tensor_shape=np.ones(gradients.shape[0]),
    )

    fod = SphericalHarmonicFOD(
        coefficients="coeffs",
        l_max=l_max,
        include_l0=False,
        rectify=False,
    )
    model = SMIKernelConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=fod,
        orientation_parameter="axis",
    )

    observed = model.signal(
        acquisition,
        {"d": 1.5, "coeffs": _positive_sh_coefficients(l_max)},
    )

    try:
        model.fit_coefficients(
            acquisition,
            observed,
            {"d": 1.5},
            include_l0=False,
            solver="bad_solver",
        )
    except ValueError as error:
        assert "solver" in str(error)
    else:
        raise AssertionError("Expected ValueError for unsupported solver.")
