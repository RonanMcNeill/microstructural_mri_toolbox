import numpy as np
import pytest

from dmri_models import Acquisition, Stick
from dmri_models.compartments import FODConvolved, SphericalHarmonicFOD, WatsonFOD


def _sample_unit_sphere(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def _watson_density(directions: np.ndarray, kappa: float, theta: float, phi: float) -> np.ndarray:
    mean_axis = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    projection = directions @ mean_axis
    from scipy.special import hyp1f1

    normaliser = 4.0 * np.pi * hyp1f1(0.5, 1.5, kappa)
    return np.exp(kappa * projection**2) / normaliser


def _fit_even_sh_coefficients(
    directions: np.ndarray,
    density: np.ndarray,
    l_max: int,
) -> np.ndarray:
    basis, _, _ = SphericalHarmonicFOD.real_even_basis(directions, l_max, cs_phase=False)
    coefficients, *_ = np.linalg.lstsq(basis, density, rcond=None)
    return coefficients


def test_sh_fod_rejects_odd_l_max() -> None:
    with pytest.raises(ValueError, match="non-negative even"):
        SphericalHarmonicFOD(coefficients="c", l_max=3)


def test_coefficient_count_validation_for_both_conventions() -> None:
    sh_without_l0 = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=False)
    sh_with_l0 = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=True)

    with pytest.raises(ValueError, match="Expected 14 SH coefficients"):
        sh_without_l0.quadrature({"c": np.zeros(13)})

    with pytest.raises(ValueError, match="Expected 15 SH coefficients"):
        sh_with_l0.quadrature({"c": np.zeros(14)})


def test_isotropic_sh_without_l0_matches_powder_average_stick() -> None:
    acquisition = Acquisition.from_b_values(
        [0.0, 1.0, 2.0],
        b_unit="ms/um^2",
        gradients=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    sh_fod = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=False)
    convolved = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=sh_fod,
        orientation_parameter="axis",
    )

    coefficients = np.zeros(sh_fod.coefficient_count(4, include_l0=False))
    signal = convolved.signal(acquisition, {"d": 1.7, "c": coefficients})
    expected = Stick(diffusivity="d", powder_average=True).signal(acquisition, {"d": 1.7})
    np.testing.assert_allclose(signal, expected, rtol=3e-3, atol=3e-3)


def test_include_l0_is_accepted_and_internally_renormalised_with_warning() -> None:
    sh_fod = SphericalHarmonicFOD(coefficients="c", l_max=0, include_l0=True)
    with pytest.warns(UserWarning, match="internally normalised"):
        directions, weights = sh_fod.quadrature({"c": np.array([2.0 / np.sqrt(4.0 * np.pi)])})
    assert directions.shape[1] == 3
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_negative_density_is_rectified_when_requested() -> None:
    sh_fod = SphericalHarmonicFOD(
        coefficients="c",
        l_max=2,
        include_l0=False,
        rectify=True,
    )
    coefficients = np.array([0.0, 0.0, 2.5, 0.0, 0.0])
    with pytest.warns(UserWarning, match="rectified"):
        _, weights = sh_fod.quadrature({"c": coefficients})
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12)


def test_negative_density_raises_when_rectification_disabled() -> None:
    sh_fod = SphericalHarmonicFOD(
        coefficients="c",
        l_max=2,
        include_l0=False,
        rectify=False,
    )
    coefficients = np.array([0.0, 0.0, 2.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="density is negative"):
        sh_fod.quadrature({"c": coefficients})


def test_b0_signal_is_unity_for_any_valid_sh_fod() -> None:
    acquisition = Acquisition.from_b_values([0.0], b_unit="ms/um^2", gradients=[[1.0, 0.0, 0.0]])
    sh_fod = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=True)
    response = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=sh_fod,
        orientation_parameter="axis",
    )
    coefficients = np.zeros(sh_fod.coefficient_count(4, include_l0=True))
    coefficients[0] = 1.0 / np.sqrt(4.0 * np.pi)
    signal = response.signal(acquisition, {"d": 1.5, "c": coefficients})
    np.testing.assert_allclose(signal, [1.0], rtol=1e-12, atol=1e-12)


def test_sh_fod_matches_watson_convolution_after_sh_fit() -> None:
    l_max = 8
    kappa = 7.5
    theta = 1.1
    phi = -0.3

    fit_directions = _sample_unit_sphere(4000, seed=7)
    density = _watson_density(fit_directions, kappa=kappa, theta=theta, phi=phi)
    coefficients = _fit_even_sh_coefficients(fit_directions, density, l_max=l_max)

    acquisition = Acquisition.from_b_values(
        [0.5, 1.0, 2.0, 3.0],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
        b_tensor_shape=1.0,
    )

    watson = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=WatsonFOD(kappa="kappa", theta="theta", phi="phi", n_theta=32, n_phi=64),
        orientation_parameter="axis",
    )
    sh_based = FODConvolved(
        response=Stick(diffusivity="d", orientation="axis"),
        fod=SphericalHarmonicFOD(
            coefficients="c",
            l_max=l_max,
            include_l0=True,
            n_theta=32,
            n_phi=64,
            rectify=True,
        ),
        orientation_parameter="axis",
    )

    params = {"d": 1.6, "kappa": kappa, "theta": theta, "phi": phi, "c": coefficients}
    expected = watson.signal(acquisition, params)
    actual = sh_based.signal(acquisition, params)
    np.testing.assert_allclose(actual, expected, rtol=2.5e-2, atol=2.5e-2)


def test_rotational_invariants_are_normalised_to_unit_l0() -> None:
    sh_fod = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=True)
    coefficients = np.zeros(sh_fod.coefficient_count(4, include_l0=True))
    coefficients[0] = 1.0 / np.sqrt(4.0 * np.pi)
    coefficients[5] = 0.12
    invariants = sh_fod.rotational_invariants({"c": coefficients}, normalise=True)
    np.testing.assert_allclose(invariants[0], 1.0, rtol=1e-12, atol=1e-12)
    assert np.all(invariants[1:] >= 0)


def test_default_crossing_coefficients_return_expected_sizes() -> None:
    for crossing in ("single", "double", "triple"):
        anisotropic = SphericalHarmonicFOD.default_crossing_coefficients(
            l_max=8,
            crossing=crossing,
            include_l0=False,
        )
        with_l0 = SphericalHarmonicFOD.default_crossing_coefficients(
            l_max=8,
            crossing=crossing,
            include_l0=True,
        )
        assert anisotropic.size == SphericalHarmonicFOD.coefficient_count(8, include_l0=False)
        assert with_l0.size == SphericalHarmonicFOD.coefficient_count(8, include_l0=True)


def test_default_crossing_coefficients_produce_nonnegative_density_after_rectification() -> None:
    sh_fod = SphericalHarmonicFOD(coefficients="c", l_max=8, include_l0=False, rectify=True)
    directions = _sample_unit_sphere(2000, seed=17)
    for crossing in ("single", "double", "triple"):
        coefficients = SphericalHarmonicFOD.default_crossing_coefficients(
            l_max=8,
            crossing=crossing,
            include_l0=False,
        )
        density = sh_fod.density(directions, {"c": coefficients})
        assert np.all(np.isfinite(density))


def test_random_unit_directions_are_unit_norm() -> None:
    directions = SphericalHarmonicFOD.random_unit_directions(128, seed=123)
    assert directions.shape == (128, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=1e-12, atol=1e-12)


def test_spherical_grid_returns_expected_shapes() -> None:
    theta, phi, directions = SphericalHarmonicFOD.spherical_grid(n_theta=20, n_phi=40)
    assert theta.shape == (20, 40)
    assert phi.shape == (20, 40)
    assert directions.shape == (800, 3)


def test_watson_density_is_finite_and_positive() -> None:
    directions = _sample_unit_sphere(1000, seed=44)
    density = SphericalHarmonicFOD.watson_density(
        directions,
        mean_axis=np.array([1.0, 1.0, 0.5]),
        kappa=8.0,
    )
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)


def test_crossing_coefficients_from_axes_matches_default_double_crossing() -> None:
    kwargs = dict(l_max=8, include_l0=False, kappa=22.0, n_theta=90, n_phi=180)
    expected = SphericalHarmonicFOD.default_crossing_coefficients(
        crossing="double",
        **kwargs,
    )
    actual = SphericalHarmonicFOD.crossing_coefficients_from_axes(
        axes=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        weights=np.array([0.5, 0.5]),
        **kwargs,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
