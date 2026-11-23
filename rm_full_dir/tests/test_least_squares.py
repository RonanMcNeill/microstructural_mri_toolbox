import numpy as np
import pytest

from dmri_models import (
    Acquisition,
    Ball,
    Component,
    LeastSquaresFitter,
    Model,
    Parameter,
    ParameterSpace,
    Simplex,
)


def two_ball_problem():
    acquisition = Acquisition.from_b_values(
        np.linspace(0, 3, 31),
        b_unit="ms/um^2",
    )
    model = Model(
        [
            Component("slow", Ball("d_slow")),
            Component("fast", Ball("d_fast")),
        ]
    )
    space = ParameterSpace(
        [
            Parameter("S0", 0.9, (0.5, 1.5)),
            Parameter("d_slow", 0.6, (0.2, 1.2)),
            Parameter("d_fast", 2.4, (1.3, 3.0)),
        ],
        simplexes=[Simplex(("slow_fraction", "fast_fraction"), (0.5, 0.5))],
    )
    truth = {
        "S0": 1.1,
        "d_slow": 0.7,
        "d_fast": 2.1,
        "slow_fraction": 0.65,
        "fast_fraction": 0.35,
    }
    data = model.signal(acquisition, truth)
    return acquisition, model, space, truth, data


def test_noiseless_fit_recovers_two_ball_parameters() -> None:
    acquisition, model, space, truth, data = two_ball_problem()
    result = LeastSquaresFitter(model, space).fit(acquisition, data)
    assert result.success
    assert result.cost < 1e-18
    assert result.parameters == pytest.approx(truth, rel=2e-6, abs=2e-7)
    np.testing.assert_allclose(result.predicted_signal, data, rtol=1e-9, atol=1e-10)


def test_multistart_fit_is_reproducible() -> None:
    acquisition, model, space, _, data = two_ball_problem()
    fitter = LeastSquaresFitter(model, space)
    first = fitter.fit(acquisition, data, n_starts=4, rng=np.random.default_rng(12))
    second = fitter.fit(acquisition, data, n_starts=4, rng=np.random.default_rng(12))
    assert first.parameters == pytest.approx(second.parameters)
    assert first.n_starts == 4


def test_sigma_weights_residuals_but_result_residuals_remain_signal_units() -> None:
    acquisition, model, space, _, data = two_ball_problem()
    noisy = data.copy()
    noisy[-1] += 0.01
    result = LeastSquaresFitter(model, space).fit(
        acquisition,
        noisy,
        sigma=np.linspace(0.01, 0.03, data.size),
    )
    np.testing.assert_allclose(result.residuals, noisy - result.predicted_signal)


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.inf])
def test_invalid_sigma_is_rejected(sigma) -> None:
    acquisition, model, space, _, data = two_ball_problem()
    with pytest.raises(ValueError, match="sigma"):
        LeastSquaresFitter(model, space).fit(acquisition, data, sigma=sigma)


def test_data_length_must_match_acquisition() -> None:
    acquisition, model, space, _, data = two_ball_problem()
    with pytest.raises(ValueError, match="one value per"):
        LeastSquaresFitter(model, space).fit(acquisition, data[:-1])


def test_batch_fit_preserves_spatial_and_measurement_shapes() -> None:
    acquisition, model, space, truth, data = two_ball_problem()
    second_truth = dict(truth, S0=0.9, slow_fraction=0.4, fast_fraction=0.6)
    batch = np.stack([data, model.signal(acquisition, second_truth)]).reshape(
        1, 2, -1
    )
    result = LeastSquaresFitter(model, space).fit_many(
        acquisition,
        batch,
        n_starts=2,
        n_jobs=2,
        rng=np.random.default_rng(22),
    )
    assert result.success.shape == (1, 2)
    assert result.predicted_signal.shape == batch.shape
    assert result.parameters["S0"].shape == (1, 2)
    np.testing.assert_allclose(result.predicted_signal, batch, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.parameters["S0"], [[1.1, 0.9]], rtol=1e-5)


def test_batch_fit_accepts_voxelwise_measurement_uncertainty() -> None:
    acquisition, model, space, _, data = two_ball_problem()
    batch = np.stack([data, data])
    sigma = np.full_like(batch, 0.02)
    result = LeastSquaresFitter(model, space).fit_many(
        acquisition,
        batch,
        sigma=sigma,
    )
    assert np.all(result.success)


def test_batch_fit_rejects_missing_spatial_dimension() -> None:
    acquisition, model, space, _, data = two_ball_problem()
    with pytest.raises(ValueError, match="spatial dimensions"):
        LeastSquaresFitter(model, space).fit_many(acquisition, data)
