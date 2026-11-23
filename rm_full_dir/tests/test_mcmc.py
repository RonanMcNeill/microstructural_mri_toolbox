import builtins

import numpy as np
import pytest

from dmri_models import (
    Acquisition,
    Ball,
    Component,
    EmceeFitter,
    GaussianNoise,
    Model,
    Parameter,
    ParameterSpace,
    Simplex,
)


def posterior_problem():
    acquisition = Acquisition.from_b_values([0, 0.5, 1, 2], b_unit="ms/um^2")
    model = Model(
        [
            Component("slow", Ball("d_slow")),
            Component("fast", Ball("d_fast")),
        ]
    )
    space = ParameterSpace(
        [
            Parameter("S0", 1.0, (0.5, 1.5)),
            Parameter("d_slow", 0.8, (0.2, 1.2)),
            Parameter("d_fast", 2.0, (1.3, 3.0)),
        ],
        simplexes=[Simplex(("slow_fraction", "fast_fraction"), (0.6, 0.4))],
    )
    data = model.signal(acquisition, space.initial_values())
    fitter = EmceeFitter(model, space, GaussianNoise(0.03))
    return acquisition, model, space, data, fitter


def test_gaussian_log_likelihood_includes_normalization() -> None:
    noise = GaussianNoise(2.0)
    actual = noise.log_likelihood([1.0, 2.0], [1.0, 2.0])
    expected = -2 * np.log(2.0 * np.sqrt(2.0 * np.pi))
    assert actual == pytest.approx(expected)


def test_log_posterior_is_finite_at_initial_values() -> None:
    acquisition, _, space, data, fitter = posterior_problem()
    actual = fitter.log_posterior(space.initial_vector(), acquisition, data)
    assert np.isfinite(actual)


def test_log_posterior_rejects_out_of_bounds_parameter() -> None:
    acquisition, _, space, data, fitter = posterior_problem()
    vector = space.initial_vector()
    vector[0] = 10.0
    assert fitter.log_posterior(vector, acquisition, data) == -np.inf


def test_uniform_simplex_prior_includes_coordinate_jacobian() -> None:
    space = ParameterSpace(
        [],
        simplexes=[Simplex(("a", "b"), (0.5, 0.5))],
    )
    centre = space.log_uniform_prior(space.pack({"a": 0.5, "b": 0.5}))
    edge = space.log_uniform_prior(space.pack({"a": 0.99, "b": 0.01}))
    assert centre > edge


def test_physical_sample_conversion_preserves_chain_shape() -> None:
    from dmri_models import MCMCResult

    _, _, space, _, _ = posterior_problem()
    chain = np.tile(space.initial_vector(), (3, 4, 1))
    result = MCMCResult(
        latent_samples=chain,
        log_probability=np.zeros((3, 4)),
        acceptance_fraction=np.ones(4),
        parameter_space=space,
        burn_in=1,
        thin=1,
    )
    physical = result.physical_samples(flat=False)
    assert physical["S0"].shape == (3, 4)
    np.testing.assert_allclose(
        physical["slow_fraction"] + physical["fast_fraction"],
        1.0,
    )
    summary = result.summary()
    assert summary["S0"]["mean"] == pytest.approx(1.0)
    assert summary["slow_fraction"]["median"] == pytest.approx(0.6)


def test_missing_emcee_has_actionable_error(monkeypatch) -> None:
    acquisition, _, _, data, fitter = posterior_problem()
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "emcee":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"microstructure-mri\[fitting\]"):
        fitter.sample(acquisition, data, n_steps=2)


def test_short_emcee_chain_has_expected_shape_and_valid_fractions() -> None:
    pytest.importorskip("emcee")
    acquisition, _, space, data, fitter = posterior_problem()
    result = fitter.sample(
        acquisition,
        data,
        n_steps=12,
        burn_in=2,
        n_walkers=24,
        rng=np.random.default_rng(13),
    )
    assert result.latent_samples.shape == (10, 24, space.dimension)
    assert result.log_probability.shape == (10, 24)
    assert np.all((result.acceptance_fraction >= 0) & (result.acceptance_fraction <= 1))
    physical = result.physical_samples(flat=False)
    np.testing.assert_allclose(
        physical["slow_fraction"] + physical["fast_fraction"],
        1.0,
        atol=1e-15,
    )
    tau = result.autocorrelation_time()
    effective = result.effective_sample_size()
    assert set(tau) == set(space.varying_names)
    assert all(value > 0 for value in effective.values())


def test_summary_rejects_invalid_credible_interval() -> None:
    from dmri_models import MCMCResult

    _, _, space, _, _ = posterior_problem()
    result = MCMCResult(
        latent_samples=np.tile(space.initial_vector(), (3, 4, 1)),
        log_probability=np.zeros((3, 4)),
        acceptance_fraction=np.ones(4),
        parameter_space=space,
        burn_in=1,
        thin=1,
    )
    with pytest.raises(ValueError, match="credible_interval"):
        result.summary(credible_interval=(0.9, 0.1))
