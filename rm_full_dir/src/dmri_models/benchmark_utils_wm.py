"""Directional-only benchmarking utilities for white-matter models."""

from __future__ import annotations

import numpy as np

from dmri_models import LeastSquaresFitter, noddi_derived_parameters, EmceeFitter, GaussianNoise


def _parameter_errors(true_parameters: dict, fit_parameters: dict) -> dict:
    """Return errors for shared scalar parameters only."""
    shared_keys = sorted(set(true_parameters).intersection(fit_parameters))
    return {
        f"{key}_error": float(fit_parameters[key] - true_parameters[key])
        for key in shared_keys
        if np.isscalar(true_parameters[key]) and np.isscalar(fit_parameters[key])
    }


def _optional_odi(parameters: dict) -> float:
    """Return ODI when the parameters belong to a NODDI-like model; otherwise NaN."""
    try:
        return float(noddi_derived_parameters(parameters)["odi"])
    except Exception:
        return np.nan


def fit_one_run_directional(
    acquisition,
    seed: int,
    true_parameters: dict,
    model,
    parameter_space,
    snr: float = 20.0,
    n_starts: int = 16,
    max_nfev: int = 3000,
):
    """Simulate one noisy directional dataset and fit it with least squares."""
    rng = np.random.default_rng(seed)

    signal_true = model.signal(acquisition, true_parameters)
    sigma = true_parameters["S0"] / snr
    signal_noisy = signal_true + rng.normal(0.0, sigma, size=signal_true.shape)

    fitter = LeastSquaresFitter(model, parameter_space)

    try:
        fit = fitter.fit(
            acquisition,
            signal_noisy,
            sigma=sigma,
            n_starts=n_starts,
            max_nfev=max_nfev,
            rng=rng,
            initialise_orientation=True,
        )

        residuals = signal_noisy - fit.predicted_signal

        odi_true = _optional_odi(true_parameters)
        odi_fit = _optional_odi(fit.parameters)

        if np.isfinite(odi_true) and np.isfinite(odi_fit):
            odi_error = float(odi_fit - odi_true)
        else:
            odi_error = np.nan

        out = {
            "success": bool(fit.success),
            "error": None,
            "fit_kind": "dir",
            "cost": fit.cost,
            "n_evaluations": fit.n_evaluations,
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "odi_true": odi_true,
            "odi_fit": odi_fit,
            "odi_error": odi_error,
            "signal_true": signal_true,
            "signal_noisy": signal_noisy,
            "predicted": fit.predicted_signal,
            "residuals": residuals,
            "fit_parameters": fit.parameters,
        }

        out.update(_parameter_errors(true_parameters, fit.parameters))
        return out

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "fit_kind": "dir",
            "cost": np.nan,
            "n_evaluations": np.nan,
            "rmse": np.nan,
            "odi_true": np.nan,
            "odi_fit": np.nan,
            "odi_error": np.nan,
            "signal_true": signal_true,
            "signal_noisy": signal_noisy,
            "predicted": np.full_like(signal_true, np.nan),
            "residuals": np.full_like(signal_true, np.nan),
            "fit_parameters": {},
        }


def run_task(
    task,
    fit_one_run_directional,
    true_parameters,
    model,
    parameter_space,
    snr,
    n_starts,
    max_nfev,
):
    """Run one directional benchmark task."""
    acq_name, acquisition, rep, seed = task

    out = fit_one_run_directional(
        acquisition=acquisition,
        seed=seed,
        true_parameters=true_parameters,
        model=model,
        parameter_space=parameter_space,
        snr=snr,
        n_starts=n_starts,
        max_nfev=max_nfev,
    )

    out["acquisition"] = acq_name
    out["rep"] = rep
    out["n_measurements"] = acquisition.n_measurements
    out["n_raw_measurements"] = acquisition.n_measurements
    out["fit_kind"] = "dir"
    return out

class GaussianPriorFitter(EmceeFitter):
    """EmceeFitter extended with optional Gaussian priors on named parameters."""

    def __init__(self, model, parameter_space, noise, *, gaussian_priors=None):
        super().__init__(model, parameter_space, noise)
        self.gaussian_priors = dict(gaussian_priors or {})

    def log_posterior(self, vector, acquisition, data):
        base = super().log_posterior(vector, acquisition, data)
        if not np.isfinite(base):
            return -np.inf

        parameters = self.parameter_space.unpack(vector)
        log_gaussian = 0.0
        for name, (mu, sigma) in self.gaussian_priors.items():
            if name in parameters:
                log_gaussian += -0.5 * ((parameters[name] - mu) / sigma) ** 2

        return base + log_gaussian

def run_task_ls_mcmc(
    task,
    true_parameters,
    model,
    parameter_space,
    snr,
    n_starts,
    max_nfev,
    mcmc_kwargs,
    gaussian_priors=None,
):
    acq_name, acquisition, rep, seed = task

    # Separate RNG streams so signal noise, LS, and MCMC are reproducible
    # but independent.
    ss = np.random.SeedSequence(seed)
    rng_signal, rng_ls, rng_mcmc_none, rng_mcmc_priors = [
        np.random.default_rng(s) for s in ss.spawn(4)
    ]

    signal_true = model.signal(acquisition, true_parameters)
    sigma = true_parameters["S0"] / snr
    signal_noisy = signal_true + rng_signal.normal(0.0, sigma, size=signal_true.shape)

    rows = []

    # 1) Least squares
    ls_fitter = LeastSquaresFitter(model, parameter_space)
    ls_fit = ls_fitter.fit(
        acquisition,
        signal_noisy,
        sigma=sigma,
        n_starts=n_starts,
        max_nfev=max_nfev,
        rng=rng_ls,
        initialise_orientation=True,
    )

    rows.append({
        "fit_stage": "ls",
        "prior_set": "none",
        "success": bool(ls_fit.success),
        "error": None,
        "cost": ls_fit.cost,
        "n_evaluations": ls_fit.n_evaluations,
        "rmse": float(np.sqrt(np.mean((signal_noisy - ls_fit.predicted_signal) ** 2))),
        "signal_true": signal_true,
        "signal_noisy": signal_noisy,
        "predicted": ls_fit.predicted_signal,
        "residuals": signal_noisy - ls_fit.predicted_signal,
        "fit_parameters": ls_fit.parameters,
    })

    # 2) MCMC with no priors
    noise = GaussianNoise(sigma=sigma)
    mcmc_fitter = GaussianPriorFitter(
        model,
        parameter_space,
        noise,
        gaussian_priors=None,
    )
    mcmc_none = mcmc_fitter.sample(
        acquisition,
        signal_noisy,
        rng=rng_mcmc_none,
        **mcmc_kwargs,
    )

    rows.append({
        "fit_stage": "mcmc",
        "prior_set": "none",
        "mcmc_result": mcmc_none,
    })

    # 3) MCMC with priors, only if provided
    if gaussian_priors:
        mcmc_prior_fitter = GaussianPriorFitter(
            model,
            parameter_space,
            noise,
            gaussian_priors=gaussian_priors,
        )
        mcmc_prior = mcmc_prior_fitter.sample(
            acquisition,
            signal_noisy,
            rng=rng_mcmc_priors,
            **mcmc_kwargs,
        )

        rows.append({
            "fit_stage": "mcmc",
            "prior_set": "gaussian",
            "mcmc_result": mcmc_prior,
        })

    for row in rows:
        row["acquisition"] = acq_name
        row["rep"] = rep
        row["n_measurements"] = acquisition.n_measurements
        row["n_raw_measurements"] = acquisition.n_measurements

    return rows