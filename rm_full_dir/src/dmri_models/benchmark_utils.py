from __future__ import annotations
from dmri_models import acquisition
import numpy as np
from dmri_models import LeastSquaresFitter, noddi_derived_parameters
from dmri_models.acquisition import Acquisition

def _parameter_errors(true_parameters: dict, fit_parameters: dict) -> dict:
    """Return errors for shared scalar parameters only."""
    shared_keys = sorted(set(true_parameters).intersection(fit_parameters))
    return {
        f"{key}_error": float(fit_parameters[key] - true_parameters[key])
        for key in shared_keys
        if np.isscalar(true_parameters[key]) and np.isscalar(fit_parameters[key])
    }


def _optional_odi(parameters: dict) -> float:
    """Return ODI when the parameters belong to a NODDI model; otherwise NaN."""
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
    rng = np.random.default_rng(seed)

    signal_true = model.signal(acquisition, true_parameters)
    sigma = true_parameters["S0"] / snr
    signal_noisy = signal_true + rng.normal(0.0, sigma, size=signal_true.shape)

    fitter = LeastSquaresFitter(model, parameter_space)

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


def fit_one_run_spherical_mean(
    acquisition,
    seed: int,
    true_parameters: dict,
    model,
    parameter_space,
    snr: float = 20.0,
    n_starts: int = 16,
    max_nfev: int = 3000,
):
    rng = np.random.default_rng(seed)

    signal_true = model.signal(acquisition, true_parameters)
    sigma = true_parameters["S0"] / snr
    signal_noisy = signal_true + rng.normal(0.0, sigma, size=signal_true.shape)

    b_vals = np.asarray(acquisition.b_values)
    b_shells = np.unique(np.round(b_vals, 4))

    sm_signal, sm_b, n_dirs = [], [], []
    for b in b_shells:
        mask = np.abs(b_vals - b) < 1e-4
        sm_signal.append(np.mean(signal_noisy[mask]))
        sm_b.append(b)
        n_dirs.append(mask.sum())

    sm_signal = np.asarray(sm_signal)
    sm_b = np.asarray(sm_b)
    n_dirs = np.asarray(n_dirs)
    sigma_sm = sigma / np.sqrt(n_dirs)

    acq_sm = Acquisition.from_pgse(
        sm_b,
        b_unit="ms/um^2",
        pulse_duration=acquisition.pulse_duration[0],
        diffusion_time=acquisition.diffusion_time[0],
        b_tensor_shape=acquisition.b_tensor_shape[0],
    )

    fitter = LeastSquaresFitter(model, parameter_space)

    fit = fitter.fit(
        acq_sm,
        sm_signal,
        sigma=sigma_sm,
        n_starts=n_starts,
        max_nfev=max_nfev,
        rng=rng,
        initialise_orientation=True,
    )

    residuals = sm_signal - fit.predicted_signal

    odi_true = _optional_odi(true_parameters)
    odi_fit = _optional_odi(fit.parameters)

    if np.isfinite(odi_true) and np.isfinite(odi_fit):
        odi_error = float(odi_fit - odi_true)
    else:
        odi_error = np.nan

    out = {
        "success": bool(fit.success),
        "error": None,
        "fit_kind": "sm",
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
        "n_fit_measurements": len(sm_b),
        "n_raw_measurements": acquisition.n_measurements,
    }

    out.update(_parameter_errors(true_parameters, fit.parameters))
    return out


def run_task(task, fit_one_run_directional, fit_one_run_spherical_mean, true_parameters, model, parameter_space, snr, n_starts, max_nfev):
    acq_name, fit_kind, acquisition, rep, seed = task

    if fit_kind == "dir":
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
    elif fit_kind == "sm":
        out = fit_one_run_spherical_mean(
            acquisition=acquisition,
            seed=seed,
            true_parameters=true_parameters,
            model=model,
            parameter_space=parameter_space,
            snr=snr,
            n_starts=n_starts,
            max_nfev=max_nfev,
        )
    else:
        raise ValueError(f"Unknown fit_kind: {fit_kind!r}")

    out["acquisition"] = acq_name
    out["rep"] = rep
    out["n_measurements"] = out.get("n_fit_measurements", acquisition.n_measurements)
    out["n_raw_measurements"] = acquisition.n_measurements
    out["fit_kind"] = fit_kind
    return out

def multi_te_acquisition(base_acq: Acquisition, echo_times) -> Acquisition:
    """Replicate a base acquisition at multiple echo times.

    Each TE produces a complete copy of the base acquisition's measurements,
    giving n_te * n_measurements total measurements.  All acquisition fields
    (b-values, gradients, timing) are identical across TEs; only echo_time
    varies.  The resulting per-measurement echo_time field is what
    T2Relaxation uses to compute per-compartment T2 weighting.

    Parameters
    ----------
    base_acq:
        The acquisition to replicate.
    echo_times:
        Sequence of echo times in ms.  Each entry produces one full copy of
        the acquisition's measurements.
    """
    echo_times = np.asarray(echo_times, dtype=float)
    n_te = len(echo_times)

    def tile1d(v):
        return None if v is None else np.tile(v, n_te)

    def tile2d(v):
        return None if v is None else np.tile(v, (n_te, 1))

    return Acquisition(
        b_values=tile1d(base_acq.b_values),
        gradients=tile2d(base_acq.gradients),
        b_tensor_shape=tile1d(base_acq.b_tensor_shape),
        pulse_duration=tile1d(base_acq.pulse_duration),
        diffusion_time=tile1d(base_acq.diffusion_time),
        gradient_strength=tile1d(base_acq.gradient_strength),
        # Each TE block covers all n measurements, so repeat each TE value n times.
        echo_time=np.repeat(echo_times, base_acq.n_measurements),
        inversion_time=tile1d(base_acq.inversion_time),
        repetition_time=tile1d(base_acq.repetition_time),
    )