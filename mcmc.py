"""Generic ensemble MCMC for composable models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..acquisition import Acquisition
from ..model import Model
from ..parameters import ParameterSpace
from .noise import GaussianNoise


@dataclass(frozen=True)
class MCMCResult:
    """Posterior samples in both sampler and named physical coordinates."""

    latent_samples: NDArray[np.float64]
    log_probability: NDArray[np.float64]
    acceptance_fraction: NDArray[np.float64]
    parameter_space: ParameterSpace
    burn_in: int
    thin: int

    def physical_samples(self, *, flat: bool = True) -> Mapping[str, NDArray[np.float64]]:
        """Return samples keyed by physical parameter name."""
        shape = self.latent_samples.shape
        vectors = self.latent_samples.reshape(-1, shape[-1])
        unpacked = [self.parameter_space.unpack(vector) for vector in vectors]
        output_shape = (vectors.shape[0],) if flat else shape[:-1]
        return {
            name: np.asarray([sample[name] for sample in unpacked]).reshape(output_shape)
            for name in self.parameter_space.names
        }

    def summary(
        self,
        *,
        credible_interval: tuple[float, float] = (0.025, 0.975),
    ) -> Mapping[str, Mapping[str, float]]:
        """Posterior mean, standard deviation, median, and credible interval."""
        lower, upper = credible_interval
        if not 0 <= lower < upper <= 1:
            raise ValueError("credible_interval must satisfy 0 <= lower < upper <= 1.")
        samples = self.physical_samples(flat=True)
        return {
            name: {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "lower": float(np.quantile(values, lower)),
                "upper": float(np.quantile(values, upper)),
            }
            for name, values in samples.items()
        }

    def autocorrelation_time(self) -> Mapping[str, float]:
        """Integrated autocorrelation time for each latent coordinate."""
        try:
            from emcee.autocorr import integrated_time
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "emcee is required for autocorrelation diagnostics."
            ) from exc
        tau = np.asarray(
            integrated_time(self.latent_samples, tol=0, quiet=True)
        ).reshape(-1)
        return dict(zip(self.parameter_space.varying_names, tau.tolist(), strict=True))

    def effective_sample_size(self) -> Mapping[str, float]:
        """Approximate ensemble effective sample size from autocorrelation time."""
        total = self.latent_samples.shape[0] * self.latent_samples.shape[1]
        return {
            name: float(total / tau)
            for name, tau in self.autocorrelation_time().items()
        }


class EmceeFitter:
    """Sample a model posterior with the affine-invariant emcee ensemble."""

    def __init__(
        self,
        model: Model,
        parameter_space: ParameterSpace,
        noise: GaussianNoise,
    ) -> None:
        if parameter_space.dimension < 1:
            raise ValueError("MCMC requires at least one varying coordinate.")
        self.model = model
        self.parameter_space = parameter_space
        self.noise = noise

    def log_posterior(
        self,
        vector: ArrayLike,
        acquisition: Acquisition,
        data: ArrayLike,
    ) -> float:
        observations = np.asarray(data, dtype=float).reshape(-1)
        if observations.shape != (acquisition.n_measurements,):
            raise ValueError(
                "data must contain exactly one value per acquisition measurement."
            )
        log_prior = self.parameter_space.log_uniform_prior(vector)
        if not np.isfinite(log_prior):
            return -np.inf
        parameters = self.parameter_space.unpack(vector)
        prediction = self.model.signal(acquisition, parameters)
        return log_prior + self.noise.log_likelihood(observations, prediction)

    def sample(
        self,
        acquisition: Acquisition,
        data: ArrayLike,
        *,
        n_steps: int = 5000,
        burn_in: int | None = None,
        thin: int = 1,
        n_walkers: int | None = None,
        rng: np.random.Generator | None = None,
        progress: bool = False,
    ) -> MCMCResult:
        try:
            import emcee
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "emcee is required for MCMC; install microstructure-mri[fitting]."
            ) from exc

        if n_steps < 2:
            raise ValueError("n_steps must be at least two.")
        if thin < 1:
            raise ValueError("thin must be at least one.")
        if burn_in is None:
            burn_in = int(0.3 * n_steps)
        if burn_in < 0 or burn_in >= n_steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps.")

        ndim = self.parameter_space.dimension
        if n_walkers is None:
            n_walkers = max(2 * ndim + 2, 24)
        if n_walkers < 2 * ndim:
            raise ValueError("n_walkers must be at least twice the parameter dimension.")

        generator = rng or np.random.default_rng()
        initial_state = self.parameter_space.random_vectors(
            n_walkers,
            rng=generator,
        )

        def posterior(vector: NDArray[np.float64]) -> float:
            return self.log_posterior(vector, acquisition, data)

        sampler = emcee.EnsembleSampler(n_walkers, ndim, posterior)
        sampler.run_mcmc(initial_state, n_steps, progress=progress)
        return MCMCResult(
            latent_samples=sampler.get_chain(discard=burn_in, thin=thin),
            log_probability=sampler.get_log_prob(discard=burn_in, thin=thin),
            acceptance_fraction=np.asarray(sampler.acceptance_fraction),
            parameter_space=self.parameter_space,
            burn_in=burn_in,
            thin=thin,
        )
