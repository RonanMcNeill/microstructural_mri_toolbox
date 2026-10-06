"""Generic nonlinear least-squares fitting for composable models."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Mapping

import numpy as np
import time
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from ..acquisition import Acquisition
from ..model import Model
from ..parameters import ParameterSpace


@dataclass(frozen=True)
class FitResult:
    """Result of a single-voxel nonlinear fit."""

    parameters: Mapping[str, float]
    predicted_signal: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: float
    success: bool
    message: str
    n_evaluations: int
    n_starts: int


@dataclass(frozen=True)
class BatchFitResult:
    """Results arranged in the spatial shape preceding the measurement axis."""

    parameters: Mapping[str, NDArray[np.float64]]
    predicted_signal: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: NDArray[np.float64]
    success: NDArray[np.bool_]
    n_evaluations: NDArray[np.int_]


def _dti_principal_direction(
    acquisition: Acquisition,
    data: NDArray[np.float64],
    *,
    b_max: float = 1.5,
) -> tuple[float, float] | None:
    """Estimate the principal diffusion direction via a log-linear DTI GLM.

    Constructs the standard 6-element log-linear design matrix
    (Basser et al. 1994) from linearly encoded diffusion-weighted
    measurements up to ``b_max``, fits the tensor by least squares, and
    returns the spherical angles ``(theta, phi)`` of the principal
    eigenvector.

    Only measurements with ``0 < b <= b_max`` are used.  The default cap
    of ``1.5 ms/µm²`` excludes high-b shells where signal is strongly
    non-Gaussian and the log-linear model breaks down, consistent with
    standard DTIFIT practice (FSL, MRtrix).

    Returns ``None`` when the acquisition lacks the necessary fields
    (gradients, at least one b=0 measurement, and at least six
    diffusion-weighted measurements with linear b-tensor shape within
    ``b_max``).
    """
    if acquisition.gradients is None:
        return None

    b_values = acquisition.b_values
    gradients = acquisition.gradients
    beta = (
        np.ones(acquisition.n_measurements)
        if acquisition.b_tensor_shape is None
        else acquisition.b_tensor_shape
    )

    # Only low-to-moderate b linearly encoded measurements obey the DTI model.
    linear_dw = (b_values > 0.0) & (b_values <= b_max) & np.isclose(beta, 1.0)
    b0_mask = b_values <= 0.05  # canonical ms/um^2 threshold

    if not np.any(b0_mask) or np.sum(linear_dw) < 6:
        return None

    s0 = float(np.mean(data[b0_mask]))
    if s0 <= 0.0 or not np.isfinite(s0):
        return None

    b_dw = b_values[linear_dw]
    g_dw = gradients[linear_dw]
    log_attenuation = np.log(np.maximum(data[linear_dw] / s0, 1e-8))

    gx, gy, gz = g_dw[:, 0], g_dw[:, 1], g_dw[:, 2]
    # Design matrix rows: b * [gx², gy², gz², 2gxgy, 2gxgz, 2gygz]
    design = b_dw[:, None] * np.column_stack([
        gx ** 2, gy ** 2, gz ** 2,
        2.0 * gx * gy, 2.0 * gx * gz, 2.0 * gy * gz,
    ])

    # log(S/S0) = -design @ [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    d_vec, _, _, _ = np.linalg.lstsq(-design, log_attenuation, rcond=None)

    tensor = np.array([
        [d_vec[0], d_vec[3], d_vec[4]],
        [d_vec[3], d_vec[1], d_vec[5]],
        [d_vec[4], d_vec[5], d_vec[2]],
    ])
    if not np.all(np.isfinite(tensor)):
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Standardise hemisphere so theta stays in [0, pi/2].
    if principal[2] < 0.0:
        principal = -principal

    theta = float(np.arccos(np.clip(principal[2], -1.0, 1.0)))
    phi = float(np.arctan2(principal[1], principal[0]))

    # Ensure theta and phi are both within [0,pi] to have them defined on the half sphere due to symmetry.
    phi = phi % np.pi

    return theta, phi


class LeastSquaresFitter:
    """Fit any Model using a shared ParameterSpace."""

    def __init__(self, model: Model, parameter_space: ParameterSpace) -> None:
        self.model = model
        self.parameter_space = parameter_space

    def fit(
        self,
        acquisition: Acquisition,
        data: ArrayLike,
        *,
        sigma: ArrayLike = 1.0,
        initial: Mapping[str, float] | None = None,
        n_starts: int = 1,
        max_nfev: int = 2000,
        rng: np.random.Generator | None = None,
        initialise_orientation: bool = True,
        dti_b_max: float = 1.5,
    ) -> FitResult:
        """Fit the model to a single measurement vector.

        Parameters
        ----------
        acquisition:
            The acquisition describing the MRI protocol.
        data:
            Observed signal, one value per measurement.
        sigma:
            Noise standard deviation, used to weight the residuals as
            ``(predicted - observed) / sigma``.

            When ``sigma`` is a uniform scalar (the default, ``1.0``), this
            reduces to ordinary unweighted least squares. The parameter
            estimates and predicted signal are **identical for any constant
            scalar sigma** — only the numerical value of the cost changes.
            You can safely omit ``sigma`` if you do not know the noise level;
            the optimised parameters will be correct.

            Provide per-measurement values (an array with one entry per
            measurement) when some measurements are more reliable than others,
            for example when noise varies across b-value shells or when you
            have an external noise map.

            ``sigma`` is only *essential* for formal chi-squared statistics or
            for comparing model fits via the cost value. For Bayesian inference
            via :class:`EmceeFitter`, use :class:`GaussianNoise` instead — the
            noise level there directly shapes the posterior.
        initial:
            Optional fixed starting point. Cannot be used with ``n_starts > 1``.
        n_starts:
            Number of random multi-starts. The best result across all starts is
            returned. Increasing this improves robustness against local minima.
        max_nfev:
            Maximum number of function evaluations per start.
        rng:
            Random number generator used to draw multi-start points.
        """
        observations = np.asarray(data, dtype=float).reshape(-1)
        if observations.shape != (acquisition.n_measurements,):
            raise ValueError(
                "data must contain exactly one value per acquisition measurement."
            )
        if not np.all(np.isfinite(observations)):
            raise ValueError("data must contain only finite values.")

        uncertainty = np.asarray(sigma, dtype=float)
        if uncertainty.ndim == 0 or uncertainty.size == 1:
            uncertainty = np.full(
                acquisition.n_measurements,
                float(uncertainty.reshape(-1)[0]),
            )
        else:
            uncertainty = uncertainty.reshape(-1)
        if uncertainty.shape != observations.shape:
            raise ValueError("sigma must be scalar or have one value per measurement.")
        if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty <= 0):
            raise ValueError("sigma must be finite and greater than zero.")

        if initial is not None and n_starts != 1:
            raise ValueError("A custom initial mapping can only be used with n_starts=1.")
        if initial is not None:
            starts = self.parameter_space.pack(initial)[None, :]
        else:
            starts = self.parameter_space.random_vectors(
                n_starts,
                rng=rng or np.random.default_rng(),
            )

        # Seed start 0 with the DTI principal eigenvector when theta/phi are free.
        if initial is None and initialise_orientation:
            free_params = {p.name for p in self.parameter_space.parameters if p.vary}
            if "theta" in free_params and "phi" in free_params:
                dti_result = _dti_principal_direction(
                    acquisition, observations, b_max=dti_b_max
                )
                if dti_result is not None:
                    init_vals = self.parameter_space.initial_values()
                    init_vals["theta"], init_vals["phi"] = dti_result
                    starts[0] = self.parameter_space.pack(init_vals)

        bounds = self.parameter_space.vector_bounds()

        def residual_function(vector: NDArray[np.float64]) -> NDArray[np.float64]:
            parameters = self.parameter_space.unpack(vector)
            prediction = self.model.signal(acquisition, parameters)
            return (prediction - observations) / uncertainty

        candidates = [
            least_squares(
                residual_function,
                start,
                bounds=bounds,
                max_nfev=max_nfev,
                xtol=1e-6,
                ftol=1e-6,
                gtol=1e-6,
            )
            for start in starts
        ]
        best = min(candidates, key=lambda candidate: candidate.cost)
        parameters = self.parameter_space.unpack(best.x)
        predicted = self.model.signal(acquisition, parameters)
        return FitResult(
            parameters=parameters,
            predicted_signal=predicted,
            residuals=observations - predicted,
            cost=float(best.cost),
            success=bool(best.success),
            message=str(best.message),
            n_evaluations=int(best.nfev),
            n_starts=n_starts,
        )

    def fit_many(
        self,
        acquisition: Acquisition,
        data: ArrayLike,
        *,
        sigma: ArrayLike = 1.0,
        n_starts: int = 1,
        max_nfev: int = 2000,
        n_jobs: int = 1,
        rng: np.random.Generator | None = None,
        initialise_orientation: bool = True,
        dti_b_max: float = 1.5,
    ) -> BatchFitResult:
        """Fit an array whose final dimension contains measurements."""
        observations = np.asarray(data, dtype=float)
        if observations.ndim < 2 or observations.shape[-1] != acquisition.n_measurements:
            raise ValueError(
                "data must have spatial dimensions followed by n_measurements."
            )
        if n_jobs < 1:
            raise ValueError("n_jobs must be at least one.")

        spatial_shape = observations.shape[:-1]
        flat_data = observations.reshape(-1, acquisition.n_measurements)
        uncertainty = np.asarray(sigma, dtype=float)
        if uncertainty.ndim == 0 or uncertainty.size == 1:
            flat_sigma = [float(uncertainty.reshape(-1)[0])] * flat_data.shape[0]
        elif uncertainty.shape == (acquisition.n_measurements,):
            flat_sigma = [uncertainty] * flat_data.shape[0]
        elif uncertainty.shape == observations.shape:
            flat_sigma = list(
                uncertainty.reshape(-1, acquisition.n_measurements)
            )
        else:
            raise ValueError(
                "sigma must be scalar, one measurement vector, or match data."
            )

        generator = rng or np.random.default_rng()
        seeds = generator.integers(
            0,
            np.iinfo(np.uint32).max,
            size=flat_data.shape[0],
            dtype=np.uint32,
        )

        def fit_index(index: int) -> FitResult:
            try:
                return self.fit(
                    acquisition,
                    flat_data[index],
                    sigma=flat_sigma[index],
                    n_starts=n_starts,
                    max_nfev=max_nfev,
                    rng=np.random.default_rng(int(seeds[index])),
                    initialise_orientation=initialise_orientation,
                    dti_b_max=dti_b_max,
                )
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as exc:

                print(
                    f"\nWARNING: voxel {index} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

                return None
                # =====================================================================
        # FIT VOXELS
        #
        # Progress is reported for both serial and parallel fitting.
        # =====================================================================

        n_voxels = flat_data.shape[0]

        start_time = time.time()


        def print_progress(
            completed: int,
        ) -> None:

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                n_voxels
                - completed
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else np.nan
            )

            eta_minutes = (
                eta_seconds
                / 60.0
            )

            print(
                f"\rFitting voxels: "
                f"{completed:,}/{n_voxels:,} "
                f"({100.0 * completed / n_voxels:.1f}%) | "
                f"{rate:.2f} vox/s | "
                f"ETA {eta_minutes:.1f} min",
                end="",
                flush=True,
            )


        # =====================================================================
        # SERIAL FITTING
        # =====================================================================

        if n_jobs == 1:

            results = []

            for index in range(
                n_voxels
            ):

                result = fit_index(
                    index
                )

                results.append(
                    result
                )

                completed = (
                    index + 1
                )

                if (
                    completed == 1
                    or completed % 10 == 0
                    or completed == n_voxels
                ):

                    print_progress(
                        completed
                    )


        # =====================================================================
        # PARALLEL FITTING
        # =====================================================================

        else:

            # Preallocate results so that output remains in the original
            # voxel order even though worker threads finish out of order.
            results = [
                None
            ] * n_voxels

            completed = 0

            with ThreadPoolExecutor(
                max_workers=n_jobs
            ) as executor:

                futures = {
                    executor.submit(
                        fit_index,
                        index,
                    ): index

                    for index in range(
                        n_voxels
                    )
                }

                for future in as_completed(
                    futures
                ):

                    index = futures[
                        future
                    ]

                    results[
                        index
                    ] = future.result()

                    completed += 1

                    if (
                        completed == 1
                        or completed % 10 == 0
                        or completed == n_voxels
                    ):

                        print_progress(
                            completed
                        )


        print()

        # =====================================================================
        # ASSEMBLE RESULTS
        #
        # Failed voxels are represented by:
        #
        #   parameters       = NaN
        #   predicted signal = NaN
        #   residuals        = NaN
        #   cost             = NaN
        #   success          = False
        #   n_evaluations    = 0
        #
        # This prevents one pathological voxel from killing a volume/chunk.
        # =====================================================================

        n_measurements = (
            acquisition.n_measurements
        )


        # ---------------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------------

        parameters = {}

        for name in self.parameter_space.names:

            values = np.full(
                n_voxels,
                np.nan,
                dtype=float,
            )

            for index, result in enumerate(
                results
            ):

                if result is None:
                    continue

                values[index] = (
                    result.parameters[name]
                )

            parameters[name] = (
                values.reshape(
                    spatial_shape
                )
            )


        # ---------------------------------------------------------------------
        # Predicted signal + residuals
        # ---------------------------------------------------------------------

        predicted = np.full(
            (
                n_voxels,
                n_measurements,
            ),
            np.nan,
            dtype=float,
        )

        residuals = np.full(
            (
                n_voxels,
                n_measurements,
            ),
            np.nan,
            dtype=float,
        )


        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        cost = np.full(
            n_voxels,
            np.nan,
            dtype=float,
        )

        success = np.zeros(
            n_voxels,
            dtype=bool,
        )

        n_evaluations = np.zeros(
            n_voxels,
            dtype=int,
        )


        # ---------------------------------------------------------------------
        # Populate successful fits
        # ---------------------------------------------------------------------

        for index, result in enumerate(
            results
        ):

            if result is None:
                continue

            predicted[
                index
            ] = result.predicted_signal

            residuals[
                index
            ] = result.residuals

            cost[
                index
            ] = result.cost

            success[
                index
            ] = result.success

            n_evaluations[
                index
            ] = result.n_evaluations


        # ---------------------------------------------------------------------
        # Report failures
        # ---------------------------------------------------------------------

        n_failed = int(
            np.sum(
                [
                    result is None
                    for result in results
                ]
            )
        )

        if n_failed > 0:

            print(
                f"\nWARNING: "
                f"{n_failed}/{n_voxels} voxels "
                f"failed fitting."
            )


        # ---------------------------------------------------------------------
        # Return
        # ---------------------------------------------------------------------

        return BatchFitResult(

            parameters=parameters,

            predicted_signal=(
                predicted.reshape(
                    observations.shape
                )
            ),

            residuals=(
                residuals.reshape(
                    observations.shape
                )
            ),

            cost=(
                cost.reshape(
                    spatial_shape
                )
            ),

            success=(
                success.reshape(
                    spatial_shape
                )
            ),

            n_evaluations=(
                n_evaluations.reshape(
                    spatial_shape
                )
            ),
        )
