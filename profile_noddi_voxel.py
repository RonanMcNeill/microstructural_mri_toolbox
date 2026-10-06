# =============================================================================
# PROFILE NODDI — FULL ACQUISITION vs b <= 2 ms/um^2
# =============================================================================
#
# Purpose
# -------
# Compare NODDI fitting using:
#
#   1. FULL acquisition:
#          b = 0, 0.5, 1, 2, 3, 5 ms/um^2
#
#   2. REDUCED acquisition:
#          b = 0, 0.5, 1, 2 ms/um^2
#
# This does NOT change:
#
#   - the NODDI model
#   - quadrature resolution
#   - fitting tolerances
#   - diffusivities
#   - initialization strategy
#   - any code used by the novel inflammation model
#
# =============================================================================


import cProfile
import io
import pstats
import time

from pathlib import Path

import nibabel as nib
import numpy as np

from dmri_models.acquisition import Acquisition
from dmri_models.fitting.least_squares import LeastSquaresFitter
from dmri_models.presets.noddi import noddi, odi_from_kappa


# =============================================================================
# 1. PATHS
# =============================================================================

PREPROC_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input"
)


DATA_PATH = (
    PREPROC_DIR
    / "data_gibbs_eddy_reoriented_global_scaled.nii.gz"
)


MASK_PATH = (
    PREPROC_DIR
    / "injection_site_mask.nii.gz"
)


BVAL_PATH = (
    PREPROC_DIR
    / "bvals"
)


BVEC_PATH = (
    PREPROC_DIR
    / "bvecs"
)


OUTPUT_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/model_fitting/noddi_profiling"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

PULSE_DURATION_MS = 5.0

DIFFUSION_TIME_MS = 12.0


N_STARTS = 1

MAX_NFEV = 2000

RNG_SEED = 12345


# Complete nominal b=2 shell extends slightly above exactly 2.0.
REDUCED_B_MAX = 2.1


# Number of repeated model.signal() calls for timing.
N_SIGNAL_CALLS = 100


# =============================================================================
# 3. ACQUISITION SUBSETTING HELPER
# =============================================================================

def subset_acquisition(
    acquisition: Acquisition,
    keep: np.ndarray,
) -> Acquisition:
    """Subset every populated per-measurement Acquisition field."""

    keep = np.asarray(
        keep,
        dtype=bool,
    )


    if keep.shape != (
        acquisition.n_measurements,
    ):

        raise ValueError(
            "keep must contain exactly one Boolean "
            "per acquisition measurement."
        )


    def subset_optional(
        value,
    ):

        if value is None:

            return None


        return np.asarray(
            value
        )[keep]


    return Acquisition(

        b_values=np.asarray(
            acquisition.b_values
        )[keep],

        gradients=(
            None

            if acquisition.gradients is None

            else np.asarray(
                acquisition.gradients
            )[keep]
        ),

        b_tensor_shape=subset_optional(
            acquisition.b_tensor_shape
        ),

        pulse_duration=subset_optional(
            acquisition.pulse_duration
        ),

        diffusion_time=subset_optional(
            acquisition.diffusion_time
        ),

        gradient_strength=subset_optional(
            acquisition.gradient_strength
        ),

        echo_time=subset_optional(
            acquisition.echo_time
        ),

        inversion_time=subset_optional(
            acquisition.inversion_time
        ),

        repetition_time=subset_optional(
            acquisition.repetition_time
        ),
    )


# =============================================================================
# 4. BUILD FULL ACQUISITION
# =============================================================================

full_acquisition = Acquisition.from_fsl(

    BVAL_PATH,

    BVEC_PATH,

    bval_unit="s/mm^2",

    b0_threshold=0.05,

    rectangular_pgse=True,

    pulse_duration=PULSE_DURATION_MS,

    diffusion_time=DIFFUSION_TIME_MS,

    b_tensor_shape=1.0,
)


# =============================================================================
# 5. BUILD REDUCED ACQUISITION
# =============================================================================

b_keep = (
    np.asarray(
        full_acquisition.b_values
    )
    <= REDUCED_B_MAX
)


reduced_acquisition = subset_acquisition(
    full_acquisition,
    b_keep,
)


print(
    "\n============================================"
)

print(
    "ACQUISITION COMPARISON"
)

print(
    "============================================"
)


print(
    f"\nFull measurements:    "
    f"{full_acquisition.n_measurements}"
)


print(
    f"Reduced measurements: "
    f"{reduced_acquisition.n_measurements}"
)


print(
    f"Removed measurements: "
    f"{np.sum(~b_keep)}"
)


print(
    "\nFull acquisition:"
)

full_acquisition.print_summary()


print(
    "\nReduced acquisition:"
)

reduced_acquisition.print_summary()


# =============================================================================
# 6. MODEL
#
# IMPORTANT:
# Both fits use exactly the same NODDI model and 24 x 48 quadrature.
# =============================================================================

preset = noddi(

    include_csf=True,

    vary_intracellular_diffusivity=False,

    vary_csf_diffusivity=False,

    vary_kappa=True,

    fod_model="watson",

    fod_quadrature_theta=24,

    fod_quadrature_phi=48,
)


fitter = LeastSquaresFitter(
    preset.model,
    preset.parameters,
)


# =============================================================================
# 7. LOAD DATA
# =============================================================================

data = np.asarray(

    nib.load(
        str(
            DATA_PATH
        )
    ).get_fdata(
        dtype=np.float32
    ),

    dtype=float,
)


mask = (

    np.asarray(

        nib.load(
            str(
                MASK_PATH
            )
        ).get_fdata(
            dtype=np.float32
        ),

        dtype=float,
    )

    > 0
)


indices = np.argwhere(
    mask
)


print(
    f"\nMask contains "
    f"{len(indices):,} voxels."
)


# =============================================================================
# 8. SELECT SAME REPRESENTATIVE VOXEL
# =============================================================================

voxel_index = (
    len(indices)
    // 2
)


x, y, z = indices[
    voxel_index
]


full_signal = np.asarray(

    data[
        x,
        y,
        z,
        :,
    ],

    dtype=float,
)


reduced_signal = (
    full_signal[
        b_keep
    ]
)


# =============================================================================
# SANITY CHECKS
# =============================================================================

if (
    full_signal.size
    != full_acquisition.n_measurements
):

    raise ValueError(
        "Full signal/acquisition measurement mismatch."
    )


if (
    reduced_signal.size
    != reduced_acquisition.n_measurements
):

    raise ValueError(
        "Reduced signal/acquisition measurement mismatch."
    )


print(
    "\n============================================"
)

print(
    "PROFILE VOXEL"
)

print(
    "============================================"
)


print(
    f"\nVoxel: "
    f"({x}, {y}, {z})"
)


print(
    f"Full signal:    "
    f"{full_signal.size} measurements"
)


print(
    f"Reduced signal: "
    f"{reduced_signal.size} measurements"
)


# =============================================================================
# 9. FIT + PROFILE FUNCTION
# =============================================================================

def run_profile(
    *,
    name: str,
    acquisition: Acquisition,
    signal: np.ndarray,
):
    """Fit and profile one acquisition for the selected voxel."""

    print(
        "\n\n============================================"
    )

    print(
        name
    )

    print(
        "============================================"
    )


    print(
        f"\nMeasurements: "
        f"{acquisition.n_measurements}"
    )


    # =========================================================================
    # WARM-UP FIT
    # =========================================================================

    print(
        "\nWarm-up fit..."
    )


    warm_start = (
        time.perf_counter()
    )


    warm_result = fitter.fit(

        acquisition,

        signal,

        sigma=1.0,

        n_starts=N_STARTS,

        max_nfev=MAX_NFEV,

        rng=np.random.default_rng(
            RNG_SEED
        ),

        initialise_orientation=True,
    )


    warm_seconds = (
        time.perf_counter()
        - warm_start
    )


    print(
        f"Warm-up runtime: "
        f"{warm_seconds:.3f} s"
    )


    print(
        f"Warm-up nfev:    "
        f"{warm_result.n_evaluations}"
    )


    # =========================================================================
    # PROFILE SECOND IDENTICAL FIT
    # =========================================================================

    print(
        "\nProfiling second identical fit..."
    )


    profiler = (
        cProfile.Profile()
    )


    profiler.enable()


    fit_start = (
        time.perf_counter()
    )


    result = fitter.fit(

        acquisition,

        signal,

        sigma=1.0,

        n_starts=N_STARTS,

        max_nfev=MAX_NFEV,

        rng=np.random.default_rng(
            RNG_SEED
        ),

        initialise_orientation=True,
    )


    fit_seconds = (
        time.perf_counter()
        - fit_start
    )


    profiler.disable()


    # =========================================================================
    # FORWARD MODEL BENCHMARK
    # =========================================================================

    parameters = dict(
        result.parameters
    )


    # Warm model once.
    preset.model.signal(
        acquisition,
        parameters,
    )


    signal_start = (
        time.perf_counter()
    )


    for _ in range(
        N_SIGNAL_CALLS
    ):

        preset.model.signal(
            acquisition,
            parameters,
        )


    signal_seconds = (
        time.perf_counter()
        - signal_start
    )


    signal_ms = (
        1000.0
        * signal_seconds
        / N_SIGNAL_CALLS
    )


    # =========================================================================
    # SAVE PROFILE
    # =========================================================================

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("<=", "le")
        .replace("=", "")
    )


    profile_path = (
        OUTPUT_DIR
        / f"{safe_name}.prof"
    )


    profiler.dump_stats(
        str(
            profile_path
        )
    )


    stream = (
        io.StringIO()
    )


    stats = pstats.Stats(
        profiler,
        stream=stream,
    )


    stats.strip_dirs()

    stats.sort_stats(
        "cumulative"
    )

    stats.print_stats(
        30
    )


    profile_text = (
        stream.getvalue()
    )


    text_path = (
        OUTPUT_DIR
        / f"{safe_name}_profile.txt"
    )


    text_path.write_text(
        profile_text
    )


    # =========================================================================
    # DERIVED ODI
    # =========================================================================

    kappa = float(
        result.parameters[
            "kappa"
        ]
    )


    odi = odi_from_kappa(
        kappa
    )


    # =========================================================================
    # FIT QUALITY
    # =========================================================================
    #
    # scipy least_squares cost is:
    #
    #       0.5 * sum(residual^2)
    #
    # Therefore:
    #
    #       RMSE = sqrt(2 * cost / N)
    #
    # when sigma=1.
    # =========================================================================

    rmse = np.sqrt(
        2.0
        * result.cost
        / acquisition.n_measurements
    )


    # =========================================================================
    # PRINT RESULTS
    # =========================================================================

    print(
        "\nResults:"
    )


    print(
        f"  Fit runtime:       "
        f"{fit_seconds:.3f} s"
    )


    print(
        f"  nfev:              "
        f"{result.n_evaluations}"
    )


    print(
        f"  Cost:              "
        f"{result.cost:.8f}"
    )


    print(
        f"  Cost / measurement:"
        f" "
        f"{result.cost / acquisition.n_measurements:.8f}"
    )


    print(
        f"  Signal RMSE:       "
        f"{rmse:.8f}"
    )


    print(
        f"  model.signal():    "
        f"{signal_ms:.2f} ms"
    )


    print(
        f"  kappa:             "
        f"{kappa:.8f}"
    )


    print(
        f"  ODI:               "
        f"{odi:.8f}"
    )


    print(
        "\nFitted parameters:"
    )


    for parameter_name, value in (
        result.parameters.items()
    ):

        print(
            f"  "
            f"{parameter_name:22s}"
            f"{float(value):.8f}"
        )


    print(
        f"\nProfile saved:\n"
        f"  {profile_path}"
    )


    print(
        f"\nProfile text saved:\n"
        f"  {text_path}"
    )


    return {

        "name":
            name,

        "n_measurements":
            acquisition.n_measurements,

        "fit_seconds":
            fit_seconds,

        "nfev":
            result.n_evaluations,

        "cost":
            float(
                result.cost
            ),

        "cost_per_measurement":
            float(
                result.cost
                / acquisition.n_measurements
            ),

        "rmse":
            float(
                rmse
            ),

        "signal_ms":
            float(
                signal_ms
            ),

        "kappa":
            kappa,

        "ODI":
            odi,

        "parameters":
            dict(
                result.parameters
            ),
    }


# =============================================================================
# 10. RUN FULL ACQUISITION
# =============================================================================

full_result = run_profile(

    name="FULL_b0_to_b5",

    acquisition=full_acquisition,

    signal=full_signal,
)


# =============================================================================
# 11. RUN REDUCED ACQUISITION
# =============================================================================

reduced_result = run_profile(

    name="REDUCED_b0_to_b2",

    acquisition=reduced_acquisition,

    signal=reduced_signal,
)


# =============================================================================
# 12. FINAL COMPARISON
# =============================================================================

print(
    "\n\n============================================"
)

print(
    "FULL vs REDUCED NODDI"
)

print(
    "============================================"
)


print(
    "\nMeasurements:"
)


print(
    f"  Full:       "
    f"{full_result['n_measurements']}"
)


print(
    f"  Reduced:    "
    f"{reduced_result['n_measurements']}"
)


# =============================================================================
# SPEED
# =============================================================================

speedup = (
    full_result[
        "fit_seconds"
    ]
    / reduced_result[
        "fit_seconds"
    ]
)


signal_speedup = (
    full_result[
        "signal_ms"
    ]
    / reduced_result[
        "signal_ms"
    ]
)


print(
    "\nRuntime:"
)


print(
    f"  Full:       "
    f"{full_result['fit_seconds']:.3f} s"
)


print(
    f"  Reduced:    "
    f"{reduced_result['fit_seconds']:.3f} s"
)


print(
    f"  Fit speedup:"
    f" {speedup:.2f}x"
)


print(
    "\nForward model:"
)


print(
    f"  Full:       "
    f"{full_result['signal_ms']:.2f} ms"
)


print(
    f"  Reduced:    "
    f"{reduced_result['signal_ms']:.2f} ms"
)


print(
    f"  Speedup:    "
    f"{signal_speedup:.2f}x"
)


# =============================================================================
# FIT QUALITY
# =============================================================================

print(
    "\nFit quality:"
)


print(
    f"  Full RMSE:    "
    f"{full_result['rmse']:.8f}"
)


print(
    f"  Reduced RMSE: "
    f"{reduced_result['rmse']:.8f}"
)


# =============================================================================
# PARAMETER COMPARISON
# =============================================================================

print(
    "\nParameter comparison:"
)


print(
    f"{'Parameter':22s}"
    f"{'Full':>14s}"
    f"{'Reduced':>14s}"
    f"{'Difference':>14s}"
)


print(
    "-" * 64
)


for parameter_name in (
    full_result[
        "parameters"
    ]
):

    full_value = float(
        full_result[
            "parameters"
        ][
            parameter_name
        ]
    )


    reduced_value = float(
        reduced_result[
            "parameters"
        ][
            parameter_name
        ]
    )


    difference = (
        reduced_value
        - full_value
    )


    print(
        f"{parameter_name:22s}"
        f"{full_value:14.6f}"
        f"{reduced_value:14.6f}"
        f"{difference:14.6f}"
    )


# =============================================================================
# ODI
# =============================================================================

odi_difference = (
    reduced_result[
        "ODI"
    ]
    - full_result[
        "ODI"
    ]
)


print(
    f"{'ODI':22s}"
    f"{full_result['ODI']:14.6f}"
    f"{reduced_result['ODI']:14.6f}"
    f"{odi_difference:14.6f}"
)


# =============================================================================
# SUMMARY
# =============================================================================

print(
    "\n============================================"
)

print(
    "SUMMARY"
)

print(
    "============================================"
)


print(
    f"\nUsing b <= {REDUCED_B_MAX:.1f} "
    f"reduced the acquisition from "
    f"{full_result['n_measurements']} to "
    f"{reduced_result['n_measurements']} measurements."
)


print(
    f"\nObserved fit speedup: "
    f"{speedup:.2f}x"
)


print(
    f"Observed forward-model speedup: "
    f"{signal_speedup:.2f}x"
)


print(
    "\nDo NOT decide from this single voxel whether "
    "the reduced acquisition is acceptable."
)


print(
    "If the parameter estimates remain reasonably similar, "
    "the next step is to compare full vs reduced NODDI "
    "across the existing 99-voxel test mask."
)