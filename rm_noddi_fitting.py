# =============================================================================
# NODDI FITTING — RAT IN-VIVO dMRI
# =============================================================================
#
# EXECUTION MODES
# ---------------
#
# LOCAL:
#
#     python rm_noddi_fitting.py
#
#     -> fits all chunks sequentially
#     -> derives ODI for each chunk
#     -> merges all chunks
#
#
# SLURM LOGIN NODE:
#
#     python rm_noddi_fitting.py
#
#     -> automatically submits N_CHUNKS array jobs
#     -> each array task fits one chunk
#     -> automatically submits a dependent merge job
#     -> merge begins only if all chunks succeed
#
#
# SLURM ARRAY WORKER:
#
#     Automatically detected from SLURM_ARRAY_TASK_ID.
#
#
# SLURM MERGE WORKER:
#
#     Automatically detected from NODDI_MERGE_ONLY=1.
#
#
# No command-line chunk arguments are required.
#
# =============================================================================


from __future__ import annotations


# =============================================================================
# IMPORTS
# =============================================================================

import os
import subprocess
import sys
import time

from pathlib import Path


import nibabel as nib
import numpy as np


from dmri_models.acquisition import Acquisition

from dmri_models.presets.noddi import noddi

from dmri_models.fitting.nifti import (
    fit_least_squares_nifti,
    merge_chunk_maps,
)


# =============================================================================
# 1. USER SETTINGS
# =============================================================================


# -----------------------------------------------------------------------------
# LOCAL PATHS
#
# Used when running outside SLURM.
# -----------------------------------------------------------------------------

LOCAL_PREPROC_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input"
)


# -----------------------------------------------------------------------------
# CLUSTER PATHS
#
# IMPORTANT:
# CHANGE THIS to the location of the SAME input directory on your cluster.
#
# It must contain:
#
#   data_gibbs_eddy_reoriented_global_scaled.nii.gz
#   bvals
#   bvecs
#   injection_site_mask.nii.gz
#
# -----------------------------------------------------------------------------

CLUSTER_PREPROC_DIR = Path(
    "/vols/Scratch/zcn545/80_post_fitting_input"
)


# -----------------------------------------------------------------------------
# Acquisition
# -----------------------------------------------------------------------------

PULSE_DURATION_MS = 5.0

DIFFUSION_TIME_MS = 12.0


# -----------------------------------------------------------------------------
# Fitting
# -----------------------------------------------------------------------------

N_STARTS = 1

MAX_NFEV = 2000

SIGMA = 1.0

RNG_SEED = 12345


# -----------------------------------------------------------------------------
# Chunking
#
# This is the ONLY chunk setting you need to change.
# -----------------------------------------------------------------------------

N_CHUNKS = 4


# =============================================================================
# 2. SLURM SETTINGS
# =============================================================================
#
# These settings are used only when the script submits itself to SLURM.
#
# Adjust partition/account/email/etc. if your cluster requires them.
# =============================================================================


SLURM_JOB_NAME = "rat_noddi"


# Maximum runtime PER CHUNK.
SLURM_TIME = "03:00:00"


# Memory PER CHUNK.
SLURM_MEMORY = "8G"


# One CPU per chunk.
#
# Your benchmarking showed that threaded fitting was slower.
SLURM_CPUS_PER_TASK = 1


# -----------------------------------------------------------------------------
# Optional SLURM settings
#
# Leave as None unless required by your cluster.
# -----------------------------------------------------------------------------

SLURM_PARTITION = None

SLURM_ACCOUNT = None


# =============================================================================
# 3. DETECT EXECUTION ENVIRONMENT
# =============================================================================


SLURM_JOB_ID = os.environ.get(
    "SLURM_JOB_ID"
)


SLURM_ARRAY_TASK_ID = os.environ.get(
    "SLURM_ARRAY_TASK_ID"
)


MERGE_ONLY = (
    os.environ.get(
        "NODDI_MERGE_ONLY",
        "0",
    )
    == "1"
)


RUNNING_INSIDE_SLURM = (
    SLURM_JOB_ID is not None
)


RUNNING_AS_ARRAY_TASK = (
    SLURM_ARRAY_TASK_ID is not None
)


# =============================================================================
# 4. CHOOSE LOCAL OR CLUSTER PATHS
# =============================================================================


if RUNNING_INSIDE_SLURM:

    PREPROC_DIR = (
        CLUSTER_PREPROC_DIR
    )

else:

    PREPROC_DIR = (
        LOCAL_PREPROC_DIR
    )


DATA_PATH = (
    PREPROC_DIR
    / "data_gibbs_eddy_reoriented_global_scaled.nii.gz"
)


BVAL_PATH = (
    PREPROC_DIR
    / "bvals"
)


BVEC_PATH = (
    PREPROC_DIR
    / "bvecs"
)


MASK_PATH = (
    PREPROC_DIR
    / "injection_site_mask.nii.gz"
)


OUTPUT_DIR = (
    PREPROC_DIR.parent
    / "model_fitting"
    / "noddi_tol1e6"
)


LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)


# =============================================================================
# 5. CURRENT SCRIPT PATH
#
# Used when the script submits itself to SLURM.
# =============================================================================


SCRIPT_PATH = Path(
    __file__
).resolve()


PYTHON_EXECUTABLE = Path(
    sys.executable
).resolve()


# =============================================================================
# 6. HELPER — ODI
# =============================================================================


def write_odi_from_kappa(
    kappa_path: Path,
) -> Path:
    """Create an ODI NIfTI from a fitted kappa NIfTI."""

    kappa_img = nib.load(
        str(
            kappa_path
        )
    )


    kappa = np.asarray(
        kappa_img.get_fdata(),
        dtype=float,
    )


    odi = np.full(
        kappa.shape,
        np.nan,
        dtype=np.float32,
    )


    valid = (
        np.isfinite(
            kappa
        )
        & (
            kappa >= 0.0
        )
    )


    odi[
        valid
    ] = (
        2.0
        / np.pi
        * np.arctan2(
            1.0,
            kappa[
                valid
            ],
        )
    ).astype(
        np.float32
    )


    # -------------------------------------------------------------------------
    # Preserve chunk suffix automatically
    #
    # noddi.kappa.chunk0001of0032.nii.gz
    #
    # ->
    #
    # noddi.ODI.chunk0001of0032.nii.gz
    # -------------------------------------------------------------------------

    odi_name = (
        kappa_path.name.replace(
            ".kappa",
            ".ODI",
            1,
        )
    )


    odi_path = (
        kappa_path.parent
        / odi_name
    )


    header = (
        kappa_img.header.copy()
    )


    header.set_data_dtype(
        np.float32
    )


    nib.save(

        nib.Nifti1Image(
            odi,
            kappa_img.affine,
            header,
        ),

        str(
            odi_path
        ),
    )


    return odi_path


# =============================================================================
# 7. HELPER — MERGE CHUNKS
# =============================================================================


def merge_all_chunks(
    *,
    output_dir: Path,
    n_chunks: int,
) -> None:
    """Merge all NODDI parameter, derived and diagnostic maps."""


    map_names = [

        # ---------------------------------------------------------------------
        # Model parameters
        # ---------------------------------------------------------------------

        "S0",
        "d_in",
        "theta",
        "phi",
        "kappa",
        "d_csf",

        "neurite_fraction",
        "extra_fraction",
        "csf_fraction",

        # ---------------------------------------------------------------------
        # Derived
        # ---------------------------------------------------------------------

        "ODI",

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        "cost",
        "success",
        "n_evaluations",
    ]


    print(
        "\n============================================"
    )

    print(
        "MERGING NODDI CHUNKS"
    )

    print(
        "============================================"
    )


    for name in map_names:


        chunk_paths = [

            output_dir
            / (
                f"noddi.{name}."
                f"chunk{i + 1:04d}"
                f"of{n_chunks:04d}.nii.gz"
            )

            for i in range(
                n_chunks
            )
        ]


        # ---------------------------------------------------------------------
        # Require every chunk
        # ---------------------------------------------------------------------

        missing = [

            path

            for path in chunk_paths

            if not path.exists()
        ]


        if missing:


            print(
                f"\nCannot merge {name}: "
                f"{len(missing)}/{n_chunks} chunks missing."
            )


            for path in missing:

                print(
                    f"  {path.name}"
                )


            raise FileNotFoundError(
                f"Missing chunk files for {name}."
            )


        # ---------------------------------------------------------------------
        # Merge
        # ---------------------------------------------------------------------

        output_path = (
            output_dir
            / f"noddi.{name}.nii.gz"
        )


        merge_chunk_maps(

            chunk_paths=chunk_paths,

            output_path=output_path,
        )


        print(
            f"Merged: "
            f"{output_path.name}"
        )


    print(
        "\nAll NODDI maps merged successfully."
    )


# =============================================================================
# 8. HELPER — SUBMIT TO SLURM
# =============================================================================


def submit_to_slurm() -> None:
    """Submit this script as a SLURM array plus dependent merge job."""


    print(
        "\n============================================"
    )

    print(
        "SLURM SUBMISSION"
    )

    print(
        "============================================"
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Check sbatch exists
    # -------------------------------------------------------------------------

    try:

        subprocess.run(
            [
                "sbatch",
                "--version",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):

        raise RuntimeError(
            "SLURM submission requested, but 'sbatch' "
            "could not be executed."
        )


    # =========================================================================
    # ARRAY JOB
    # =========================================================================


    array_command = [

        "sbatch",

        "--parsable",

        f"--job-name={SLURM_JOB_NAME}",

        f"--array=0-{N_CHUNKS - 1}",

        f"--cpus-per-task={SLURM_CPUS_PER_TASK}",

        f"--mem={SLURM_MEMORY}",

        f"--time={SLURM_TIME}",

        (
            f"--output="
            f"{LOG_DIR}/noddi_%A_%a.out"
        ),

        (
            f"--error="
            f"{LOG_DIR}/noddi_%A_%a.err"
        ),
    ]


    if SLURM_PARTITION is not None:

        array_command.append(
            f"--partition={SLURM_PARTITION}"
        )


    if SLURM_ACCOUNT is not None:

        array_command.append(
            f"--account={SLURM_ACCOUNT}"
        )


    # -------------------------------------------------------------------------
    # Environment variables
    #
    # Setting numerical libraries to one thread prevents the nested
    # threading behaviour that slowed your local fits.
    # -------------------------------------------------------------------------

    export_values = (

        "ALL,"
        "OMP_NUM_THREADS=1,"
        "OPENBLAS_NUM_THREADS=1,"
        "MKL_NUM_THREADS=1,"
        "NUMEXPR_NUM_THREADS=1"

    )


    array_command.append(
        f"--export={export_values}"
    )


    # -------------------------------------------------------------------------
    # Wrap Python invocation
    # -------------------------------------------------------------------------

    array_wrap = (
        f'"{PYTHON_EXECUTABLE}" '
        f'"{SCRIPT_PATH}"'
    )


    array_command.extend([
        "--wrap",
        array_wrap,
    ])


    print(
        "\nSubmitting fitting array:"
    )

    print(
        " ".join(
            array_command
        )
    )


    array_submission = subprocess.run(

        array_command,

        check=True,

        capture_output=True,

        text=True,
    )


    array_job_id = (
        array_submission.stdout
        .strip()
        .split(";")[0]
    )


    print(
        f"\nArray job ID: "
        f"{array_job_id}"
    )


    # =========================================================================
    # DEPENDENT MERGE JOB
    # =========================================================================


    merge_command = [

        "sbatch",

        "--parsable",

        f"--job-name={SLURM_JOB_NAME}_merge",

        "--cpus-per-task=1",

        "--mem=4G",

        "--time=00:30:00",

        (
            f"--dependency="
            f"afterok:{array_job_id}"
        ),

        (
            f"--output="
            f"{LOG_DIR}/noddi_merge_%j.out"
        ),

        (
            f"--error="
            f"{LOG_DIR}/noddi_merge_%j.err"
        ),
    ]


    if SLURM_PARTITION is not None:

        merge_command.append(
            f"--partition={SLURM_PARTITION}"
        )


    if SLURM_ACCOUNT is not None:

        merge_command.append(
            f"--account={SLURM_ACCOUNT}"
        )


    merge_export = (

        "ALL,"
        "NODDI_MERGE_ONLY=1,"
        "OMP_NUM_THREADS=1,"
        "OPENBLAS_NUM_THREADS=1,"
        "MKL_NUM_THREADS=1,"
        "NUMEXPR_NUM_THREADS=1"

    )


    merge_command.append(
        f"--export={merge_export}"
    )


    merge_wrap = (
        f'"{PYTHON_EXECUTABLE}" '
        f'"{SCRIPT_PATH}"'
    )


    merge_command.extend([
        "--wrap",
        merge_wrap,
    ])


    print(
        "\nSubmitting dependent merge job:"
    )

    print(
        " ".join(
            merge_command
        )
    )


    merge_submission = subprocess.run(

        merge_command,

        check=True,

        capture_output=True,

        text=True,
    )


    merge_job_id = (
        merge_submission.stdout
        .strip()
        .split(";")[0]
    )


    print(
        f"\nMerge job ID: "
        f"{merge_job_id}"
    )


    print(
        "\n============================================"
    )

    print(
        "PIPELINE SUBMITTED"
    )

    print(
        "============================================"
    )


    print(
        f"\nFitting array: "
        f"{array_job_id}"
    )

    print(
        f"Merge job:     "
        f"{merge_job_id}"
    )

    print(
        f"\nChunks:        "
        f"{N_CHUNKS}"
    )

    print(
        f"Logs:          "
        f"{LOG_DIR}"
    )

    print(
        "\nThe merge job will begin automatically "
        "only after every fitting chunk succeeds."
    )


# =============================================================================
# 9. DETERMINE WHETHER THIS IS A SLURM LOGIN NODE
# =============================================================================
#
# If:
#
#   - we are NOT already inside a SLURM allocation, and
#   - sbatch is available,
#
# then assume this is a SLURM login/submission node.
#
# The script submits itself and exits.
#
# =============================================================================


def sbatch_available() -> bool:

    try:

        result = subprocess.run(
            [
                "sbatch",
                "--version",
            ],
            capture_output=True,
            text=True,
        )

        return (
            result.returncode
            == 0
        )

    except FileNotFoundError:

        return False


# =============================================================================
# 10. AUTOMATIC SLURM SUBMISSION
# =============================================================================


if (
    not RUNNING_INSIDE_SLURM
    and sbatch_available()
):

    submit_to_slurm()

    raise SystemExit(
        0
    )


# =============================================================================
# 11. CHECK INPUT FILES
# =============================================================================


print(
    "\n============================================"
)

print(
    "CHECKING INPUT FILES"
)

print(
    "============================================"
)


required_paths = {

    "dMRI":
        DATA_PATH,

    "bvals":
        BVAL_PATH,

    "bvecs":
        BVEC_PATH,

    "mask":
        MASK_PATH,
}


for name, path in required_paths.items():


    print(
        f"{name:8s}: "
        f"{path}"
    )


    if not path.exists():


        raise FileNotFoundError(
            f"\nCould not find {name}:\n"
            f"{path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 12. MERGE-ONLY MODE
# =============================================================================
#
# A dependent SLURM merge job enters here.
#
# It does not need to load the 4D diffusion data or construct the model.
# =============================================================================


if MERGE_ONLY:


    merge_start = (
        time.perf_counter()
    )


    merge_all_chunks(

        output_dir=OUTPUT_DIR,

        n_chunks=N_CHUNKS,
    )


    merge_seconds = (
        time.perf_counter()
        - merge_start
    )


    print(
        f"\nMerge completed in "
        f"{merge_seconds:.1f} seconds."
    )


    print(
        "\nFinal merged maps:"
    )


    for path in sorted(
        OUTPUT_DIR.glob(
            "noddi.*.nii.gz"
        )
    ):


        if ".chunk" not in path.name:

            print(
                f"  {path.name}"
            )


    raise SystemExit(
        0
    )


# =============================================================================
# 13. INSPECT NIFTI
# =============================================================================


data_img = nib.load(
    str(
        DATA_PATH
    )
)


data_shape = (
    data_img.shape
)


if len(
    data_shape
) != 4:


    raise ValueError(
        "Diffusion data must be 4D.\n"
        f"Found shape: "
        f"{data_shape}"
    )


spatial_shape = (
    data_shape[:3]
)


n_volumes = (
    data_shape[-1]
)


print(
    "\n============================================"
)

print(
    "DIFFUSION DATA"
)

print(
    "============================================"
)


print(
    f"Shape:        "
    f"{data_shape}"
)

print(
    f"Spatial:      "
    f"{spatial_shape}"
)

print(
    f"Measurements: "
    f"{n_volumes}"
)

print(
    f"Voxel size:   "
    f"{data_img.header.get_zooms()[:3]}"
)


# =============================================================================
# 14. CHECK MASK
# =============================================================================


mask_img = nib.load(
    str(
        MASK_PATH
    )
)


mask = (
    np.asarray(
        mask_img.get_fdata(),
        dtype=float,
    )
    > 0
)


if (
    mask.shape
    != spatial_shape
):


    raise ValueError(
        "\nMask/data shape mismatch:\n"
        f"Data: {spatial_shape}\n"
        f"Mask: {mask.shape}"
    )


n_mask_voxels = int(
    np.sum(
        mask
    )
)


print(
    f"\nMask voxels: "
    f"{n_mask_voxels:,}"
)


if n_mask_voxels == 0:


    raise ValueError(
        "Brain mask contains no voxels."
    )


# =============================================================================
# 15. BUILD ACQUISITION
# =============================================================================


acquisition = Acquisition.from_fsl(

    BVAL_PATH,

    BVEC_PATH,

    bval_unit="s/mm^2",

    b0_threshold=0.05,

    rectangular_pgse=True,

    pulse_duration=(
        PULSE_DURATION_MS
    ),

    diffusion_time=(
        DIFFUSION_TIME_MS
    ),

    b_tensor_shape=1.0,
)


# =============================================================================
# 16. CHECK ACQUISITION
# =============================================================================


if (
    acquisition.n_measurements
    != n_volumes
):


    raise ValueError(
        "\nACQUISITION / NIFTI MISMATCH\n"
        f"NIfTI volumes: "
        f"{n_volumes}\n"
        f"Acquisition:   "
        f"{acquisition.n_measurements}"
    )


print(
    "\n============================================"
)

print(
    "ACQUISITION"
)

print(
    "============================================"
)


acquisition.print_summary()


# =============================================================================
# 17. SIGNAL-SCALE CHECK
# =============================================================================


b_values = np.asarray(
    acquisition.b_values
)


b0_mask = (
    b_values
    <= 0.05
)


data = np.asarray(

    data_img.get_fdata(
        dtype=np.float32
    ),

    dtype=float,
)


mean_b0 = np.mean(

    data[
        ...,
        b0_mask
    ],

    axis=-1,
)


brain_b0 = (

    mean_b0[
        mask
        & np.isfinite(
            mean_b0
        )
    ]
)


print(
    "\n============================================"
)

print(
    "SCALED b0 SIGNAL CHECK"
)

print(
    "============================================"
)


for percentile in [
    1,
    5,
    25,
    50,
    75,
    95,
    99,
]:


    print(
        f"{percentile:2d}%: "
        f"{np.percentile(brain_b0, percentile):.3f}"
    )


# =============================================================================
# 18. CREATE NODDI PRESET
# =============================================================================


preset = noddi(

    include_csf=True,

    vary_intracellular_diffusivity=False,

    vary_csf_diffusivity=False,

    vary_kappa=True,

    fod_model="watson",

    # Explicitly preserve full quadrature.
    fod_quadrature_theta=24,

    fod_quadrature_phi=48,
)


# =============================================================================
# 19. FIT ONE CHUNK
# =============================================================================


def fit_one_chunk(
    chunk_index: int,
):


    print(
        "\n============================================"
    )

    print(
        f"FITTING CHUNK "
        f"{chunk_index + 1}/{N_CHUNKS}"
    )

    print(
        "============================================"
    )


    chunk_start = (
        time.perf_counter()
    )


    result = fit_least_squares_nifti(

        acquisition=acquisition,

        model=preset.model,

        parameter_space=preset.parameters,

        data_path=DATA_PATH,

        output_dir=OUTPUT_DIR,

        output_prefix="noddi",

        mask_path=MASK_PATH,

        sigma=SIGMA,

        n_starts=N_STARTS,

        max_nfev=MAX_NFEV,

        # Keep every fitting process serial.
        n_jobs=1,

        rng_seed=(
            RNG_SEED
            + chunk_index
        ),

        initialise_orientation=True,

        chunk_index=chunk_index,

        n_chunks=N_CHUNKS,
    )


    # -------------------------------------------------------------------------
    # ODI
    # -------------------------------------------------------------------------

    odi_path = write_odi_from_kappa(

        Path(
            result.parameter_paths[
                "kappa"
            ]
        )
    )


    # -------------------------------------------------------------------------
    # Runtime
    # -------------------------------------------------------------------------

    chunk_seconds = (
        time.perf_counter()
        - chunk_start
    )


    seconds_per_voxel = (

        chunk_seconds
        / result.n_voxels_fit

        if result.n_voxels_fit > 0

        else np.nan
    )


    print(
        f"\nChunk "
        f"{chunk_index + 1}/{N_CHUNKS} complete."
    )


    print(
        f"  Voxels:        "
        f"{result.n_voxels_fit:,}"
    )

    print(
        f"  Runtime:       "
        f"{chunk_seconds / 60:.2f} min"
    )

    print(
        f"  Seconds/voxel: "
        f"{seconds_per_voxel:.2f}"
    )

    print(
        f"  ODI:           "
        f"{odi_path.name}"
    )


    return (
        result,
        chunk_seconds,
    )


# =============================================================================
# 20. EXECUTE FITTING
# =============================================================================


# -----------------------------------------------------------------------------
# SLURM ARRAY WORKER
# -----------------------------------------------------------------------------


if RUNNING_AS_ARRAY_TASK:


    chunk_index = int(
        SLURM_ARRAY_TASK_ID
    )


    if (
        chunk_index < 0
        or chunk_index >= N_CHUNKS
    ):


        raise ValueError(
            f"SLURM_ARRAY_TASK_ID="
            f"{chunk_index}, "
            f"but N_CHUNKS="
            f"{N_CHUNKS}."
        )


    print(
        "\nExecution mode: "
        "SLURM ARRAY WORKER"
    )


    print(
        f"SLURM job ID: "
        f"{SLURM_JOB_ID}"
    )


    print(
        f"Array task:   "
        f"{chunk_index}"
    )


    fit_one_chunk(
        chunk_index
    )


    print(
        "\nSLURM chunk finished successfully."
    )


# -----------------------------------------------------------------------------
# LOCAL
# -----------------------------------------------------------------------------


else:


    print(
        "\nExecution mode: LOCAL / SEQUENTIAL"
    )


    print(
        f"Fitting {N_CHUNKS} chunks sequentially."
    )


    local_start = (
        time.perf_counter()
    )


    chunk_runtimes = []


    for chunk_index in range(
        N_CHUNKS
    ):


        _, runtime = fit_one_chunk(
            chunk_index
        )


        chunk_runtimes.append(
            runtime
        )


    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    merge_start = (
        time.perf_counter()
    )


    merge_all_chunks(

        output_dir=OUTPUT_DIR,

        n_chunks=N_CHUNKS,
    )


    merge_seconds = (
        time.perf_counter()
        - merge_start
    )


    total_seconds = (
        time.perf_counter()
        - local_start
    )


    chunk_runtimes = np.asarray(
        chunk_runtimes,
        dtype=float,
    )


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(
        "\n============================================"
    )

    print(
        "NODDI FIT COMPLETE"
    )

    print(
        "============================================"
    )


    print(
        f"\nTotal runtime: "
        f"{total_seconds / 3600:.2f} h"
    )


    print(
        f"Merge runtime: "
        f"{merge_seconds:.1f} s"
    )


    print(
        "\nChunk runtimes:"
    )


    print(
        f"  Fastest: "
        f"{np.min(chunk_runtimes)/60:.2f} min"
    )


    print(
        f"  Median:  "
        f"{np.median(chunk_runtimes)/60:.2f} min"
    )


    print(
        f"  Slowest: "
        f"{np.max(chunk_runtimes)/60:.2f} min"
    )


print(
    "\nDone."
)