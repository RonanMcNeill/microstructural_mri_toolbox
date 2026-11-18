#!/usr/bin/env python3
# =============================================================================
# BATCH NODDI FITTING — RAT IN-VIVO dMRI
# =============================================================================
#
# EXECUTION
# ---------
#
# LOCAL:
#
#     python rm_noddi_fitting.py
#
#     -> fits N_CHUNKS sequentially
#     -> derives ODI for each chunk
#     -> merges all chunk maps automatically
#
#
# CLUSTER WITH FSL_SUB:
#
#     python rm_noddi_fitting.py
#
#     -> submits one independent fsl_sub job per chunk
#     -> chunk jobs may run concurrently
#     -> writes clean status files for monitoring
#     -> submits one dependent merge job
#     -> merge runs after all chunk jobs complete
#
#
# IMPORTANT
# ---------
#
# Native fsl_sub arrays are NOT used because --array_native is broken in
# the installed fsl_sub SLURM plugin.
#
# Each chunk is therefore submitted as an independent ordinary fsl_sub job.
#
# =============================================================================


from __future__ import annotations


# =============================================================================
# IMPORTS / THREAD LIMITS
# =============================================================================

import os


# -----------------------------------------------------------------------------
# Prevent hidden numerical-library threading.
#
# This MUST happen before importing NumPy / SciPy / dmri_models.
# -----------------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import subprocess
import sys
import time

from datetime import datetime
from pathlib import Path

# Locate the src directory containing dmri_models
PROJECT_SRC = Path(__file__).resolve().parents[2]

# Make the package importable in SLURM jobs
sys.path.insert(0, str(PROJECT_SRC))


import nibabel as nib
import numpy as np


# ====================== BATCH DATASET CONFIGURATION ======================
# A parent invocation dispatches 16 independent pipelines. Each SLURM worker
# receives its immutable dataset/mask identifier via NODDI_BATCH_CASE.
import argparse

BATCH_ROOT = Path('/vols/Scratch/zcn545/LPS_Dose_Pilot/MRI')
ANIMALS = ('M02841978', 'M02841979', 'M02841980', 'M02841981')
TIMEPOINTS = ('pre', 'post')
MASKS = {
    'injection': 'injection_site_mask.nii.gz',
    'contralateral': 'contralateral_site_mask.nii.gz',
}
BATCH_CASES = [
    (animal, timepoint, label)
    for animal in ANIMALS
    for timepoint in TIMEPOINTS
    for label in MASKS
]
BATCH_CASE_ENV = os.environ.get('NODDI_BATCH_CASE')
BATCH_N_STARTS = 1
BATCH_N_CHUNKS = 8
BATCH_QUEUE = 'short'
BATCH_MERGE_QUEUE = 'short'
BATCH_JOB_MINUTES = 180
BATCH_JOB_RAM_GB = 8
BATCH_MERGE_MINUTES = 30
BATCH_MERGE_RAM_GB = 4
NORMALISED_FILENAME = 'data_b0_normalized.nii.gz'


def case_paths(case_id):
    animal, timepoint, label = BATCH_CASES[case_id]
    folder = BATCH_ROOT / animal / timepoint / 'diffusion' / 'preprocessing' / 'fitting_input'
    return folder, MASKS[label], f'noddi_prefit_{label}', (animal, timepoint, label)


def normalise_dataset(folder):
    """Create a b0-normalised copy, leaving data.nii.gz untouched."""
    source = folder / 'data.nii.gz'
    target = folder / NORMALISED_FILENAME
    if target.exists():
        print(f'Using existing normalised data: {target}', flush=True)
        return
    img = nib.load(str(source))
    bvals = np.loadtxt(folder / 'bvals').reshape(-1)
    if len(img.shape) != 4 or bvals.size != img.shape[3]:
        raise ValueError(f'Incompatible data/bvals: {folder}')
    # bvals on disk are in s/mm^2; 50 s/mm^2 is the b0 cutoff.
    b0 = bvals <= 50.0
    if not np.any(b0):
        raise ValueError(f'No b0 measurements in {folder}')
    raw = np.asarray(img.dataobj, dtype=np.float32)
    baseline = np.mean(raw[..., b0], axis=-1, dtype=np.float32)
    valid = np.isfinite(baseline) & (baseline > 0)
    # Non-valid baseline voxels are excluded by the mask validation below.
    np.divide(raw, baseline[..., None], out=raw,
              where=valid[..., None])
    raw[~valid] = 0.0
    if not np.all(np.isfinite(raw)):
        raise ValueError(f'Nonfinite normalised signal in {folder}')
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    temp = folder / 'data_b0_normalized.tmp.nii.gz'
    nib.save(nib.Nifti1Image(raw, img.affine, hdr), str(temp))
    temp.replace(target)
    print(f'Created {target}', flush=True)


def validate_case(case_id, prepare=False):
    folder, mask_name, output_name, description = case_paths(case_id)
    for filename in ('data.nii.gz', 'bvals', 'bvecs', mask_name):
        if not (folder / filename).exists():
            raise FileNotFoundError(folder / filename)
    img = nib.load(str(folder / 'data.nii.gz'))
    mask_img = nib.load(str(folder / mask_name))
    if img.shape[:3] != mask_img.shape or not np.allclose(img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f'Mask geometry mismatch: {folder / mask_name}')
    mask = np.asarray(mask_img.dataobj) > 0
    voxels = int(mask.sum())
    if voxels == 0:
        raise ValueError(f'Empty mask: {folder / mask_name}')
    bvals = np.loadtxt(folder / 'bvals').reshape(-1)
    if bvals.size != img.shape[-1] or not np.any(bvals <= 50):
        raise ValueError(f'Invalid bvals or missing b0: {folder}')
    if prepare:
        normalise_dataset(folder)
        # Validate b0 baseline for every fitted voxel.
        raw = nib.load(str(folder / 'data.nii.gz'))
        b0_indices = np.flatnonzero(bvals <= 50)
        baseline = np.mean(np.asarray(raw.dataobj, dtype=np.float32)[..., b0_indices], axis=-1)
        if not np.all(np.isfinite(baseline[mask]) & (baseline[mask] > 0)):
            raise ValueError(f'Non-positive or non-finite b0 within mask: {folder / mask_name}')
    return voxels


def dispatch_batch():
    parser = argparse.ArgumentParser(description='NODDI prefit batch launcher')
    parser.add_argument('--submit', action='store_true', help='Actually submit SLURM jobs')
    parser.add_argument('--case', type=int, choices=range(len(BATCH_CASES)),
                        help='Limit to one case (0-15), for testing')
    args = parser.parse_args()
    indices = [args.case] if args.case is not None else list(range(len(BATCH_CASES)))
    print(f'Batch mode: {len(indices)} cases; submit={args.submit}', flush=True)
    for case_id in indices:
        folder, mask_name, output_name, desc = case_paths(case_id)
        voxels = validate_case(case_id, prepare=False)
        print(f'[{case_id:02d}] {desc}: {voxels} voxels; output={folder / output_name}', flush=True)
    if not args.submit:
        print('DRY RUN ONLY. Pass --submit to normalise and submit jobs.')
        return
    if not ON_CLUSTER:
        raise RuntimeError('Batch submission requires fsl_sub on the cluster')
    for case_id in indices:
        folder, mask_name, output_name, desc = case_paths(case_id)
        out = folder / output_name
        if (out / 'qc' / 'qc_summary.txt').exists() or (out / 'status').exists():
            raise RuntimeError(f'Existing outputs/status detected; refusing to overwrite: {out}')
        validate_case(case_id, prepare=True)
        env = dict(os.environ)
        env['NODDI_BATCH_CASE'] = str(case_id)
        print(f'Submitting case {case_id}: {desc}', flush=True)
        subprocess.run([str(Path(sys.executable).resolve()), str(Path(__file__).resolve())],
                       env=env, check=True)


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
# Local input directory
# -----------------------------------------------------------------------------

LOCAL_PREPROC_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input"
)


# -----------------------------------------------------------------------------
# Cluster input directory
# -----------------------------------------------------------------------------

CLUSTER_PREPROC_DIR = Path(
    "/vols/Scratch/zcn545/80_post_fitting_input"
)


# -----------------------------------------------------------------------------
# Input filenames
# -----------------------------------------------------------------------------

DATA_FILENAME = (
    "data_gibbs_eddy_reoriented_global_scaled.nii.gz"
)

BVAL_FILENAME = (
    "bvals"
)

BVEC_FILENAME = (
    "bvecs"
)

MASK_FILENAME = (
    "injection_site_mask.nii.gz"
)


# -----------------------------------------------------------------------------
# Output directory name
# -----------------------------------------------------------------------------

OUTPUT_FOLDER_NAME = (
    "noddi_1_start_roi"
)


# =============================================================================
# 2. ACQUISITION SETTINGS
# =============================================================================

PULSE_DURATION_MS = 5.0

DIFFUSION_TIME_MS = 12.0


# =============================================================================
# 3. FITTING SETTINGS
# =============================================================================

N_STARTS = 1

MAX_NFEV = 2000

SIGMA = 1.0

RNG_SEED = 12345


# =============================================================================
# 4. CHUNKING SETTINGS
# =============================================================================
#
# Current 99-voxel cluster integration test:
#
#     N_CHUNKS = 4
#
# Once everything is confirmed:
#
#     MASK_FILENAME = "nodif_brain_mask.nii.gz"
#     N_CHUNKS = 32
#
# =============================================================================

N_CHUNKS = 32


# =============================================================================
# 5. FSL_SUB SETTINGS
# =============================================================================

CLUSTER_JOB_NAME = (
    "rat_noddi"
)


# -----------------------------------------------------------------------------
# Resources PER fitting chunk
# -----------------------------------------------------------------------------

CLUSTER_TIME_MINUTES = 2000

CLUSTER_RAM_GB = 8


# -----------------------------------------------------------------------------
# Merge job resources
# -----------------------------------------------------------------------------

MERGE_TIME_MINUTES = 300

MERGE_RAM_GB = 4


# =============================================================================
# 6. EXECUTION ENVIRONMENT
# =============================================================================
#
# Parent:
#
#     no special environment variable
#
# Chunk worker:
#
#     NODDI_CHUNK_INDEX=<zero-based index>
#
# Merge worker:
#
#     NODDI_MERGE_ONLY=1
#
# =============================================================================


MERGE_ONLY = (
    os.environ.get(
        "NODDI_MERGE_ONLY",
        "0",
    )
    == "1"
)


CHUNK_INDEX_ENV = (
    os.environ.get(
        "NODDI_CHUNK_INDEX"
    )
)


RUNNING_AS_CHUNK_WORKER = (
    CHUNK_INDEX_ENV is not None
)


if RUNNING_AS_CHUNK_WORKER:

    CHUNK_INDEX = int(
        CHUNK_INDEX_ENV
    )

    if (
        CHUNK_INDEX < 0
        or CHUNK_INDEX >= N_CHUNKS
    ):

        raise ValueError(
            f"NODDI_CHUNK_INDEX={CHUNK_INDEX}, "
            f"but N_CHUNKS={N_CHUNKS}."
        )

else:

    CHUNK_INDEX = None


# =============================================================================
# 7. FSL_SUB AVAILABILITY
# =============================================================================


def fsl_sub_available() -> bool:
    """Return True if fsl_sub is available."""

    try:

        result = subprocess.run(
            [
                "fsl_sub",
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


ON_CLUSTER = (
    fsl_sub_available()
)


# =============================================================================
# 8. CHOOSE LOCAL OR CLUSTER PATHS
# =============================================================================


if BATCH_CASE_ENV is not None:
    BATCH_CASE_ID = int(BATCH_CASE_ENV)
    if not 0 <= BATCH_CASE_ID < len(BATCH_CASES):
        raise ValueError(f'Invalid NODDI_BATCH_CASE={BATCH_CASE_ENV}')
    PREPROC_DIR, MASK_FILENAME, OUTPUT_FOLDER_NAME, CASE_DESCRIPTION = case_paths(BATCH_CASE_ID)
    DATA_FILENAME = NORMALISED_FILENAME
    N_STARTS = BATCH_N_STARTS
    N_CHUNKS = BATCH_N_CHUNKS
    CLUSTER_TIME_MINUTES = BATCH_JOB_MINUTES
    CLUSTER_RAM_GB = BATCH_JOB_RAM_GB
    MERGE_TIME_MINUTES = BATCH_MERGE_MINUTES
    MERGE_RAM_GB = BATCH_MERGE_RAM_GB
    CLUSTER_JOB_NAME = f'noddi_{BATCH_CASE_ID:02d}'
else:
    PREPROC_DIR = CLUSTER_PREPROC_DIR if ON_CLUSTER else LOCAL_PREPROC_DIR


DATA_PATH = (
    PREPROC_DIR
    / DATA_FILENAME
)


BVAL_PATH = (
    PREPROC_DIR
    / BVAL_FILENAME
)


BVEC_PATH = (
    PREPROC_DIR
    / BVEC_FILENAME
)


MASK_PATH = (
    PREPROC_DIR
    / MASK_FILENAME
)


OUTPUT_DIR = (
    PREPROC_DIR
    / OUTPUT_FOLDER_NAME
)


LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)


STATUS_DIR = (
    OUTPUT_DIR
    / "status"
)


# =============================================================================
# 9. SCRIPT / PYTHON PATHS
# =============================================================================


SCRIPT_PATH = Path(
    __file__
).resolve()


PYTHON_EXECUTABLE = Path(
    sys.executable
).resolve()


# =============================================================================
# 10. TIME HELPER
# =============================================================================


def timestamp() -> str:
    """Return a readable current timestamp."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =============================================================================
# 11. CHUNK SIZE HELPERS
# =============================================================================


def get_mask_indices() -> np.ndarray:
    """Return flat indices of all fitted voxels."""

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

    return np.flatnonzero(
        mask.reshape(-1)
    )


def get_chunk_sizes() -> list[int]:
    """Return voxel count assigned to each chunk."""

    flat_indices = (
        get_mask_indices()
    )

    chunks = np.array_split(
        flat_indices,
        N_CHUNKS,
    )

    return [
        int(
            len(chunk)
        )
        for chunk in chunks
    ]


# =============================================================================
# 12. STATUS FILE HELPERS
# =============================================================================


def status_path_for_chunk(
    chunk_index: int,
) -> Path:

    return (
        STATUS_DIR
        / (
            f"chunk_"
            f"{chunk_index + 1:04d}.status"
        )
    )


def write_chunk_status(
    *,
    chunk_index: int,
    status: str,
    n_voxels: int,
    job_id: str | None = None,
    completed: int | None = None,
    elapsed_seconds: float | None = None,
    rate: float | None = None,
    eta_seconds: float | None = None,
    runtime_seconds: float | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    message: str | None = None,
) -> None:
    """Write a clean human-readable status file for one chunk."""

    STATUS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    lines = [
        (
            f"chunk="
            f"{chunk_index + 1}/{N_CHUNKS}"
        ),
        (
            f"chunk_index="
            f"{chunk_index}"
        ),
        (
            f"status="
            f"{status}"
        ),
        (
            f"voxels="
            f"{n_voxels}"
        ),
        (
            f"updated="
            f"{timestamp()}"
        ),
    ]


    if job_id is not None:

        lines.append(
            f"job_id={job_id}"
        )


    if runtime_seconds is not None:

        lines.extend([
            (
                f"runtime_seconds="
                f"{runtime_seconds:.1f}"
            ),
            (
                f"runtime_minutes="
                f"{runtime_seconds / 60.0:.2f}"
            ),
        ])


        if (
            runtime_seconds > 0
            and n_voxels > 0
        ):

            lines.extend([
                (
                    f"seconds_per_voxel="
                    f"{runtime_seconds / n_voxels:.2f}"
                ),
                (
                    f"voxels_per_second="
                    f"{n_voxels / runtime_seconds:.3f}"
                ),
            ])

    if started_at is not None:
        lines.append(f"started_at={started_at}")

    if finished_at is not None:
        lines.append(f"finished_at={finished_at}")
        
    if message is not None:

        lines.append(
            f"message={message}"
        )


    status_path_for_chunk(
        chunk_index
    ).write_text(
        "\n".join(
            lines
        )
        + "\n"
    )


# =============================================================================
# 13. PIPELINE STATUS SUMMARY
# =============================================================================


def print_status_summary() -> None:
    """Print current status of all chunks."""

    print(
        "\n============================================"
    )

    print(
        "CHUNK STATUS"
    )

    print(
        "============================================"
    )


    for chunk_index in range(
        N_CHUNKS
    ):

        path = status_path_for_chunk(
            chunk_index
        )


        if not path.exists():

            print(
                f"Chunk "
                f"{chunk_index + 1:3d}/{N_CHUNKS}: "
                f"UNKNOWN"
            )

            continue


        contents = (
            path.read_text()
        )


        values = {}

        for line in contents.splitlines():

            if "=" not in line:
                continue

            key, value = line.split(
                "=",
                1,
            )

            values[key] = value


        status = values.get(
            "status",
            "UNKNOWN",
        )

        voxels = values.get(
            "voxels",
            "?",
        )

        job_id = values.get(
            "job_id",
            "-",
        )

        runtime = values.get(
            "runtime_minutes"
        )


        text = (
            f"Chunk "
            f"{chunk_index + 1:3d}/{N_CHUNKS}: "
            f"{status:10s} | "
            f"{voxels:>5s} voxels | "
            f"job {job_id}"
        )


        if runtime is not None:

            text += (
                f" | "
                f"{runtime} min"
            )


        print(
            text
        )


# =============================================================================
# 14. ODI HELPER
# =============================================================================


def write_odi_from_kappa(
    kappa_path: Path,
) -> Path:
    """Create ODI NIfTI from a fitted kappa NIfTI."""

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
# 15. MERGE CHUNKS
# =============================================================================


def merge_all_chunks(
    *,
    output_dir: Path,
    n_chunks: int,
) -> None:
    """Merge all NODDI parameter, ODI and diagnostic maps."""

    map_names = [
        "S0",
        "d_in",
        "theta",
        "phi",
        "kappa",
        "d_csf",
        "neurite_fraction",
        "extra_fraction",
        "csf_fraction",
        "ODI",
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


        missing = [
            path
            for path in chunk_paths
            if not path.exists()
        ]


        if missing:

            print(
                f"\nCannot merge {name}: "
                f"{len(missing)}/{n_chunks} "
                f"chunks missing."
            )


            for path in missing:

                print(
                    f"  {path.name}"
                )


            raise FileNotFoundError(
                f"Missing chunk files for "
                f"{name}."
            )


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
# AUTOMATIC POST-FITTING QUALITY CONTROL
# =============================================================================

def generate_qc_report() -> None:
    """Generate ROI statistics and histograms from merged NODDI maps."""

    import csv
    import matplotlib
    matplotlib.use("Agg")  # Required for headless SLURM jobs
    import matplotlib.pyplot as plt

    qc_dir = OUTPUT_DIR / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    mask_img = nib.load(str(MASK_PATH))
    mask = np.asarray(mask_img.dataobj) > 0
    n_voxels = int(mask.sum())

    parameters = [
        "S0",
        "d_in",
        "theta",
        "phi",
        "kappa",
        "d_csf",
        "neurite_fraction",
        "extra_fraction",
        "csf_fraction",
        "ODI",
        "cost",
        "success",
        "n_evaluations",
    ]

    rows = []
    maps = {}

    for name in parameters:

        path = OUTPUT_DIR / f"noddi.{name}.nii.gz"

        if not path.exists():
            raise FileNotFoundError(f"Missing merged map: {path}")

        img = nib.load(str(path))
        data = np.asarray(img.dataobj)

        if data.shape != mask.shape:
            raise ValueError(f"Shape mismatch for {name}")

        maps[name] = data

        roi_values = data[mask]
        finite = roi_values[np.isfinite(roi_values)]

        row = {
            "parameter": name,
            "n_mask_voxels": n_voxels,
            "n_finite": len(finite),
            "n_nan": int(np.isnan(roi_values).sum()),
            "n_inf": int(np.isinf(roi_values).sum()),
        }

        if len(finite):
            row.update({
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "median": float(np.median(finite)),
                "min": float(np.min(finite)),
                "p05": float(np.percentile(finite, 5)),
                "p25": float(np.percentile(finite, 25)),
                "p75": float(np.percentile(finite, 75)),
                "p95": float(np.percentile(finite, 95)),
                "max": float(np.max(finite)),
                "n_zero": int(np.sum(finite == 0)),
            })

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.hist(
                finite,
                bins=30,
                edgecolor="black",
                alpha=0.8,
            )

            ax.axvline(
                np.mean(finite),
                linestyle="--",
                label=f"Mean = {np.mean(finite):.4g}",
            )

            ax.axvline(
                np.median(finite),
                linestyle=":",
                label=f"Median = {np.median(finite):.4g}",
            )

            ax.set_title(f"NODDI: {name} (n={len(finite)})")
            ax.set_xlabel(name)
            ax.set_ylabel("Voxel count")
            ax.legend()
            fig.tight_layout()

            fig.savefig(
                qc_dir / f"hist_{name}.png",
                dpi=200,
            )
            plt.close(fig)

        rows.append(row)

    # Save statistics for all parameters
    columns = [
        "parameter", "n_mask_voxels", "n_finite",
        "n_nan", "n_inf", "mean", "std",
        "median", "min", "p05", "p25",
        "p75", "p95", "max", "n_zero",
    ]

    with (qc_dir / "parameter_statistics.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Fitting success
    success = maps["success"][mask]
    n_success = int(np.sum(success == 1))
    success_pct = 100 * n_success / n_voxels

    # Fraction simplex check
    fractions = (
        maps["neurite_fraction"][mask]
        + maps["extra_fraction"][mask]
        + maps["csf_fraction"][mask]
    )

    finite_fractions = fractions[np.isfinite(fractions)]
    fraction_error = np.abs(finite_fractions - 1.0)
    
    # -------------------------------------------------------------------------
    # FITTING RUNTIME STATISTICS
    # -------------------------------------------------------------------------

    chunk_runtimes = []
    chunk_start_times = []
    chunk_end_times = []
    completed_voxels = 0

    for chunk_index in range(N_CHUNKS):

        status_file = status_path_for_chunk(chunk_index)

        if not status_file.exists():
            continue

        values = {}

        for line in status_file.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value

        if values.get("status") != "COMPLETE":
            continue

        if "runtime_seconds" not in values:
            continue

        chunk_runtimes.append(
            float(values["runtime_seconds"])
        )

        completed_voxels += int(values.get("voxels", 0))

        if "started_at" in values:
            chunk_start_times.append(
                datetime.fromisoformat(values["started_at"])
            )

        if "finished_at" in values:
            chunk_end_times.append(
                datetime.fromisoformat(values["finished_at"])
            )

    # Sum of individual chunk runtimes, not elapsed pipeline time
    total_chunk_seconds = sum(chunk_runtimes)

    mean_chunk_seconds = (
        float(np.mean(chunk_runtimes))
        if chunk_runtimes
        else float("nan")
    )

    mean_seconds_per_voxel = (
        total_chunk_seconds / completed_voxels
        if completed_voxels > 0
        else float("nan")
    )

    # Wall-clock window from first chunk start to last chunk finish
    if (
        len(chunk_runtimes) == N_CHUNKS
        and len(chunk_start_times) == N_CHUNKS
        and len(chunk_end_times) == N_CHUNKS
    ):
        elapsed_fitting_seconds = (
            max(chunk_end_times) - min(chunk_start_times)
        ).total_seconds()
    else:
        elapsed_fitting_seconds = None

    effective_throughput = (
        completed_voxels / (elapsed_fitting_seconds / 60)
        if elapsed_fitting_seconds is not None
        and elapsed_fitting_seconds > 0
        else None
    )
    
    with (qc_dir / "qc_summary.txt").open("w") as f:
        f.write("NODDI FITTING QUALITY CONTROL\n")
        f.write("=============================\n\n")
        f.write(f"Mask: {MASK_PATH}\n")
        f.write(f"Voxels: {n_voxels}\n")
        f.write(f"N starts: {N_STARTS}\n")
        f.write(f"N chunks: {N_CHUNKS}\n\n")

        f.write(f"Successful fits: {n_success}/{n_voxels}\n")
        f.write(f"Success rate: {success_pct:.2f}%\n\n")
        
        f.write("\nFITTING RUNTIME\n")
        f.write("=" * 44 + "\n\n")

        f.write("PARALLEL EXECUTION\n")
        f.write("-" * 44 + "\n")

        f.write(f"Number of chunks: {N_CHUNKS}\n")
        f.write(
            f"Completed chunks: {len(chunk_runtimes)}/{N_CHUNKS}\n"
        )

        if elapsed_fitting_seconds is not None:
            f.write(
                f"Elapsed fitting time: "
                f"{elapsed_fitting_seconds / 60:.2f} minutes\n"
            )
        else:
            f.write("Elapsed fitting time: unavailable\n")

        f.write(
            f"Summed chunk runtime: "
            f"{total_chunk_seconds / 60:.2f} minutes\n"
        )

        if chunk_runtimes:
            f.write(
                f"Mean chunk runtime: "
                f"{mean_chunk_seconds / 60:.2f} minutes\n"
            )

            f.write(
                f"Longest chunk runtime: "
                f"{max(chunk_runtimes) / 60:.2f} minutes\n"
            )

        f.write("\nCOMPUTATIONAL PERFORMANCE\n")
        f.write("-" * 44 + "\n")

        f.write(f"Total fitted voxels: {completed_voxels}\n")

        f.write(
            f"Mean chunk runtime per voxel: "
            f"{mean_seconds_per_voxel:.2f} seconds\n"
        )

        if effective_throughput is not None:
            f.write(
                f"Effective throughput: "
                f"{effective_throughput:.2f} voxels/minute\n"
            )

        f.write("\n")
    
        f.write("FRACTION SUM CHECK\n")

        if len(fraction_error):

            mean_error = np.mean(fraction_error)
            max_error = np.max(fraction_error)

            mean_text = (
                "<0.0001"
                if mean_error < 0.0001
                else f"{mean_error:.6g}"
            )

            max_text = (
                "<0.0001"
                if max_error < 0.0001
                else f"{max_error:.6g}"
            )

            f.write(
                f"Mean absolute error: {mean_text}\n"
            )

            f.write(
                f"Maximum absolute error: {max_text}\n"
            )
        else:
            f.write("No finite fraction sums available.\n")

    print(f"\nQC report saved to: {qc_dir}")
    print(f"Successful fits: {n_success}/{n_voxels}")

# =============================================================================
# 16. FSL_SUB HELPERS
# =============================================================================


def get_job_id(
    stdout: str,
    *,
    description: str,
) -> str:
    """Extract final non-empty line returned by fsl_sub."""

    lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip()
    ]


    if not lines:

        raise RuntimeError(
            f"fsl_sub returned no job ID "
            f"for {description}."
        )


    return lines[-1]


def run_fsl_sub(
    command: list[str],
    *,
    description: str,
) -> str:
    """Submit one fsl_sub command and return job ID."""

    try:

        submission = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as exc:

        print(
            f"\nFSL submission failed: "
            f"{description}"
        )

        print(
            "\nCommand:"
        )

        print(
            " ".join(
                command
            )
        )

        print(
            "\nSTDOUT:"
        )

        print(
            exc.stdout
        )

        print(
            "\nSTDERR:"
        )

        print(
            exc.stderr
        )

        raise


    return get_job_id(
        submission.stdout,
        description=description,
    )


# =============================================================================
# 17. SUBMIT PIPELINE WITH FSL_SUB
# =============================================================================


def submit_with_fsl_sub() -> None:
    """
    Submit one independent fsl_sub job per chunk.

    A dependent merge job is then submitted.
    """

    print(
        "\n============================================"
    )

    print(
        "FSL_SUB CLUSTER SUBMISSION"
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


    STATUS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    python_executable = str(
        PYTHON_EXECUTABLE
    )


    script_path = str(
        SCRIPT_PATH
    )


    # -------------------------------------------------------------------------
    # Determine voxel allocation before submission.
    # -------------------------------------------------------------------------

    chunk_sizes = (
        get_chunk_sizes()
    )


    n_total_voxels = int(
        np.sum(
            chunk_sizes
        )
    )


    print(
        f"\nPython:\n"
        f"  {python_executable}"
    )


    print(
        f"\nScript:\n"
        f"  {script_path}"
    )


    print(
        f"\nTotal voxels: "
        f"{n_total_voxels:,}"
    )


    print(
        f"Number of chunks: "
        f"{N_CHUNKS}"
    )


    print(
        "\nChunk allocation:"
    )


    for index, size in enumerate(
        chunk_sizes
    ):

        print(
            f"  Chunk "
            f"{index + 1:3d}/{N_CHUNKS}: "
            f"{size:,} voxels"
        )


    # =========================================================================
    # SUBMIT CHUNKS
    # =========================================================================

    chunk_job_ids = []


    for chunk_index in range(
        N_CHUNKS
    ):

        chunk_number = (
            chunk_index + 1
        )


        # ---------------------------------------------------------------------
        # Initial status
        # -------------------------------------------------------------------------

        write_chunk_status(
            chunk_index=chunk_index,
            status="SUBMITTING",
            n_voxels=(
                chunk_sizes[
                    chunk_index
                ]
            ),
        )

        chunk_command = [

            "fsl_sub",
        
            "--name",
            (
                f"{CLUSTER_JOB_NAME}_"
                f"{chunk_number:03d}"
            ),
        
            "--queue", BATCH_QUEUE,
            "--logdir",
            str(
                LOG_DIR
            ),
        
            "--jobtime",
            str(
                CLUSTER_TIME_MINUTES
            ),
        
            "--jobram",
            str(
                CLUSTER_RAM_GB
            ),
        
            "--export",
            (
                f"NODDI_CHUNK_INDEX="
                f"{chunk_index},NODDI_BATCH_CASE={BATCH_CASE_ID}"
            ),
        
            python_executable,
        
            script_path,
        ]


        print(
            f"\nSubmitting chunk "
            f"{chunk_number}/{N_CHUNKS} "
            f"({chunk_sizes[chunk_index]:,} voxels)..."
        )


        job_id = run_fsl_sub(
            chunk_command,
            description=(
                f"chunk "
                f"{chunk_number}/{N_CHUNKS}"
            ),
        )


        chunk_job_ids.append(
            job_id
        )


        write_chunk_status(
            chunk_index=chunk_index,
            status="QUEUED",
            n_voxels=(
                chunk_sizes[
                    chunk_index
                ]
            ),
            job_id=job_id,
        )


        print(
            f"  Job ID: "
            f"{job_id}"
        )


    # =========================================================================
    # SUBMIT MERGE JOB
    # =========================================================================

    jobhold_value = (
        ",".join(
            chunk_job_ids
        )
    )


    merge_command = [
    
        "fsl_sub",
    
        "--name",
        f"{CLUSTER_JOB_NAME}_merge",
        "--queue", BATCH_MERGE_QUEUE,
    
        "--logdir",
        str(
            LOG_DIR
        ),
    
        "--jobtime",
        str(
            MERGE_TIME_MINUTES
        ),
    
        "--jobram",
        str(
            MERGE_RAM_GB
        ),
    
        "--jobhold",
        jobhold_value,
    
        "--export",
        f"NODDI_MERGE_ONLY=1,NODDI_BATCH_CASE={BATCH_CASE_ID}",
    
        python_executable,
    
        script_path,
    ]

    print(
        "\nSubmitting dependent merge job..."
    )


    merge_job_id = run_fsl_sub(
        merge_command,
        description="merge job",
    )


    # =========================================================================
    # REPORT
    # =========================================================================

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
        "\nChunk jobs:"
    )


    for index, job_id in enumerate(
        chunk_job_ids
    ):

        print(
            f"  "
            f"{index + 1:3d}/{N_CHUNKS}: "
            f"{job_id} "
            f"({chunk_sizes[index]:,} voxels)"
        )


    print(
        f"\nMerge job: "
        f"{merge_job_id}"
    )


    print(
        f"\nStatus directory:\n"
        f"  {STATUS_DIR}"
    )


    print(
        f"\nLog directory:\n"
        f"  {LOG_DIR}"
    )


    print(
        "\nTo monitor cleanly:"
    )


    print(
        f"\n  watch -n 5 "
        f"'cat {STATUS_DIR}/*.status'"
    )


# Batch parent dispatch (workers and case-specific submitters skip this).
if BATCH_CASE_ENV is None:
    dispatch_batch()
    raise SystemExit(0)


# =============================================================================
# 18. AUTOMATIC CLUSTER SUBMISSION
# =============================================================================


if (
    ON_CLUSTER
    and not RUNNING_AS_CHUNK_WORKER
    and not MERGE_ONLY
):

    submit_with_fsl_sub()

    raise SystemExit(
        0
    )


# =============================================================================
# 19. CHECK INPUT FILES
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


STATUS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 20. MERGE-ONLY MODE
# =============================================================================


if MERGE_ONLY:

    print(
        "\n============================================"
    )

    print(
        "MERGE-ONLY MODE"
    )

    print(
        "============================================"
    )


    print_status_summary()


    merge_start = (
        time.perf_counter()
    )


    merge_all_chunks(
        output_dir=OUTPUT_DIR,
        n_chunks=N_CHUNKS,
    )

    generate_qc_report()

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
# 21. INSPECT NIFTI
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
# 22. CHECK MASK
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
# 23. BUILD ACQUISITION
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
# 24. CHECK ACQUISITION
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
# 25. SIGNAL-SCALE CHECK
# =============================================================================


b_values = np.asarray(
    acquisition.b_values
)


b0_mask = (
    b_values
    <= 0.05
)


print(
    f"\nNumber of b0 measurements: "
    f"{np.sum(b0_mask)}"
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
# 26. CREATE NODDI PRESET
# =============================================================================


preset = noddi(

    include_csf=True,

    vary_intracellular_diffusivity=False,

    vary_csf_diffusivity=False,

    vary_kappa=True,

    fod_model="watson",

    # Preserve full quadrature.
    fod_quadrature_theta=24,

    fod_quadrature_phi=48,
)


# =============================================================================
# 27. MODEL SUMMARY
# =============================================================================


print(
    "\n============================================"
)

print(
    "NODDI MODEL"
)

print(
    "============================================"
)


print(
    "\nParameters:"
)


for parameter in (
    preset.parameters.parameters
):

    status = (
        "FIT"
        if parameter.vary
        else "FIXED"
    )

    print(
        f"  "
        f"{parameter.name:22s}"
        f"{status}"
    )


print(
    "\nSimplex parameters:"
)


for simplex in (
    preset.parameters.simplexes
):

    print(
        "  "
        + ", ".join(
            simplex.names
        )
    )


print(
    "\nOrientation strategy:"
)

print(
    "  DTI principal direction -> "
    "initial theta/phi"
)

print(
    "  theta/phi remain free during "
    "NODDI fitting"
)


# =============================================================================
# 28. FIT CONFIGURATION
# =============================================================================


print(
    "\n============================================"
)

print(
    "FIT CONFIGURATION"
)

print(
    "============================================"
)


print(
    f"Input:          "
    f"{DATA_PATH}"
)

print(
    f"Mask:           "
    f"{MASK_PATH}"
)

print(
    f"Output:         "
    f"{OUTPUT_DIR}"
)

print(
    f"Voxels:         "
    f"{n_mask_voxels:,}"
)

print(
    f"Measurements:   "
    f"{acquisition.n_measurements}"
)

print(
    f"N starts:       "
    f"{N_STARTS}"
)

print(
    f"Max nfev:       "
    f"{MAX_NFEV}"
)

print(
    f"Random seed:    "
    f"{RNG_SEED}"
)

print(
    f"N chunks:       "
    f"{N_CHUNKS}"
)


# =============================================================================
# 29. FIT ONE CHUNK
# =============================================================================


def fit_one_chunk(
    chunk_index: int,
):

    chunk_sizes = (
        get_chunk_sizes()
    )


    expected_voxels = (
        chunk_sizes[
            chunk_index
        ]
    )


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


    print(
        f"\nExpected voxels: "
        f"{expected_voxels:,}"
    )


    # -------------------------------------------------------------------------
    # Preserve scheduler job ID in status file when available.
    # -------------------------------------------------------------------------

    scheduler_job_id = (
        os.environ.get(
            "SLURM_JOB_ID"
        )
    )


    write_chunk_status(
        chunk_index=chunk_index,
        status="RUNNING",
        n_voxels=expected_voxels,
        job_id=scheduler_job_id,
        message=(
            f"started {timestamp()}"
        ),
    )

    chunk_start_datetime = datetime.now()
    
    chunk_start = (
        time.perf_counter()
    )
    
        # =========================================================================
    # LIVE INTRA-CHUNK PROGRESS
    # =========================================================================

    def progress_callback(
        *,
        completed,
        total,
        elapsed,
        rate,
        eta_seconds,
    ):

        write_chunk_status(
            chunk_index=chunk_index,
            status="RUNNING",
            n_voxels=total,
            job_id=scheduler_job_id,
            completed=completed,
            elapsed_seconds=elapsed,
            rate=rate,
            eta_seconds=eta_seconds,
            message=(
                f"fitting voxel "
                f"{completed}/{total}"
            ),
        )

    try:

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

            # Serial inside each cluster job.
            n_jobs=1,

            rng_seed=(
                RNG_SEED
                + chunk_index
            ),

            initialise_orientation=True,

            chunk_index=chunk_index,
    
            n_chunks=N_CHUNKS,
    
            progress_callback=(
                progress_callback
            ),
        )


        # ---------------------------------------------------------------------
        # Derive ODI
        # ---------------------------------------------------------------------

        odi_path = write_odi_from_kappa(
            Path(
                result.parameter_paths[
                    "kappa"
                ]
            )
        )


        # ---------------------------------------------------------------------
        # Runtime
        # ---------------------------------------------------------------------

        chunk_seconds = (
            time.perf_counter()
            - chunk_start
        )

        chunk_end_datetime = datetime.now()

        seconds_per_voxel = (
            chunk_seconds
            / result.n_voxels_fit

            if result.n_voxels_fit > 0

            else np.nan
        )


        write_chunk_status(
            chunk_index=chunk_index,
            status="COMPLETE",
            n_voxels=(
                result.n_voxels_fit
            ),
            job_id=scheduler_job_id,
            runtime_seconds=(
                chunk_seconds
            ),
            started_at=chunk_start_datetime.isoformat(),
            finished_at=chunk_end_datetime.isoformat(),
            message=(
                f"finished {timestamp()}"
            ),
        )


        print(
            f"\nChunk "
            f"{chunk_index + 1}/{N_CHUNKS} "
            f"complete."
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


    except Exception as exc:

        chunk_seconds = (
            time.perf_counter()
            - chunk_start
        )


        write_chunk_status(
            chunk_index=chunk_index,
            status="FAILED",
            n_voxels=expected_voxels,
            job_id=scheduler_job_id,
            runtime_seconds=(
                chunk_seconds
            ),
            message=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )


        raise


# =============================================================================
# 30. EXECUTE FITTING
# =============================================================================


# -----------------------------------------------------------------------------
# FSL_SUB CHUNK WORKER
# -----------------------------------------------------------------------------

if RUNNING_AS_CHUNK_WORKER:

    print(
        "\n============================================"
    )

    print(
        "FSL_SUB CHUNK WORKER"
    )

    print(
        "============================================"
    )


    print(
        f"\nChunk index: "
        f"{CHUNK_INDEX}"
    )

    print(
        f"Chunk:       "
        f"{CHUNK_INDEX + 1}/{N_CHUNKS}"
    )

    print(
        f"Python:      "
        f"{sys.executable}"
    )


    fit_one_chunk(
        CHUNK_INDEX
    )


    print(
        "\nCluster chunk finished successfully."
    )


# -----------------------------------------------------------------------------
# LOCAL EXECUTION
# -----------------------------------------------------------------------------

else:

    print(
        "\n============================================"
    )

    print(
        "LOCAL EXECUTION"
    )

    print(
        "============================================"
    )


    print(
        f"\nFitting "
        f"{N_CHUNKS} chunks sequentially."
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
    
    generate_qc_report()

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


    print_status_summary()


    print(
        f"\nTotal runtime: "
        f"{total_seconds / 60:.2f} min"
    )

    print(
        f"Total runtime: "
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
        f"{np.min(chunk_runtimes) / 60:.2f} min"
    )

    print(
        f"  Median:  "
        f"{np.median(chunk_runtimes) / 60:.2f} min"
    )

    print(
        f"  Slowest: "
        f"{np.max(chunk_runtimes) / 60:.2f} min"
    )


print(
    "\nDone."
)