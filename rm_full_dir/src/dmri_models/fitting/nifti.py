"""High-level NIfTI helpers for voxel-wise least-squares fitting.

These utilities are designed for large dMRI volumes and cluster workflows:
- fit one chunk of voxels per job from a .nii/.nii.gz file;
- write per-parameter NIfTI maps with NaNs outside the fitted chunk;
- merge chunk maps back into a full-brain map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..acquisition import Acquisition
from ..model import Model
from ..parameters import ParameterSpace
from .least_squares import LeastSquaresFitter


@dataclass(frozen=True)
class NiftiLeastSquaresResult:
    """Paths and metadata produced by :func:`fit_least_squares_nifti`."""

    output_dir: Path
    parameter_paths: Mapping[str, Path]
    diagnostic_paths: Mapping[str, Path]
    chunk_index: int | None
    n_chunks: int | None
    n_voxels_fit: int


def _require_nibabel():
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - exercised via tests with skip.
        raise ImportError(
            "nibabel is required for NIfTI fitting helpers. "
            "Install with: pip install microstructure-mri[io]"
        ) from exc
    return nib


def _validate_chunking(
    chunk_index: int | None,
    n_chunks: int | None,
) -> tuple[int | None, int | None]:
    if (chunk_index is None) ^ (n_chunks is None):
        raise ValueError("chunk_index and n_chunks must be provided together.")
    if n_chunks is None:
        return None, None
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least one.")
    if chunk_index < 0 or chunk_index >= n_chunks:
        raise ValueError("chunk_index must satisfy 0 <= chunk_index < n_chunks.")
    return int(chunk_index), int(n_chunks)


def _chunk_indices(flat_indices: np.ndarray, chunk_index: int, n_chunks: int) -> np.ndarray:
    chunks = np.array_split(flat_indices, n_chunks)
    return np.asarray(chunks[chunk_index], dtype=int)


def _chunk_suffix(chunk_index: int | None, n_chunks: int | None) -> str:
    if chunk_index is None or n_chunks is None:
        return ""
    return f".chunk{chunk_index + 1:04d}of{n_chunks:04d}"


def fit_least_squares_nifti(
    *,
    acquisition: Acquisition,
    model: Model,
    parameter_space: ParameterSpace,
    data_path: str | Path,
    output_dir: str | Path,
    output_prefix: str = "fit",
    mask_path: str | Path | None = None,
    sigma: float | np.ndarray = 1.0,
    n_starts: int = 1,
    max_nfev: int = 2000,
    n_jobs: int = 1,
    rng_seed: int | None = None,
    initialise_orientation: bool = True,
    chunk_index: int | None = None,
    n_chunks: int | None = None,
) -> NiftiLeastSquaresResult:
    """Fit a model to all (or chunked) voxels in a 4D diffusion NIfTI.

    The input NIfTI must have shape ``(X, Y, Z, N)`` where ``N`` matches the
    number of measurements in ``acquisition``.

    For cluster use, provide ``chunk_index`` and ``n_chunks`` so each job
    processes one disjoint voxel subset and writes chunk-suffixed maps.
    """
    nib = _require_nibabel()

    chunk_index, n_chunks = _validate_chunking(chunk_index, n_chunks)

    data_img = nib.load(str(data_path))
    data = np.asarray(data_img.get_fdata(dtype=np.float32), dtype=float)
    if data.ndim != 4:
        raise ValueError("data_path must point to a 4D NIfTI array (X, Y, Z, measurements).")
    if data.shape[-1] != acquisition.n_measurements:
        raise ValueError(
            "Last NIfTI dimension must match acquisition.n_measurements "
            f"({data.shape[-1]} != {acquisition.n_measurements})."
        )

    spatial_shape = data.shape[:3]
    if mask_path is None:
        mask = np.ones(spatial_shape, dtype=bool)
    else:
        mask_img = nib.load(str(mask_path))
        mask_data = np.asarray(mask_img.get_fdata(dtype=np.float32), dtype=float)
        if mask_data.shape != spatial_shape:
            raise ValueError("mask_path must have the same 3D shape as the data volume.")
        mask = mask_data > 0.0

    flat_mask_indices = np.flatnonzero(mask.reshape(-1))
    fit_indices = flat_mask_indices
    if chunk_index is not None and n_chunks is not None:
        fit_indices = _chunk_indices(flat_mask_indices, chunk_index, n_chunks)

    fitter = LeastSquaresFitter(model, parameter_space)
    flat_data = data.reshape(-1, acquisition.n_measurements)

    if fit_indices.size == 0:
        result_parameters = {
            name: np.empty((0,), dtype=float) for name in parameter_space.names
        }
        result_cost = np.empty((0,), dtype=float)
        result_success = np.empty((0,), dtype=float)
        result_n_evaluations = np.empty((0,), dtype=float)
    else:
        fit_data = flat_data[fit_indices]
        fit_result = fitter.fit_many(
            acquisition,
            fit_data,
            sigma=sigma,
            n_starts=n_starts,
            max_nfev=max_nfev,
            n_jobs=n_jobs,
            rng=np.random.default_rng(rng_seed),
            initialise_orientation=initialise_orientation,
        )
        result_parameters = fit_result.parameters
        result_cost = np.asarray(fit_result.cost, dtype=float)
        result_success = np.asarray(fit_result.success, dtype=float)
        result_n_evaluations = np.asarray(fit_result.n_evaluations, dtype=float)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = _chunk_suffix(chunk_index, n_chunks)

    parameter_paths: dict[str, Path] = {}
    for name in parameter_space.names:
        volume = np.full(spatial_shape, np.nan, dtype=np.float32)
        if fit_indices.size > 0:
            volume.reshape(-1)[fit_indices] = np.asarray(result_parameters[name], dtype=np.float32)
        path = output_root / f"{output_prefix}.{name}{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(volume, data_img.affine, data_img.header), str(path))
        parameter_paths[name] = path

    cost_map = np.full(spatial_shape, np.nan, dtype=np.float32)
    success_map = np.full(spatial_shape, np.nan, dtype=np.float32)
    nfev_map = np.full(spatial_shape, np.nan, dtype=np.float32)
    if fit_indices.size > 0:
        cost_map.reshape(-1)[fit_indices] = result_cost.astype(np.float32)
        success_map.reshape(-1)[fit_indices] = result_success.astype(np.float32)
        nfev_map.reshape(-1)[fit_indices] = result_n_evaluations.astype(np.float32)

    diagnostic_paths = {
        "cost": output_root / f"{output_prefix}.cost{suffix}.nii.gz",
        "success": output_root / f"{output_prefix}.success{suffix}.nii.gz",
        "n_evaluations": output_root / f"{output_prefix}.n_evaluations{suffix}.nii.gz",
    }
    nib.save(
        nib.Nifti1Image(cost_map, data_img.affine, data_img.header),
        str(diagnostic_paths["cost"]),
    )
    nib.save(
        nib.Nifti1Image(success_map, data_img.affine, data_img.header),
        str(diagnostic_paths["success"]),
    )
    nib.save(
        nib.Nifti1Image(nfev_map, data_img.affine, data_img.header),
        str(diagnostic_paths["n_evaluations"]),
    )

    return NiftiLeastSquaresResult(
        output_dir=output_root,
        parameter_paths=parameter_paths,
        diagnostic_paths=diagnostic_paths,
        chunk_index=chunk_index,
        n_chunks=n_chunks,
        n_voxels_fit=int(fit_indices.size),
    )


def merge_chunk_maps(
    *,
    chunk_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Merge chunk maps written by :func:`fit_least_squares_nifti`.

    The chunk files should contain NaNs outside their fitted voxel subsets.
    """
    if not chunk_paths:
        raise ValueError("chunk_paths must not be empty.")

    nib = _require_nibabel()

    first = nib.load(str(chunk_paths[0]))
    merged = np.asarray(first.get_fdata(dtype=np.float32), dtype=float)

    for path in chunk_paths[1:]:
        part = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=float)
        if part.shape != merged.shape:
            raise ValueError("All chunk maps must have identical shapes.")
        use = np.isfinite(part)
        merged[use] = part[use]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(merged.astype(np.float32), first.affine, first.header), str(out))
    return out
