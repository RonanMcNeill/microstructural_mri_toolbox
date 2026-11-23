from __future__ import annotations

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from dmri_models import (
    Acquisition,
    Ball,
    Component,
    Model,
    Parameter,
    ParameterSpace,
    Simplex,
    fit_least_squares_nifti,
    merge_chunk_maps,
)


def _toy_setup():
    acquisition = Acquisition.from_b_values(
        np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=float),
        b_unit="ms/um^2",
    )
    model = Model(
        [
            Component("fast", Ball(diffusivity="d_fast"), fraction="f_fast"),
            Component("slow", Ball(diffusivity="d_slow"), fraction="f_slow"),
        ]
    )
    parameter_space = ParameterSpace(
        parameters=[
            Parameter("S0", 1.0, (0.5, 1.5)),
            Parameter("d_fast", 2.0, (1.0, 3.0)),
            Parameter("d_slow", 0.8, (0.2, 1.5)),
        ],
        simplexes=[Simplex(("f_fast", "f_slow"), (0.6, 0.4))],
    )
    return acquisition, model, parameter_space


def _write_synthetic_nifti(tmp_path):
    acquisition, model, parameter_space = _toy_setup()
    shape = (2, 2, 1)
    n_meas = acquisition.n_measurements

    truth = {
        (0, 0, 0): {"S0": 1.0, "d_fast": 2.2, "d_slow": 0.7, "f_fast": 0.70, "f_slow": 0.30},
        (0, 1, 0): {"S0": 1.0, "d_fast": 2.1, "d_slow": 0.8, "f_fast": 0.60, "f_slow": 0.40},
        (1, 0, 0): {"S0": 1.0, "d_fast": 2.3, "d_slow": 0.6, "f_fast": 0.55, "f_slow": 0.45},
    }

    data = np.zeros(shape + (n_meas,), dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    for index, params in truth.items():
        data[index] = model.signal(acquisition, params).astype(np.float32)
        mask[index] = 1

    affine = np.eye(4)
    data_path = tmp_path / "dwi.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), data_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    return acquisition, model, parameter_space, data_path, mask_path, mask


def test_fit_least_squares_nifti_writes_maps(tmp_path) -> None:
    acquisition, model, parameter_space, data_path, mask_path, mask = _write_synthetic_nifti(
        tmp_path
    )

    result = fit_least_squares_nifti(
        acquisition=acquisition,
        model=model,
        parameter_space=parameter_space,
        data_path=data_path,
        mask_path=mask_path,
        output_dir=tmp_path / "fit_full",
        output_prefix="toy",
        sigma=1.0,
        n_starts=1,
        max_nfev=400,
        n_jobs=1,
        rng_seed=0,
    )

    assert result.n_voxels_fit == int(mask.sum())
    for path in result.parameter_paths.values():
        assert path.exists()

    f_fast_map = np.asarray(
        nib.load(str(result.parameter_paths["f_fast"])).get_fdata(dtype=np.float32),
        dtype=float,
    )
    assert f_fast_map.shape == mask.shape
    assert np.all(np.isfinite(f_fast_map[mask > 0]))
    assert np.all(np.isnan(f_fast_map[mask == 0]))


def test_fit_least_squares_nifti_chunking_and_merge(tmp_path) -> None:
    acquisition, model, parameter_space, data_path, mask_path, mask = _write_synthetic_nifti(
        tmp_path
    )

    full = fit_least_squares_nifti(
        acquisition=acquisition,
        model=model,
        parameter_space=parameter_space,
        data_path=data_path,
        mask_path=mask_path,
        output_dir=tmp_path / "fit_full",
        output_prefix="toy",
        sigma=1.0,
        n_starts=1,
        max_nfev=400,
        n_jobs=1,
        rng_seed=0,
    )

    chunk0 = fit_least_squares_nifti(
        acquisition=acquisition,
        model=model,
        parameter_space=parameter_space,
        data_path=data_path,
        mask_path=mask_path,
        output_dir=tmp_path / "fit_chunks",
        output_prefix="toy",
        sigma=1.0,
        n_starts=1,
        max_nfev=400,
        n_jobs=1,
        rng_seed=0,
        chunk_index=0,
        n_chunks=2,
    )
    chunk1 = fit_least_squares_nifti(
        acquisition=acquisition,
        model=model,
        parameter_space=parameter_space,
        data_path=data_path,
        mask_path=mask_path,
        output_dir=tmp_path / "fit_chunks",
        output_prefix="toy",
        sigma=1.0,
        n_starts=1,
        max_nfev=400,
        n_jobs=1,
        rng_seed=0,
        chunk_index=1,
        n_chunks=2,
    )

    merged_path = merge_chunk_maps(
        chunk_paths=[
            chunk0.parameter_paths["d_fast"],
            chunk1.parameter_paths["d_fast"],
        ],
        output_path=tmp_path / "fit_chunks" / "toy.d_fast.merged.nii.gz",
    )

    merged = np.asarray(nib.load(str(merged_path)).get_fdata(dtype=np.float32), dtype=float)
    reference = np.asarray(
        nib.load(str(full.parameter_paths["d_fast"])).get_fdata(dtype=np.float32),
        dtype=float,
    )
    np.testing.assert_allclose(merged[mask > 0], reference[mask > 0], rtol=1e-5, atol=1e-6)
    assert np.all(np.isnan(merged[mask == 0]))


def test_fit_least_squares_nifti_requires_complete_chunk_spec(tmp_path) -> None:
    acquisition, model, parameter_space, data_path, mask_path, _ = _write_synthetic_nifti(
        tmp_path
    )

    with pytest.raises(ValueError, match="chunk_index and n_chunks"):
        fit_least_squares_nifti(
            acquisition=acquisition,
            model=model,
            parameter_space=parameter_space,
            data_path=data_path,
            mask_path=mask_path,
            output_dir=tmp_path / "bad_chunk",
            chunk_index=0,
            n_chunks=None,
        )
