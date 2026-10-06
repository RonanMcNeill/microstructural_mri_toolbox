"""Packaged example acquisitions from major diffusion MRI studies."""

from __future__ import annotations

from importlib.resources import files

import numpy as np

from .acquisition import Acquisition


def _fibonacci_directions(n: int) -> np.ndarray:
    """Deterministic approximately uniform unit directions on the sphere."""
    if n < 1:
        raise ValueError("n must be at least one.")
    indices = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / n
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    radius = np.sqrt(np.maximum(0.0, 1.0 - z**2))
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))


def _paths(name: str):
    directory = files("dmri_models").joinpath("data", "acquisitions", name)
    return directory.joinpath("bvals"), directory.joinpath("bvecs")


def _pgse_preset(
    name: str,
    *,
    pulse_duration: float,
    diffusion_time: float,
    echo_time: float | None = None,
    repetition_time: float | None = None,
) -> Acquisition:
    bvals, bvecs = _paths(name)
    return Acquisition.from_fsl(
        bvals,
        bvecs,
        bval_unit="s/mm^2",
        b0_threshold=0.05,
        rectangular_pgse=True,
        pulse_duration=pulse_duration,
        diffusion_time=diffusion_time,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )


def hcp() -> Acquisition:
    """HCP diffusion acquisition: 288 measurements, δ=10.6 ms, Δ=43.1 ms."""
    return _pgse_preset(
        "hcp",
        pulse_duration=10.6,
        diffusion_time=43.1,
        echo_time=89.5,
        repetition_time=5520.0,
    )


def uk_biobank() -> Acquisition:
    """UK Biobank diffusion acquisition: 105 measurements, δ=21.4 ms, Δ=45.5 ms."""
    return _pgse_preset(
        "uk_biobank",
        pulse_duration=21.4,
        diffusion_time=45.5,
        echo_time=92.0,
    )


def bigmac_highres() -> Acquisition:
    """BigMac 0.6-mm diffusion acquisition with 136 measurements."""
    return _pgse_preset(
        "bigmac_0p6mm",
        pulse_duration=7.0,
        diffusion_time=13.0,
        echo_time=25.4,
        repetition_time=10000.0,
    )


def bigmac_lowres() -> Acquisition:
    """BigMac 1.0-mm diffusion acquisition with 260 measurements."""
    return _pgse_preset(
        "bigmac_1p0mm",
        pulse_duration=14.0,
        diffusion_time=24.0,
        echo_time=42.4,
        repetition_time=3500.0,
    )


def sandi_full_acq() -> Acquisition:
    """Palombo-style linear-encoding SANDI scheme.

    Shell layout: b = 0, 500, 1000, 2000, 3000, 4000, 6000 s/mm²
    (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0 ms/µm²), 50 directions per non-zero shell,
    and 50 b=0 measurements. PGSE timing: δ = 24 ms, Δ = 39 ms.
    """
    shell_specs = [
        (0.0,   50, 1),
        (500.0,  50, 1),
        (1000.0, 50, 1),
        (2000.0, 50, 1),
        (3000.0, 50, 1),
        (4000.0, 50, 1),
        (6000.0, 50, 1),
    ]

    delta = 24.0
    big_delta = 39.0

    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
    )

def _build_rm_acq(shell_specs) -> Acquisition:
    """
    shell_specs entries:
        (bval, n_dirs, n_reps, pulse_duration, diffusion_time, echo_time)
    """
    total_dirs = sum(n_dirs for bval, n_dirs, *_ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []
    delta_blocks = []
    big_delta_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps, delta, big_delta, te in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, 2500.0, dtype=float))
        delta_blocks.append(np.full(n_meas, delta, dtype=float))
        big_delta_blocks.append(np.full(n_meas, big_delta, dtype=float))

    return Acquisition.from_pgse(
        np.concatenate(b_blocks),
        b_unit="s/mm^2",
        pulse_duration=np.concatenate(delta_blocks),
        diffusion_time=np.concatenate(big_delta_blocks),
        gradients=np.vstack(g_blocks),
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=np.concatenate(te_blocks),
        repetition_time=np.concatenate(tr_blocks),
    )

def rm_full() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (3000.0, 30, 1),
        (4000.0, 30, 1),
        (5000.0, 30, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_full_te_contrast() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1, 5.0, 12.0, 26.0),
        (500.0, 30, 1, 5.0, 12.0, 26.0),
        (1000.0, 30, 1, 5.0, 12.0, 26.0),
        (2000.0, 30, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 5.0, 12.0, 50.0),  # extra run
        (4000.0, 30, 1, 5.0, 12.0, 26.0),
        (5000.0, 30, 1, 5.0, 12.0, 26.0),
    ]
    return _build_rm_acq(shell_specs)

def rm_full_tdr_contrast() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1, 5.0, 12.0, 26.0),
        (500.0, 30, 1, 5.0, 12.0, 26.0),
        (1000.0, 30, 1, 5.0, 12.0, 26.0),
        (2000.0, 30, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 12.0, 32.0, 50.0),  # extra run
        (4000.0, 30, 1, 5.0, 12.0, 26.0),
        (5000.0, 30, 1, 5.0, 12.0, 26.0),
    ]
    return _build_rm_acq(shell_specs)

def rm_full_both_contrasts() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1, 5.0, 12.0, 26.0),
        (500.0, 30, 1, 5.0, 12.0, 26.0),
        (1000.0, 30, 1, 5.0, 12.0, 26.0),
        (2000.0, 30, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 5.0, 12.0, 26.0),   # original
        (3000.0, 30, 1, 5.0, 12.0, 50.0),    # TE only
        (3000.0, 30, 1, 12.0, 32.0, 50.0),   # TDR only
        (4000.0, 30, 1, 5.0, 12.0, 26.0),
        (5000.0, 30, 1, 5.0, 12.0, 26.0),
    ]
    return _build_rm_acq(shell_specs)

def rm_no_b0p5() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (3000.0, 30, 1),
        (4000.0, 30, 1),
        (5000.0, 30, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b5() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (3000.0, 45, 1),
        (4000.0, 45, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b3_or_b5() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (4000.0, 90, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b3_or_b4() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (5000.0, 90, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b3() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (4000.0, 45, 1),
        (5000.0, 45, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b4_or_b5() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (3000.0, 90, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b3_b4_or_b5() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  60, 1),
        (1000.0, 60, 1),
        (2000.0, 60, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b4() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (3000.0, 45, 1),
        (5000.0, 45, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_no_b2_or_b4() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (3000.0, 30, 1),
        (5000.0, 30, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_11_shells_bmax_10() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  50, 1),
        (1000.0, 50, 1),
        (2000.0, 50, 1),
        (3000.0, 50, 1),
        (4000.0, 50, 1),
        (5000.0, 50, 1),
        (6000.0, 50, 1),
        (7000.0, 50, 1),
        (8000.0, 50, 1),
        (9000.0, 50, 1),
        (10000.0, 50, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_10_shells_bmax_5() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  18, 1),
        (1000.0, 18, 1),
        (1500.0, 18, 1),
        (2000.0, 18, 1),
        (2500.0, 18, 1),
        (3000.0, 18, 1),
        (3500.0, 18, 1),
        (4000.0, 18, 1),
        (4500.0, 18, 1),
        (5000.0, 18, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_30dirs() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  30, 1),
        (1000.0, 30, 1),
        (2000.0, 30, 1),
        (3000.0, 30, 1),
        (5000.0, 30, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_50dirs() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  50, 1),
        (1000.0, 50, 1),
        (2000.0, 50, 1),
        (3000.0, 50, 1),
        (5000.0, 50, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_60dirs() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  60, 1),
        (1000.0, 60, 1),
        (2000.0, 60, 1),
        (3000.0, 60, 1),
        (5000.0, 60, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_inc_dirs_high() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  10, 1),
        (1000.0, 50, 1),
        (2000.0, 50, 1),
        (3000.0, 75, 1),
        (5000.0, 120, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_inc_dirs_mod() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  10, 1),
        (1000.0, 30, 1),
        (2000.0, 50, 1),
        (3000.0, 60, 1),
        (5000.0, 80, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )

def rm_inc_dirs_low() -> Acquisition:
    shell_specs = [
        (0.0,   20, 1),
        (500.0,  10, 1),
        (1000.0, 20, 1),
        (2000.0, 30, 1),
        (3000.0, 40, 1),
        (5000.0, 60, 1),
    ]

    delta = 5.0
    big_delta = 12.0
    te = 26.0
    tr = 2500.0

    # total number of unique non-zero directions needed
    total_dirs = sum(n_dirs for bval, n_dirs, _ in shell_specs if bval != 0.0)
    master_dirs = _fibonacci_directions(total_dirs)

    b_blocks = []
    g_blocks = []
    te_blocks = []
    tr_blocks = []

    cursor = 0
    for bval, n_dirs, n_reps in shell_specs:
        n_meas = n_dirs * n_reps

        if bval == 0.0:
            grads = np.zeros((n_meas, 3), dtype=float)
        else:
            dirs = master_dirs[cursor:cursor + n_dirs]
            cursor += n_dirs
            grads = np.repeat(dirs, n_reps, axis=0)

        b_blocks.append(np.full(n_meas, bval, dtype=float))
        g_blocks.append(grads)
        te_blocks.append(np.full(n_meas, te, dtype=float))
        tr_blocks.append(np.full(n_meas, tr, dtype=float))

    bvals = np.concatenate(b_blocks)
    gradients = np.vstack(g_blocks)
    echo_time = np.concatenate(te_blocks)
    repetition_time = np.concatenate(tr_blocks)

    return Acquisition.from_pgse(
        bvals,
        b_unit="s/mm^2",
        pulse_duration=delta,
        diffusion_time=big_delta,
        gradients=gradients,
        normalize_gradients=False,
        b_tensor_shape=1.0,
        echo_time=echo_time,
        repetition_time=repetition_time,
    )


def names() -> tuple[str, ...]:
    """Names accepted by :func:`load`."""
    return (
        "hcp",
        "uk_biobank",
        "bigmac_highres",
        "bigmac_lowres",
        "sandi_full_acq",
        "rm_full",
        "rm_no_b0p5",
        "rm_no_b5",
        "rm_no_b4_or_b5",
        "rm_no_b3_b4_or_b5",
        "rm_no_b4",
        "rm_no_b3_or_b4",
        "rm_no_b2_or_b4",
        "rm_30dirs",
        "rm_50dirs",
        "rm_60dirs",
        "rm_inc_dirs_high",
        "rm_inc_dirs_mod",
        "rm_inc_dirs_low",
        "rm_11_shells_bmax_10",
        "rm_10_shells_bmax_5",
        "rm_full_te_contrast",
        "rm_full_tdr_contrast",
        "rm_full_both_contrasts",
        "rm_no_b3",
        "rm_no_b3_or_b5",
    )


def load(name: str) -> Acquisition:
    """Load a packaged acquisition by name."""
    loaders = {
        "hcp": hcp,
        "uk_biobank": uk_biobank,
        "bigmac_highres": bigmac_highres,
        "bigmac_lowres": bigmac_lowres,
        "sandi_full_acq": sandi_full_acq,
        "rm_full": rm_full,
        "rm_no_b0p5": rm_no_b0p5,
        "rm_no_b5": rm_no_b5,
        "rm_no_b4_or_b5": rm_no_b4_or_b5,
        "rm_no_b3_b4_or_b5": rm_no_b3_b4_or_b5,
        "rm_no_b4": rm_no_b4,
        "rm_no_b3_or_b4": rm_no_b3_or_b4,
        "rm_no_b2_or_b4": rm_no_b2_or_b4,
        "rm_30dirs": rm_30dirs,
        "rm_50dirs": rm_50dirs,
        "rm_60dirs": rm_60dirs,
        "rm_inc_dirs_high": rm_inc_dirs_high,
        "rm_inc_dirs_mod": rm_inc_dirs_mod,
        "rm_inc_dirs_low": rm_inc_dirs_low,
        "rm_11_shells_bmax_10": rm_11_shells_bmax_10,
        "rm_10_shells_bmax_5": rm_10_shells_bmax_5,
        "rm_full_te_contrast": rm_full_te_contrast,
        "rm_full_tdr_contrast": rm_full_tdr_contrast,
        "rm_full_both_contrasts": rm_full_both_contrasts,
        "rm_no_b3": rm_no_b3,
        "rm_no_b3_or_b5": rm_no_b3_or_b5,
    }
    try:
        return loaders[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown acquisition {name!r}; choose from {names()}.") from exc

# =============================================================================
# Extended RM acquisition sweep
# =============================================================================
#
# 120 acquisitions:
#   5 shell families x 4 volume budgets x 6 allocations
#
# All acquisitions use:
#   b=0: 20 measurements
#   delta = 5 ms
#   diffusion time = 12 ms
#   TE = 26 ms
#   TR = 2500 ms
#
# b-values are specified in s/mm^2 because _build_rm_acq() expects that unit.
# Each non-zero shell uses one measurement per direction (n_reps=1).
# =============================================================================

_EXTENDED_RM_SPECS = {
    "rm_no_b4_or_b5_200_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 45, 1, 5.0, 12.0, 26.0),
        (1000.0, 45, 1, 5.0, 12.0, 26.0),
        (2000.0, 45, 1, 5.0, 12.0, 26.0),
        (3000.0, 45, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_200_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 55, 1, 5.0, 12.0, 26.0),
        (1000.0, 48, 1, 5.0, 12.0, 26.0),
        (2000.0, 40, 1, 5.0, 12.0, 26.0),
        (3000.0, 37, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_200_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 37, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 48, 1, 5.0, 12.0, 26.0),
        (3000.0, 55, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_200_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 34, 1, 5.0, 12.0, 26.0),
        (1000.0, 34, 1, 5.0, 12.0, 26.0),
        (2000.0, 43, 1, 5.0, 12.0, 26.0),
        (3000.0, 69, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_200_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 38, 1, 5.0, 12.0, 26.0),
        (1000.0, 52, 1, 5.0, 12.0, 26.0),
        (2000.0, 52, 1, 5.0, 12.0, 26.0),
        (3000.0, 38, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_200_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 54, 1, 5.0, 12.0, 26.0),
        (1000.0, 36, 1, 5.0, 12.0, 26.0),
        (2000.0, 36, 1, 5.0, 12.0, 26.0),
        (3000.0, 54, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_300_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 70, 1, 5.0, 12.0, 26.0),
        (1000.0, 70, 1, 5.0, 12.0, 26.0),
        (2000.0, 70, 1, 5.0, 12.0, 26.0),
        (3000.0, 70, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_300_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 86, 1, 5.0, 12.0, 26.0),
        (1000.0, 74, 1, 5.0, 12.0, 26.0),
        (2000.0, 63, 1, 5.0, 12.0, 26.0),
        (3000.0, 57, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_300_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 57, 1, 5.0, 12.0, 26.0),
        (1000.0, 63, 1, 5.0, 12.0, 26.0),
        (2000.0, 74, 1, 5.0, 12.0, 26.0),
        (3000.0, 86, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_300_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 53, 1, 5.0, 12.0, 26.0),
        (1000.0, 53, 1, 5.0, 12.0, 26.0),
        (2000.0, 67, 1, 5.0, 12.0, 26.0),
        (3000.0, 107, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_300_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 58, 1, 5.0, 12.0, 26.0),
        (1000.0, 82, 1, 5.0, 12.0, 26.0),
        (2000.0, 82, 1, 5.0, 12.0, 26.0),
        (3000.0, 58, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_300_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 84, 1, 5.0, 12.0, 26.0),
        (1000.0, 56, 1, 5.0, 12.0, 26.0),
        (2000.0, 56, 1, 5.0, 12.0, 26.0),
        (3000.0, 84, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_400_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 95, 1, 5.0, 12.0, 26.0),
        (1000.0, 95, 1, 5.0, 12.0, 26.0),
        (2000.0, 95, 1, 5.0, 12.0, 26.0),
        (3000.0, 95, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_400_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 116, 1, 5.0, 12.0, 26.0),
        (1000.0, 100, 1, 5.0, 12.0, 26.0),
        (2000.0, 85, 1, 5.0, 12.0, 26.0),
        (3000.0, 79, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_400_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 79, 1, 5.0, 12.0, 26.0),
        (1000.0, 85, 1, 5.0, 12.0, 26.0),
        (2000.0, 100, 1, 5.0, 12.0, 26.0),
        (3000.0, 116, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_400_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 72, 1, 5.0, 12.0, 26.0),
        (1000.0, 72, 1, 5.0, 12.0, 26.0),
        (2000.0, 90, 1, 5.0, 12.0, 26.0),
        (3000.0, 146, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_400_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 79, 1, 5.0, 12.0, 26.0),
        (1000.0, 111, 1, 5.0, 12.0, 26.0),
        (2000.0, 111, 1, 5.0, 12.0, 26.0),
        (3000.0, 79, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_400_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 114, 1, 5.0, 12.0, 26.0),
        (1000.0, 76, 1, 5.0, 12.0, 26.0),
        (2000.0, 76, 1, 5.0, 12.0, 26.0),
        (3000.0, 114, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_500_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 120, 1, 5.0, 12.0, 26.0),
        (1000.0, 120, 1, 5.0, 12.0, 26.0),
        (2000.0, 120, 1, 5.0, 12.0, 26.0),
        (3000.0, 120, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_500_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 147, 1, 5.0, 12.0, 26.0),
        (1000.0, 127, 1, 5.0, 12.0, 26.0),
        (2000.0, 108, 1, 5.0, 12.0, 26.0),
        (3000.0, 98, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_500_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 98, 1, 5.0, 12.0, 26.0),
        (1000.0, 108, 1, 5.0, 12.0, 26.0),
        (2000.0, 127, 1, 5.0, 12.0, 26.0),
        (3000.0, 147, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_500_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 91, 1, 5.0, 12.0, 26.0),
        (1000.0, 91, 1, 5.0, 12.0, 26.0),
        (2000.0, 114, 1, 5.0, 12.0, 26.0),
        (3000.0, 184, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_500_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 100, 1, 5.0, 12.0, 26.0),
        (1000.0, 140, 1, 5.0, 12.0, 26.0),
        (2000.0, 140, 1, 5.0, 12.0, 26.0),
        (3000.0, 100, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_or_b5_500_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 144, 1, 5.0, 12.0, 26.0),
        (1000.0, 96, 1, 5.0, 12.0, 26.0),
        (2000.0, 96, 1, 5.0, 12.0, 26.0),
        (3000.0, 144, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_200_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 45, 1, 5.0, 12.0, 26.0),
        (1000.0, 45, 1, 5.0, 12.0, 26.0),
        (2000.0, 45, 1, 5.0, 12.0, 26.0),
        (5000.0, 45, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_200_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 55, 1, 5.0, 12.0, 26.0),
        (1000.0, 48, 1, 5.0, 12.0, 26.0),
        (2000.0, 40, 1, 5.0, 12.0, 26.0),
        (5000.0, 37, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_200_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 37, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 48, 1, 5.0, 12.0, 26.0),
        (5000.0, 55, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_200_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 34, 1, 5.0, 12.0, 26.0),
        (1000.0, 34, 1, 5.0, 12.0, 26.0),
        (2000.0, 43, 1, 5.0, 12.0, 26.0),
        (5000.0, 69, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_200_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 38, 1, 5.0, 12.0, 26.0),
        (1000.0, 52, 1, 5.0, 12.0, 26.0),
        (2000.0, 52, 1, 5.0, 12.0, 26.0),
        (5000.0, 38, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_200_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 54, 1, 5.0, 12.0, 26.0),
        (1000.0, 36, 1, 5.0, 12.0, 26.0),
        (2000.0, 36, 1, 5.0, 12.0, 26.0),
        (5000.0, 54, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_300_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 70, 1, 5.0, 12.0, 26.0),
        (1000.0, 70, 1, 5.0, 12.0, 26.0),
        (2000.0, 70, 1, 5.0, 12.0, 26.0),
        (5000.0, 70, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_300_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 86, 1, 5.0, 12.0, 26.0),
        (1000.0, 74, 1, 5.0, 12.0, 26.0),
        (2000.0, 63, 1, 5.0, 12.0, 26.0),
        (5000.0, 57, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_300_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 57, 1, 5.0, 12.0, 26.0),
        (1000.0, 63, 1, 5.0, 12.0, 26.0),
        (2000.0, 74, 1, 5.0, 12.0, 26.0),
        (5000.0, 86, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_300_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 53, 1, 5.0, 12.0, 26.0),
        (1000.0, 53, 1, 5.0, 12.0, 26.0),
        (2000.0, 67, 1, 5.0, 12.0, 26.0),
        (5000.0, 107, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_300_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 58, 1, 5.0, 12.0, 26.0),
        (1000.0, 82, 1, 5.0, 12.0, 26.0),
        (2000.0, 82, 1, 5.0, 12.0, 26.0),
        (5000.0, 58, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_300_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 84, 1, 5.0, 12.0, 26.0),
        (1000.0, 56, 1, 5.0, 12.0, 26.0),
        (2000.0, 56, 1, 5.0, 12.0, 26.0),
        (5000.0, 84, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_400_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 95, 1, 5.0, 12.0, 26.0),
        (1000.0, 95, 1, 5.0, 12.0, 26.0),
        (2000.0, 95, 1, 5.0, 12.0, 26.0),
        (5000.0, 95, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_400_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 116, 1, 5.0, 12.0, 26.0),
        (1000.0, 100, 1, 5.0, 12.0, 26.0),
        (2000.0, 85, 1, 5.0, 12.0, 26.0),
        (5000.0, 79, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_400_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 79, 1, 5.0, 12.0, 26.0),
        (1000.0, 85, 1, 5.0, 12.0, 26.0),
        (2000.0, 100, 1, 5.0, 12.0, 26.0),
        (5000.0, 116, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_400_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 72, 1, 5.0, 12.0, 26.0),
        (1000.0, 72, 1, 5.0, 12.0, 26.0),
        (2000.0, 90, 1, 5.0, 12.0, 26.0),
        (5000.0, 146, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_400_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 79, 1, 5.0, 12.0, 26.0),
        (1000.0, 111, 1, 5.0, 12.0, 26.0),
        (2000.0, 111, 1, 5.0, 12.0, 26.0),
        (5000.0, 79, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_400_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 114, 1, 5.0, 12.0, 26.0),
        (1000.0, 76, 1, 5.0, 12.0, 26.0),
        (2000.0, 76, 1, 5.0, 12.0, 26.0),
        (5000.0, 114, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_500_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 120, 1, 5.0, 12.0, 26.0),
        (1000.0, 120, 1, 5.0, 12.0, 26.0),
        (2000.0, 120, 1, 5.0, 12.0, 26.0),
        (5000.0, 120, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_500_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 147, 1, 5.0, 12.0, 26.0),
        (1000.0, 127, 1, 5.0, 12.0, 26.0),
        (2000.0, 108, 1, 5.0, 12.0, 26.0),
        (5000.0, 98, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_500_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 98, 1, 5.0, 12.0, 26.0),
        (1000.0, 108, 1, 5.0, 12.0, 26.0),
        (2000.0, 127, 1, 5.0, 12.0, 26.0),
        (5000.0, 147, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_500_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 91, 1, 5.0, 12.0, 26.0),
        (1000.0, 91, 1, 5.0, 12.0, 26.0),
        (2000.0, 114, 1, 5.0, 12.0, 26.0),
        (5000.0, 184, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_500_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 100, 1, 5.0, 12.0, 26.0),
        (1000.0, 140, 1, 5.0, 12.0, 26.0),
        (2000.0, 140, 1, 5.0, 12.0, 26.0),
        (5000.0, 100, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b4_500_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 144, 1, 5.0, 12.0, 26.0),
        (1000.0, 96, 1, 5.0, 12.0, 26.0),
        (2000.0, 96, 1, 5.0, 12.0, 26.0),
        (5000.0, 144, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_200_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 45, 1, 5.0, 12.0, 26.0),
        (1000.0, 45, 1, 5.0, 12.0, 26.0),
        (2000.0, 45, 1, 5.0, 12.0, 26.0),
        (4000.0, 45, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_200_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 55, 1, 5.0, 12.0, 26.0),
        (1000.0, 48, 1, 5.0, 12.0, 26.0),
        (2000.0, 40, 1, 5.0, 12.0, 26.0),
        (4000.0, 37, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_200_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 37, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 48, 1, 5.0, 12.0, 26.0),
        (4000.0, 55, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_200_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 34, 1, 5.0, 12.0, 26.0),
        (1000.0, 34, 1, 5.0, 12.0, 26.0),
        (2000.0, 43, 1, 5.0, 12.0, 26.0),
        (4000.0, 69, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_200_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 38, 1, 5.0, 12.0, 26.0),
        (1000.0, 52, 1, 5.0, 12.0, 26.0),
        (2000.0, 52, 1, 5.0, 12.0, 26.0),
        (4000.0, 38, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_200_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 54, 1, 5.0, 12.0, 26.0),
        (1000.0, 36, 1, 5.0, 12.0, 26.0),
        (2000.0, 36, 1, 5.0, 12.0, 26.0),
        (4000.0, 54, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_300_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 70, 1, 5.0, 12.0, 26.0),
        (1000.0, 70, 1, 5.0, 12.0, 26.0),
        (2000.0, 70, 1, 5.0, 12.0, 26.0),
        (4000.0, 70, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_300_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 86, 1, 5.0, 12.0, 26.0),
        (1000.0, 74, 1, 5.0, 12.0, 26.0),
        (2000.0, 63, 1, 5.0, 12.0, 26.0),
        (4000.0, 57, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_300_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 57, 1, 5.0, 12.0, 26.0),
        (1000.0, 63, 1, 5.0, 12.0, 26.0),
        (2000.0, 74, 1, 5.0, 12.0, 26.0),
        (4000.0, 86, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_300_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 53, 1, 5.0, 12.0, 26.0),
        (1000.0, 53, 1, 5.0, 12.0, 26.0),
        (2000.0, 67, 1, 5.0, 12.0, 26.0),
        (4000.0, 107, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_300_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 58, 1, 5.0, 12.0, 26.0),
        (1000.0, 82, 1, 5.0, 12.0, 26.0),
        (2000.0, 82, 1, 5.0, 12.0, 26.0),
        (4000.0, 58, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_300_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 84, 1, 5.0, 12.0, 26.0),
        (1000.0, 56, 1, 5.0, 12.0, 26.0),
        (2000.0, 56, 1, 5.0, 12.0, 26.0),
        (4000.0, 84, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_400_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 95, 1, 5.0, 12.0, 26.0),
        (1000.0, 95, 1, 5.0, 12.0, 26.0),
        (2000.0, 95, 1, 5.0, 12.0, 26.0),
        (4000.0, 95, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_400_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 116, 1, 5.0, 12.0, 26.0),
        (1000.0, 100, 1, 5.0, 12.0, 26.0),
        (2000.0, 85, 1, 5.0, 12.0, 26.0),
        (4000.0, 79, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_400_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 79, 1, 5.0, 12.0, 26.0),
        (1000.0, 85, 1, 5.0, 12.0, 26.0),
        (2000.0, 100, 1, 5.0, 12.0, 26.0),
        (4000.0, 116, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_400_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 72, 1, 5.0, 12.0, 26.0),
        (1000.0, 72, 1, 5.0, 12.0, 26.0),
        (2000.0, 90, 1, 5.0, 12.0, 26.0),
        (4000.0, 146, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_400_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 79, 1, 5.0, 12.0, 26.0),
        (1000.0, 111, 1, 5.0, 12.0, 26.0),
        (2000.0, 111, 1, 5.0, 12.0, 26.0),
        (4000.0, 79, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_400_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 114, 1, 5.0, 12.0, 26.0),
        (1000.0, 76, 1, 5.0, 12.0, 26.0),
        (2000.0, 76, 1, 5.0, 12.0, 26.0),
        (4000.0, 114, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_500_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 120, 1, 5.0, 12.0, 26.0),
        (1000.0, 120, 1, 5.0, 12.0, 26.0),
        (2000.0, 120, 1, 5.0, 12.0, 26.0),
        (4000.0, 120, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_500_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 147, 1, 5.0, 12.0, 26.0),
        (1000.0, 127, 1, 5.0, 12.0, 26.0),
        (2000.0, 108, 1, 5.0, 12.0, 26.0),
        (4000.0, 98, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_500_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 98, 1, 5.0, 12.0, 26.0),
        (1000.0, 108, 1, 5.0, 12.0, 26.0),
        (2000.0, 127, 1, 5.0, 12.0, 26.0),
        (4000.0, 147, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_500_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 91, 1, 5.0, 12.0, 26.0),
        (1000.0, 91, 1, 5.0, 12.0, 26.0),
        (2000.0, 114, 1, 5.0, 12.0, 26.0),
        (4000.0, 184, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_500_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 100, 1, 5.0, 12.0, 26.0),
        (1000.0, 140, 1, 5.0, 12.0, 26.0),
        (2000.0, 140, 1, 5.0, 12.0, 26.0),
        (4000.0, 100, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b3_or_b5_500_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 144, 1, 5.0, 12.0, 26.0),
        (1000.0, 96, 1, 5.0, 12.0, 26.0),
        (2000.0, 96, 1, 5.0, 12.0, 26.0),
        (4000.0, 144, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_200_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 36, 1, 5.0, 12.0, 26.0),
        (1000.0, 36, 1, 5.0, 12.0, 26.0),
        (2000.0, 36, 1, 5.0, 12.0, 26.0),
        (3000.0, 36, 1, 5.0, 12.0, 26.0),
        (5000.0, 36, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_200_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 46, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 34, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 5.0, 12.0, 26.0),
        (5000.0, 30, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_200_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 30, 1, 5.0, 12.0, 26.0),
        (1000.0, 30, 1, 5.0, 12.0, 26.0),
        (2000.0, 34, 1, 5.0, 12.0, 26.0),
        (3000.0, 40, 1, 5.0, 12.0, 26.0),
        (5000.0, 46, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_200_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 28, 1, 5.0, 12.0, 26.0),
        (1000.0, 28, 1, 5.0, 12.0, 26.0),
        (2000.0, 28, 1, 5.0, 12.0, 26.0),
        (3000.0, 41, 1, 5.0, 12.0, 26.0),
        (5000.0, 55, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_200_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 26, 1, 5.0, 12.0, 26.0),
        (1000.0, 38, 1, 5.0, 12.0, 26.0),
        (2000.0, 52, 1, 5.0, 12.0, 26.0),
        (3000.0, 38, 1, 5.0, 12.0, 26.0),
        (5000.0, 26, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_200_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 51, 1, 5.0, 12.0, 26.0),
        (1000.0, 26, 1, 5.0, 12.0, 26.0),
        (2000.0, 26, 1, 5.0, 12.0, 26.0),
        (3000.0, 26, 1, 5.0, 12.0, 26.0),
        (5000.0, 51, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_300_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 56, 1, 5.0, 12.0, 26.0),
        (1000.0, 56, 1, 5.0, 12.0, 26.0),
        (2000.0, 56, 1, 5.0, 12.0, 26.0),
        (3000.0, 56, 1, 5.0, 12.0, 26.0),
        (5000.0, 56, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_300_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 71, 1, 5.0, 12.0, 26.0),
        (1000.0, 62, 1, 5.0, 12.0, 26.0),
        (2000.0, 53, 1, 5.0, 12.0, 26.0),
        (3000.0, 47, 1, 5.0, 12.0, 26.0),
        (5000.0, 47, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_300_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 47, 1, 5.0, 12.0, 26.0),
        (1000.0, 47, 1, 5.0, 12.0, 26.0),
        (2000.0, 53, 1, 5.0, 12.0, 26.0),
        (3000.0, 62, 1, 5.0, 12.0, 26.0),
        (5000.0, 71, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_300_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 43, 1, 5.0, 12.0, 26.0),
        (1000.0, 43, 1, 5.0, 12.0, 26.0),
        (2000.0, 43, 1, 5.0, 12.0, 26.0),
        (3000.0, 65, 1, 5.0, 12.0, 26.0),
        (5000.0, 86, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_300_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 40, 1, 5.0, 12.0, 26.0),
        (1000.0, 60, 1, 5.0, 12.0, 26.0),
        (2000.0, 80, 1, 5.0, 12.0, 26.0),
        (3000.0, 60, 1, 5.0, 12.0, 26.0),
        (5000.0, 40, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_300_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 80, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 40, 1, 5.0, 12.0, 26.0),
        (3000.0, 40, 1, 5.0, 12.0, 26.0),
        (5000.0, 80, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_400_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 76, 1, 5.0, 12.0, 26.0),
        (1000.0, 76, 1, 5.0, 12.0, 26.0),
        (2000.0, 76, 1, 5.0, 12.0, 26.0),
        (3000.0, 76, 1, 5.0, 12.0, 26.0),
        (5000.0, 76, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_400_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 97, 1, 5.0, 12.0, 26.0),
        (1000.0, 84, 1, 5.0, 12.0, 26.0),
        (2000.0, 71, 1, 5.0, 12.0, 26.0),
        (3000.0, 64, 1, 5.0, 12.0, 26.0),
        (5000.0, 64, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_400_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 64, 1, 5.0, 12.0, 26.0),
        (1000.0, 64, 1, 5.0, 12.0, 26.0),
        (2000.0, 71, 1, 5.0, 12.0, 26.0),
        (3000.0, 84, 1, 5.0, 12.0, 26.0),
        (5000.0, 97, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_400_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 58, 1, 5.0, 12.0, 26.0),
        (1000.0, 58, 1, 5.0, 12.0, 26.0),
        (2000.0, 58, 1, 5.0, 12.0, 26.0),
        (3000.0, 88, 1, 5.0, 12.0, 26.0),
        (5000.0, 118, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_400_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 55, 1, 5.0, 12.0, 26.0),
        (1000.0, 81, 1, 5.0, 12.0, 26.0),
        (2000.0, 108, 1, 5.0, 12.0, 26.0),
        (3000.0, 81, 1, 5.0, 12.0, 26.0),
        (5000.0, 55, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_400_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 109, 1, 5.0, 12.0, 26.0),
        (1000.0, 54, 1, 5.0, 12.0, 26.0),
        (2000.0, 54, 1, 5.0, 12.0, 26.0),
        (3000.0, 54, 1, 5.0, 12.0, 26.0),
        (5000.0, 109, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_500_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 96, 1, 5.0, 12.0, 26.0),
        (1000.0, 96, 1, 5.0, 12.0, 26.0),
        (2000.0, 96, 1, 5.0, 12.0, 26.0),
        (3000.0, 96, 1, 5.0, 12.0, 26.0),
        (5000.0, 96, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_500_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 122, 1, 5.0, 12.0, 26.0),
        (1000.0, 106, 1, 5.0, 12.0, 26.0),
        (2000.0, 90, 1, 5.0, 12.0, 26.0),
        (3000.0, 81, 1, 5.0, 12.0, 26.0),
        (5000.0, 81, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_500_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 81, 1, 5.0, 12.0, 26.0),
        (1000.0, 81, 1, 5.0, 12.0, 26.0),
        (2000.0, 90, 1, 5.0, 12.0, 26.0),
        (3000.0, 106, 1, 5.0, 12.0, 26.0),
        (5000.0, 122, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_500_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 74, 1, 5.0, 12.0, 26.0),
        (1000.0, 74, 1, 5.0, 12.0, 26.0),
        (2000.0, 74, 1, 5.0, 12.0, 26.0),
        (3000.0, 111, 1, 5.0, 12.0, 26.0),
        (5000.0, 147, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_500_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 69, 1, 5.0, 12.0, 26.0),
        (1000.0, 102, 1, 5.0, 12.0, 26.0),
        (2000.0, 138, 1, 5.0, 12.0, 26.0),
        (3000.0, 102, 1, 5.0, 12.0, 26.0),
        (5000.0, 69, 1, 5.0, 12.0, 26.0),
    ],
    "rm_no_b4_500_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 136, 1, 5.0, 12.0, 26.0),
        (1000.0, 69, 1, 5.0, 12.0, 26.0),
        (2000.0, 69, 1, 5.0, 12.0, 26.0),
        (3000.0, 69, 1, 5.0, 12.0, 26.0),
        (5000.0, 137, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_200_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 18, 1, 5.0, 12.0, 26.0),
        (1000.0, 18, 1, 5.0, 12.0, 26.0),
        (1500.0, 18, 1, 5.0, 12.0, 26.0),
        (2000.0, 18, 1, 5.0, 12.0, 26.0),
        (2500.0, 18, 1, 5.0, 12.0, 26.0),
        (3000.0, 18, 1, 5.0, 12.0, 26.0),
        (3500.0, 18, 1, 5.0, 12.0, 26.0),
        (4000.0, 18, 1, 5.0, 12.0, 26.0),
        (4500.0, 18, 1, 5.0, 12.0, 26.0),
        (5000.0, 18, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_200_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 23, 1, 5.0, 12.0, 26.0),
        (1000.0, 22, 1, 5.0, 12.0, 26.0),
        (1500.0, 20, 1, 5.0, 12.0, 26.0),
        (2000.0, 19, 1, 5.0, 12.0, 26.0),
        (2500.0, 17, 1, 5.0, 12.0, 26.0),
        (3000.0, 16, 1, 5.0, 12.0, 26.0),
        (3500.0, 16, 1, 5.0, 12.0, 26.0),
        (4000.0, 16, 1, 5.0, 12.0, 26.0),
        (4500.0, 16, 1, 5.0, 12.0, 26.0),
        (5000.0, 15, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_200_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 15, 1, 5.0, 12.0, 26.0),
        (1000.0, 15, 1, 5.0, 12.0, 26.0),
        (1500.0, 15, 1, 5.0, 12.0, 26.0),
        (2000.0, 15, 1, 5.0, 12.0, 26.0),
        (2500.0, 16, 1, 5.0, 12.0, 26.0),
        (3000.0, 18, 1, 5.0, 12.0, 26.0),
        (3500.0, 19, 1, 5.0, 12.0, 26.0),
        (4000.0, 21, 1, 5.0, 12.0, 26.0),
        (4500.0, 22, 1, 5.0, 12.0, 26.0),
        (5000.0, 24, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_200_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 13, 1, 5.0, 12.0, 26.0),
        (1000.0, 13, 1, 5.0, 12.0, 26.0),
        (1500.0, 13, 1, 5.0, 12.0, 26.0),
        (2000.0, 14, 1, 5.0, 12.0, 26.0),
        (2500.0, 15, 1, 5.0, 12.0, 26.0),
        (3000.0, 17, 1, 5.0, 12.0, 26.0),
        (3500.0, 20, 1, 5.0, 12.0, 26.0),
        (4000.0, 23, 1, 5.0, 12.0, 26.0),
        (4500.0, 25, 1, 5.0, 12.0, 26.0),
        (5000.0, 27, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_200_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 13, 1, 5.0, 12.0, 26.0),
        (1000.0, 16, 1, 5.0, 12.0, 26.0),
        (1500.0, 19, 1, 5.0, 12.0, 26.0),
        (2000.0, 20, 1, 5.0, 12.0, 26.0),
        (2500.0, 22, 1, 5.0, 12.0, 26.0),
        (3000.0, 22, 1, 5.0, 12.0, 26.0),
        (3500.0, 20, 1, 5.0, 12.0, 26.0),
        (4000.0, 19, 1, 5.0, 12.0, 26.0),
        (4500.0, 16, 1, 5.0, 12.0, 26.0),
        (5000.0, 13, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_200_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 23, 1, 5.0, 12.0, 26.0),
        (1000.0, 19, 1, 5.0, 12.0, 26.0),
        (1500.0, 16, 1, 5.0, 12.0, 26.0),
        (2000.0, 14, 1, 5.0, 12.0, 26.0),
        (2500.0, 14, 1, 5.0, 12.0, 26.0),
        (3000.0, 14, 1, 5.0, 12.0, 26.0),
        (3500.0, 16, 1, 5.0, 12.0, 26.0),
        (4000.0, 19, 1, 5.0, 12.0, 26.0),
        (4500.0, 22, 1, 5.0, 12.0, 26.0),
        (5000.0, 23, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_300_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 28, 1, 5.0, 12.0, 26.0),
        (1000.0, 28, 1, 5.0, 12.0, 26.0),
        (1500.0, 28, 1, 5.0, 12.0, 26.0),
        (2000.0, 28, 1, 5.0, 12.0, 26.0),
        (2500.0, 28, 1, 5.0, 12.0, 26.0),
        (3000.0, 28, 1, 5.0, 12.0, 26.0),
        (3500.0, 28, 1, 5.0, 12.0, 26.0),
        (4000.0, 28, 1, 5.0, 12.0, 26.0),
        (4500.0, 28, 1, 5.0, 12.0, 26.0),
        (5000.0, 28, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_300_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 37, 1, 5.0, 12.0, 26.0),
        (1000.0, 34, 1, 5.0, 12.0, 26.0),
        (1500.0, 32, 1, 5.0, 12.0, 26.0),
        (2000.0, 29, 1, 5.0, 12.0, 26.0),
        (2500.0, 27, 1, 5.0, 12.0, 26.0),
        (3000.0, 25, 1, 5.0, 12.0, 26.0),
        (3500.0, 24, 1, 5.0, 12.0, 26.0),
        (4000.0, 24, 1, 5.0, 12.0, 26.0),
        (4500.0, 24, 1, 5.0, 12.0, 26.0),
        (5000.0, 24, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_300_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 23, 1, 5.0, 12.0, 26.0),
        (1000.0, 23, 1, 5.0, 12.0, 26.0),
        (1500.0, 23, 1, 5.0, 12.0, 26.0),
        (2000.0, 23, 1, 5.0, 12.0, 26.0),
        (2500.0, 26, 1, 5.0, 12.0, 26.0),
        (3000.0, 28, 1, 5.0, 12.0, 26.0),
        (3500.0, 30, 1, 5.0, 12.0, 26.0),
        (4000.0, 32, 1, 5.0, 12.0, 26.0),
        (4500.0, 35, 1, 5.0, 12.0, 26.0),
        (5000.0, 37, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_300_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 21, 1, 5.0, 12.0, 26.0),
        (1000.0, 21, 1, 5.0, 12.0, 26.0),
        (1500.0, 21, 1, 5.0, 12.0, 26.0),
        (2000.0, 21, 1, 5.0, 12.0, 26.0),
        (2500.0, 23, 1, 5.0, 12.0, 26.0),
        (3000.0, 27, 1, 5.0, 12.0, 26.0),
        (3500.0, 31, 1, 5.0, 12.0, 26.0),
        (4000.0, 35, 1, 5.0, 12.0, 26.0),
        (4500.0, 39, 1, 5.0, 12.0, 26.0),
        (5000.0, 41, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_300_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 21, 1, 5.0, 12.0, 26.0),
        (1000.0, 25, 1, 5.0, 12.0, 26.0),
        (1500.0, 29, 1, 5.0, 12.0, 26.0),
        (2000.0, 31, 1, 5.0, 12.0, 26.0),
        (2500.0, 34, 1, 5.0, 12.0, 26.0),
        (3000.0, 34, 1, 5.0, 12.0, 26.0),
        (3500.0, 31, 1, 5.0, 12.0, 26.0),
        (4000.0, 29, 1, 5.0, 12.0, 26.0),
        (4500.0, 25, 1, 5.0, 12.0, 26.0),
        (5000.0, 21, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_300_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 36, 1, 5.0, 12.0, 26.0),
        (1000.0, 29, 1, 5.0, 12.0, 26.0),
        (1500.0, 25, 1, 5.0, 12.0, 26.0),
        (2000.0, 22, 1, 5.0, 12.0, 26.0),
        (2500.0, 22, 1, 5.0, 12.0, 26.0),
        (3000.0, 22, 1, 5.0, 12.0, 26.0),
        (3500.0, 25, 1, 5.0, 12.0, 26.0),
        (4000.0, 29, 1, 5.0, 12.0, 26.0),
        (4500.0, 34, 1, 5.0, 12.0, 26.0),
        (5000.0, 36, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_400_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 38, 1, 5.0, 12.0, 26.0),
        (1000.0, 38, 1, 5.0, 12.0, 26.0),
        (1500.0, 38, 1, 5.0, 12.0, 26.0),
        (2000.0, 38, 1, 5.0, 12.0, 26.0),
        (2500.0, 38, 1, 5.0, 12.0, 26.0),
        (3000.0, 38, 1, 5.0, 12.0, 26.0),
        (3500.0, 38, 1, 5.0, 12.0, 26.0),
        (4000.0, 38, 1, 5.0, 12.0, 26.0),
        (4500.0, 38, 1, 5.0, 12.0, 26.0),
        (5000.0, 38, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_400_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 50, 1, 5.0, 12.0, 26.0),
        (1000.0, 46, 1, 5.0, 12.0, 26.0),
        (1500.0, 43, 1, 5.0, 12.0, 26.0),
        (2000.0, 40, 1, 5.0, 12.0, 26.0),
        (2500.0, 36, 1, 5.0, 12.0, 26.0),
        (3000.0, 33, 1, 5.0, 12.0, 26.0),
        (3500.0, 33, 1, 5.0, 12.0, 26.0),
        (4000.0, 33, 1, 5.0, 12.0, 26.0),
        (4500.0, 33, 1, 5.0, 12.0, 26.0),
        (5000.0, 33, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_400_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 31, 1, 5.0, 12.0, 26.0),
        (1000.0, 31, 1, 5.0, 12.0, 26.0),
        (1500.0, 31, 1, 5.0, 12.0, 26.0),
        (2000.0, 32, 1, 5.0, 12.0, 26.0),
        (2500.0, 35, 1, 5.0, 12.0, 26.0),
        (3000.0, 38, 1, 5.0, 12.0, 26.0),
        (3500.0, 41, 1, 5.0, 12.0, 26.0),
        (4000.0, 44, 1, 5.0, 12.0, 26.0),
        (4500.0, 47, 1, 5.0, 12.0, 26.0),
        (5000.0, 50, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_400_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 28, 1, 5.0, 12.0, 26.0),
        (1000.0, 28, 1, 5.0, 12.0, 26.0),
        (1500.0, 28, 1, 5.0, 12.0, 26.0),
        (2000.0, 28, 1, 5.0, 12.0, 26.0),
        (2500.0, 31, 1, 5.0, 12.0, 26.0),
        (3000.0, 37, 1, 5.0, 12.0, 26.0),
        (3500.0, 42, 1, 5.0, 12.0, 26.0),
        (4000.0, 48, 1, 5.0, 12.0, 26.0),
        (4500.0, 54, 1, 5.0, 12.0, 26.0),
        (5000.0, 56, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_400_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 28, 1, 5.0, 12.0, 26.0),
        (1000.0, 34, 1, 5.0, 12.0, 26.0),
        (1500.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 43, 1, 5.0, 12.0, 26.0),
        (2500.0, 45, 1, 5.0, 12.0, 26.0),
        (3000.0, 45, 1, 5.0, 12.0, 26.0),
        (3500.0, 43, 1, 5.0, 12.0, 26.0),
        (4000.0, 40, 1, 5.0, 12.0, 26.0),
        (4500.0, 34, 1, 5.0, 12.0, 26.0),
        (5000.0, 28, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_400_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 49, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (1500.0, 33, 1, 5.0, 12.0, 26.0),
        (2000.0, 30, 1, 5.0, 12.0, 26.0),
        (2500.0, 30, 1, 5.0, 12.0, 26.0),
        (3000.0, 30, 1, 5.0, 12.0, 26.0),
        (3500.0, 33, 1, 5.0, 12.0, 26.0),
        (4000.0, 40, 1, 5.0, 12.0, 26.0),
        (4500.0, 46, 1, 5.0, 12.0, 26.0),
        (5000.0, 49, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_500_uniform": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 48, 1, 5.0, 12.0, 26.0),
        (1000.0, 48, 1, 5.0, 12.0, 26.0),
        (1500.0, 48, 1, 5.0, 12.0, 26.0),
        (2000.0, 48, 1, 5.0, 12.0, 26.0),
        (2500.0, 48, 1, 5.0, 12.0, 26.0),
        (3000.0, 48, 1, 5.0, 12.0, 26.0),
        (3500.0, 48, 1, 5.0, 12.0, 26.0),
        (4000.0, 48, 1, 5.0, 12.0, 26.0),
        (4500.0, 48, 1, 5.0, 12.0, 26.0),
        (5000.0, 48, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_500_low": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 62, 1, 5.0, 12.0, 26.0),
        (1000.0, 58, 1, 5.0, 12.0, 26.0),
        (1500.0, 54, 1, 5.0, 12.0, 26.0),
        (2000.0, 50, 1, 5.0, 12.0, 26.0),
        (2500.0, 46, 1, 5.0, 12.0, 26.0),
        (3000.0, 42, 1, 5.0, 12.0, 26.0),
        (3500.0, 42, 1, 5.0, 12.0, 26.0),
        (4000.0, 42, 1, 5.0, 12.0, 26.0),
        (4500.0, 42, 1, 5.0, 12.0, 26.0),
        (5000.0, 42, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_500_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 40, 1, 5.0, 12.0, 26.0),
        (1000.0, 40, 1, 5.0, 12.0, 26.0),
        (1500.0, 40, 1, 5.0, 12.0, 26.0),
        (2000.0, 40, 1, 5.0, 12.0, 26.0),
        (2500.0, 44, 1, 5.0, 12.0, 26.0),
        (3000.0, 48, 1, 5.0, 12.0, 26.0),
        (3500.0, 51, 1, 5.0, 12.0, 26.0),
        (4000.0, 55, 1, 5.0, 12.0, 26.0),
        (4500.0, 59, 1, 5.0, 12.0, 26.0),
        (5000.0, 63, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_500_strong_high": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 36, 1, 5.0, 12.0, 26.0),
        (1000.0, 36, 1, 5.0, 12.0, 26.0),
        (1500.0, 36, 1, 5.0, 12.0, 26.0),
        (2000.0, 36, 1, 5.0, 12.0, 26.0),
        (2500.0, 39, 1, 5.0, 12.0, 26.0),
        (3000.0, 46, 1, 5.0, 12.0, 26.0),
        (3500.0, 53, 1, 5.0, 12.0, 26.0),
        (4000.0, 60, 1, 5.0, 12.0, 26.0),
        (4500.0, 67, 1, 5.0, 12.0, 26.0),
        (5000.0, 71, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_500_middle": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 36, 1, 5.0, 12.0, 26.0),
        (1000.0, 43, 1, 5.0, 12.0, 26.0),
        (1500.0, 50, 1, 5.0, 12.0, 26.0),
        (2000.0, 54, 1, 5.0, 12.0, 26.0),
        (2500.0, 57, 1, 5.0, 12.0, 26.0),
        (3000.0, 57, 1, 5.0, 12.0, 26.0),
        (3500.0, 54, 1, 5.0, 12.0, 26.0),
        (4000.0, 50, 1, 5.0, 12.0, 26.0),
        (4500.0, 43, 1, 5.0, 12.0, 26.0),
        (5000.0, 36, 1, 5.0, 12.0, 26.0),
    ],
    "rm_10_shells_500_edge": [
        (0.0, 20, 1, 5.0, 12.0, 26.0),
        (500.0, 62, 1, 5.0, 12.0, 26.0),
        (1000.0, 50, 1, 5.0, 12.0, 26.0),
        (1500.0, 42, 1, 5.0, 12.0, 26.0),
        (2000.0, 38, 1, 5.0, 12.0, 26.0),
        (2500.0, 38, 1, 5.0, 12.0, 26.0),
        (3000.0, 38, 1, 5.0, 12.0, 26.0),
        (3500.0, 42, 1, 5.0, 12.0, 26.0),
        (4000.0, 50, 1, 5.0, 12.0, 26.0),
        (4500.0, 58, 1, 5.0, 12.0, 26.0),
        (5000.0, 62, 1, 5.0, 12.0, 26.0),
    ],
}


def _make_extended_rm_acquisition(name: str) -> Acquisition:
    """Build one extended RM acquisition from its exact shell specification."""
    return _build_rm_acq(_EXTENDED_RM_SPECS[name])


def _make_extended_loader(name: str):
    def loader() -> Acquisition:
        return _make_extended_rm_acquisition(name)

    loader.__name__ = name
    loader.__qualname__ = name
    loader.__doc__ = f"Extended RM acquisition: {name}."
    return loader


# Make each acquisition available as a normal module-level function.
for _name in _EXTENDED_RM_SPECS:
    globals()[_name] = _make_extended_loader(_name)


_EXTENDED_RM_NAMES = tuple(_EXTENDED_RM_SPECS.keys())

# Preserve all pre-existing names/loaders while adding the sweep.
_ORIGINAL_NAMES = names
_ORIGINAL_LOAD = load


def names() -> tuple[str, ...]:
    """Names accepted by :func:`load`, including the extended RM sweep."""
    return _ORIGINAL_NAMES() + _EXTENDED_RM_NAMES


def load(name: str) -> Acquisition:
    """Load a packaged acquisition by name, including the extended RM sweep."""
    if name in _EXTENDED_RM_SPECS:
        return _make_extended_rm_acquisition(name)
    return _ORIGINAL_LOAD(name)
