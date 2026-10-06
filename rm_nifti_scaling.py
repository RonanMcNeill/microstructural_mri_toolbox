from pathlib import Path

import nibabel as nib
import numpy as np


DATA_PATH = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input/data.nii.gz"
)

BVAL_PATH = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input/bvals"
)

MASK_PATH = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input/nodif_brain_mask.nii.gz"
)


# =============================================================================
# LOAD
# =============================================================================

img = nib.load(
    str(DATA_PATH)
)

data = np.asarray(
    img.get_fdata(dtype=np.float32),
    dtype=float,
)

bvals = np.loadtxt(
    BVAL_PATH
).reshape(-1)

mask = (
    nib.load(
        str(MASK_PATH)
    )
    .get_fdata()
    > 0
)


# =============================================================================
# b0 VOLUMES
# =============================================================================

b0 = (
    bvals <= 50
)

mean_b0 = np.mean(
    data[..., b0],
    axis=-1,
)


# =============================================================================
# BRAIN-ONLY STATISTICS
# =============================================================================

values = mean_b0[
    mask
    & np.isfinite(mean_b0)
    & (mean_b0 > 0)
]


print(
    f"Brain voxels: {len(values)}"
)

print(
    f"b0 volumes:   {b0.sum()}"
)

print("\nBrain mean-b0 intensity:")

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
        f"  {percentile:2d}%: "
        f"{np.percentile(values, percentile):.2f}"
    )


# =============================================================================
# GLOBAL REFERENCE SIGNAL
# =============================================================================

S_REF = np.median(
    values
)

print(
    f"\nGlobal reference signal: {S_REF:.2f}"
)


# =============================================================================
# WHAT WOULD S0 LOOK LIKE AFTER SCALING?
# =============================================================================

scaled_b0 = (
    values
    / S_REF
)

print(
    "\nExpected S0 after global scaling:"
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
        f"  {percentile:2d}%: "
        f"{np.percentile(scaled_b0, percentile):.3f}"
    )

# =============================================================================
# GLOBAL SIGNAL SCALING
# =============================================================================

S_REF = np.median(
    values
)

scaled_data = (
    data / S_REF
).astype(
    np.float32
)


# =============================================================================
# SAVE
# =============================================================================

SCALED_DATA_PATH = (
    DATA_PATH.parent
    / "data_gibbs_eddy_reoriented_global_scaled.nii.gz"
)

header = img.header.copy()

header.set_data_dtype(
    np.float32
)

scaled_img = nib.Nifti1Image(
    scaled_data,
    img.affine,
    header,
)

nib.save(
    scaled_img,
    str(
        SCALED_DATA_PATH
    ),
)


print(
    f"\nSaved globally scaled dMRI:\n"
    f"{SCALED_DATA_PATH}"
)

print(
    f"\nScaling factor = {S_REF:.4f}"
)

SCALE_INFO_PATH = (
    DATA_PATH.parent
    / "data_gibbs_eddy_reoriented_global_scaled.txt"
)

with open(
    SCALE_INFO_PATH,
    "w"
) as f:

    f.write(
        f"Global scaling factor: {S_REF:.8f}\n"
    )

    f.write(
        "Definition: median voxelwise mean-b0 "
        "intensity within brain mask\n"
    )

    f.write(
        f"Number of brain voxels: "
        f"{len(values)}\n"
    )

    f.write(
        f"Number of b0 volumes: "
        f"{b0.sum()}\n"
    )

    