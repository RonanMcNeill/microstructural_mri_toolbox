from pathlib import Path
import nibabel as nib
import numpy as np

DATA_PATH = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input/data.nii.gz"
)

BVAL_PATH = Path(
    "/Users/Ronan/Documents/PhD/Data/80_post_fitting_input/bvals"
)

img = nib.load(str(DATA_PATH))

data = np.asarray(
    img.get_fdata(dtype=np.float32),
    dtype=float,
)

bvals = np.loadtxt(
    BVAL_PATH
).reshape(-1)

# conventional FSL b-values: s/mm²
b0 = bvals <= 50

print(
    f"Number of b0 volumes: {b0.sum()}"
)

mean_b0 = np.mean(
    data[..., b0],
    axis=-1,
)

# Ignore zero/background voxels for this quick diagnostic
positive = (
    np.isfinite(mean_b0)
    & (mean_b0 > 0)
)

values = mean_b0[
    positive
]

print("\nMean b0 intensity:")
print(
    f"  min:     {np.min(values):.4f}"
)
print(
    f"  1%:      {np.percentile(values, 1):.4f}"
)
print(
    f"  5%:      {np.percentile(values, 5):.4f}"
)
print(
    f"  median:  {np.median(values):.4f}"
)
print(
    f"  95%:     {np.percentile(values, 95):.4f}"
)
print(
    f"  99%:     {np.percentile(values, 99):.4f}"
)
print(
    f"  max:     {np.max(values):.4f}"
)