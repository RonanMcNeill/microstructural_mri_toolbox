# =============================================================================
# SUMMARISE NODDI FIT
# =============================================================================

from pathlib import Path

import nibabel as nib
import numpy as np


FIT_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/model_fitting/noddi_tol1e6"
)

NFEV_PATH = (
    FIT_DIR
    / "noddi.n_evaluations.nii.gz"
)

SUCCESS_PATH = (
    FIT_DIR
    / "noddi.success.nii.gz"
)

COST_PATH = (
    FIT_DIR
    / "noddi.cost.nii.gz"
)


nfev = nib.load(
    str(NFEV_PATH)
).get_fdata()

success = nib.load(
    str(SUCCESS_PATH)
).get_fdata()

cost = nib.load(
    str(COST_PATH)
).get_fdata()


valid = (
    np.isfinite(nfev)
    & np.isfinite(success)
)


nfev_values = nfev[
    valid
]

success_values = success[
    valid
]

cost_values = cost[
    valid
]


print("\n===================================")
print("FIT SUMMARY")
print("===================================")

print(
    f"Voxels:  {len(nfev_values)}"
)

print(
    f"Success: "
    f"{100*np.mean(success_values):.1f}%"
)

print("\nn_evaluations:")

for p in [
    0,
    25,
    50,
    75,
    95,
    100,
]:

    print(
        f"  {p:3d}%: "
        f"{np.percentile(nfev_values, p):.0f}"
    )


print("\nCost:")

for p in [
    0,
    25,
    50,
    75,
    95,
    100,
]:

    print(
        f"  {p:3d}%: "
        f"{np.percentile(cost_values, p):.4g}"
    )