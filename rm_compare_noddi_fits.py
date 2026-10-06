# =============================================================================
# COMPREHENSIVE COMPARISON OF TWO NODDI FITS
#
# Compares:
#   - fitted parameter values
#   - convergence diagnostics
#   - voxelwise differences
#   - correlations
#   - relative errors
#   - side-by-side spatial maps
#
# For each parameter:
#
#       REFERENCE | TEST | DIFFERENCE
#
# Reference and test use identical display limits.
# Difference maps use symmetric limits around zero.
#
# =============================================================================

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# 1. DIRECTORIES
# =============================================================================

REFERENCE_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/model_fitting/noddi_reference"
)

TEST_DIR = Path(
    "/Users/Ronan/Documents/PhD/Data/model_fitting/noddi_tol1e6"
)

OUTPUT_DIR = (
    TEST_DIR
    / "comparison_to_reference"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

PARAMETERS = [
    "S0",
    "theta",
    "phi",
    "kappa",
    "neurite_fraction",
    "extra_fraction",
    "csf_fraction",
]


PARAMETER_LABELS = {
    "S0": r"$S_0$",
    "theta": r"$\theta$",
    "phi": r"$\phi$",
    "kappa": r"$\kappa$",
    "neurite_fraction": r"$f_n$",
    "extra_fraction": r"$f_{ex}$",
    "csf_fraction": r"$f_{FW}$",
}


# =============================================================================
# 3. DIAGNOSTICS
# =============================================================================

DIAGNOSTICS = [
    "cost",
    "success",
    "n_evaluations",
]


# =============================================================================
# 4. HELPERS
# =============================================================================

def load_map(
    directory,
    name,
):

    path = (
        directory
        / f"noddi.{name}.nii.gz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find:\n{path}"
        )

    img = nib.load(
        str(path)
    )

    data = np.asarray(
        img.get_fdata(),
        dtype=float,
    )

    return img, data


def angular_difference(
    test,
    reference,
    period,
):
    """
    Signed shortest angular difference.

    Returns values in:
        [-period/2, +period/2)
    """

    return (
        (
            test
            - reference
            + period / 2
        )
        % period
        - period / 2
    )


def calculate_difference(
    parameter,
    test,
    reference,
):
    """
    Calculate appropriate signed difference.

    Fractions etc:
        test - reference

    theta:
        wrapped with period pi

    phi:
        wrapped with period pi because the fibre axis is antipodally
        symmetric in the current orientation convention.
    """

    if parameter == "theta":

        return angular_difference(
            test,
            reference,
            np.pi,
        )

    elif parameter == "phi":

        return angular_difference(
            test,
            reference,
            np.pi,
        )

    else:

        return (
            test
            - reference
        )


# =============================================================================
# 5. CHECK MAP SHAPES
# =============================================================================

reference_img, first_reference = load_map(
    REFERENCE_DIR,
    PARAMETERS[0],
)

_, first_test = load_map(
    TEST_DIR,
    PARAMETERS[0],
)


if (
    first_reference.shape
    != first_test.shape
):

    raise ValueError(
        "Reference and test volumes have "
        "different spatial shapes."
    )


spatial_shape = (
    first_reference.shape
)


print(
    "\nSpatial shape:",
    spatial_shape,
)


# =============================================================================
# 6. FIND COMMON FITTED VOXELS
#
# Use S0 to establish where both fits contain finite results.
# =============================================================================

common_mask = (
    np.isfinite(
        first_reference
    )
    & np.isfinite(
        first_test
    )
)


n_common = int(
    np.sum(
        common_mask
    )
)


print(
    f"Common fitted voxels: "
    f"{n_common}"
)


if n_common == 0:

    raise ValueError(
        "No common finite fitted voxels "
        "between reference and test."
    )


# =============================================================================
# 7. AUTOMATICALLY CHOOSE DISPLAY SLICE
#
# Find the slice along each dimension containing the largest number
# of fitted voxels.
#
# Then choose the orientation with the largest peak voxel count.
# =============================================================================

counts_axis0 = np.sum(
    common_mask,
    axis=(1, 2),
)

counts_axis1 = np.sum(
    common_mask,
    axis=(0, 2),
)

counts_axis2 = np.sum(
    common_mask,
    axis=(0, 1),
)


best_indices = [
    int(
        np.argmax(
            counts_axis0
        )
    ),
    int(
        np.argmax(
            counts_axis1
        )
    ),
    int(
        np.argmax(
            counts_axis2
        )
    ),
]


best_counts = [
    int(
        np.max(
            counts_axis0
        )
    ),
    int(
        np.max(
            counts_axis1
        )
    ),
    int(
        np.max(
            counts_axis2
        )
    ),
]


DISPLAY_AXIS = int(
    np.argmax(
        best_counts
    )
)

DISPLAY_SLICE = (
    best_indices[
        DISPLAY_AXIS
    ]
)


print(
    "\nAutomatic display selection:"
)

print(
    f"  axis:  {DISPLAY_AXIS}"
)

print(
    f"  slice: {DISPLAY_SLICE}"
)

print(
    f"  fitted voxels in slice: "
    f"{best_counts[DISPLAY_AXIS]}"
)


# =============================================================================
# 8. SLICE HELPER
# =============================================================================

def extract_slice(
    volume,
    axis,
    index,
):

    if axis == 0:

        return volume[
            index,
            :,
            :
        ]

    elif axis == 1:

        return volume[
            :,
            index,
            :
        ]

    elif axis == 2:

        return volume[
            :,
            :,
            index
        ]

    else:

        raise ValueError(
            "axis must be 0, 1 or 2"
        )


# =============================================================================
# 9. PARAMETER COMPARISON
# =============================================================================

summary_rows = []


for parameter in PARAMETERS:

    print(
        "\n============================================"
    )

    print(
        parameter
    )

    print(
        "============================================"
    )


    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    _, reference = load_map(
        REFERENCE_DIR,
        parameter,
    )

    _, test = load_map(
        TEST_DIR,
        parameter,
    )


    if (
        reference.shape
        != test.shape
    ):

        raise ValueError(
            f"Shape mismatch for {parameter}"
        )


    # -------------------------------------------------------------------------
    # Common valid voxels
    # -------------------------------------------------------------------------

    valid = (
        np.isfinite(
            reference
        )
        & np.isfinite(
            test
        )
    )


    ref_values = (
        reference[
            valid
        ]
    )

    test_values = (
        test[
            valid
        ]
    )


    # -------------------------------------------------------------------------
    # Difference
    # -------------------------------------------------------------------------

    difference = calculate_difference(
        parameter,
        test_values,
        ref_values,
    )


    absolute_difference = np.abs(
        difference
    )


    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    bias = float(
        np.mean(
            difference
        )
    )

    mae = float(
        np.mean(
            absolute_difference
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )

    median_abs_difference = float(
        np.median(
            absolute_difference
        )
    )

    p95_abs_difference = float(
        np.percentile(
            absolute_difference,
            95,
        )
    )

    max_abs_difference = float(
        np.max(
            absolute_difference
        )
    )


    # -------------------------------------------------------------------------
    # Correlation
    # -------------------------------------------------------------------------

    if (
        np.std(ref_values) > 0
        and np.std(test_values) > 0
    ):

        correlation = float(
            np.corrcoef(
                ref_values,
                test_values,
            )[0, 1]
        )

    else:

        correlation = np.nan


    # -------------------------------------------------------------------------
    # Relative MAE
    #
    # Normalise by the dynamic range of the reference values.
    #
    # This is preferable to dividing voxelwise by values that may approach 0.
    # -------------------------------------------------------------------------

    reference_range = (
        np.max(
            ref_values
        )
        - np.min(
            ref_values
        )
    )


    if reference_range > 0:

        normalized_mae = (
            mae
            / reference_range
        )

    else:

        normalized_mae = np.nan


    # -------------------------------------------------------------------------
    # Store
    # -------------------------------------------------------------------------

    summary_rows.append({

        "parameter":
            parameter,

        "n_voxels":
            len(
                ref_values
            ),

        "reference_mean":
            np.mean(
                ref_values
            ),

        "test_mean":
            np.mean(
                test_values
            ),

        "bias":
            bias,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "median_abs_difference":
            median_abs_difference,

        "p95_abs_difference":
            p95_abs_difference,

        "max_abs_difference":
            max_abs_difference,

        "normalized_MAE":
            normalized_mae,

        "correlation":
            correlation,
    })


    # -------------------------------------------------------------------------
    # Print
    # -------------------------------------------------------------------------

    print(
        f"Voxels:                 "
        f"{len(ref_values)}"
    )

    print(
        f"Reference mean:          "
        f"{np.mean(ref_values):.6g}"
    )

    print(
        f"Test mean:               "
        f"{np.mean(test_values):.6g}"
    )

    print(
        f"Bias:                    "
        f"{bias:.6g}"
    )

    print(
        f"MAE:                     "
        f"{mae:.6g}"
    )

    print(
        f"RMSE:                    "
        f"{rmse:.6g}"
    )

    print(
        f"Median |difference|:     "
        f"{median_abs_difference:.6g}"
    )

    print(
        f"95th % |difference|:     "
        f"{p95_abs_difference:.6g}"
    )

    print(
        f"Maximum |difference|:    "
        f"{max_abs_difference:.6g}"
    )

    print(
        f"Normalized MAE:          "
        f"{100*normalized_mae:.4f}%"
    )

    print(
        f"Correlation:             "
        f"{correlation:.6f}"
    )


# =============================================================================
# 10. SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


print(
    "\n\n============================================"
)

print(
    "PARAMETER SUMMARY"
)

print(
    "============================================\n"
)


print(
    summary_df.to_string(
        index=False
    )
)


summary_path = (
    OUTPUT_DIR
    / "parameter_comparison.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
)


print(
    f"\nSaved:\n{summary_path}"
)


# =============================================================================
# 11. DIAGNOSTIC COMPARISON
# =============================================================================

diagnostic_rows = []


for diagnostic in DIAGNOSTICS:

    _, reference = load_map(
        REFERENCE_DIR,
        diagnostic,
    )

    _, test = load_map(
        TEST_DIR,
        diagnostic,
    )


    valid = (
        np.isfinite(
            reference
        )
        & np.isfinite(
            test
        )
    )


    ref_values = (
        reference[
            valid
        ]
    )

    test_values = (
        test[
            valid
        ]
    )


    if diagnostic == "success":

        diagnostic_rows.append({

            "diagnostic":
                diagnostic,

            "reference":
                np.mean(
                    ref_values
                ),

            "test":
                np.mean(
                    test_values
                ),

            "difference":
                np.mean(
                    test_values
                )
                - np.mean(
                    ref_values
                ),
        })


    else:

        diagnostic_rows.append({

            "diagnostic":
                diagnostic,

            "reference":
                np.median(
                    ref_values
                ),

            "test":
                np.median(
                    test_values
                ),

            "difference":
                np.median(
                    test_values
                )
                - np.median(
                    ref_values
                ),
        })


diagnostic_df = pd.DataFrame(
    diagnostic_rows
)


print(
    "\n\n============================================"
)

print(
    "CONVERGENCE DIAGNOSTICS"
)

print(
    "============================================\n"
)


print(
    diagnostic_df.to_string(
        index=False
    )
)


diagnostic_df.to_csv(
    OUTPUT_DIR
    / "diagnostic_comparison.csv",
    index=False,
)


# =============================================================================
# 12. SIDE-BY-SIDE PARAMETER MAPS
#
# One figure per parameter:
#
#     reference | test | difference
#
# =============================================================================

for parameter in PARAMETERS:

    _, reference = load_map(
        REFERENCE_DIR,
        parameter,
    )

    _, test = load_map(
        TEST_DIR,
        parameter,
    )


    # -------------------------------------------------------------------------
    # Full difference volume
    # -------------------------------------------------------------------------

    difference = np.full(
        reference.shape,
        np.nan,
        dtype=float,
    )


    valid = (
        np.isfinite(
            reference
        )
        & np.isfinite(
            test
        )
    )


    difference[
        valid
    ] = calculate_difference(
        parameter,
        test[
            valid
        ],
        reference[
            valid
        ],
    )


    # -------------------------------------------------------------------------
    # Extract display slice
    # -------------------------------------------------------------------------

    ref_slice = extract_slice(
        reference,
        DISPLAY_AXIS,
        DISPLAY_SLICE,
    )

    test_slice = extract_slice(
        test,
        DISPLAY_AXIS,
        DISPLAY_SLICE,
    )

    diff_slice = extract_slice(
        difference,
        DISPLAY_AXIS,
        DISPLAY_SLICE,
    )


    # -------------------------------------------------------------------------
    # Shared reference/test display limits
    #
    # Use robust 2–98 percentile limits to prevent isolated outliers from
    # dominating the display.
    # -------------------------------------------------------------------------

    combined = np.concatenate([
        reference[
            np.isfinite(
                reference
            )
        ],
        test[
            np.isfinite(
                test
            )
        ],
    ])


    vmin = np.percentile(
        combined,
        2,
    )

    vmax = np.percentile(
        combined,
        98,
    )


    # -------------------------------------------------------------------------
    # Symmetric difference scale
    #
    # Use 98th percentile of |difference|.
    # -------------------------------------------------------------------------

    finite_difference = (
        difference[
            np.isfinite(
                difference
            )
        ]
    )


    diff_limit = np.percentile(
        np.abs(
            finite_difference
        ),
        98,
    )


    if (
        not np.isfinite(
            diff_limit
        )
        or diff_limit == 0
    ):

        diff_limit = 1e-12


    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            12,
            4.2,
        ),
    )


    # Reference
    image_reference = axes[0].imshow(
        np.rot90(
            ref_slice
        ),
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(
        "Reference\n"
        r"$10^{-10}$"
    )


    # Test
    image_test = axes[1].imshow(
        np.rot90(
            test_slice
        ),
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title(
        "Test\n"
        r"$10^{-6}$"
    )


    # Difference
    image_difference = axes[2].imshow(
        np.rot90(
            diff_slice
        ),
        vmin=-diff_limit,
        vmax=diff_limit,
        cmap="coolwarm",
    )

    axes[2].set_title(
        "Difference\n"
        r"test $-$ reference"
    )


    # -------------------------------------------------------------------------
    # Remove axes
    # -------------------------------------------------------------------------

    for ax in axes:

        ax.axis(
            "off"
        )


    # -------------------------------------------------------------------------
    # Shared parameter colourbar
    # -------------------------------------------------------------------------

    cbar_parameter = fig.colorbar(
        image_reference,
        ax=axes[:2],
        fraction=0.035,
        pad=0.02,
    )

    cbar_parameter.set_label(
        PARAMETER_LABELS[
            parameter
        ],
        fontsize=11,
    )


    # -------------------------------------------------------------------------
    # Difference colourbar
    # -------------------------------------------------------------------------

    cbar_difference = fig.colorbar(
        image_difference,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )

    cbar_difference.set_label(
        "Difference",
        fontsize=11,
    )


    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------

    fig.suptitle(
        PARAMETER_LABELS[
            parameter
        ],
        fontsize=15,
        y=0.98,
    )


    fig.subplots_adjust(
        left=0.02,
        right=0.95,
        bottom=0.05,
        top=0.84,
        wspace=0.08,
    )


    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    figure_path = (
        OUTPUT_DIR
        / (
            f"comparison_"
            f"{parameter}.png"
        )
    )


    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.show()

    plt.close(
        fig
    )


# =============================================================================
# 13. SCATTER PLOTS
#
# Reference vs test provides another useful check for equivalence.
# =============================================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(
        16,
        8,
    ),
)

axes = axes.flatten()


for ax, parameter in zip(
    axes,
    PARAMETERS,
):

    _, reference = load_map(
        REFERENCE_DIR,
        parameter,
    )

    _, test = load_map(
        TEST_DIR,
        parameter,
    )


    valid = (
        np.isfinite(
            reference
        )
        & np.isfinite(
            test
        )
    )


    x = reference[
        valid
    ]

    y = test[
        valid
    ]


    ax.scatter(
        x,
        y,
        s=18,
        alpha=0.5,
        edgecolors="none",
    )


    value_min = min(
        np.min(x),
        np.min(y),
    )

    value_max = max(
        np.max(x),
        np.max(y),
    )


    ax.plot(
        [
            value_min,
            value_max,
        ],
        [
            value_min,
            value_max,
        ],
        linestyle="--",
        linewidth=1.2,
    )


    ax.set_xlim(
        value_min,
        value_max,
    )

    ax.set_ylim(
        value_min,
        value_max,
    )


    ax.set_aspect(
        "equal",
        adjustable="box",
    )


    ax.set_xlabel(
        "Reference"
    )

    ax.set_ylabel(
        "Test"
    )


    ax.set_title(
        PARAMETER_LABELS[
            parameter
        ]
    )


    ax.grid(
        alpha=0.15,
    )


# Hide unused panel
for ax in axes[
    len(PARAMETERS):
]:

    ax.set_visible(
        False
    )


fig.tight_layout()


scatter_path = (
    OUTPUT_DIR
    / "reference_vs_test_scatter.png"
)


fig.savefig(
    scatter_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


print(
    f"\nSaved scatter comparison:\n"
    f"{scatter_path}"
)