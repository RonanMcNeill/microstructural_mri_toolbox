import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

d_in = 2.2

# Fraction ranges
f_soma = np.linspace(0.0, 0.4, 201)
f_neurite = np.linspace(0.1, 0.7, 201)

# Fixed free-water fractions to compare
f_fw_values = [0.05, 0.15, 0.25, 0.35]

# Create 2D grid
FS, FN = np.meshgrid(f_soma, f_neurite)


# ============================================================
# Tortuosity model
# ============================================================

def calculate_diffusivities(FS, FN, f_fw, d_in=2.2):
    """
    Calculate extracellular parallel and perpendicular diffusivities.

    Model:

        d_ex_parallel =
            d_in * (1 - f_soma / (1 - f_neurite - f_fw))

        d_ex_perpendicular =
            d_ex_parallel
            * (1 - f_neurite / (1 - f_soma - f_fw))

    Assumes alpha_s = alpha_n = 1.
    """

    # Remaining extracellular fraction
    f_ex = 1.0 - FS - FN - f_fw

    # Only retain physically possible compartment combinations
    valid = f_ex > 0

    with np.errstate(divide="ignore", invalid="ignore"):

        # Soma-related obstruction
        soma_factor = 1.0 - FS / (1.0 - FN - f_fw)

        # Neurite-related transverse obstruction
        neurite_factor = 1.0 - FN / (1.0 - FS - f_fw)

        # Parallel extracellular diffusivity
        d_ex_parallel = d_in * soma_factor

        # Perpendicular extracellular diffusivity
        d_ex_perpendicular = (
            d_in
            * soma_factor
            * neurite_factor
        )

    # Mask physically impossible combinations
    d_ex_parallel = np.where(valid, d_ex_parallel, np.nan)
    d_ex_perpendicular = np.where(
        valid,
        d_ex_perpendicular,
        np.nan
    )

    return d_ex_parallel, d_ex_perpendicular, f_ex


# ============================================================
# Plot heatmaps
# ============================================================

fig, axes = plt.subplots(
    2,
    len(f_fw_values),
    figsize=(16, 8),
    sharex=True,
    sharey=True
)

for col, f_fw in enumerate(f_fw_values):

    d_par, d_perp, f_ex = calculate_diffusivities(
        FS,
        FN,
        f_fw,
        d_in=d_in
    )

    # --------------------------------------------------------
    # Parallel diffusivity
    # --------------------------------------------------------

    im_par = axes[0, col].pcolormesh(
        FS,
        FN,
        d_par,
        shading="auto",
        vmin=0,
        vmax=d_in
    )

    axes[0, col].set_title(
        rf"$f_{{fw}}={f_fw:.2f}$"
    )

    # --------------------------------------------------------
    # Perpendicular diffusivity
    # --------------------------------------------------------

    im_perp = axes[1, col].pcolormesh(
        FS,
        FN,
        d_perp,
        shading="auto",
        vmin=0,
        vmax=d_in
    )

    # --------------------------------------------------------
    # Mark boundary where f_ex = 0
    # --------------------------------------------------------

    # f_soma + f_neurite + f_fw = 1
    boundary = 1.0 - f_fw - f_soma

    valid_boundary = (
        (boundary >= f_neurite.min())
        & (boundary <= f_neurite.max())
    )

    for row in range(2):
        axes[row, col].plot(
            f_soma[valid_boundary],
            boundary[valid_boundary],
            "--",
            linewidth=1.5
        )

        axes[row, col].set_xlim(0, 0.4)
        axes[row, col].set_ylim(0.1, 0.7)

    axes[1, col].set_xlabel(
        r"$f_{\mathrm{soma}}$"
    )


# ============================================================
# Labels
# ============================================================

axes[0, 0].set_ylabel(
    r"$f_{\mathrm{neurite}}$"
    "\n\n"
    r"$d_{ex\parallel}$"
)

axes[1, 0].set_ylabel(
    r"$f_{\mathrm{neurite}}$"
    "\n\n"
    r"$d_{ex\perp}$"
)


# ============================================================
# Shared colour bar
# ============================================================

cbar = fig.colorbar(
    im_perp,
    ax=axes,
    shrink=0.85,
    pad=0.02
)

cbar.set_label(
    r"Extracellular diffusivity ($\mu m^2/ms$)"
)


# ============================================================
# Overall title
# ============================================================

fig.suptitle(
    rf"Tortuosity model with $d_{{in}}={d_in}$"
    "\n"
    r"Dashed line: $f_{ex}=0$",
    fontsize=14
)

plt.show()