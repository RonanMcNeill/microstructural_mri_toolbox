"""Reusable plotting utilities for diffusion MRI model exploration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .compartments import SphericalHarmonicFOD


def _get_matplotlib_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "matplotlib is required for plotting. Install the package with the "
            "'diagnostics' or 'notebooks' extras."
        ) from exc
    return plt


def fod_surface_mesh(
    fod: SphericalHarmonicFOD,
    coefficients: np.ndarray,
    *,
    coefficient_name: str | None = None,
    n_theta: int = 120,
    n_phi: int = 240,
    rectify_density: bool = True,
) -> dict[str, np.ndarray]:
    """Build a 3D FOD surface mesh from SH coefficients.

    Returns a dictionary with Cartesian coordinates (x, y, z), the raw density,
    and the radial map used for plotting.
    """
    key = coefficient_name or fod.coefficients
    theta_grid, phi_grid, directions = SphericalHarmonicFOD.spherical_grid(
        n_theta=n_theta,
        n_phi=n_phi,
    )
    density = np.asarray(fod.density(directions, {key: coefficients}), dtype=float)
    if rectify_density:
        density = np.clip(density, 0.0, None)

    radii = density.reshape(theta_grid.shape)
    x = radii * np.sin(theta_grid) * np.cos(phi_grid)
    y = radii * np.sin(theta_grid) * np.sin(phi_grid)
    z = radii * np.cos(theta_grid)

    return {
        "theta": theta_grid,
        "phi": phi_grid,
        "density": density,
        "radii": radii,
        "x": x,
        "y": y,
        "z": z,
    }


def plot_fod_surface(
    title: str,
    fod: SphericalHarmonicFOD,
    coefficients: np.ndarray,
    *,
    coefficient_name: str | None = None,
    n_theta: int = 120,
    n_phi: int = 240,
    colour_mode: str = "multid_xyz",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (7.0, 6.0),
    show: bool = True,
):
    """Plot an SH-FOD surface.

    Parameters
    ----------
    colour_mode:
        "multid_xyz" reproduces the MultiD/SMI-style RGB mapping based on
        absolute Cartesian surface coordinates.
        "colormap" uses a standard scalar colormap over radial amplitude.
    """
    plt = _get_matplotlib_pyplot()
    mesh = fod_surface_mesh(
        fod,
        coefficients,
        coefficient_name=coefficient_name,
        n_theta=n_theta,
        n_phi=n_phi,
        rectify_density=True,
    )

    radii = mesh["radii"]
    if colour_mode == "multid_xyz":
        stacked = np.stack((np.abs(mesh["x"]), np.abs(mesh["y"]), np.abs(mesh["z"])), axis=2)
        normaliser = np.max(stacked, axis=2, keepdims=True)
        normaliser = np.maximum(normaliser, 1e-12)
        colours = stacked / normaliser
    elif colour_mode == "colormap":
        normalised = (radii - radii.min()) / (radii.max() - radii.min() + 1e-12)
        colours = getattr(plt.cm, cmap)(normalised)
    else:
        raise ValueError("colour_mode must be 'multid_xyz' or 'colormap'.")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mesh["x"],
        mesh["y"],
        mesh["z"],
        facecolors=colours,
        linewidth=0.0,
        antialiased=True,
    )
    ax.set_title(title)
    ax.set_axis_off()

    r_max = float(np.max(radii))
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)

    if show:
        plt.show()
    return ax


def plot_signal_agreement(
    clean_signal: np.ndarray,
    *,
    observed_signal: np.ndarray | None = None,
    reconstructed_signal: np.ndarray | None = None,
    title: str = "Signal agreement",
    figsize: tuple[float, float] = (6.0, 5.0),
    show: bool = True,
):
    """Scatter agreement plot against a clean reference signal."""
    plt = _get_matplotlib_pyplot()

    clean = np.asarray(clean_signal, dtype=float).reshape(-1)
    if observed_signal is None and reconstructed_signal is None:
        raise ValueError("At least one of observed_signal or reconstructed_signal must be provided.")

    fig, ax = plt.subplots(figsize=figsize)
    if observed_signal is not None:
        observed = np.asarray(observed_signal, dtype=float).reshape(-1)
        if observed.size != clean.size:
            raise ValueError("observed_signal must match clean_signal length.")
        ax.scatter(clean, observed, s=14, alpha=0.6, label="Observed")

    if reconstructed_signal is not None:
        reconstructed = np.asarray(reconstructed_signal, dtype=float).reshape(-1)
        if reconstructed.size != clean.size:
            raise ValueError("reconstructed_signal must match clean_signal length.")
        ax.scatter(clean, reconstructed, s=14, alpha=0.6, label="Reconstructed")

    min_v = float(np.min(clean))
    max_v = float(np.max(clean))
    ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)
    ax.set_xlabel("Clean signal")
    ax.set_ylabel("Estimated / observed signal")
    ax.set_title(title)
    ax.legend()

    if show:
        plt.show()
    return ax
