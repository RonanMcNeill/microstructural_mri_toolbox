import numpy as np
import pytest

from dmri_models import fod_surface_mesh
from dmri_models.compartments import SphericalHarmonicFOD


def test_fod_surface_mesh_shapes() -> None:
    fod = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=False)
    coefficients = np.zeros(SphericalHarmonicFOD.coefficient_count(4, include_l0=False))

    mesh = fod_surface_mesh(fod, coefficients, coefficient_name="c", n_theta=24, n_phi=48)
    assert mesh["x"].shape == (24, 48)
    assert mesh["y"].shape == (24, 48)
    assert mesh["z"].shape == (24, 48)
    assert mesh["radii"].shape == (24, 48)
    assert mesh["density"].shape == (24 * 48,)


def test_plotting_functions_smoke() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from dmri_models import plot_fod_surface, plot_signal_agreement

    fod = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=False)
    coefficients = np.zeros(SphericalHarmonicFOD.coefficient_count(4, include_l0=False))

    ax_surface = plot_fod_surface(
        "smoke",
        fod,
        coefficients,
        coefficient_name="c",
        n_theta=18,
        n_phi=36,
        colour_mode="multid_xyz",
        show=False,
    )
    assert ax_surface is not None

    ax_surface_colormap = plot_fod_surface(
        "smoke-cmap",
        fod,
        coefficients,
        coefficient_name="c",
        n_theta=18,
        n_phi=36,
        colour_mode="colormap",
        show=False,
    )
    assert ax_surface_colormap is not None

    clean = np.linspace(0.1, 1.0, 20)
    observed = clean + 0.01
    reconstructed = clean - 0.01
    ax_scatter = plot_signal_agreement(
        clean,
        observed_signal=observed,
        reconstructed_signal=reconstructed,
        show=False,
    )
    assert ax_scatter is not None


def test_plot_fod_surface_rejects_unknown_colour_mode() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from dmri_models import plot_fod_surface

    fod = SphericalHarmonicFOD(coefficients="c", l_max=4, include_l0=False)
    coefficients = np.zeros(SphericalHarmonicFOD.coefficient_count(4, include_l0=False))

    with pytest.raises(ValueError, match="colour_mode"):
        plot_fod_surface(
            "bad",
            fod,
            coefficients,
            coefficient_name="c",
            colour_mode="unknown",
            show=False,
        )
