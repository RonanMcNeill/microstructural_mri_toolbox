"""Regression values generated with FSL DIVE commit f75b7e0.

These tests intentionally combine DIVE comparisons with tighter independent
tests elsewhere. DIVE uses 100-direction numerical spherical means and a
slightly rounded gyromagnetic-ratio constant, so tolerances reflect those
specific reference limitations.
"""

import numpy as np

from dmri_models import Acquisition, Ball, Sphere, Stick, Tensor, Zeppelin


B = np.array([0.0, 0.5, 1.0, 2.0])
GRADIENTS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    ]
)


def test_pgse_gradient_scale_matches_dive() -> None:
    acquisition = Acquisition.from_pgse(
        B,
        b_unit="ms/um^2",
        pulse_duration=30,
        diffusion_time=50,
    )
    dive_gradient_t_per_m = (
        np.array([0, 13.930846274693657, 19.70119173700648, 27.861692549387314])
        * 1e-3
    )
    np.testing.assert_allclose(
        acquisition.gradient_strength,
        dive_gradient_t_per_m,
        rtol=1e-5,
        atol=1e-12,
    )


def test_directional_gaussian_compartments_match_dive() -> None:
    acquisition = Acquisition.from_b_values(
        B,
        b_unit="ms/um^2",
        gradients=GRADIENTS,
    )
    np.testing.assert_allclose(
        Ball("d").signal(acquisition, {"d": 1.2}),
        [1, 0.5488116360940264, 0.30119421191220214, 0.09071795328941251],
    )
    np.testing.assert_allclose(
        Stick("d", "axis").signal(
            acquisition,
            {"d": 1.7, "axis": np.array([0, 0, 1])},
        ),
        [1, 1, 0.18268352405273466, 0.32195827153767587],
    )
    np.testing.assert_allclose(
        Zeppelin("d_parallel", "d_perpendicular", "axis").signal(
            acquisition,
            {
                "d_parallel": 1.7,
                "d_perpendicular": 0.5,
                "axis": np.array([0, 0, 1]),
            },
        ),
        [1, 0.7788007830714049, 0.18268352405273466, 0.16529888822158656],
    )
    np.testing.assert_allclose(
        Tensor(("d_x", "d_y", "d_z")).signal(
            acquisition,
            {"d_x": 0.3, "d_y": 0.8, "d_z": 1.7},
        ),
        [1, 0.6703200460356393, 0.18268352405273466, 0.15463826454925475],
    )


def test_sphere_matches_dive_when_given_identical_physical_gradient() -> None:
    dive_gradient_t_per_m = (
        np.array([0, 13.930846274693657, 19.70119173700648, 27.861692549387314])
        * 1e-3
    )
    acquisition = Acquisition.from_b_values(
        B,
        b_unit="ms/um^2",
        pulse_duration=30,
        diffusion_time=50,
        gradient_strength=dive_gradient_t_per_m,
    )
    np.testing.assert_allclose(
        Sphere("radius", "d").signal(acquisition, {"radius": 10.0, "d": 3.0}),
        [1, 0.9103240457200821, 0.8286898682161780, 0.6867268976841464],
        rtol=2e-14,
        atol=1e-15,
    )


def test_analytical_means_agree_with_dive_numerical_means() -> None:
    acquisition = Acquisition.from_b_values(B, b_unit="ms/um^2")
    np.testing.assert_allclose(
        Stick("d", powder_average=True).spherical_mean(acquisition, {"d": 1.7}),
        [1, 0.7786488333446249, 0.6390421201497767, 0.4815018726977640],
        rtol=1.5e-2,
    )
    np.testing.assert_allclose(
        Zeppelin("d_parallel", "d_perpendicular", "axis").spherical_mean(
            acquisition,
            {
                "d_parallel": 1.7,
                "d_perpendicular": 0.5,
                "axis": np.array([0, 0, 1]),
            },
        ),
        [1, 0.6488107600781332, 0.4329072746265550, 0.2060968532697298],
        rtol=1.5e-2,
    )
    np.testing.assert_allclose(
        Tensor(("d_x", "d_y", "d_z")).spherical_mean(
            acquisition,
            {"d_x": 0.3, "d_y": 0.8, "d_z": 1.7},
        ),
        [1, 0.6377959881875571, 0.41955795042189054, 0.1969240792125696],
        rtol=1.5e-2,
    )
