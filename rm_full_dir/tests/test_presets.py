import numpy as np
import pytest

from dmri_models import (
    Acquisition,
    noddi,
    noddi_derived_parameters,
    odi_from_kappa,
    sandi,
)


def test_sandi_regression_with_corrected_b_to_gradient_conversion() -> None:
    acquisition = Acquisition.from_pgse(
        [0, 0.5, 1.0, 2.0],
        b_unit="ms/um^2",
        pulse_duration=30,
        diffusion_time=50,
    )
    preset = sandi()
    signal = preset.model.signal(acquisition, preset.parameters.initial_values())
    # Generated from the source SANDI compartment functions after converting
    # canonical b-values to SI correctly when deriving gradient strength.
    np.testing.assert_allclose(
        signal,
        [
            1.0,
            0.7135372527212032,
            0.5548043759521122,
            0.3973717354979421,
        ],
        rtol=1e-13,
        atol=1e-15,
    )


def test_sandi_fractions_are_a_single_simplex() -> None:
    preset = sandi()
    values = preset.parameters.initial_values()
    fractions = [
        values["soma_fraction"],
        values["neurite_fraction"],
        values["extra_fraction"],
    ]
    assert sum(fractions) == 1.0
    assert all(fraction > 0 for fraction in fractions)


def test_sandi_sphere_diffusivity_is_fixed_by_default_and_can_vary() -> None:
    fixed = sandi()
    varying = sandi(vary_sphere_diffusivity=True)
    assert "d_sphere" not in fixed.parameters.varying_names
    assert "d_sphere" in varying.parameters.varying_names


def test_noddi_defaults_include_three_compartments_and_fixed_diffusivities() -> None:
    preset = noddi()
    values = preset.parameters.initial_values()
    assert len(preset.model.components) == 3
    assert preset.model.components[0].name == "neurite"
    assert preset.model.components[1].name == "extra"
    assert preset.model.components[2].name == "csf"
    assert np.isclose(
        values["neurite_fraction"] + values["extra_fraction"] + values["csf_fraction"],
        1.0,
    )
    assert "d_in" not in preset.parameters.varying_names
    assert "kappa" in preset.parameters.varying_names
    assert "theta" in preset.parameters.varying_names
    assert "phi" in preset.parameters.varying_names
    assert "d_csf" not in preset.parameters.varying_names


def test_noddi_supports_bingham_fod_parameterisation() -> None:
    preset = noddi(fod_model="bingham")
    values = preset.parameters.initial_values()
    assert "kappa1" in values
    assert "kappa2" in values
    assert "psi" in values


def test_noddi_without_csf_uses_two_fraction_simplex() -> None:
    preset = noddi(include_csf=False)
    values = preset.parameters.initial_values()
    assert len(preset.model.components) == 2
    assert set(values) >= {"neurite_fraction", "extra_fraction"}
    assert "csf_fraction" not in values
    assert np.isclose(values["neurite_fraction"] + values["extra_fraction"], 1.0)


def test_noddi_can_expose_diffusivities_to_the_optimiser() -> None:
    preset = noddi(
        vary_intracellular_diffusivity=True,
        vary_csf_diffusivity=True,
        vary_kappa=True,
    )
    assert "d_in" in preset.parameters.varying_names
    assert "d_csf" in preset.parameters.varying_names
    assert "kappa" in preset.parameters.varying_names


def test_odi_from_kappa_behaviour() -> None:
    assert odi_from_kappa(0.0) == pytest.approx(1.0)
    assert odi_from_kappa(1e6) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="kappa"):
        odi_from_kappa(-1.0)


def test_noddi_derived_parameter_utility() -> None:
    derived = noddi_derived_parameters(
        {
            "kappa": 1.0,
            "neurite_fraction": 0.6,
            "csf_fraction": 0.1,
        }
    )
    assert "odi" in derived
    assert derived["odi"] == pytest.approx(odi_from_kappa(1.0))
    assert derived["v_ic_tissue"] == pytest.approx(0.6 / 0.9)

