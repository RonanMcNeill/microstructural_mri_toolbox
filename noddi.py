"""Neurite Orientation Dispersion and Density Imaging preset."""

from __future__ import annotations

import numpy as np
from typing import Mapping

from ..compartments import (
    Ball,
    BinghamFOD,
    FODConvolved,
    NODDIExtraNeurite,
    Stick,
    WatsonFOD,
)
from ..model import Component, Model
from ..parameters import Parameter, ParameterSpace, Simplex
from .sandi import ModelPreset


def odi_from_kappa(kappa: float) -> float:
    """Convert Watson concentration to NODDI orientation dispersion index."""
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and non-negative.")
    return float((2.0 / np.pi) * np.arctan2(1.0, kappa))


def noddi_derived_parameters(
    values: Mapping[str, float],
) -> dict[str, float]:
    """Return derived NODDI metrics from a fitted parameter dictionary.

    Returned metrics include ODI for Watson fits and tissue intra-neurite
    fraction where relevant keys are available.
    """
    derived: dict[str, float] = {}
    if "kappa" in values:
        derived["odi"] = odi_from_kappa(float(values["kappa"]))
    if "neurite_fraction" in values and "csf_fraction" in values:
        f_neurite = float(values["neurite_fraction"])
        f_csf = float(values["csf_fraction"])
        if f_csf >= 1.0:
            raise ValueError("csf_fraction must be less than one to compute v_ic.")
        derived["v_ic_tissue"] = f_neurite / max(1e-12, 1.0 - f_csf)
    elif "neurite_fraction" in values:
        derived["v_ic_tissue"] = float(values["neurite_fraction"])
    return derived


def noddi(
    *,
    include_csf: bool = True,
    vary_intracellular_diffusivity: bool = False,
    vary_csf_diffusivity: bool = False,
    vary_kappa: bool = True,
    fod_model: str = "watson",
    fod_quadrature_theta: int = 24,
    fod_quadrature_phi: int = 48,
) -> ModelPreset:
    """Create a fitting-ready diffusion-only NODDI preset.

    Notes
    -----
    - The neurite stick and extra-neurite tensor responses are convolved with a
      Watson FOD parameterised by ``kappa`` and mean orientation ``theta``/``phi``.
    - ``kappa`` can be converted to ODI via :func:`odi_from_kappa`.
    - The extra-neurite radial diffusivity follows NODDI tortuosity:
      ``d_perp = d_par * (1 - v_ic)`` with
      ``v_ic = f_neurite / (1 - f_csf)`` when CSF is included.
    """
    if fod_model == "watson":
        fod = WatsonFOD(
            kappa="kappa",
            theta="theta",
            phi="phi",
            n_theta=fod_quadrature_theta,
            n_phi=fod_quadrature_phi,
        )
    elif fod_model == "bingham":
        fod = BinghamFOD(
            kappa1="kappa1",
            kappa2="kappa2",
            theta="theta",
            phi="phi",
            psi="psi",
            n_theta=fod_quadrature_theta,
            n_phi=fod_quadrature_phi,
        )
    else:
        raise ValueError("fod_model must be 'watson' or 'bingham'.")

    components = [
        Component(
            "neurite",
            FODConvolved(
                response=Stick(diffusivity="d_in", orientation="axis"),
                fod=fod,
                orientation_parameter="axis",
            ),
        ),
        Component(
            "extra",
            FODConvolved(
                response=NODDIExtraNeurite(
                    parallel_diffusivity="d_in",
                    orientation="axis",
                    neurite_fraction="neurite_fraction",
                    csf_fraction="csf_fraction" if include_csf else None,
                ),
                fod=fod,
                orientation_parameter="axis",
            ),
        ),
    ]
    if include_csf:
        components.append(Component("csf", Ball(diffusivity="d_csf")))

    model = Model(components)

    parameter_list = [
        Parameter("S0", 1.0, (0.05, 2.0)),
        Parameter("d_in", 2.2, (0.5, 3.0), vary=vary_intracellular_diffusivity),
        Parameter("theta", np.pi / 2.0, (0.0, np.pi)),
        Parameter("phi", 0.0, (-np.pi, np.pi)),
    ]
    if fod_model == "watson":
        parameter_list.append(Parameter("kappa", 1.0, (0.0, 64.0), vary=vary_kappa))
    else:
        parameter_list.extend(
            [
                Parameter("kappa1", -4.0, (-128.0, 0.0), vary=vary_kappa),
                Parameter("kappa2", -1.0, (-128.0, 0.0), vary=vary_kappa),
                Parameter("psi", 0.0, (-np.pi, np.pi)),
            ]
        )
    if include_csf:
        parameter_list.append(
            Parameter("d_csf", 3.0, (1.5, 3.5), vary=vary_csf_diffusivity)
        )

    if include_csf:
        simplexes = [
            Simplex(
                ("neurite_fraction", "extra_fraction", "csf_fraction"),
                (0.6, 0.3, 0.1),
            )
        ]
    else:
        simplexes = [
            Simplex(
                ("neurite_fraction", "extra_fraction"),
                (0.65, 0.35),
            )
        ]

    parameters = ParameterSpace(parameter_list, simplexes=simplexes)
    return ModelPreset(model=model, parameters=parameters)
