"""Soma and Neurite Density Imaging model preset."""

from __future__ import annotations

from dataclasses import dataclass

from ..compartments import Ball, Sphere, Stick
from ..model import Component, Model
from ..parameters import Parameter, ParameterSpace, Simplex


@dataclass(frozen=True)
class ModelPreset:
    """A model bundled with fitting-ready parameter definitions."""

    model: Model
    parameters: ParameterSpace


def sandi(
    *,
    vary_sphere_diffusivity: bool = False,
    vary_neurite_diffusivity: bool = False,
    vary_extra_diffusivity: bool = False,
    d_sphere: float = 3.0,
    d_in: float = 2.0,
    d_ex: float = 1.5,
) -> ModelPreset:
    """Create the powder-averaged, diffusion-only SANDI model.

    The soma uses the timing-aware GPD sphere. The neurite is a
    powder-averaged stick, and the extra-cellular compartment is an isotropic
    Gaussian ball. Acquisition data must therefore describe rectangular PGSE
    timing and gradient strength, for example via ``Acquisition.from_pgse``.

    Following the article approach (Palombo et al. 2020), diffusivities are
    typically fixed at literature values and only the soma radius and volume
    fractions are fitted. This yields a well-conditioned inverse problem with
    five non-zero b-shells and three unknowns (r_soma, f_soma, f_neurite).

    Parameters
    ----------
    vary_sphere_diffusivity:
        If True, the intra-soma diffusivity ``d_sphere`` is a free parameter.
        If False (default), it is held fixed at ``d_sphere``.
    vary_neurite_diffusivity:
        If True, the intra-neurite diffusivity ``d_in`` is a free parameter.
        If False (default), it is held fixed at ``d_in``.
    vary_extra_diffusivity:
        If True, the extra-axonal diffusivity ``d_ex`` is a free parameter.
        If False (default), it is held fixed at ``d_ex``.
    d_sphere:
        Intra-soma diffusivity (µm²/ms). Fixed or initial value depending on
        ``vary_sphere_diffusivity``. Defaults to 3.0 µm²/ms (free water at
        body temperature).
    d_in:
        Intra-neurite diffusivity (µm²/ms). Fixed or initial value depending
        on ``vary_neurite_diffusivity``. Defaults to 2.0 µm²/ms.
    d_ex:
        Extra-axonal diffusivity (µm²/ms). Fixed or initial value depending
        on ``vary_extra_diffusivity``. Defaults to 1.5 µm²/ms.
    """
    model = Model(
        [
            Component("soma", Sphere(radius="r_soma", diffusivity="d_sphere")),
            Component(
                "neurite",
                Stick(diffusivity="d_in", powder_average=True),
            ),
            Component("extra", Ball(diffusivity="d_ex")),
        ]
    )
    parameters = ParameterSpace(
        [
            Parameter("S0", 1.0, (0.05, 2.0)),
            Parameter("r_soma", 10.0, (2.0, 15.0)),
            Parameter("d_in", d_in, (0.1, 3.0), vary=vary_neurite_diffusivity),
            Parameter("d_ex", d_ex, (0.1, 3.0), vary=vary_extra_diffusivity),
            Parameter(
                "d_sphere",
                d_sphere,
                (0.1, 3.5),
                vary=vary_sphere_diffusivity,
            ),
        ],
        simplexes=[
            Simplex(
                ("soma_fraction", "neurite_fraction", "extra_fraction"),
                (0.3, 0.4, 0.3),
            )
        ],
    )
    return ModelPreset(model=model, parameters=parameters)

