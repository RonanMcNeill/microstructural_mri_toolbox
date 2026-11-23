"""Composable diffusion MRI biophysical signal models."""

from . import acquisitions
from .acquisition import Acquisition, pgse_gradient_strength
from .compartments import (
    Ball,
    BinghamFOD,
    FODConvolved,
    SMIKernelConvolved,
    Sphere,
    SphericalHarmonicFOD,
    Stick,
    T2Relaxation,
    Tensor,
    WatsonFOD,
    Zeppelin,
)
from .fitting import (
    EmceeFitter,
    BatchFitResult,
    FitResult,
    GaussianNoise,
    LeastSquaresFitter,
    MCMCResult,
    NiftiLeastSquaresResult,
    fit_least_squares_nifti,
    merge_chunk_maps,
)
from .model import Component, Model
from .parameters import Parameter, ParameterSpace, Simplex
from .plotting import fod_surface_mesh, plot_fod_surface, plot_signal_agreement
from .presets import ModelPreset, noddi, noddi_derived_parameters, odi_from_kappa, sandi

__all__ = [
    "Acquisition",
    "acquisitions",
    "Ball",
    "BatchFitResult",
    "Component",
    "BinghamFOD",
    "EmceeFitter",
    "FitResult",
    "FODConvolved",
    "GaussianNoise",
    "LeastSquaresFitter",
    "MCMCResult",
    "NiftiLeastSquaresResult",
    "Model",
    "ModelPreset",
    "fit_least_squares_nifti",
    "merge_chunk_maps",
    "noddi",
    "noddi_derived_parameters",
    "odi_from_kappa",
    "Parameter",
    "ParameterSpace",
    "plot_fod_surface",
    "plot_signal_agreement",
    "fod_surface_mesh",
    "Simplex",
    "Sphere",
    "SphericalHarmonicFOD",
    "SMIKernelConvolved",
    "Stick",
    "T2Relaxation",
    "Tensor",
    "WatsonFOD",
    "Zeppelin",
    "pgse_gradient_strength",
    "sandi",
]

__version__ = "0.1.0.dev0"
