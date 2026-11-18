"""Generic fitting interfaces."""

from .least_squares import BatchFitResult, FitResult, LeastSquaresFitter
from .mcmc import EmceeFitter, MCMCResult
from .nifti import NiftiLeastSquaresResult, fit_least_squares_nifti, merge_chunk_maps
from .noise import GaussianNoise

__all__ = [
    "EmceeFitter",
    "BatchFitResult",
    "FitResult",
    "GaussianNoise",
    "LeastSquaresFitter",
    "MCMCResult",
    "NiftiLeastSquaresResult",
    "fit_least_squares_nifti",
    "merge_chunk_maps",
]
