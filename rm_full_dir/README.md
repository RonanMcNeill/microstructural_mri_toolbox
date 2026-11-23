# microstructure-mri

Composable simulation and fitting of microstructure MRI models, focused on diffusion MRI.

This repository provides:
- explicit acquisition handling with unit-safe constructors,
- composable compartment models,
- ready-to-fit preset models (SANDI and NODDI variants),
- nonlinear fitting and ensemble MCMC,
- spherical mean simulation utilities,
- orientation-distribution (Watson/Bingham) convolution models,
- hypergeometric-ratio signal formulations for Ball-and-Rackets style models.

## Why this package

The design goal is to make model development and fitting reproducible and easy to inspect:
- build models from small, testable components,
- keep physical parameter names throughout simulation and fitting,
- enforce simplex constraints for fractions automatically,
- make acquisition assumptions explicit in code.

## Unit conventions

Canonical in-memory units:
- distance: um
- time: ms
- diffusivity: um^2/ms
- b-value: ms/um^2
- gradient strength: T/m

Notes:
- `Acquisition.from_b_values` requires explicit `b_unit` and never infers units from magnitude.
- `Acquisition.from_fsl` defaults to FSL-style b-values in `s/mm^2` and converts to canonical units.
- `b0_threshold` is always interpreted in canonical `ms/um^2`.

## Installation and setup

Use the existing local environment:

```bash
micromamba activate /Users/ahoward2/fsl/envs/microstructure-mri
```

Or create a fresh environment:

```bash
micromamba create -f environment.yml
micromamba activate microstructure-mri
```

Run tests:

```bash
pytest
```

## Quick start

```python
from dmri_models import Acquisition, Ball, Component, Model

acquisition = Acquisition.from_b_values(
    [0, 1000, 2000],
    b_unit="s/mm^2",
)

model = Model(
    [
        Component("slow", Ball(diffusivity="d_slow")),
        Component("fast", Ball(diffusivity="d_fast")),
    ]
)

parameters = {
    "S0": 1.0,
    "slow_fraction": 0.7,
    "fast_fraction": 0.3,
    "d_slow": 0.8,
    "d_fast": 2.0,
}

signal = model.signal(acquisition, parameters)
```

## Core concepts

### 1) Acquisition

Main constructors:
- `Acquisition.from_b_values(...)`
- `Acquisition.from_pgse(...)`
- `Acquisition.from_fsl(...)`

Use `from_pgse` when timing-aware models are involved (for example restricted spheres), because `from_pgse` derives gradient strength from

`b = gamma^2 * G^2 * delta^2 * (Delta - delta/3)`.

Helpful inspection utilities:
- `acquisition.summary(b_tolerance=0.1)`
- `acquisition.print_summary(b_tolerance=0.1)`

### 2) Compartments and model composition

Common compartments:
- `Ball`: isotropic Gaussian diffusion.
- `Stick`: zero-radius cylinder; oriented or powder-averaged.
- `Zeppelin`: axisymmetric Gaussian tensor.
- `Tensor`: general Gaussian tensor.
- `Sphere`: restricted diffusion in impermeable spheres (timing-aware).
- `T2Relaxation`: wraps any compartment with mono-exponential T2 weighting.

Models are built as weighted mixtures of named components:

```python
from dmri_models import Component, Model

model = Model([
    Component("compartment_a", ...),
    Component("compartment_b", ...),
])
```

Fractions are named automatically as `<component_name>_fraction` unless overridden.

### 3) ParameterSpace

`ParameterSpace` keeps named physical parameters while exposing optimiser-friendly vectors.

`Simplex` coordinates use additive log-ratios internally, ensuring fractions are always positive and sum to one during fitting and MCMC.

### 4) Fitting APIs

Deterministic fit:
- `LeastSquaresFitter.fit(...)` for one voxel/signal.
- `LeastSquaresFitter.fit_many(...)` for arrays with measurements on the last axis.
- `fit_least_squares_nifti(...)` for voxel-wise fitting from a 4D `.nii`/`.nii.gz`.
- `merge_chunk_maps(...)` to merge cluster chunk outputs back into full maps.

Bayesian fit:
- `EmceeFitter(...).sample(...)`
- diagnostics from `MCMCResult`: `summary()`, `autocorrelation_time()`, `effective_sample_size()`.

### Voxel-wise NIfTI fitting (cluster friendly)

```python
from pathlib import Path

from dmri_models import fit_least_squares_nifti, merge_chunk_maps

# Full-volume fit on one machine:
result = fit_least_squares_nifti(
    acquisition=acquisition,
    model=model,
    parameter_space=parameter_space,
    data_path="dwi.nii.gz",
    mask_path="mask.nii.gz",  # optional
    output_dir="fit_outputs",
    output_prefix="subject01",
    n_starts=4,
    n_jobs=8,
)

# Cluster mode: one chunk per job (example: 32 jobs total).
chunk_result = fit_least_squares_nifti(
    acquisition=acquisition,
    model=model,
    parameter_space=parameter_space,
    data_path="dwi.nii.gz",
    mask_path="mask.nii.gz",
    output_dir="fit_outputs/chunks",
    output_prefix="subject01",
    chunk_index=job_index,  # 0..31
    n_chunks=32,
    n_starts=2,
    n_jobs=1,
)

# Merge one parameter map after all chunk jobs finish.
chunk_paths = sorted(Path("fit_outputs/chunks").glob("subject01.d_in.chunk*.nii.gz"))
merge_chunk_maps(
    chunk_paths=chunk_paths,
    output_path="fit_outputs/subject01.d_in.nii.gz",
)
```

## Acquisition presets

Packaged presets are standard editable `Acquisition` objects:
- `dmri_models.acquisitions.hcp()`
- `dmri_models.acquisitions.uk_biobank()`
- `dmri_models.acquisitions.bigmac_highres()`
- `dmri_models.acquisitions.bigmac_lowres()`
- `dmri_models.acquisitions.sandi_full_acq()`

## Model presets

### SANDI

```python
from dmri_models import sandi

preset = sandi(vary_sphere_diffusivity=False)
model = preset.model
space = preset.parameters
```

### NODDI

```python
from dmri_models import noddi

preset = noddi(
    include_csf=True,
    vary_intracellular_diffusivity=False,
    vary_csf_diffusivity=False,
    vary_kappa=True,
    fod_model="watson",  # or "bingham"
)
```

Utilities:
- `odi_from_kappa(kappa)`
- `noddi_derived_parameters(values)`

## FOD and hypergeometric-ratio options

Advanced orientation-dispersion components are in `dmri_models.compartments`.

Quadrature-convolution pathway:
- `FODConvolved(response=..., fod=WatsonFOD(...), orientation_parameter="axis")`
- `FODConvolved(response=..., fod=BinghamFOD(...), orientation_parameter="axis")`

Hypergeometric-ratio pathway:
- `WatsonStickHypergeometric`
- `BinghamStickHypergeometric`
- `WatsonZeppelinHypergeometric`

These implement Ball-and-Rackets style signal expressions using ratios of normalising constants with matrix arguments.

SH-FOD utilities:
- `SphericalHarmonicFOD.random_unit_directions(...)`
- `SphericalHarmonicFOD.spherical_grid(...)`
- `SphericalHarmonicFOD.watson_density(...)`
- `SphericalHarmonicFOD.default_crossing_coefficients(...)`
- `SphericalHarmonicFOD.crossing_coefficients_from_axes(...)`

SMI-like pathway:
- `SMIKernelConvolved.kernel_by_order(...)`
- `SMIKernelConvolved.fit_coefficients(...)`

Reusable plotting utilities:
- `plot_fod_surface(...)`
- `plot_signal_agreement(...)`
- `fod_surface_mesh(...)`

By default, `plot_fod_surface` now uses a MultiD-style RGB colouring based on
absolute Cartesian FOD surface coordinates (`colour_mode="multid_xyz"`).
Set `colour_mode="colormap"` to use amplitude-based colormaps.

## Spherical mean simulation

Use model-level spherical mean consistently across mixed compartment types:

```python
mean_signal = model.spherical_mean_signal(
    acquisition,
    parameters,
    n_directions=256,
)
```

Components with analytical spherical means use them directly; others are averaged numerically over sampled directions.

## Notebook tutorials

Current notebooks in `examples/notebooks`:
- `01_arbitrary_compartment_simulation.ipynb`
- `02_noddi_simulation_and_nonlinear_fit.ipynb`
- `03_sandi_mcmc_degeneracy.ipynb`
- `04_ball_and_rackets_nonlinear_fit.ipynb`
- `05_api_walkthrough_options.ipynb` (comprehensive syntax walkthrough)
- `06_custom_model_from_scratch.ipynb`
- `07_sandi_gaussian_priors.ipynb`
- `08_spherical_harmonic_fod_visualisation.ipynb` (SH-FOD surface visualisation)
- `09_smi_like_sh_fitting_simulation.ipynb` (SMI-like SH coefficient fitting demo)

## Practical tips

- Start from a preset (`sandi()` or `noddi()`) if you are new to the codebase.
- Use `acquisition.print_summary()` before fitting to validate protocol assumptions.
- For restricted compartments (`Sphere`), ensure PGSE timing fields are present.
- For multi-start nonlinear fitting, increase `n_starts` before tightening bounds.
- Use simplex constraints for all mutually exclusive fractions.

## Running selected tests

```bash
pytest tests/test_model.py -q
pytest tests/test_presets.py -q
```

## Project TODO (prioritised)

This checklist is intended as a practical roadmap for the next development phase.

### P0: scientific correctness and regression safety

- [ ] Confirm SANDI implementation and assumptions against the reference paper.
    Suggested steps:
    1. Write a short validation note in `docs/` covering equations, parameter constraints, units, and acquisition assumptions.
    2. Add regression tests for canonical simulation settings (including timing-aware sphere behaviour).
    3. Add one notebook cell that reproduces at least one published-style SANDI curve.
    Acceptance criteria:
    - tests are deterministic and pass in CI,
    - validation note links each implemented term to paper notation.

- [ ] Validate NODDI implementation (Watson and Bingham variants) against equations and expected limiting behaviour.
    Suggested steps:
    1. Verify tortuosity coupling and fraction definitions with and without CSF.
    2. Add tests for limiting cases (kappa -> 0, kappa -> large, isotropic/anisotropic special cases).
    3. Add notebook comparisons between quadrature and hypergeometric pathways on HCP-style acquisition.
    Acceptance criteria:
    - all NODDI limits match expected analytical or numerical references,
    - no unexplained discrepancies between pathways.

- [ ] Check hypergeometric-ratio implementations thoroughly.
    Suggested steps:
    1. Build a parameter sweep test matrix over b-values, diffusivities, and concentration parameters.
    2. Compare quadrature and hypergeometric outputs with strict tolerances and report max errors.
    3. Add stress tests for high concentration and high b-value stability.
    Acceptance criteria:
    - error envelope is documented,
    - numerical stability is verified for realistic and edge-case regimes.

### P1: expand FOD formalisms

- [ ] Add additional FOD formalisms inspired by Coelho SMI-style modelling.
    Suggested options:
    - alternative parametric families beyond Watson/Bingham,
    - parameter mappings that support SMI-style interpretation workflows.
    Acceptance criteria:
    - each new formalism has clear parameter definitions and at least one fitted example.

- [ ] Add spherical harmonics (SH) FOD representation.
    Suggested steps:
    1. Implement an SH basis FOD class with configurable order.
    2. Add optional regularisation for stable coefficient estimation.
    3. Provide helper utilities for coefficient normalisation and visualisation.
    Acceptance criteria:
    - SH FOD can be used with existing response convolution interfaces,
    - notebook example shows end-to-end use.

- [ ] Add rotational invariants derived from FOD representations.
    Suggested steps:
    1. Define a stable API for invariant summaries from Watson/Bingham/SH forms.
    2. Add unit tests for invariance under coordinate frame rotation.
    3. Add notebook examples showing interpretation of invariants.
    Acceptance criteria:
    - invariants are numerically consistent after random rotations,
    - documentation explains interpretation and limitations.

### P1: scale fitting across voxels and cluster environments

- [ ] Parallelise fitting across voxels for workstation and cluster execution.
    Suggested steps:
    1. Extend `fit_many` workflows with chunked processing and reproducible seeding.
    2. Add optional backends for local multiprocessing and cluster job arrays (for example Slurm-style array splits).
    3. Add lightweight checkpointing and restart support for long jobs.
    Acceptance criteria:
    - same results are obtained in serial vs parallel runs (within tolerance),
    - example scripts for local and cluster execution are provided.

### P2: usability and onboarding improvements

- [ ] Add a short "decision tree" section to guide model/preset choice by use-case.
- [ ] Add API reference pages generated from docstrings.
- [ ] Add a small benchmark notebook with runtime/memory profiles for key workflows.

## Suggested implementation order

1. Finish P0 validation tasks (SANDI, NODDI, hypergeometric checks).
2. Add SH FOD + rotational invariants as the first non-parametric extension.
3. Add cluster-oriented voxel parallelisation and checkpointing.
4. Add final documentation and benchmarking polish.
