#!/usr/bin/env python3
"""Real-data NODDI+soma Tort2 fitting, chunked via fsl_sub/SLURM.

Requires a completed, co-registered NODDI pre-fit in NODDI_PREFIT_FOLDER.
This is a real-data implementation, not the notebook's simulated-truth experiment.
Run with python; requires numpy scipy nibabel matplotlib and dmri_models.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.optimize import least_squares
SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
from dmri_models.acquisition import Acquisition
from dmri_models import Ball, Component, Model, ModelPreset, Parameter, ParameterSpace, Simplex, Sphere, Stick
from dmri_models.compartments import WatsonFOD, FODConvolved

# ============================ USER SETTINGS ==============================
LOCAL_PREPROC_DIR = Path("/Users/Ronan/Documents/PhD/Data/80_post_fitting_input")
CLUSTER_PREPROC_DIR = Path("/vols/Scratch/zcn545/80_post_fitting_input")
DATA_FILENAME = "data.nii.gz"
BVAL_FILENAME = "bvals"
BVEC_FILENAME = "bvecs"
MASK_FILENAME = os.environ.get("SOMA_MASK_FILENAME", "injection_site_mask.nii.gz")
NODDI_PREFIT_FOLDER = os.environ.get("SOMA_PREFIT_FOLDER", "noddi_1_start_roi")  # Must exist for each dataset
OUTPUT_FOLDER_NAME = os.environ.get("SOMA_OUTPUT_FOLDER", "noddi_soma_10starts")
N_CHUNKS = 5  # Per mask; increase after a successful test
N_STARTS = 10
MAX_NFEV = 2000
RNG_SEED = 12345
PULSE_DURATION_MS = 5.0
DIFFUSION_TIME_MS = 12.0
# Gaussian-style prior widths: provisional simulation-informed assumptions.
# NOT measured uncertainty; validate by sensitivity analysis on real data.
PRIOR_SCALE = 1.0

KAPPA_PRIOR_SD = 0.30 * PRIOR_SCALE
Q_NEURITE_PRIOR_SD = 0.04 * PRIOR_SCALE
Q_EXTRA_PRIOR_SD = 0.12 * PRIOR_SCALE
Q_FW_PRIOR_SD = 0.08 * PRIOR_SCALE
# Set the signal noise SD in the units of your globally scaled NIfTI.
SNR = 20.0
DTI_MAX_B = 1.1  # acquisition.b_values in ms/um^2; adjust for actual shells
DTI_MIN_DIRECTIONS = 6
# SLURM partitions
CLUSTER_QUEUE = "short"
MERGE_QUEUE = "short"

# Resources per fitting chunk
CLUSTER_TIME_MINUTES = 300
CLUSTER_RAM_GB = 8

# Merge resources
MERGE_TIME_MINUTES = 30
MERGE_RAM_GB = 4

SCRIPT_PATH = Path(__file__).resolve()
PYTHON_EXECUTABLE = str(Path(sys.executable).resolve())
CHUNK_ENV = os.environ.get("NODDI_SOMA_CHUNK_INDEX")
MERGE_ONLY = os.environ.get("NODDI_SOMA_MERGE_ONLY") == "1"
def on_cluster():
    try:
        return subprocess.run(["fsl_sub", "--version"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except FileNotFoundError:
        return False
ON_CLUSTER = on_cluster()
PREPROC_DIR = Path(os.environ["SOMA_INPUT_DIR"]) if "SOMA_INPUT_DIR" in os.environ else (CLUSTER_PREPROC_DIR if ON_CLUSTER else LOCAL_PREPROC_DIR)
DATA_PATH = PREPROC_DIR / DATA_FILENAME
MASK_PATH = PREPROC_DIR / MASK_FILENAME
BVAL_PATH = PREPROC_DIR / BVAL_FILENAME
BVEC_PATH = PREPROC_DIR / BVEC_FILENAME
PREFIT_DIR = PREPROC_DIR / NODDI_PREFIT_FOLDER
OUTPUT_DIR = PREPROC_DIR / OUTPUT_FOLDER_NAME
STATUS_DIR = OUTPUT_DIR / "status"
LOG_DIR = OUTPUT_DIR / "logs"
PREFIX = "noddi_soma"
PARAMETERS = ["S0", "d_in", "d_soma", "d_csf", "r_soma", "kappa", "theta", "phi",
              "neurite_fraction", "extra_fraction", "soma_fraction", "csf_fraction",
              "q_neurite", "q_extra", "q_fw", "ODI", "cost", "signal_rmse",
              "success", "n_evaluations"]

def stamp():
    return datetime.now(timezone.utc).isoformat()
def status_path(i):
    return STATUS_DIR / f"chunk_{i+1:04d}.status"
def status(i, state, **kwargs):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"chunk={i+1}/{N_CHUNKS}", f"status={state}", f"updated={stamp()}"]
    lines += [f"{k}={v}" for k,v in kwargs.items()]
    status_path(i).write_text("\n".join(lines) + "\n")
def read_status(i):
    p = status_path(i)
    if not p.exists(): return {}
    return dict(line.split("=",1) for line in p.read_text().splitlines() if "=" in line)
def load_volume(path, shape):
    img = nib.load(str(path))
    if img.shape != shape: raise ValueError(f"Shape mismatch: {path}: {img.shape} vs {shape}")
    return np.asarray(img.dataobj, dtype=float)

def load_inputs():
    for p in (DATA_PATH, MASK_PATH, BVAL_PATH, BVEC_PATH):
        if not p.exists(): raise FileNotFoundError(p)
    img = nib.load(str(DATA_PATH))
    if len(img.shape) != 4: raise ValueError("Data must be 4D")
    mask_img = nib.load(str(MASK_PATH))
    if mask_img.shape != img.shape[:3] or not np.allclose(mask_img.affine, img.affine, atol=1e-4):
        raise ValueError("Mask must match dMRI shape and affine")
    mask = np.asarray(mask_img.dataobj) > 0
    if not mask.any(): raise ValueError("Empty mask")
    acq = Acquisition.from_fsl(BVAL_PATH, BVEC_PATH, bval_unit="s/mm^2", b0_threshold=0.05,
        rectangular_pgse=True, pulse_duration=PULSE_DURATION_MS,
        diffusion_time=DIFFUSION_TIME_MS, b_tensor_shape=1.0)
    if acq.n_measurements != img.shape[-1]: raise ValueError("Acquisition measurement mismatch")
    if not np.any(np.asarray(acq.b_values) <= 0.05): raise ValueError("No b0 volumes for voxelwise normalisation")
    prefit = {}
    for name in ("kappa", "neurite_fraction", "extra_fraction", "csf_fraction"):
        p = PREFIT_DIR / f"noddi.{name}.nii.gz"
        if not p.exists(): raise FileNotFoundError(f"Missing NODDI pre-fit: {p}")
        prefit[name] = load_volume(p, mask.shape)
    return img, mask, acq, prefit

# ========================== NOTEBOOK TORT2 MODEL =========================
def _nonnegative_parameter(parameters, name):
    value = float(parameters[name])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass
class WhiteMatterTortuosityZeppelin:
    """Extra-neurite Zeppelin using the compartment-aware Tort2 approximation.

    d_ex_parallel =
        d_in * (1 - f_soma / (1 - f_neurite - f_csf))^alpha

    d_ex_perp =
        d_ex_parallel *
        (1 - f_neurite / (1 - f_soma - f_csf))^beta

    Currently alpha = beta = 1.
    """

    parallel_diffusivity: str
    orientation: str
    neurite_fraction: str = "neurite_fraction"
    soma_fraction: str = "soma_fraction"
    csf_fraction: str | None = "csf_fraction"

    def signal(self, acquisition, parameters):
        if acquisition.gradients is None:
            raise ValueError("A tensor requires acquisition gradient directions.")

        d_in = _nonnegative_parameter(
            parameters,
            self.parallel_diffusivity,
        )

        f_neurite = float(parameters[self.neurite_fraction])
        f_soma = float(parameters[self.soma_fraction])

        if not np.isfinite(f_neurite) or f_neurite < 0:
            raise ValueError("neurite fraction must be finite and non-negative")
        if not np.isfinite(f_soma) or f_soma < 0:
            raise ValueError("soma fraction must be finite and non-negative")

        if self.csf_fraction is None:
            f_csf = 0.0
        else:
            f_csf = float(parameters[self.csf_fraction])
            if not np.isfinite(f_csf) or f_csf < 0 or f_csf >= 1.0:
                raise ValueError("csf fraction must be finite, non-negative and < 1")

       # -------------------------------------------------------------
        # Compartment-aware Tort2 approximation
        #
        # alpha = beta = 1
        # -------------------------------------------------------------

        available_parallel = (
            1.0
            - f_neurite
            - f_csf
        )

        available_perpendicular = (
            1.0
            - f_soma
            - f_csf
        )

        if available_parallel <= 0:
            raise ValueError(
                "No extracellular+soma space available for "
                "parallel tortuosity calculation."
            )

        if available_perpendicular <= 0:
            raise ValueError(
                "No extracellular+neurite space available for "
                "perpendicular tortuosity calculation."
            )

        soma_factor = (
            1.0
            - f_soma
            / available_parallel
        )

        neurite_factor = (
            1.0
            - f_neurite
            / available_perpendicular
        )

        if soma_factor < 0:
            raise ValueError(
                "Soma tortuosity factor became negative."
            )

        if neurite_factor < 0:
            raise ValueError(
                "Neurite tortuosity factor became negative."
            )

        alpha = 1.0
        beta = 1.0

        d_parallel = (
            d_in
            * soma_factor**alpha
        )

        d_perpendicular = (
            d_parallel
            * neurite_factor**beta
        )

        orientation = np.asarray(
            parameters[self.orientation],
            dtype=float,
        ).reshape(-1)
        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"{self.orientation} must be a finite three-vector")

        norm = np.linalg.norm(orientation)
        if np.isclose(norm, 0.0):
            raise ValueError(f"{self.orientation} must be non-zero")
        orientation = orientation / norm

        projection = acquisition.gradients @ orientation
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )

        directional_factor = (
            beta * (projection ** 2 - 1.0 / 3.0)
            + 1.0 / 3.0
        )

        adc_term = (
            (d_parallel - d_perpendicular)
            * directional_factor
            + d_perpendicular
        )

        return np.exp(-acquisition.b_values * adc_term)


def tort2_model():
    theta = 1.0
    phi = 1.2

    fod = WatsonFOD(
        kappa="kappa",
        theta="theta",
        phi="phi",
        n_theta=20,
        n_phi=40,
    )

    model = Model([
        Component(
            "neurite",
            compartment=FODConvolved(
                response=Stick(
                    diffusivity="d_in",
                    orientation="axis",
                ),
                fod=fod,
                orientation_parameter="axis",
            ),
        ),
        Component(
            "extra",
            compartment=FODConvolved(
                response=WhiteMatterTortuosityZeppelin(
                    parallel_diffusivity="d_in",
                    orientation="axis",
                    neurite_fraction="neurite_fraction",
                    soma_fraction="soma_fraction",
                    csf_fraction="csf_fraction",
                ),
                fod=fod,
                orientation_parameter="axis",
            ),
        ),
        Component(
            "soma",
            compartment=Sphere(
                radius="r_soma",
                diffusivity="d_soma",
            ),
        ),
        Component(
            "csf",
            compartment=Ball(
                diffusivity="d_csf",
            ),
        ),
    ])

    parameters = ParameterSpace([
        Parameter("S0", 1.0, (0.05, 2.0)),
        Parameter("d_in", 2.2, (0.5, 3.0), vary=False),
        Parameter("d_soma", 3.0, (0.5, 3.5), vary=False),
        Parameter("d_csf", 3.0, (0.5, 3.5), vary=False),
        Parameter("r_soma", 6.0, (1.0, 15.0), vary=True),
        Parameter("kappa", 2.0, (0.0, 64.0), vary=True),
        Parameter("theta", theta, (0.0, np.pi), vary=False),
        Parameter("phi", phi, (-np.pi, np.pi), vary=False),
    ], simplexes=[
        Simplex(
            ("neurite_fraction", "extra_fraction", "soma_fraction", "csf_fraction"),
            (0.50, 0.25, 0.10, 0.15),
        )
    ])

    return ModelPreset(model=model, parameters=parameters)

def sigmoid(x):
    x = np.clip(float(x), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return np.log(p / (1.0 - p))


def stick_breaking_to_fractions(u):
    """Map 3 unconstrained values to 4 exact simplex fractions."""
    u = np.asarray(u, dtype=float)
    v1, v2, v3 = [sigmoid(value) for value in u]

    f_neurite = v1
    f_extra = (1.0 - v1) * v2
    f_soma = (1.0 - v1) * (1.0 - v2) * v3
    f_fw = (1.0 - v1) * (1.0 - v2) * (1.0 - v3)

    return {
        "neurite_fraction": float(f_neurite),
        "extra_fraction": float(f_extra),
        "soma_fraction": float(f_soma),
        "csf_fraction": float(f_fw),
    }


def fractions_to_stick_breaking(
    f_neurite,
    f_extra,
    f_soma,
    f_fw,
):
    """Convert physical simplex fractions to stick-breaking coordinates."""
    fractions = np.array([
        f_neurite,
        f_extra,
        f_soma,
        f_fw,
    ], dtype=float)
    fractions = np.clip(fractions, 1e-8, None)
    fractions /= fractions.sum()

    f_n, f_e, f_s, _ = fractions

    v1 = f_n
    v2 = f_e / max(1e-12, 1.0 - f_n)
    remaining = 1.0 - f_n - f_e
    v3 = f_s / max(1e-12, remaining)

    return np.array([
        logit(v1),
        logit(v2),
        logit(v3),
    ], dtype=float)


def fractions_to_q(fractions):
    """Return the relative composition of the non-soma pool."""
    f_soma = float(fractions["soma_fraction"])
    non_soma = max(1e-12, 1.0 - f_soma)

    return {
        "q_neurite": float(
            fractions["neurite_fraction"] / non_soma
        ),
        "q_extra": float(
            fractions["extra_fraction"] / non_soma
        ),
        "q_fw": float(
            fractions["csf_fraction"] / non_soma
        ),
    }


def q_from_noddi_fractions(noddi):
    """Normalize the three NODDI fractions into non-soma composition q."""
    raw = np.array([
        noddi["neurite_fraction"],
        noddi["extra_fraction"],
        noddi["csf_fraction"],
    ], dtype=float)
    raw = np.clip(raw, 1e-10, None)
    raw /= raw.sum()

    return {
        "q_neurite": float(raw[0]),
        "q_extra": float(raw[1]),
        "q_fw": float(raw[2]),
    }



def physical_fractions_from_fsoma(f_soma,q):
    return {"neurite_fraction":(1-f_soma)*q["q_neurite"],
            "extra_fraction":(1-f_soma)*q["q_extra"],
            "soma_fraction":f_soma,"csf_fraction":(1-f_soma)*q["q_fw"]}


# ========================= REAL-DATA FITTING =============================
def dti_angles(signal, acquisition):
    """Linear log-tensor fit using b0 and selected low-b directions."""
    b = np.asarray(acquisition.b_values, float)
    g = np.asarray(acquisition.gradients, float)
    b0 = b <= 0.05
    dw = (b > 0.05) & (b <= DTI_MAX_B)
    if b0.sum() == 0 or dw.sum() < DTI_MIN_DIRECTIONS:
        raise ValueError("Insufficient b0/low-b DTI measurements; review DTI_MAX_B")
    s0 = float(np.mean(signal[b0]))
    if not np.isfinite(s0) or s0 <= 0: raise ValueError("Invalid b0 for DTI")
    gd = g[dw]
    bd = b[dw]
    design = np.column_stack((gd[:,0]**2, gd[:,1]**2, gd[:,2]**2,
        2*gd[:,0]*gd[:,1], 2*gd[:,0]*gd[:,2], 2*gd[:,1]*gd[:,2])) * bd[:,None]
    if np.linalg.matrix_rank(design) < 6: raise ValueError("Rank-deficient DTI design")
    y = -np.log(np.maximum(signal[dw], 1e-8) / s0)
    d = np.linalg.lstsq(design, y, rcond=None)[0]
    tensor = np.array([[d[0],d[3],d[4]], [d[3],d[1],d[5]], [d[4],d[5],d[2]]])
    eigval,eigvec = np.linalg.eigh(tensor)
    axis = eigvec[:,np.argmax(eigval)]
    if axis[2] < 0: axis = -axis  # canonical axial representation
    theta = float(np.arccos(np.clip(axis[2], -1, 1)))
    phi = float(np.arctan2(axis[1], axis[0]))
    return theta,phi

def fit_voxel(signal, acquisition, prefit, theta, phi, seed):
    # Approximate voxelwise noise SD assuming SNR = 20
    sigma_data = float(np.max(np.abs(signal)) / SNR)

    if not np.isfinite(sigma_data) or sigma_data <= 0:
        raise ValueError("Invalid voxelwise noise estimate")
    # These are conditional Gaussian-style penalties, not empirical posteriors.
    q0 = q_from_noddi_fractions(prefit)
    kapp0 = float(prefit["kappa"])
    if not np.isfinite(kapp0) or kapp0 < 0: raise ValueError("Invalid NODDI kappa")
    prior_sd = np.array([KAPPA_PRIOR_SD, Q_NEURITE_PRIOR_SD,
                         Q_EXTRA_PRIOR_SD, Q_FW_PRIOR_SD],float)
    if np.any(prior_sd <= 0):
        raise ValueError("Prior SDs must be positive")
    f_soma0 = 0.10
    initial = physical_fractions_from_fsoma(f_soma0, q0)
    x0 = np.array([np.clip(np.mean(signal[np.asarray(acquisition.b_values)<=0.05]),0.051,1.999),
                   7.0, np.clip(kapp0, 0.0001, 63.999),
                   *fractions_to_stick_breaking(initial["neurite_fraction"],
                      initial["extra_fraction"], initial["soma_fraction"],initial["csf_fraction"])])
    lower = np.array([0.05,1.0,0.0,-20,-20,-20],float)
    upper = np.array([2.0,15.0,64.0,20,20,20],float)
    x0 = np.clip(x0, lower+1e-7,upper-1e-7)
    preset = tort2_model()
    def unpack(x):
        return {"S0":float(x[0]), "r_soma":float(x[1]),"kappa":float(x[2]),
                "theta":theta,"phi":phi,"d_in":2.2,"d_soma":3.0,"d_csf":3.0,
                **stick_breaking_to_fractions(x[3:6])}
    def prediction(x):
        return np.asarray(preset.model.signal(acquisition,unpack(x)),float)
    def residual(x):
        vals=unpack(x)
        q=fractions_to_q(vals)
        penalties=np.array([vals["kappa"]-kapp0, q["q_neurite"]-q0["q_neurite"],
                            q["q_extra"]-q0["q_extra"],q["q_fw"]-q0["q_fw"]])/prior_sd
        return np.r_[
            (prediction(x) - signal) / sigma_data,
            penalties,
        ]
    rng=np.random.default_rng(seed)
    starts=[x0]
    for _ in range(N_STARTS-1):
        z=x0.copy()
        z[:3]+=rng.normal(0,[0.03,1.0,1.0])
        z[3:]+=rng.normal(0,0.25,3)
        starts.append(np.clip(z,lower+1e-7,upper-1e-7))
    candidates=[]
    for z in starts:
        try:
            fit=least_squares(residual,z,bounds=(lower,upper),max_nfev=MAX_NFEV)
            if np.isfinite(fit.cost): candidates.append(fit)
        except (ValueError,FloatingPointError):
            continue
    if not candidates: raise RuntimeError("All multistart attempts failed")
    best=min(candidates,key=lambda r:r.cost)
    vals=unpack(best.x)
    q=fractions_to_q(vals)
    pred=prediction(best.x)
    return {**vals,**q,"ODI":float(2/np.pi*np.arctan2(1.0,vals["kappa"])),
        "cost":float(best.cost),"signal_rmse":float(np.sqrt(np.mean((pred-signal)**2))),
        "success":float(best.success),"n_evaluations":float(best.nfev)}

def chunk_name(name,i):
    return OUTPUT_DIR / f"{PREFIX}.{name}.chunk{i+1:04d}of{N_CHUNKS:04d}.nii.gz"
def fit_chunk(i):
    img,mask,acq,prefit=load_inputs()
    indices=np.array_split(np.flatnonzero(mask),N_CHUNKS)[i]
    started=stamp(); clock=time.perf_counter()
    status(i,"RUNNING",voxels=len(indices),started_at=started,job_id=os.environ.get("SLURM_JOB_ID","local"))
    maps={name:np.full(mask.shape,np.nan,dtype=np.float32) for name in PARAMETERS}
    try:
        for j,idx in enumerate(indices):
            xyz=np.unravel_index(int(idx),mask.shape)
            signal=np.asarray(img.dataobj[xyz+(slice(None),)],float)
            b0 = np.asarray(acq.b_values) <= 0.05
            b0_mean = float(np.mean(signal[b0]))
            if not np.isfinite(b0_mean) or b0_mean <= 0:
                maps["success"][xyz] = 0
                print(f"Voxel {xyz} failed: invalid mean b0", flush=True)
                continue
            signal = signal / b0_mean
            try:
                theta,phi=dti_angles(signal,acq)
                pre={name:float(v[xyz]) for name,v in prefit.items()}
                fitted=fit_voxel(signal,acq,pre,theta,phi,RNG_SEED+int(idx))
                for name in PARAMETERS: maps[name][xyz]=fitted[name]
            except Exception as exc:
                # Retain failed voxels as NaN; count and report failures.
                print(f"Voxel {xyz} failed: {exc}",flush=True)
                maps["success"][xyz]=0
            if (j+1)%5==0 or j+1==len(indices):
                status(i,"RUNNING",voxels=len(indices),completed=j+1,
                    started_at=started,elapsed_seconds=round(time.perf_counter()-clock,1))
        OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
        for name,arr in maps.items():
            hdr=img.header.copy(); hdr.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(arr,img.affine,hdr),str(chunk_name(name,i)))
        runtime=time.perf_counter()-clock
        status(i,"COMPLETE",voxels=len(indices),runtime_seconds=f"{runtime:.3f}",
            started_at=started,finished_at=stamp(),n_success=int(np.sum(maps["success"]==1)))
    except Exception as exc:
        status(i,"FAILED",voxels=len(indices),started_at=started,finished_at=stamp(),error=str(exc))
        raise

def merge_and_qc():
    img,mask,acq,_=load_inputs()
    for i in range(N_CHUNKS):
        if read_status(i).get("status")!="COMPLETE":
            raise RuntimeError(f"Chunk {i+1} did not complete; refusing to merge")
    merged={}
    for name in PARAMETERS:
        out=np.full(mask.shape,np.nan,dtype=np.float32)
        for i in range(N_CHUNKS):
            p=chunk_name(name,i)
            if not p.exists():raise FileNotFoundError(p)
            chunk=load_volume(p,mask.shape)
            valid=np.isfinite(chunk)
            out[valid]=chunk[valid]
        merged[name]=out
        nib.save(nib.Nifti1Image(out,img.affine),str(OUTPUT_DIR/f"{PREFIX}.{name}.nii.gz"))
    qc=OUTPUT_DIR/"qc";qc.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[]
    for name,arr in merged.items():
        values=arr[mask]; v=values[np.isfinite(values)]
        row={"parameter":name,"n_mask":int(mask.sum()),"n_finite":len(v),"n_nan":int(np.isnan(values).sum())}
        if len(v):
            row.update(mean=float(v.mean()),std=float(v.std()),median=float(np.median(v)),
                min=float(v.min()),p05=float(np.percentile(v,5)),p95=float(np.percentile(v,95)),
                max=float(v.max()),n_zero=int((v==0).sum()))
            fig,ax=plt.subplots(figsize=(7,5));ax.hist(v,bins=30,edgecolor="black")
            ax.set(title=name,xlabel=name,ylabel="Voxels");fig.tight_layout()
            fig.savefig(qc/f"hist_{name}.png",dpi=180);plt.close(fig)
        rows.append(row)
    fields=["parameter","n_mask","n_finite","n_nan","mean","std","median","min","p05","p95","max","n_zero"]
    with (qc/"parameter_statistics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    records=[read_status(i) for i in range(N_CHUNKS)]
    durations=[float(r["runtime_seconds"]) for r in records]
    starts=[datetime.fromisoformat(r["started_at"]) for r in records]
    ends=[datetime.fromisoformat(r["finished_at"]) for r in records]
    elapsed=(max(ends)-min(starts)).total_seconds()
    fractions=sum(merged[n][mask] for n in ("neurite_fraction","extra_fraction","soma_fraction","csf_fraction"))
    errors=np.abs(fractions[np.isfinite(fractions)]-1)
    def fmt(x):return "<0.0001" if x<0.0001 else f"{x:.6g}"
    with (qc/"qc_summary.txt").open("w") as f:
        f.write(f"NODDI+SOMA TORT2 QC\nMask: {MASK_PATH}\nVoxels: {mask.sum()}\n")
        f.write(f"N starts: {N_STARTS}\nChunks: {N_CHUNKS}\n")
        f.write(f"Successful voxels: {int(np.sum(merged['success'][mask]==1))}/{mask.sum()}\n")
        f.write(f"Elapsed fitting window: {elapsed/60:.2f} min\n")
        f.write(f"Summed chunk runtime: {sum(durations)/60:.2f} min (not CPU time)\n")
        f.write(f"Longest chunk runtime: {max(durations)/60:.2f} min\n")
        f.write(f"Mean runtime per voxel: {sum(durations)/mask.sum():.2f} s\n")
        if len(errors): f.write(f"Fraction sum mean abs error: {fmt(float(errors.mean()))}\nFraction sum max abs error: {fmt(float(errors.max()))}\n")
        f.write("Signal scaling: voxelwise mean b0 normalisation\n")
        f.write(f"Assumed SNR: {SNR:.1f}\nPrior SDs (kappa,q_n,q_ex,q_fw): {KAPPA_PRIOR_SD}, {Q_NEURITE_PRIOR_SD}, {Q_EXTRA_PRIOR_SD}, {Q_FW_PRIOR_SD}\n")
    print(f"Merged {len(PARAMETERS)} maps; QC in {qc}")

def submit():
    img,mask,acq,prefit=load_inputs()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True);LOG_DIR.mkdir(exist_ok=True);STATUS_DIR.mkdir(exist_ok=True)
    jobs=[]
    for i in range(N_CHUNKS):
        cmd=["fsl_sub","--name",f"soma_{i+1:03d}","--queue", CLUSTER_QUEUE,"--logdir",str(LOG_DIR),
             "--jobtime",str(CLUSTER_TIME_MINUTES),"--jobram",str(CLUSTER_RAM_GB),
             "--export",f"NODDI_SOMA_CHUNK_INDEX={i},SOMA_INPUT_DIR={PREPROC_DIR},SOMA_MASK_FILENAME={MASK_FILENAME},SOMA_OUTPUT_FOLDER={OUTPUT_FOLDER_NAME},SOMA_PREFIT_FOLDER={NODDI_PREFIT_FOLDER}",PYTHON_EXECUTABLE,str(SCRIPT_PATH)]
        result=subprocess.run(cmd,check=True,capture_output=True,text=True)
        job=result.stdout.strip().splitlines()[-1];jobs.append(job)
        status(i,"QUEUED",voxels=len(np.array_split(np.flatnonzero(mask),N_CHUNKS)[i]),job_id=job)
        print(f"Chunk {i+1}/{N_CHUNKS}: {job}")
    cmd=["fsl_sub","--name","soma_merge","--queue", MERGE_QUEUE,"--logdir",str(LOG_DIR),"--jobtime",str(MERGE_TIME_MINUTES),
         "--jobram",str(MERGE_RAM_GB),"--jobhold",",".join(jobs),"--export",f"NODDI_SOMA_MERGE_ONLY=1,SOMA_INPUT_DIR={PREPROC_DIR},SOMA_MASK_FILENAME={MASK_FILENAME},SOMA_OUTPUT_FOLDER={OUTPUT_FOLDER_NAME},SOMA_PREFIT_FOLDER={NODDI_PREFIT_FOLDER}",
         PYTHON_EXECUTABLE,str(SCRIPT_PATH)]
    result=subprocess.run(cmd,check=True,capture_output=True,text=True)
    print("Merge job:",result.stdout.strip())

# =========================== BATCH SUBMISSION ============================
ANIMALS = ("M02841978", "M02841979", "M02841980", "M02841981")
TIMEPOINTS = ("pre", "post")
MASKS = ("injection_site_mask.nii.gz", "contralateral_site_mask.nii.gz")
BATCH_ROOT = Path("/home/fs0/zcn545/meeting_softlinks/my_scratch/LPS_Dose_Pilot/MRI")
# Use the same input folder layout for each dataset.
# NODDI pre-fit maps MUST exist in the chosen prefit subfolder for every dataset/mask.
BATCH_MODE = os.environ.get("SOMA_BATCH_MODE", "1") == "1"

def submit_batch():
    import shlex
    tasks = []
    for animal in ANIMALS:
        for tp in TIMEPOINTS:
            directory = BATCH_ROOT / animal / tp / "diffusion/preprocessing/fitting_input"
            for mask_file in MASKS:
                label = "injection" if mask_file.startswith("injection") else "contralateral"
                # Distinct prefit directories per mask avoid using the wrong ROI's priors.
                prefit_folder = f"noddi_prefit_{label}"
                out_folder = f"noddi_soma_{label}_{N_STARTS}starts"
                tasks.append((directory, mask_file, prefit_folder, out_folder))
    print(f"\nChecking {len(tasks)} dataset/mask fits before submission...", flush=True)
    problems = []
    for directory, mask_file, prefit_folder, out_folder in tasks:
        required = [directory / x for x in (DATA_FILENAME, BVAL_FILENAME, BVEC_FILENAME, mask_file)]
        required += [directory / prefit_folder / f"noddi.{x}.nii.gz" for x in
                     ("kappa", "neurite_fraction", "extra_fraction", "csf_fraction")]
        for path in required:
            if not path.exists(): problems.append(str(path))
    if problems:
        print("Missing inputs (nothing submitted):")
        for path in problems: print("  ", path)
        raise SystemExit("Run NODDI pre-fitting for each ROI first, or set the correct prefit folder names.")
    if os.environ.get("SOMA_DRY_RUN", "1") == "1":
        for directory, mask_file, prefit_folder, out_folder in tasks:
            print(directory, mask_file, '->', out_folder, 'prefit=', prefit_folder)
        print("DRY RUN ONLY. Set SOMA_DRY_RUN=0 to submit.")
        return
    for directory, mask_file, prefit_folder, out_folder in tasks:
        env = os.environ.copy()
        env.update(SOMA_BATCH_MODE="0", SOMA_INPUT_DIR=str(directory),
                   SOMA_MASK_FILENAME=mask_file, SOMA_OUTPUT_FOLDER=out_folder,
                   SOMA_PREFIT_FOLDER=prefit_folder)
        print("Submitting", directory, mask_file, flush=True)
        subprocess.run([PYTHON_EXECUTABLE, str(SCRIPT_PATH)], env=env, check=True)

if __name__=="__main__":
    if CHUNK_ENV is not None: fit_chunk(int(CHUNK_ENV))
    elif MERGE_ONLY: merge_and_qc()
    elif BATCH_MODE and ON_CLUSTER: submit_batch()
    elif ON_CLUSTER: submit()
    else:
        for i in range(N_CHUNKS):fit_chunk(i)
        merge_and_qc()
