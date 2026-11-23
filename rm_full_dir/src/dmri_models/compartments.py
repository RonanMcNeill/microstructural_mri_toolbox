"""Composable diffusion and relaxation signal components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol
import warnings

import numpy as np
from scipy.special import erf, erfi, factorial, hyp1f1, i0e, lpmv

from .acquisition import Acquisition

Parameters = Mapping[str, float | np.ndarray]
_GAMMA_RAD_PER_MS_PER_T = 2.6752218744e8 * 1e-3

# First 60 roots used by the Camino/FSL DIVE GPD sphere implementation.
_SPHERE_ROOTS = np.array(
    [
        2.08157597781810,
        5.94036999057271,
        9.20584014293667,
        12.4044450219020,
        15.5792364103872,
        18.7426455847748,
        21.8996964794928,
        25.0528252809930,
        28.2033610039524,
        31.3520917265645,
        34.4995149213670,
        37.6459603230864,
        40.7916552312719,
        43.9367614714198,
        47.0813974121542,
        50.2256516491831,
        53.3695918204908,
        56.5132704621986,
        59.6567290035279,
        62.8000005565198,
        65.9431119046553,
        69.0860849466452,
        72.2289377620154,
        75.3716854092873,
        78.5143405319308,
        81.6569138240367,
        84.7994143922025,
        87.9418500396598,
        91.0842274914688,
        94.2265525745684,
        97.3688303629010,
        100.511065295271,
        103.653261271734,
        106.795421732944,
        109.937549725876,
        113.079647958579,
        116.221718846033,
        116.221718846033,
        119.363764548757,
        122.505787005472,
        125.647787960854,
        128.789768989223,
        131.931731514843,
        135.073676829384,
        138.215606107009,
        141.357520417437,
        144.499420737305,
        147.641307960079,
        150.783182904724,
        153.925046323312,
        157.066898907715,
        166.492397790874,
        169.634212946261,
        172.776020008465,
        175.917819411203,
        179.059611557741,
        182.201396823524,
        185.343175558534,
        188.484948089409,
        191.626714721361,
    ],
    dtype=float,
)


class Compartment(Protocol):
    """Contract implemented by all unit-normalised signal compartments."""

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        ...


class OrientationDistribution(Protocol):
    """Interface for orientation distributions used in response/FOD convolution."""

    def quadrature(self, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        """Return unit directions and normalised integration weights."""
        ...


def _unit_vector_from_spherical_angles(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def _spherical_product_quadrature(
    n_theta: int,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_theta < 2 or n_phi < 4:
        raise ValueError("Quadrature requires at least n_theta=2 and n_phi=4.")
    cos_theta_nodes, cos_theta_weights = np.polynomial.legendre.leggauss(n_theta)
    phi_nodes = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    cos_grid, phi_grid = np.meshgrid(cos_theta_nodes, phi_nodes, indexing="ij")
    sin_grid = np.sqrt(np.maximum(0.0, 1.0 - cos_grid**2))

    directions = np.column_stack(
        (
            (sin_grid * np.cos(phi_grid)).reshape(-1),
            (sin_grid * np.sin(phi_grid)).reshape(-1),
            cos_grid.reshape(-1),
        )
    )
    weights = np.repeat(cos_theta_weights, n_phi) * (2.0 * np.pi / n_phi)
    return directions, weights


def _validate_even_l_max(l_max: int) -> None:
    if l_max < 0 or l_max % 2 != 0:
        raise ValueError("l_max must be a non-negative even integer.")


def _even_sh_indices(l_max: int) -> tuple[np.ndarray, np.ndarray]:
    _validate_even_l_max(l_max)
    l_values: list[int] = []
    m_values: list[int] = []
    for l_order in range(0, l_max + 1, 2):
        for m_degree in range(-l_order, l_order + 1):
            l_values.append(l_order)
            m_values.append(m_degree)
    return np.asarray(l_values, dtype=int), np.asarray(m_values, dtype=int)


def _real_even_sh_basis(
    directions: np.ndarray,
    l_max: int,
    *,
    cs_phase: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions must have shape (N, 3).")
    if not np.all(np.isfinite(directions)):
        raise ValueError("directions must be finite.")

    unit = np.asarray(directions, dtype=float).copy()
    norms = np.linalg.norm(unit, axis=1)
    if np.any(np.isclose(norms, 0.0)):
        raise ValueError("directions must not contain zero vectors.")
    unit /= norms[:, None]

    l_values, m_values = _even_sh_indices(l_max)
    x = unit[:, 0]
    y = unit[:, 1]
    z = unit[:, 2]
    phi = np.arctan2(y, x)
    cos_theta = np.clip(z, -1.0, 1.0)

    basis = np.empty((unit.shape[0], l_values.size), dtype=float)
    for index, (l_order, m_degree) in enumerate(zip(l_values, m_values, strict=True)):
        abs_m = abs(m_degree)
        norm = np.sqrt(
            ((2.0 * l_order + 1.0) / (4.0 * np.pi))
            * factorial(l_order - abs_m, exact=False)
            / factorial(l_order + abs_m, exact=False)
        )
        legendre = lpmv(abs_m, l_order, cos_theta)
        if m_degree > 0:
            phi_term = np.cos(m_degree * phi)
        elif m_degree < 0:
            phi_term = np.sin(-m_degree * phi)
        else:
            phi_term = np.ones_like(phi)
        extra = 1.0 if m_degree == 0 else np.sqrt(2.0)
        if cs_phase:
            extra *= (-1.0) ** m_degree
        basis[:, index] = extra * norm * phi_term * legendre
    return basis, l_values, m_values


def _rotational_invariants(
    coefficients: np.ndarray,
    l_values: np.ndarray,
    *,
    normalise: bool,
) -> np.ndarray:
    unique_l = np.unique(l_values)
    invariants = np.empty(unique_l.size, dtype=float)
    for index, l_order in enumerate(unique_l):
        scale = 4.0 * np.pi * (2.0 * l_order + 1.0)
        invariants[index] = np.sum(coefficients[l_values == l_order] ** 2) / scale
    if normalise:
        if invariants[0] <= 0:
            raise ValueError("Cannot normalise rotational invariants with non-positive l=0 term.")
        invariants = invariants / invariants[0]
    return invariants


def _single_measurement_acquisition(
    acquisition: Acquisition,
    index: int,
) -> Acquisition:
    gradients = None
    if acquisition.gradients is not None:
        gradients = acquisition.gradients[index : index + 1, :]

    def _slice_optional(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        return values[index : index + 1]

    return Acquisition(
        b_values=acquisition.b_values[index : index + 1],
        gradients=gradients,
        b_tensor_shape=_slice_optional(acquisition.b_tensor_shape),
        pulse_duration=_slice_optional(acquisition.pulse_duration),
        diffusion_time=_slice_optional(acquisition.diffusion_time),
        gradient_strength=_slice_optional(acquisition.gradient_strength),
        echo_time=_slice_optional(acquisition.echo_time),
        inversion_time=_slice_optional(acquisition.inversion_time),
        repetition_time=_slice_optional(acquisition.repetition_time),
    )


def _normalised_gradient_directions(acquisition: Acquisition) -> np.ndarray:
    if acquisition.gradients is None:
        raise ValueError("SMI-style SH convolution requires acquisition gradients.")

    directions = np.asarray(acquisition.gradients, dtype=float).copy()
    norms = np.linalg.norm(directions, axis=1)
    nonzero = norms > 1e-12
    directions[nonzero] /= norms[nonzero, None]
    directions[~nonzero] = np.array([0.0, 0.0, 1.0])
    return directions


def _orientation_with_projection(
    gradient: np.ndarray,
    projection: float,
) -> np.ndarray:
    if not np.isfinite(projection) or projection < -1.0 or projection > 1.0:
        raise ValueError("projection must be finite and in [-1, 1].")

    if abs(gradient[2]) < 0.9:
        reference = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)

    tangent = np.cross(gradient, reference)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-12:
        tangent = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent_norm = 1.0
    tangent /= tangent_norm

    orthogonal = np.sqrt(max(0.0, 1.0 - projection**2))
    direction = projection * gradient + orthogonal * tangent
    direction /= np.linalg.norm(direction)
    return direction


def _orthonormal_frame_from_angles(
    theta: float,
    phi: float,
    psi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e3 = _unit_vector_from_spherical_angles(theta, phi)

    # Build a stable tangent basis around e3.
    if abs(e3[2]) < 0.9:
        reference = np.array([0.0, 0.0, 1.0])
    else:
        reference = np.array([1.0, 0.0, 0.0])
    e1_base = np.cross(reference, e3)
    e1_base /= np.linalg.norm(e1_base)
    e2_base = np.cross(e3, e1_base)

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    e1 = cos_psi * e1_base + sin_psi * e2_base
    e2 = -sin_psi * e1_base + cos_psi * e2_base
    return e1, e2, e3


def _bingham_shape_matrix_from_angles(
    kappa1: float,
    kappa2: float,
    theta: float,
    phi: float,
    psi: float,
) -> np.ndarray:
    e1, e2, _ = _orthonormal_frame_from_angles(theta, phi, psi)
    return kappa1 * np.outer(e1, e1) + kappa2 * np.outer(e2, e2)


def _watson_shape_matrix_from_angles(
    kappa: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    mean_axis = _unit_vector_from_spherical_angles(theta, phi)
    return kappa * np.outer(mean_axis, mean_axis)


def _hypergeometric_normaliser_quadrature(
    shape_matrix: np.ndarray,
    *,
    n_theta: int,
    n_phi: int,
) -> float:
    """Approximate 4*pi*1F1(1/2, 3/2, Z) via spherical quadrature.

    For symmetric matrix Z, this equals the sphere integral of exp(n^T Z n).
    """
    if shape_matrix.shape != (3, 3) or not np.all(np.isfinite(shape_matrix)):
        raise ValueError("shape_matrix must be a finite 3x3 matrix.")
    if not np.allclose(shape_matrix, shape_matrix.T, atol=1e-12):
        raise ValueError("shape_matrix must be symmetric.")

    directions, solid_angle_weights = _spherical_product_quadrature(n_theta, n_phi)
    quadratic = np.einsum("ni,ij,nj->n", directions, shape_matrix, directions)
    max_quadratic = np.max(quadratic)
    # Stabilise the exponential for highly concentrated distributions.
    return float(
        np.exp(max_quadratic)
        * np.sum(solid_angle_weights * np.exp(quadratic - max_quadratic))
    )


@dataclass(frozen=True)
class WatsonFOD:
    """Watson orientation distribution, parameterised by kappa and mean axis."""

    kappa: str
    theta: str
    phi: str
    n_theta: int = 24
    n_phi: int = 48

    @staticmethod
    def normalisation_constant(kappa: float) -> float:
        """Exact Watson normalisation constant on S2.

        For density exp(kappa * (mu dot n)^2), the normalisation is
        4*pi*1F1(1/2, 3/2, kappa).
        """
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and non-negative.")
        return float(4.0 * np.pi * hyp1f1(0.5, 1.5, kappa))

    def quadrature(self, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        kappa = _nonnegative_parameter(parameters, self.kappa)
        theta = float(parameters[self.theta])
        phi = float(parameters[self.phi])
        if not np.isfinite(theta) or not np.isfinite(phi):
            raise ValueError("Watson orientation angles must be finite.")

        mean_axis = _unit_vector_from_spherical_angles(theta, phi)
        directions, solid_angle_weights = _spherical_product_quadrature(
            self.n_theta,
            self.n_phi,
        )
        projection = directions @ mean_axis

        # Use hypergeometric normalisation then discretely renormalise for quadrature.
        density = np.exp(kappa * projection**2) / self.normalisation_constant(kappa)
        weighted = solid_angle_weights * density
        normaliser = np.sum(weighted)
        if normaliser <= 0 or not np.isfinite(normaliser):
            raise ValueError("Invalid Watson quadrature normalisation.")
        return directions, weighted / normaliser


@dataclass(frozen=True)
class BinghamFOD:
    """Bingham orientation distribution with an explicit principal frame.

    The density is proportional to
    exp(kappa1 * (n dot e1)^2 + kappa2 * (n dot e2)^2),
    where e1/e2/e3 form an orthonormal frame. The major axis e3 is
    parameterised by (theta, phi), and psi rotates e1/e2 around e3.
    """

    kappa1: str
    kappa2: str
    theta: str
    phi: str
    psi: str
    n_theta: int = 24
    n_phi: int = 48

    def quadrature(self, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        kappa1 = float(parameters[self.kappa1])
        kappa2 = float(parameters[self.kappa2])
        theta = float(parameters[self.theta])
        phi = float(parameters[self.phi])
        psi = float(parameters[self.psi])

        if not all(np.isfinite(item) for item in (kappa1, kappa2, theta, phi, psi)):
            raise ValueError("Bingham parameters must be finite.")

        e1, e2, _ = _orthonormal_frame_from_angles(theta, phi, psi)
        directions, solid_angle_weights = _spherical_product_quadrature(
            self.n_theta,
            self.n_phi,
        )
        p1 = directions @ e1
        p2 = directions @ e2

        log_density = kappa1 * p1**2 + kappa2 * p2**2
        log_density -= np.max(log_density)
        weighted = solid_angle_weights * np.exp(log_density)
        normaliser = np.sum(weighted)
        if normaliser <= 0 or not np.isfinite(normaliser):
            raise ValueError("Invalid Bingham quadrature normalisation.")
        return directions, weighted / normaliser


@dataclass(frozen=True)
class SphericalHarmonicFOD:
    """Even-order real SH FOD with SMI-style basis and quadrature export.

    Coefficients can be provided either including or excluding the l=0,m=0 term.
    The resulting density is rectified (if requested) and always normalised over
    the sphere before conversion to quadrature weights.
    """

    coefficients: str
    l_max: int
    include_l0: bool = False
    cs_phase: bool = False
    n_theta: int = 24
    n_phi: int = 48
    rectify: bool = True
    warn_on_renormalisation: bool = True
    renormalisation_tolerance: float = 1e-3

    def __post_init__(self) -> None:
        _validate_even_l_max(self.l_max)

    @staticmethod
    def coefficient_count(l_max: int, *, include_l0: bool) -> int:
        _validate_even_l_max(l_max)
        full_count = int((l_max + 2) * (l_max + 1) // 2)
        return full_count if include_l0 else full_count - 1

    @staticmethod
    def random_unit_directions(
        n_directions: int,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample random unit vectors on S2 for simulation or fitting setup."""
        if n_directions < 1:
            raise ValueError("n_directions must be at least one.")
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(n_directions, 3))
        norms = np.linalg.norm(vectors, axis=1)
        near_zero = norms < 1e-12
        while np.any(near_zero):
            vectors[near_zero] = rng.normal(size=(int(np.sum(near_zero)), 3))
            norms = np.linalg.norm(vectors, axis=1)
            near_zero = norms < 1e-12
        return vectors / norms[:, None]

    @staticmethod
    def spherical_grid(
        *,
        n_theta: int = 120,
        n_phi: int = 240,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a regular spherical grid and flattened Cartesian directions."""
        if n_theta < 2 or n_phi < 4:
            raise ValueError("Grid requires at least n_theta=2 and n_phi=4.")
        theta = np.linspace(0.0, np.pi, n_theta)
        phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)
        directions = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        return theta_grid, phi_grid, directions

    @staticmethod
    def watson_density(
        directions: np.ndarray,
        mean_axis: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """Evaluate a normalised Watson density on arbitrary directions."""
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and non-negative.")
        axis = np.asarray(mean_axis, dtype=float).reshape(-1)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise ValueError("mean_axis must be a finite three-vector.")
        axis_norm = np.linalg.norm(axis)
        if np.isclose(axis_norm, 0.0):
            raise ValueError("mean_axis must be non-zero.")
        axis = axis / axis_norm

        vectors = np.asarray(directions, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 3:
            raise ValueError("directions must have shape (N, 3).")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("directions must be finite.")
        vector_norms = np.linalg.norm(vectors, axis=1)
        if np.any(np.isclose(vector_norms, 0.0)):
            raise ValueError("directions must not contain zero vectors.")
        vectors = vectors / vector_norms[:, None]

        projection = vectors @ axis
        normaliser = 4.0 * np.pi * hyp1f1(0.5, 1.5, kappa)
        return np.exp(kappa * projection**2) / normaliser

    @staticmethod
    def fit_coefficients_from_density(
        directions: np.ndarray,
        density: np.ndarray,
        l_max: int,
        *,
        include_l0: bool,
        cs_phase: bool = False,
    ) -> np.ndarray:
        """Fit real even-order SH coefficients to sampled angular density."""
        density_values = np.asarray(density, dtype=float).reshape(-1)
        basis, _, _ = _real_even_sh_basis(
            np.asarray(directions, dtype=float),
            l_max,
            cs_phase=cs_phase,
        )
        if density_values.size != basis.shape[0]:
            raise ValueError("density length must match the number of sample directions.")
        if not np.all(np.isfinite(density_values)):
            raise ValueError("density must contain finite values.")

        coefficients, *_ = np.linalg.lstsq(basis, density_values, rcond=None)
        if include_l0:
            return coefficients
        return coefficients[1:]

    @staticmethod
    def default_crossing_coefficients(
        l_max: int,
        *,
        crossing: str,
        include_l0: bool,
        kappa: float = 22.0,
        weights: np.ndarray | None = None,
        n_theta: int = 90,
        n_phi: int = 180,
        cs_phase: bool = False,
    ) -> np.ndarray:
        """Return default SH coefficients for single/double/triple crossings."""
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and non-negative.")
        if crossing not in {"single", "double", "triple"}:
            raise ValueError("crossing must be one of: 'single', 'double', 'triple'.")

        if crossing == "single":
            axes = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
        elif crossing == "double":
            axes = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=float,
            )
        else:
            axes = np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        return SphericalHarmonicFOD.crossing_coefficients_from_axes(
            l_max,
            axes=axes,
            weights=weights,
            kappa=kappa,
            include_l0=include_l0,
            n_theta=n_theta,
            n_phi=n_phi,
            cs_phase=cs_phase,
        )

    @staticmethod
    def crossing_coefficients_from_axes(
        l_max: int,
        *,
        axes: np.ndarray,
        weights: np.ndarray | None = None,
        kappa: float = 22.0,
        include_l0: bool,
        n_theta: int = 90,
        n_phi: int = 180,
        cs_phase: bool = False,
    ) -> np.ndarray:
        """Fit SH coefficients for an arbitrary weighted crossing configuration."""
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("kappa must be finite and non-negative.")

        axis_array = np.asarray(axes, dtype=float)
        if axis_array.ndim != 2 or axis_array.shape[1] != 3:
            raise ValueError("axes must have shape (N, 3).")
        if axis_array.shape[0] < 1:
            raise ValueError("At least one crossing axis must be provided.")
        if not np.all(np.isfinite(axis_array)):
            raise ValueError("axes must be finite.")

        axis_norms = np.linalg.norm(axis_array, axis=1)
        if np.any(np.isclose(axis_norms, 0.0)):
            raise ValueError("axes must not contain zero vectors.")
        axis_array = axis_array / axis_norms[:, None]

        if weights is None:
            crossing_weights = np.ones(axis_array.shape[0], dtype=float) / axis_array.shape[0]
        else:
            crossing_weights = np.asarray(weights, dtype=float).reshape(-1)
            if crossing_weights.size != axis_array.shape[0]:
                raise ValueError("weights length must match the number of axes.")
            if np.any(crossing_weights < 0) or not np.all(np.isfinite(crossing_weights)):
                raise ValueError("weights must be finite and non-negative.")
            if np.isclose(np.sum(crossing_weights), 0.0):
                raise ValueError("weights must sum to a positive value.")
            crossing_weights = crossing_weights / np.sum(crossing_weights)

        directions, solid_angle_weights = _spherical_product_quadrature(n_theta, n_phi)
        density = np.zeros(directions.shape[0], dtype=float)
        for axis, weight in zip(axis_array, crossing_weights, strict=True):
            density += float(weight) * SphericalHarmonicFOD.watson_density(
                directions,
                axis,
                kappa,
            )

        integral = np.sum(solid_angle_weights * density)
        if integral <= 0 or not np.isfinite(integral):
            raise ValueError("Generated crossing density could not be normalised.")
        density /= integral

        return SphericalHarmonicFOD.fit_coefficients_from_density(
            directions,
            density,
            l_max,
            include_l0=include_l0,
            cs_phase=cs_phase,
        )

    @staticmethod
    def real_even_basis(
        directions: np.ndarray,
        l_max: int,
        *,
        cs_phase: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _real_even_sh_basis(
            np.asarray(directions, dtype=float),
            l_max,
            cs_phase=cs_phase,
        )

    def _full_coefficients(self, parameters: Parameters) -> np.ndarray:
        coefficients = np.asarray(parameters[self.coefficients], dtype=float).reshape(-1)
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("SH coefficients must be finite.")

        expected = self.coefficient_count(self.l_max, include_l0=self.include_l0)
        if coefficients.size != expected:
            raise ValueError(
                f"Expected {expected} SH coefficients for l_max={self.l_max} "
                f"(include_l0={self.include_l0}), got {coefficients.size}."
            )

        if self.include_l0:
            return coefficients

        isotropic = np.array([1.0 / np.sqrt(4.0 * np.pi)], dtype=float)
        return np.concatenate((isotropic, coefficients))

    def density(self, directions: np.ndarray, parameters: Parameters) -> np.ndarray:
        basis, _, _ = self.real_even_basis(directions, self.l_max, cs_phase=self.cs_phase)
        return basis @ self._full_coefficients(parameters)

    def rotational_invariants(
        self,
        parameters: Parameters,
        *,
        normalise: bool = True,
    ) -> np.ndarray:
        coefficients = self._full_coefficients(parameters)
        _, l_values, _ = self.real_even_basis(
            np.array([[0.0, 0.0, 1.0]], dtype=float),
            self.l_max,
            cs_phase=self.cs_phase,
        )
        return _rotational_invariants(coefficients, l_values, normalise=normalise)

    def quadrature(self, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        directions, solid_angle_weights = _spherical_product_quadrature(
            self.n_theta,
            self.n_phi,
        )

        density = np.asarray(self.density(directions, parameters), dtype=float)
        if density.shape != (directions.shape[0],):
            raise ValueError("SH FOD density must have shape (N,).")
        if not np.all(np.isfinite(density)):
            raise ValueError("SH FOD density must be finite.")

        if np.any(density < 0):
            if self.rectify:
                warnings.warn(
                    "SH FOD had negative values and was rectified before normalisation.",
                    UserWarning,
                    stacklevel=2,
                )
                density = np.clip(density, 0.0, None)
            else:
                raise ValueError(
                    "SH FOD density is negative on the quadrature grid; "
                    "set rectify=True to clip and renormalise."
                )

        integral = float(np.sum(solid_angle_weights * density))
        if integral <= 0.0 or not np.isfinite(integral):
            raise ValueError("Invalid SH FOD normalisation integral.")

        if self.warn_on_renormalisation and abs(integral - 1.0) > self.renormalisation_tolerance:
            warnings.warn(
                "SH FOD coefficients did not integrate to one and were internally normalised.",
                UserWarning,
                stacklevel=2,
            )

        weighted = solid_angle_weights * density / integral
        return directions, weighted


@dataclass(frozen=True)
class SMIKernelConvolved:
    """SMI-like SH convolution for axisymmetric response kernels.

    This class uses a numerical Legendre projection of the response kernel for
    each measurement and applies the SH convolution rule via Funk-Hecke.
    """

    response: Compartment
    fod: SphericalHarmonicFOD
    orientation_parameter: str
    n_projections: int = 200

    def __post_init__(self) -> None:
        if self.n_projections < 16:
            raise ValueError("n_projections must be at least 16 for stable kernels.")

    def kernel_by_order(
        self,
        acquisition: Acquisition,
        parameters: Parameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project the response onto even Legendre orders for each measurement."""
        if acquisition.gradients is None:
            raise ValueError("SMI-style SH convolution requires acquisition gradients.")

        nodes, weights = np.polynomial.legendre.leggauss(self.n_projections)
        orders = np.arange(0, self.fod.l_max + 1, 2, dtype=int)
        legendre_matrix = np.stack(
            [
                np.polynomial.legendre.legval(nodes, [0.0] * order + [1.0])
                for order in orders
            ],
            axis=0,
        )

        gradients = _normalised_gradient_directions(acquisition)
        kernels = np.empty((acquisition.n_measurements, orders.size), dtype=float)
        base_parameters = dict(parameters)

        for measurement_index in range(acquisition.n_measurements):
            local_acquisition = _single_measurement_acquisition(
                acquisition,
                measurement_index,
            )

            gradient = gradients[measurement_index]
            responses = np.empty(nodes.size, dtype=float)
            if np.linalg.norm(acquisition.gradients[measurement_index]) < 1e-12:
                # For b0-like entries with no direction, the axisymmetric response
                # is orientation independent, so a constant projection is sufficient.
                base_parameters[self.orientation_parameter] = np.array(
                    [0.0, 0.0, 1.0],
                    dtype=float,
                )
                response_value = float(
                    np.asarray(
                        self.response.signal(local_acquisition, base_parameters),
                        dtype=float,
                    )[0]
                )
                responses.fill(response_value)
            else:
                for node_index, projection in enumerate(nodes):
                    # Construct one representative direction with this cosine
                    # projection against the measurement gradient.
                    base_parameters[self.orientation_parameter] = _orientation_with_projection(
                        gradient,
                        float(projection),
                    )
                    responses[node_index] = float(
                        np.asarray(
                            self.response.signal(local_acquisition, base_parameters),
                            dtype=float,
                        )[0]
                    )

            kernels[measurement_index, :] = 2.0 * np.pi * (
                legendre_matrix @ (weights * responses)
            )

        return orders, kernels

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        """Evaluate SH-convolved signal using projected response kernels."""
        gradients = _normalised_gradient_directions(acquisition)
        basis, l_values, _ = self.fod.real_even_basis(
            gradients,
            self.fod.l_max,
            cs_phase=self.fod.cs_phase,
        )
        coefficients = self.fod._full_coefficients(parameters)

        orders, kernels = self.kernel_by_order(acquisition, parameters)
        order_to_column = {int(order): index for index, order in enumerate(orders)}
        kernel_columns = np.asarray([order_to_column[int(order)] for order in l_values])

        return np.sum(
            basis * coefficients[None, :] * kernels[:, kernel_columns],
            axis=1,
        )

    def fit_coefficients(
        self,
        acquisition: Acquisition,
        signal: np.ndarray,
        parameters: Parameters,
        *,
        include_l0: bool | None = None,
        ridge: float = 0.0,
        solver: str = "lstsq",
    ) -> np.ndarray:
        """Estimate SH coefficients from signal via linear least squares.

        Args:
            acquisition: Measurement acquisition.
            signal: Observed signal values, one per measurement.
            parameters: Response-compartment parameters.
            include_l0: Whether to fit the l=0 SH term. If ``None``, defaults
                to ``self.fod.include_l0``.
            ridge: Non-negative Tikhonov regularisation strength. When
                positive, the fit solves normal equations with ridge penalty.
            solver: Linear solver used when ``ridge == 0``. Supported values
                are ``"lstsq"`` and ``"pinv"``.
        """
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if solver not in {"lstsq", "pinv"}:
            raise ValueError("solver must be one of {'lstsq', 'pinv'}.")

        observed = np.asarray(signal, dtype=float).reshape(-1)
        if observed.size != acquisition.n_measurements:
            raise ValueError("signal length must match acquisition.n_measurements.")

        gradients = _normalised_gradient_directions(acquisition)
        basis, l_values, _ = self.fod.real_even_basis(
            gradients,
            self.fod.l_max,
            cs_phase=self.fod.cs_phase,
        )
        orders, kernels = self.kernel_by_order(acquisition, parameters)
        order_to_column = {int(order): index for index, order in enumerate(orders)}
        kernel_columns = np.asarray([order_to_column[int(order)] for order in l_values])
        design = basis * kernels[:, kernel_columns]

        def _solve_linear(design_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
            if ridge == 0.0:
                if solver == "lstsq":
                    solution, *_ = np.linalg.lstsq(design_matrix, targets, rcond=None)
                    return solution
                return np.linalg.pinv(design_matrix) @ targets
            lhs = design_matrix.T @ design_matrix
            lhs += ridge * np.eye(lhs.shape[0], dtype=float)
            rhs = design_matrix.T @ targets
            return np.linalg.solve(lhs, rhs)

        if include_l0 is None:
            include_l0 = self.fod.include_l0

        l0_mask = l_values == 0
        if np.sum(l0_mask) != 1:
            raise ValueError("Unexpected SH basis layout: expected exactly one l=0 term.")

        if include_l0:
            return _solve_linear(design, observed)

        c0 = 1.0 / np.sqrt(4.0 * np.pi)
        observed_residual = observed - design[:, l0_mask][:, 0] * c0
        anisotropic = _solve_linear(design[:, ~l0_mask], observed_residual)
        return anisotropic


@dataclass(frozen=True)
class FODConvolved:
    """Convolve an oriented response compartment with an orientation distribution."""

    response: Compartment
    fod: OrientationDistribution
    orientation_parameter: str

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        directions, weights = self.fod.quadrature(parameters)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("FOD quadrature directions must have shape (N, 3).")
        if weights.ndim != 1 or weights.shape[0] != directions.shape[0]:
            raise ValueError("FOD quadrature weights must match direction count.")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("FOD quadrature weights must be finite and non-negative.")

        output = np.zeros(acquisition.n_measurements, dtype=float)
        base_parameters = dict(parameters)
        for direction, weight in zip(directions, weights, strict=True):
            base_parameters[self.orientation_parameter] = direction
            output += float(weight) * np.asarray(
                self.response.signal(acquisition, base_parameters),
                dtype=float,
            )
        return output


@dataclass(frozen=True)
class WatsonStickHypergeometric:
    """Watson-stick signal via the hypergeometric-ratio formulation.

    This computes the ratio of normalising constants implied by the ball and
    rackets model with Watson ODF,
    C(B - b d beta gg^T) / C(B), where C(Z)=4*pi*1F1(1/2,3/2,Z),
    and applies the b-tensor-shape isotropic offset for beta < 1.
    """

    diffusivity: str
    kappa: str
    theta: str
    phi: str
    n_theta: int = 24
    n_phi: int = 48

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        d_parallel = _positive_parameter(parameters, self.diffusivity)
        kappa = _nonnegative_parameter(parameters, self.kappa)
        theta = float(parameters[self.theta])
        phi = float(parameters[self.phi])
        if not all(np.isfinite(item) for item in (theta, phi)):
            raise ValueError("Watson orientation angles must be finite.")

        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )

        gradients = acquisition.gradients
        if gradients is None:
            if np.allclose(beta, 0.0):
                gradients = np.zeros((acquisition.n_measurements, 3), dtype=float)
            else:
                raise ValueError(
                    "WatsonStickHypergeometric requires gradients for beta > 0."
                )

        shape_matrix = _watson_shape_matrix_from_angles(kappa, theta, phi)
        denominator = _hypergeometric_normaliser_quadrature(
            shape_matrix,
            n_theta=self.n_theta,
            n_phi=self.n_phi,
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("Invalid Watson hypergeometric denominator.")

        signal = np.empty(acquisition.n_measurements, dtype=float)
        for index, (b_value, beta_value) in enumerate(
            zip(acquisition.b_values, beta, strict=True)
        ):
            attenuation_matrix = (
                shape_matrix
                - (b_value * d_parallel * beta_value)
                * np.outer(gradients[index], gradients[index])
            )
            numerator = _hypergeometric_normaliser_quadrature(
                attenuation_matrix,
                n_theta=self.n_theta,
                n_phi=self.n_phi,
            )
            isotropic_offset = np.exp(-b_value * d_parallel * (1.0 - beta_value) / 3.0)
            signal[index] = isotropic_offset * (numerator / denominator)
        return signal


@dataclass(frozen=True)
class BinghamStickHypergeometric:
    """Bingham-stick signal via the hypergeometric-ratio formulation.

    This computes the ratio C(B - b d beta gg^T) / C(B), with
    C(Z)=4*pi*1F1(1/2,3/2,Z), using stable spherical quadrature to evaluate
    the matrix-argument hypergeometric normaliser.
    """

    diffusivity: str
    kappa1: str
    kappa2: str
    theta: str
    phi: str
    psi: str
    n_theta: int = 24
    n_phi: int = 48

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        d_parallel = _positive_parameter(parameters, self.diffusivity)
        kappa1 = float(parameters[self.kappa1])
        kappa2 = float(parameters[self.kappa2])
        theta = float(parameters[self.theta])
        phi = float(parameters[self.phi])
        psi = float(parameters[self.psi])

        if not all(np.isfinite(item) for item in (kappa1, kappa2, theta, phi, psi)):
            raise ValueError("Bingham parameters must be finite.")

        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )

        gradients = acquisition.gradients
        if gradients is None:
            if np.allclose(beta, 0.0):
                gradients = np.zeros((acquisition.n_measurements, 3), dtype=float)
            else:
                raise ValueError(
                    "BinghamStickHypergeometric requires gradients for beta > 0."
                )

        shape_matrix = _bingham_shape_matrix_from_angles(
            kappa1,
            kappa2,
            theta,
            phi,
            psi,
        )
        denominator = _hypergeometric_normaliser_quadrature(
            shape_matrix,
            n_theta=self.n_theta,
            n_phi=self.n_phi,
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("Invalid Bingham hypergeometric denominator.")

        signal = np.empty(acquisition.n_measurements, dtype=float)
        for index, (b_value, beta_value) in enumerate(
            zip(acquisition.b_values, beta, strict=True)
        ):
            attenuation_matrix = (
                shape_matrix
                - (b_value * d_parallel * beta_value)
                * np.outer(gradients[index], gradients[index])
            )
            numerator = _hypergeometric_normaliser_quadrature(
                attenuation_matrix,
                n_theta=self.n_theta,
                n_phi=self.n_phi,
            )
            isotropic_offset = np.exp(-b_value * d_parallel * (1.0 - beta_value) / 3.0)
            signal[index] = isotropic_offset * (numerator / denominator)
        return signal


@dataclass(frozen=True)
class WatsonZeppelinHypergeometric:
    """Watson-zeppelin signal via the hypergeometric-ratio formulation.

    For a zeppelin response, the anisotropic term depends on
    delta_d = d_parallel - d_perpendicular, giving
    C(B - b * beta * delta_d * gg^T) / C(B), with
    C(Z)=4*pi*1F1(1/2,3/2,Z). The isotropic zeppelin contribution is applied as a
    multiplicative offset.
    """

    parallel_diffusivity: str
    perpendicular_diffusivity: str
    kappa: str
    theta: str
    phi: str
    n_theta: int = 24
    n_phi: int = 48

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        d_parallel = _nonnegative_parameter(parameters, self.parallel_diffusivity)
        d_perpendicular = _nonnegative_parameter(
            parameters,
            self.perpendicular_diffusivity,
        )
        delta_diffusivity = d_parallel - d_perpendicular

        kappa = _nonnegative_parameter(parameters, self.kappa)
        theta = float(parameters[self.theta])
        phi = float(parameters[self.phi])
        if not all(np.isfinite(item) for item in (theta, phi)):
            raise ValueError("Watson orientation angles must be finite.")

        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )

        gradients = acquisition.gradients
        if gradients is None:
            if np.allclose(beta, 0.0):
                gradients = np.zeros((acquisition.n_measurements, 3), dtype=float)
            else:
                raise ValueError(
                    "WatsonZeppelinHypergeometric requires gradients for beta > 0."
                )

        shape_matrix = _watson_shape_matrix_from_angles(kappa, theta, phi)
        denominator = _hypergeometric_normaliser_quadrature(
            shape_matrix,
            n_theta=self.n_theta,
            n_phi=self.n_phi,
        )
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("Invalid Watson hypergeometric denominator.")

        signal = np.empty(acquisition.n_measurements, dtype=float)
        for index, (b_value, beta_value) in enumerate(
            zip(acquisition.b_values, beta, strict=True)
        ):
            attenuation_matrix = (
                shape_matrix
                - (b_value * beta_value * delta_diffusivity)
                * np.outer(gradients[index], gradients[index])
            )
            numerator = _hypergeometric_normaliser_quadrature(
                attenuation_matrix,
                n_theta=self.n_theta,
                n_phi=self.n_phi,
            )
            isotropic_zeppelin = np.exp(
                -b_value
                * (
                    d_perpendicular
                    + delta_diffusivity * (1.0 - beta_value) / 3.0
                )
            )
            signal[index] = isotropic_zeppelin * (numerator / denominator)
        return signal


@dataclass(frozen=True)
class NODDIExtraNeurite:
    """Extra-neurite zeppelin response with NODDI tortuosity coupling.

    The radial diffusivity is computed as ``d_perp = d_par * (1 - v_ic)``, where
    ``v_ic = f_ic / (1 - f_iso)`` when CSF is present and ``v_ic = f_ic`` when
    CSF is not modelled.
    """

    parallel_diffusivity: str
    orientation: str
    neurite_fraction: str = "neurite_fraction"
    csf_fraction: str | None = "csf_fraction"

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        if acquisition.gradients is None:
            raise ValueError("A tensor requires acquisition gradient directions.")
        d_parallel = _nonnegative_parameter(parameters, self.parallel_diffusivity)

        f_ic = float(parameters[self.neurite_fraction])
        if not np.isfinite(f_ic) or f_ic < 0:
            raise ValueError("neurite fraction must be finite and non-negative.")
        if self.csf_fraction is None:
            v_ic = f_ic
        else:
            f_iso = float(parameters[self.csf_fraction])
            if not np.isfinite(f_iso) or f_iso < 0:
                raise ValueError("csf fraction must be finite and non-negative.")
            if f_iso >= 1.0:
                raise ValueError("csf fraction must be less than one.")
            v_ic = f_ic / max(1e-12, 1.0 - f_iso)
        #if v_ic < 0 or v_ic > 1:
        #   raise ValueError("Computed intra-cellular fraction must lie in [0, 1].")
        # Numerical tolerance for simplex-boundary floating-point error.
        #
        # The simplex guarantees physically that:
        #
        #     0 <= f_ic / (1 - f_iso) <= 1
        #
        # but floating-point arithmetic can produce values such as
        # 1.0000000000000002 at the boundary.
        tol = 1e-10

        if v_ic < -tol or v_ic > 1.0 + tol:
            raise ValueError(
                "Computed intra-cellular fraction must lie in [0, 1]. "
                f"Got v_ic={v_ic:.17g}, "
                f"f_ic={f_ic:.17g}, "
                f"f_iso={f_iso:.17g}"
            )

        v_ic = float(
            np.clip(
                v_ic,
                0.0,
                1.0,
            )
        )
        d_perpendicular = d_parallel * (1.0 - v_ic)

        orientation = np.asarray(parameters[self.orientation], dtype=float).reshape(-1)
        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"{self.orientation} must be a finite three-vector.")
        norm = np.linalg.norm(orientation)
        if np.isclose(norm, 0.0):
            raise ValueError(f"{self.orientation} must be non-zero.")
        orientation = orientation / norm

        projection = acquisition.gradients @ orientation
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        directional_factor = beta * (projection**2 - 1.0 / 3.0) + 1.0 / 3.0
        adc_term = (
            (d_parallel - d_perpendicular) * directional_factor
            + d_perpendicular
        )
        return np.exp(-acquisition.b_values * adc_term)

@dataclass
class NODDISomaExtraNeurite:
    """Extra-neurite Zeppelin response with soma-aware tortuosity coupling.

    The coupling is:

        d_ex_parallel =
            d_in * (1 - f_soma / (1 - f_csf))

        d_ex_perp =
            d_ex_parallel *
            (1 - f_neurite / (1 - f_csf))

    When ``csf_fraction`` is None, the fractions are interpreted without
    CSF normalisation.

    Parameters
    ----------
    parallel_diffusivity:
        Name of the intrinsic / neurite parallel diffusivity parameter,
        typically ``"d_in"``.

    orientation:
        Name of the orientation parameter in the parameter dictionary.

    neurite_fraction:
        Name of the neurite fraction parameter.

    soma_fraction:
        Name of the soma fraction parameter.

    csf_fraction:
        Name of the free-water/CSF fraction parameter. Set to ``None``
        when the model does not contain a CSF compartment.
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
            raise ValueError(
                "Neurite fraction must be finite and non-negative."
            )

        if not np.isfinite(f_soma) or f_soma < 0:
            raise ValueError(
                "Soma fraction must be finite and non-negative."
            )

        if self.csf_fraction is None:
            f_csf = 0.0
        else:
            f_csf = float(parameters[self.csf_fraction])

            if not np.isfinite(f_csf) or f_csf < 0:
                raise ValueError(
                    "CSF fraction must be finite and non-negative."
                )

            if f_csf >= 1.0:
                raise ValueError(
                    "CSF fraction must be less than one."
                )

        available_fraction = max(1e-12, 1.0 - f_csf)

        soma_factor = 1.0 - f_soma / available_fraction
        neurite_factor = 1.0 - f_neurite / available_fraction

        if soma_factor < 0:
            raise ValueError(
                "Invalid soma fraction: derived extra-neurite parallel "
                "diffusivity would be negative."
            )

        if neurite_factor < 0:
            raise ValueError(
                "Invalid neurite fraction: derived extra-neurite radial "
                "diffusivity would be negative."
            )

        d_parallel = d_in * soma_factor
        d_perpendicular = d_parallel * neurite_factor

        if d_parallel < 0 or d_perpendicular < 0:
            raise ValueError(
                "Derived extra-neurite diffusivities must be non-negative."
            )

        orientation = np.asarray(
            parameters[self.orientation],
            dtype=float,
        ).reshape(-1)

        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            raise ValueError(
                f"{self.orientation} must be a finite three-vector."
            )

        norm = np.linalg.norm(orientation)

        if np.isclose(norm, 0.0):
            raise ValueError(
                f"{self.orientation} must be non-zero."
            )

        orientation = orientation / norm

        projection = acquisition.gradients @ orientation

        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )

        directional_factor = (
            beta * (projection**2 - 1.0 / 3.0)
            + 1.0 / 3.0
        )

        adc_term = (
            (d_parallel - d_perpendicular)
            * directional_factor
            + d_perpendicular
        )

        return np.exp(
            -acquisition.b_values * adc_term
        )

from dataclasses import dataclass

import numpy as np


@dataclass
class NODDISomaDispersionExtraNeurite:
    """Extra-neurite Zeppelin with soma- and ODI-aware tortuosity coupling.

    The extra-neurite diffusivities are derived from the fitted parameters:

        ODI = (2 / pi) * arctan2(1, kappa)

        v_ic = f_neurite / (1 - f_csf)

        d_ex_parallel =
            d_in
            * (1 - f_soma / (1 - f_csf))
            * (1 - (2/3) * v_ic * ODI)

        d_ex_perp =
            d_in
            * (1 - f_soma / (1 - f_csf))
            * (1 - v_ic + (1/3) * v_ic * ODI)

    When ``csf_fraction`` is None, the CSF normalisation is omitted and
    ``v_ic = f_neurite``.

    Parameters
    ----------
    parallel_diffusivity:
        Name of the intrinsic/parallel diffusivity parameter, typically
        ``"d_in"``.

    orientation:
        Name of the orientation parameter.

    neurite_fraction:
        Name of the neurite fraction parameter.

    soma_fraction:
        Name of the soma fraction parameter.

    csf_fraction:
        Name of the CSF fraction parameter. Set to ``None`` when CSF is
        not included in the model.

    kappa:
        Name of the Watson concentration parameter. ODI is derived from
        this parameter and is NOT independently fitted.
    """

    parallel_diffusivity: str
    orientation: str
    neurite_fraction: str = "neurite_fraction"
    soma_fraction: str = "soma_fraction"
    csf_fraction: str | None = "csf_fraction"
    kappa: str = "kappa"

    def signal(self, acquisition, parameters):
        if acquisition.gradients is None:
            raise ValueError(
                "A tensor requires acquisition gradient directions."
            )

        # ------------------------------------------------------------------
        # Intrinsic diffusivity
        # ------------------------------------------------------------------

        d_in = _nonnegative_parameter(
            parameters,
            self.parallel_diffusivity,
        )

        # ------------------------------------------------------------------
        # Fractions
        # ------------------------------------------------------------------

        f_neurite = float(
            parameters[self.neurite_fraction]
        )

        f_soma = float(
            parameters[self.soma_fraction]
        )

        if not np.isfinite(f_neurite) or f_neurite < 0:
            raise ValueError(
                "Neurite fraction must be finite and non-negative."
            )

        if not np.isfinite(f_soma) or f_soma < 0:
            raise ValueError(
                "Soma fraction must be finite and non-negative."
            )

        # ------------------------------------------------------------------
        # CSF fraction
        # ------------------------------------------------------------------

        if self.csf_fraction is None:
            f_csf = 0.0
        else:
            f_csf = float(
                parameters[self.csf_fraction]
            )

            if not np.isfinite(f_csf) or f_csf < 0:
                raise ValueError(
                    "CSF fraction must be finite and non-negative."
                )

            if f_csf >= 1.0:
                raise ValueError(
                    "CSF fraction must be less than one."
                )

        available_fraction = max(
            1e-12,
            1.0 - f_csf,
        )

        # ------------------------------------------------------------------
        # Watson kappa -> NODDI ODI
        # ------------------------------------------------------------------

        kappa = float(parameters[self.kappa])

        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError(
                "kappa must be finite and non-negative."
            )

        odi = (2.0 / np.pi) * np.arctan2(1.0, kappa)

        # ------------------------------------------------------------------
        # Tissue-normalised intracellular fraction
        # ------------------------------------------------------------------

        v_ic = f_neurite / available_fraction

        if v_ic < 0 or v_ic > 1:
            raise ValueError(
                "Computed intracellular volume fraction must lie in [0, 1]."
            )

        # ------------------------------------------------------------------
        # Soma correction
        # ------------------------------------------------------------------

        soma_factor = (
            1.0
            - f_soma / available_fraction
        )

        if soma_factor < 0:
            raise ValueError(
                "Invalid soma fraction: derived extra-neurite diffusivity "
                "would be negative."
            )

        # ------------------------------------------------------------------
        # Dispersion-aware tortuosity coupling
        # ------------------------------------------------------------------

        parallel_factor = (
            1.0
            - (2.0 / 3.0) * v_ic * odi
        )

        perpendicular_factor = (
            1.0
            - v_ic
            + (1.0 / 3.0) * v_ic * odi
        )

        if parallel_factor < 0:
            raise ValueError(
                "Derived parallel diffusivity would be negative. "
                "Check the kappa/fraction combination."
            )

        if perpendicular_factor < 0:
            raise ValueError(
                "Derived perpendicular diffusivity would be negative. "
                "Check the kappa/fraction combination."
            )

        d_parallel = (
            d_in
            * soma_factor
            * parallel_factor
        )

        d_perpendicular = (
            d_in
            * soma_factor
            * perpendicular_factor
        )

        if d_parallel < 0 or d_perpendicular < 0:
            raise ValueError(
                "Derived extra-neurite diffusivities must be non-negative."
            )

        # ------------------------------------------------------------------
        # Orientation
        # ------------------------------------------------------------------

        orientation = np.asarray(
            parameters[self.orientation],
            dtype=float,
        ).reshape(-1)

        if (
            orientation.shape != (3,)
            or not np.all(np.isfinite(orientation))
        ):
            raise ValueError(
                f"{self.orientation} must be a finite three-vector."
            )

        norm = np.linalg.norm(orientation)

        if np.isclose(norm, 0.0):
            raise ValueError(
                f"{self.orientation} must be non-zero."
            )

        orientation = orientation / norm

        # ------------------------------------------------------------------
        # Directional diffusion signal
        # ------------------------------------------------------------------

        projection = acquisition.gradients @ orientation

        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )

        directional_factor = (
            beta * (projection**2 - 1.0 / 3.0)
            + 1.0 / 3.0
        )

        adc_term = (
            (d_parallel - d_perpendicular)
            * directional_factor
            + d_perpendicular
        )

        return np.exp(
            -acquisition.b_values * adc_term
        )


def _positive_parameter(parameters: Parameters, name: str) -> float:
    value = float(parameters[name])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return value


def _nonnegative_parameter(parameters: Parameters, name: str) -> float:
    value = float(parameters[name])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _axisymmetric_spherical_mean(
    b_values: np.ndarray,
    beta: np.ndarray,
    d_parallel: float,
    d_perpendicular: float,
) -> np.ndarray:
    """Analytical rotational mean of an axisymmetric Gaussian tensor."""
    d_iso = (d_parallel + 2.0 * d_perpendicular) / 3.0
    alpha = b_values * beta * (d_parallel - d_perpendicular)
    factor = np.empty_like(alpha, dtype=float)
    positive = alpha > 1e-7
    negative = alpha < -1e-7
    small = ~(positive | negative)

    root = np.sqrt(alpha[positive])
    factor[positive] = np.sqrt(np.pi) * erf(root) / (2.0 * root)
    root = np.sqrt(-alpha[negative])
    factor[negative] = np.sqrt(np.pi) * erfi(root) / (2.0 * root)
    a = alpha[small]
    factor[small] = 1.0 - a / 3.0 + a**2 / 10.0 - a**3 / 42.0

    constant = (1.0 - beta) * d_iso + beta * d_perpendicular
    return np.exp(-b_values * constant) * factor


@dataclass(frozen=True)
class Ball:
    """Gaussian isotropic diffusion compartment."""

    diffusivity: str

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        d_iso = _positive_parameter(parameters, self.diffusivity)
        return np.exp(-acquisition.b_values * d_iso)

    def spherical_mean(
        self, acquisition: Acquisition, parameters: Parameters
    ) -> np.ndarray:
        return self.signal(acquisition, parameters)


@dataclass(frozen=True)
class Stick:
    """Zero-radius cylinder, oriented or powder averaged."""

    diffusivity: str
    orientation: str | None = None
    powder_average: bool = False

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        d_parallel = _positive_parameter(parameters, self.diffusivity)
        bd = acquisition.b_values * d_parallel

        if self.powder_average:
            return self.spherical_mean(acquisition, parameters)

        if self.orientation is None:
            raise ValueError("An oriented stick requires an orientation parameter name.")
        if acquisition.gradients is None:
            raise ValueError("An oriented stick requires acquisition gradient directions.")

        orientation = np.asarray(parameters[self.orientation], dtype=float).reshape(-1)
        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"{self.orientation} must be a finite three-vector.")
        norm = np.linalg.norm(orientation)
        if np.isclose(norm, 0.0):
            raise ValueError(f"{self.orientation} must be non-zero.")
        orientation = orientation / norm
        projection = acquisition.gradients @ orientation
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        directional_factor = beta * (projection**2 - 1.0 / 3.0) + 1.0 / 3.0
        return np.exp(-bd * directional_factor)

    def spherical_mean(
        self, acquisition: Acquisition, parameters: Parameters
    ) -> np.ndarray:
        d_parallel = _positive_parameter(parameters, self.diffusivity)
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        return _axisymmetric_spherical_mean(
            acquisition.b_values,
            beta,
            d_parallel,
            0.0,
        )


@dataclass(frozen=True)
class Zeppelin:
    """Axisymmetric Gaussian tensor with arbitrary b-tensor shape."""

    parallel_diffusivity: str
    perpendicular_diffusivity: str
    orientation: str

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        if acquisition.gradients is None:
            raise ValueError("A tensor requires acquisition gradient directions.")
        d_parallel = _nonnegative_parameter(parameters, self.parallel_diffusivity)
        d_perpendicular = _nonnegative_parameter(
            parameters, self.perpendicular_diffusivity
        )

        orientation = np.asarray(parameters[self.orientation], dtype=float).reshape(-1)
        if orientation.shape != (3,) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"{self.orientation} must be a finite three-vector.")
        norm = np.linalg.norm(orientation)
        if np.isclose(norm, 0.0):
            raise ValueError(f"{self.orientation} must be non-zero.")
        orientation = orientation / norm

        projection = acquisition.gradients @ orientation
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        directional_factor = beta * (projection**2 - 1.0 / 3.0) + 1.0 / 3.0
        adc_term = (
            (d_parallel - d_perpendicular) * directional_factor
            + d_perpendicular
        )
        return np.exp(-acquisition.b_values * adc_term)

    def spherical_mean(
        self, acquisition: Acquisition, parameters: Parameters
    ) -> np.ndarray:
        d_parallel = _nonnegative_parameter(parameters, self.parallel_diffusivity)
        d_perpendicular = _nonnegative_parameter(
            parameters, self.perpendicular_diffusivity
        )
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        return _axisymmetric_spherical_mean(
            acquisition.b_values,
            beta,
            d_parallel,
            d_perpendicular,
        )


@dataclass(frozen=True)
class Tensor:
    """General Gaussian diffusion tensor with three eigenvalues.

    ``eigenvectors`` names an optional 3×3 orthonormal matrix parameter whose
    columns correspond to the three eigenvalues. If omitted, the laboratory
    axes are used.
    """

    eigenvalues: tuple[str, str, str]
    eigenvectors: str | None = None

    def _diffusion_tensor(self, parameters: Parameters) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(
            [_nonnegative_parameter(parameters, name) for name in self.eigenvalues]
        )
        if self.eigenvectors is None:
            rotation = np.eye(3)
        else:
            rotation = np.asarray(parameters[self.eigenvectors], dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError(f"{self.eigenvectors} must be a finite 3x3 matrix.")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
                raise ValueError(f"{self.eigenvectors} must be orthonormal.")
        return rotation @ np.diag(values) @ rotation.T, values

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        if acquisition.gradients is None:
            raise ValueError("A tensor requires acquisition gradient directions.")
        diffusion_tensor, _ = self._diffusion_tensor(parameters)
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        identity = np.eye(3)
        outer = np.einsum(
            "ni,nj->nij",
            acquisition.gradients,
            acquisition.gradients,
        )
        b_tensors = acquisition.b_values[:, None, None] * (
            ((1.0 - beta) / 3.0)[:, None, None] * identity
            + beta[:, None, None] * outer
        )
        attenuation = np.einsum("nij,ij->n", b_tensors, diffusion_tensor)
        return np.exp(-attenuation)

    def spherical_mean(
        self,
        acquisition: Acquisition,
        parameters: Parameters,
        *,
        quadrature_order: int = 64,
    ) -> np.ndarray:
        """Rotational mean for linear or spherical b-tensor encoding."""
        _, eigenvalues = self._diffusion_tensor(parameters)
        beta = (
            np.ones(acquisition.n_measurements)
            if acquisition.b_tensor_shape is None
            else acquisition.b_tensor_shape
        )
        result = np.empty(acquisition.n_measurements, dtype=float)
        spherical = np.isclose(beta, 0.0)
        linear = np.isclose(beta, 1.0)
        if np.any(~(spherical | linear)):
            raise ValueError(
                "General Tensor spherical_mean currently supports beta=0 or beta=1."
            )
        result[spherical] = np.exp(
            -acquisition.b_values[spherical] * np.mean(eigenvalues)
        )
        if np.any(linear):
            nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
            u = (nodes + 1.0) / 2.0
            weights = weights / 2.0
            d_x, d_y, d_z = eigenvalues
            b = acquisition.b_values[linear, None]
            one_minus_u2 = 1.0 - u[None, :] ** 2
            exponent = -b * (
                d_z * u[None, :] ** 2
                + 0.5 * (d_x + d_y) * one_minus_u2
            )
            bessel_argument = (
                0.5 * b * (d_x - d_y) * one_minus_u2
            )
            result[linear] = np.sum(
                np.exp(exponent + np.abs(bessel_argument))
                * i0e(bessel_argument)
                * weights[None, :],
                axis=1,
            )
        return result


@dataclass(frozen=True)
class Sphere:
    """Water restricted inside impermeable spheres using the GPD approximation.

    The implementation requires rectangular-PGSE timing and gradient strength;
    b-value alone is insufficient to specify restricted diffusion attenuation.
    """

    radius: str
    diffusivity: str

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        missing = [
            name
            for name in ("pulse_duration", "diffusion_time", "gradient_strength")
            if getattr(acquisition, name) is None
        ]
        if missing:
            raise ValueError(
                "A GPA sphere requires acquisition fields: " + ", ".join(missing)
            )

        radius = _positive_parameter(parameters, self.radius)
        diffusivity = _positive_parameter(parameters, self.diffusivity)
        delta = acquisition.pulse_duration
        big_delta = acquisition.diffusion_time
        gradient_t_per_m = acquisition.gradient_strength
        assert delta is not None
        assert big_delta is not None
        assert gradient_t_per_m is not None

        roots_over_radius = _SPHERE_ROOTS[:, None] / radius
        diffusion_rates = diffusivity * roots_over_radius**2
        numerator = (
            2.0 * diffusion_rates * delta[None, :]
            - 2.0
            + 2.0 * np.exp(-diffusion_rates * delta[None, :])
            + 2.0 * np.exp(-diffusion_rates * big_delta[None, :])
            - np.exp(-diffusion_rates * (big_delta - delta)[None, :])
            - np.exp(-diffusion_rates * (big_delta + delta)[None, :])
        )
        denominator = (
            diffusion_rates**2
            * roots_over_radius**2
            * (radius**2 * roots_over_radius**2 - 2.0)
        )
        gpd_sum = np.sum(numerator / denominator, axis=0)

        # The GPA expression uses gradient strength in T/µm.
        gradient_t_per_um = gradient_t_per_m * 1e-6
        log_attenuation = (
            -2.0
            * _GAMMA_RAD_PER_MS_PER_T**2
            * gradient_t_per_um**2
            * gpd_sum
        )
        return np.exp(log_attenuation)


@dataclass(frozen=True)
class T2Relaxation:
    """Apply mono-exponential T2 relaxation to any compartment."""

    compartment: Compartment
    t2: str

    def signal(self, acquisition: Acquisition, parameters: Parameters) -> np.ndarray:
        if acquisition.echo_time is None:
            raise ValueError("T2 relaxation requires echo_time in the acquisition.")
        t2 = _positive_parameter(parameters, self.t2)
        return self.compartment.signal(acquisition, parameters) * np.exp(
            -acquisition.echo_time / t2
        )
