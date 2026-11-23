"""Named model parameters and constraint-preserving vector conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Parameter:
    """A scalar physical parameter used by simulation and fitting."""

    name: str
    initial: float
    bounds: tuple[float, float] = (-np.inf, np.inf)
    vary: bool = True

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not self.name:
            raise ValueError("Parameter names must be non-empty.")
        if not np.isfinite(self.initial):
            raise ValueError(f"Initial value for {self.name} must be finite.")
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {self.name}: {self.bounds}.")
        if not lower <= self.initial <= upper:
            raise ValueError(
                f"Initial value for {self.name} lies outside {self.bounds}."
            )


@dataclass(frozen=True)
class Simplex:
    """A group of positive fractions constrained to sum to one."""

    names: tuple[str, ...]
    initial: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.names) < 2:
            raise ValueError("A simplex requires at least two fractions.")
        if len(self.names) != len(self.initial):
            raise ValueError("Simplex names and initial fractions must have equal length.")
        if len(self.names) != len(set(self.names)):
            raise ValueError("Simplex fraction names must be unique.")
        fractions = np.asarray(self.initial, dtype=float)
        if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0):
            raise ValueError("Initial simplex fractions must be finite and positive.")
        if not np.isclose(fractions.sum(), 1.0, atol=1e-10):
            raise ValueError("Initial simplex fractions must sum to one.")


class ParameterSpace:
    """Map named physical parameters to an optimiser-friendly vector.

    Ordinary varying parameters remain in physical coordinates and use explicit
    bounds. Each K-fraction simplex uses K-1 additive log-ratio coordinates,
    which guarantees positive fractions summing exactly to one.
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        *,
        simplexes: Sequence[Simplex] = (),
    ) -> None:
        self.parameters = tuple(parameters)
        self.simplexes = tuple(simplexes)

        names = [parameter.name for parameter in self.parameters]
        simplex_names = [name for simplex in self.simplexes for name in simplex.names]
        all_names = names + simplex_names
        if len(all_names) != len(set(all_names)):
            raise ValueError("Every parameter and fraction name must be unique.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters) + tuple(
            name for simplex in self.simplexes for name in simplex.names
        )

    @property
    def varying_names(self) -> tuple[str, ...]:
        ordinary = tuple(
            parameter.name for parameter in self.parameters if parameter.vary
        )
        simplex = tuple(
            f"{group.names[index]}/{group.names[-1]}"
            for group in self.simplexes
            for index in range(len(group.names) - 1)
        )
        return ordinary + simplex

    @property
    def dimension(self) -> int:
        return len(self.varying_names)

    def initial_values(self) -> dict[str, float]:
        values = {parameter.name: parameter.initial for parameter in self.parameters}
        for simplex in self.simplexes:
            values.update(zip(simplex.names, simplex.initial, strict=True))
        return values

    def pack(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        """Pack physical values into ordinary and log-ratio coordinates."""
        vector: list[float] = []
        for parameter in self.parameters:
            value = float(values[parameter.name])
            if not np.isfinite(value):
                raise ValueError(f"{parameter.name} must be finite.")
            lower, upper = parameter.bounds
            if not lower <= value <= upper:
                raise ValueError(f"{parameter.name} lies outside {parameter.bounds}.")
            if parameter.vary:
                vector.append(value)

        for simplex in self.simplexes:
            fractions = np.asarray([values[name] for name in simplex.names], dtype=float)
            if not np.all(np.isfinite(fractions)) or np.any(fractions <= 0):
                raise ValueError("Simplex fractions must be finite and positive.")
            if not np.isclose(fractions.sum(), 1.0, atol=1e-10):
                raise ValueError("Simplex fractions must sum to one.")
            vector.extend(np.log(fractions[:-1] / fractions[-1]))
        return np.asarray(vector, dtype=float)

    def unpack(self, vector: ArrayLike) -> dict[str, float]:
        """Convert optimiser coordinates into named physical parameters."""
        packed = np.asarray(vector, dtype=float).reshape(-1)
        if packed.size != self.dimension:
            raise ValueError(
                f"Parameter vector has length {packed.size}; expected {self.dimension}."
            )
        if not np.all(np.isfinite(packed)):
            raise ValueError("Parameter vector must contain only finite values.")

        values: dict[str, float] = {}
        cursor = 0
        for parameter in self.parameters:
            if parameter.vary:
                value = float(packed[cursor])
                cursor += 1
            else:
                value = parameter.initial
            lower, upper = parameter.bounds
            if not lower <= value <= upper:
                raise ValueError(f"{parameter.name} lies outside {parameter.bounds}.")
            values[parameter.name] = value

        for simplex in self.simplexes:
            n_coordinates = len(simplex.names) - 1
            logits = np.concatenate(
                (packed[cursor : cursor + n_coordinates], np.array([0.0]))
            )
            cursor += n_coordinates
            logits -= np.max(logits)
            fractions = np.exp(logits)
            fractions /= fractions.sum()
            values.update(zip(simplex.names, fractions.tolist(), strict=True))
        return values

    def initial_vector(self) -> NDArray[np.float64]:
        return self.pack(self.initial_values())

    def vector_bounds(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = [
            parameter.bounds[0] for parameter in self.parameters if parameter.vary
        ]
        upper = [
            parameter.bounds[1] for parameter in self.parameters if parameter.vary
        ]
        n_simplex_coordinates = sum(
            len(simplex.names) - 1 for simplex in self.simplexes
        )
        lower.extend([-np.inf] * n_simplex_coordinates)
        upper.extend([np.inf] * n_simplex_coordinates)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    def log_uniform_prior(self, vector: ArrayLike) -> float:
        """Uniform physical-space prior, expressed in fitting coordinates.

        Ordinary parameters have uniform density within their declared bounds.
        Simplex fractions have a uniform Dirichlet(1) density. The additive
        log-ratio transformation therefore contributes ``sum(log(fraction))``
        as its Jacobian term.
        """
        try:
            values = self.unpack(vector)
        except ValueError:
            return -np.inf
        log_density = 0.0
        for simplex in self.simplexes:
            fractions = np.asarray([values[name] for name in simplex.names])
            if np.any(fractions <= 0):
                return -np.inf
            log_density += float(np.sum(np.log(fractions)))
        return log_density

    def random_vectors(
        self,
        count: int,
        *,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Generate valid multistart coordinates, including the declared initial."""
        if count < 1:
            raise ValueError("count must be at least one.")
        vectors = np.empty((count, self.dimension), dtype=float)
        vectors[0] = self.initial_vector()
        if count == 1:
            return vectors

        for row in range(1, count):
            values = self.initial_values()
            for parameter in self.parameters:
                if not parameter.vary:
                    continue
                lower, upper = parameter.bounds
                if np.isfinite(lower) and np.isfinite(upper):
                    values[parameter.name] = rng.uniform(lower, upper)
                else:
                    scale = max(abs(parameter.initial), 1.0)
                    candidate = parameter.initial + rng.normal(0.0, scale)
                    if np.isfinite(lower):
                        candidate = max(candidate, np.nextafter(lower, np.inf))
                    if np.isfinite(upper):
                        candidate = min(candidate, np.nextafter(upper, -np.inf))
                    values[parameter.name] = candidate
            for simplex in self.simplexes:
                fractions = rng.dirichlet(np.ones(len(simplex.names)))
                values.update(zip(simplex.names, fractions.tolist(), strict=True))
            vectors[row] = self.pack(values)
        return vectors
