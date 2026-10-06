"""Composition of compartments into biophysical signal models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import warnings

import numpy as np

from .acquisition import Acquisition
from .compartments import Compartment, Parameters


def _sample_unit_sphere(
    n_directions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vectors = rng.normal(size=(n_directions, 3))
    norms = np.linalg.norm(vectors, axis=1)
    near_zero = norms < 1e-12
    while np.any(near_zero):
        vectors[near_zero] = rng.normal(size=(int(np.sum(near_zero)), 3))
        norms = np.linalg.norm(vectors, axis=1)
        near_zero = norms < 1e-12
    return vectors / norms[:, None]


def _repeat_field(
    values: np.ndarray | None,
    n_directions: int,
) -> np.ndarray | None:
    if values is None:
        return None
    return np.repeat(values, n_directions)


@dataclass(frozen=True)
class Component:
    """A named model component with an independently addressable fraction."""

    name: str
    compartment: Compartment
    fraction: str | None = None

    @property
    def fraction_parameter(self) -> str:
        return self.fraction or f"{self.name}_fraction"


@dataclass(frozen=True)
class Model:
    """A signal-weighted mixture of arbitrary compartments."""

    components: Sequence[Component]
    baseline: str = "S0"

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("A model requires at least one component.")
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("Component names must be unique.")

    def signal(
        self,
        acquisition: Acquisition,
        parameters: Parameters,
    ) -> np.ndarray:
        fractions = np.asarray(
            [float(parameters[item.fraction_parameter]) for item in self.components],
            dtype=float,
        )
        if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
            raise ValueError("Signal fractions must be finite and non-negative.")
        if not np.isclose(fractions.sum(), 1.0, atol=1e-8):
            raise ValueError("Signal fractions must sum to one.")

        baseline = float(parameters[self.baseline])
        if not np.isfinite(baseline) or baseline < 0:
            raise ValueError(f"{self.baseline} must be finite and non-negative.")

        mixture = np.zeros(acquisition.n_measurements, dtype=float)
        for fraction, component in zip(fractions, self.components, strict=True):
            component_signal = np.asarray(
                component.compartment.signal(acquisition, parameters),
                dtype=float,
            )
            if component_signal.shape != mixture.shape:
                raise ValueError(
                    f"Component {component.name!r} returned shape "
                    f"{component_signal.shape}; expected {mixture.shape}."
                )
            mixture += fraction * component_signal
        return baseline * mixture

    def spherical_mean_signal(
        self,
        acquisition: Acquisition,
        parameters: Parameters,
        *,
        n_directions: int = 256,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Return a model-level spherical mean consistently across components.

        Components with an analytical ``spherical_mean`` implementation use it.
        Components without this method are simulated over random directions and
        numerically averaged.
        """
        if n_directions < 4:
            raise ValueError("n_directions must be at least four.")
        if n_directions < 30:
            warnings.warn(
                "n_directions < 30 may produce inaccurate spherical mean estimates.",
                UserWarning,
                stacklevel=2,
            )

        fractions = np.asarray(
            [float(parameters[item.fraction_parameter]) for item in self.components],
            dtype=float,
        )
        if not np.all(np.isfinite(fractions)) or np.any(fractions < 0):
            raise ValueError("Signal fractions must be finite and non-negative.")
        if not np.isclose(fractions.sum(), 1.0, atol=1e-8):
            raise ValueError("Signal fractions must sum to one.")

        baseline = float(parameters[self.baseline])
        if not np.isfinite(baseline) or baseline < 0:
            raise ValueError(f"{self.baseline} must be finite and non-negative.")

        generator = rng or np.random.default_rng()
        mixture = np.zeros(acquisition.n_measurements, dtype=float)
        for fraction, component in zip(fractions, self.components, strict=True):
            spherical_mean = getattr(component.compartment, "spherical_mean", None)
            if callable(spherical_mean):
                component_signal = np.asarray(
                    spherical_mean(acquisition, parameters),
                    dtype=float,
                )
            else:
                directions = _sample_unit_sphere(n_directions, generator)
                sampled = Acquisition(
                    b_values=np.repeat(acquisition.b_values, n_directions),
                    gradients=np.tile(directions, (acquisition.n_measurements, 1)),
                    b_tensor_shape=_repeat_field(
                        acquisition.b_tensor_shape,
                        n_directions,
                    ),
                    pulse_duration=_repeat_field(
                        acquisition.pulse_duration,
                        n_directions,
                    ),
                    diffusion_time=_repeat_field(
                        acquisition.diffusion_time,
                        n_directions,
                    ),
                    gradient_strength=_repeat_field(
                        acquisition.gradient_strength,
                        n_directions,
                    ),
                    echo_time=_repeat_field(acquisition.echo_time, n_directions),
                    inversion_time=_repeat_field(
                        acquisition.inversion_time,
                        n_directions,
                    ),
                    repetition_time=_repeat_field(
                        acquisition.repetition_time,
                        n_directions,
                    ),
                )
                component_signal = np.asarray(
                    component.compartment.signal(sampled, parameters),
                    dtype=float,
                ).reshape(acquisition.n_measurements, n_directions).mean(axis=1)

            if component_signal.shape != mixture.shape:
                raise ValueError(
                    f"Component {component.name!r} returned shape "
                    f"{component_signal.shape}; expected {mixture.shape}."
                )
            mixture += fraction * component_signal
        return baseline * mixture

