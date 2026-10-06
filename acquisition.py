"""Diffusion MRI acquisition representation and unit conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

BValueUnit = Literal["ms/um^2", "s/mm^2"]
_GAMMA_RAD_PER_MS_PER_T = 2.6752218744e8 * 1e-3


def _convert_b_values(values: ArrayLike, unit: BValueUnit) -> NDArray[np.float64]:
    b_values = np.asarray(values, dtype=float).reshape(-1)
    if unit == "ms/um^2":
        converted = b_values
    elif unit == "s/mm^2":
        converted = b_values * 1e-3
    else:
        raise ValueError(
            f"Unsupported b-value unit {unit!r}; use 'ms/um^2' or 's/mm^2'."
        )
    if not np.all(np.isfinite(converted)):
        raise ValueError("b-values must be finite.")
    if np.any(converted < 0):
        raise ValueError("b-values must be non-negative.")
    return converted


def _measurement_array(
    value: ArrayLike | None,
    n_measurements: int,
    name: str,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.size == 1:
        array = np.full(n_measurements, float(array.reshape(-1)[0]))
    else:
        array = array.reshape(-1)
        if array.size != n_measurements:
            raise ValueError(
                f"{name} has {array.size} values but the acquisition has "
                f"{n_measurements} measurements."
            )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def pgse_gradient_strength(
    b_values: ArrayLike,
    pulse_duration: ArrayLike,
    diffusion_time: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate rectangular-PGSE gradient strength in T/m.

    Inputs use the package's canonical units: b in ms/µm² and timing in ms.
    The expression is

    ``b = gamma² G² delta² (Delta - delta/3)``,

    with ``gamma`` in rad/ms/T and ``G`` converted from T/m to T/µm.
    """
    b = np.asarray(b_values, dtype=float).reshape(-1)
    delta = _measurement_array(pulse_duration, b.size, "pulse_duration")
    big_delta = _measurement_array(diffusion_time, b.size, "diffusion_time")
    assert delta is not None
    assert big_delta is not None
    if np.any(delta <= 0):
        raise ValueError("pulse_duration must be greater than zero.")
    if np.any(big_delta < delta):
        raise ValueError("diffusion_time must be greater than or equal to pulse_duration.")

    denominator = (
        _GAMMA_RAD_PER_MS_PER_T**2
        * delta**2
        * (big_delta - delta / 3.0)
    )
    # The equation produces T/µm. Convert to the public gradient unit, T/m.
    return np.sqrt(b / denominator) * 1e6


def _gradient_array(
    gradients: ArrayLike | None,
    b_values: NDArray[np.float64],
    normalize: bool,
) -> NDArray[np.float64] | None:
    if gradients is None:
        return None
    array = np.asarray(gradients, dtype=float)
    if array.ndim != 2:
        raise ValueError("gradients must have shape (n_measurements, 3) or (3, n_measurements).")
    if array.shape == (b_values.size, 3):
        pass
    elif array.shape == (3, b_values.size):
        array = array.T
    else:
        raise ValueError(
            "gradients must have shape (n_measurements, 3) or (3, n_measurements)."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("gradients must contain only finite values.")

    array = array.copy()
    norms = np.linalg.norm(array, axis=1)
    diffusion_weighted = b_values > 0
    if np.any(diffusion_weighted & np.isclose(norms, 0.0)):
        raise ValueError("Every non-zero b-value requires a non-zero gradient direction.")
    if normalize:
        array[diffusion_weighted] /= norms[diffusion_weighted, None]
    elif np.any(diffusion_weighted & ~np.isclose(norms, 1.0, atol=1e-5)):
        raise ValueError("Non-zero gradient directions must be unit vectors.")
    return array


@dataclass(frozen=True)
class Acquisition:
    """A flattened collection of MRI measurements in canonical units.

    Canonical units are ms, µm, ms/µm² for b-values, and T/m for gradient
    strength. Each populated field has one value per measurement.
    """

    b_values: NDArray[np.float64]
    gradients: NDArray[np.float64] | None = None
    b_tensor_shape: NDArray[np.float64] | None = None
    pulse_duration: NDArray[np.float64] | None = None
    diffusion_time: NDArray[np.float64] | None = None
    gradient_strength: NDArray[np.float64] | None = None
    echo_time: NDArray[np.float64] | None = None
    inversion_time: NDArray[np.float64] | None = None
    repetition_time: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        """Validate direct construction in canonical units as strictly as factories."""
        b_values = _convert_b_values(self.b_values, "ms/um^2")
        n = b_values.size
        values = {
            "b_values": b_values,
            "gradients": _gradient_array(self.gradients, b_values, normalize=False),
            "b_tensor_shape": _measurement_array(
                self.b_tensor_shape, n, "b_tensor_shape"
            ),
            "pulse_duration": _measurement_array(
                self.pulse_duration, n, "pulse_duration"
            ),
            "diffusion_time": _measurement_array(
                self.diffusion_time, n, "diffusion_time"
            ),
            "gradient_strength": _measurement_array(
                self.gradient_strength, n, "gradient_strength"
            ),
            "echo_time": _measurement_array(self.echo_time, n, "echo_time"),
            "inversion_time": _measurement_array(
                self.inversion_time, n, "inversion_time"
            ),
            "repetition_time": _measurement_array(
                self.repetition_time, n, "repetition_time"
            ),
        }
        for name, value in values.items():
            if value is not None:
                value = np.array(value, dtype=float, copy=True)
                value.flags.writeable = False
            object.__setattr__(self, name, value)

        if self.pulse_duration is not None and np.any(self.pulse_duration <= 0):
            raise ValueError("pulse_duration must be greater than zero.")
        if self.diffusion_time is not None and np.any(self.diffusion_time <= 0):
            raise ValueError("diffusion_time must be greater than zero.")
        if (
            self.pulse_duration is not None
            and self.diffusion_time is not None
            and np.any(self.diffusion_time < self.pulse_duration)
        ):
            raise ValueError(
                "diffusion_time must be greater than or equal to pulse_duration."
            )
        if self.gradient_strength is not None and np.any(self.gradient_strength < 0):
            raise ValueError("gradient_strength must be non-negative.")
        if self.b_tensor_shape is not None and np.any(
            (self.b_tensor_shape < -0.5) | (self.b_tensor_shape > 1.0)
        ):
            raise ValueError("b_tensor_shape must lie between -0.5 and 1.")
        for name in ("echo_time", "inversion_time", "repetition_time"):
            value = getattr(self, name)
            if value is not None and np.any(value < 0):
                raise ValueError(f"{name} must be non-negative.")

    @classmethod
    def from_b_values(
        cls,
        b_values: ArrayLike,
        *,
        b_unit: BValueUnit,
        gradients: ArrayLike | None = None,
        normalize_gradients: bool = True,
        b_tensor_shape: ArrayLike | None = None,
        pulse_duration: ArrayLike | None = None,
        diffusion_time: ArrayLike | None = None,
        gradient_strength: ArrayLike | None = None,
        echo_time: ArrayLike | None = None,
        inversion_time: ArrayLike | None = None,
        repetition_time: ArrayLike | None = None,
    ) -> "Acquisition":
        """Construct an acquisition while explicitly converting b-value units."""
        b = _convert_b_values(b_values, b_unit)
        n = b.size
        return cls(
            b_values=b,
            gradients=_gradient_array(gradients, b, normalize_gradients),
            b_tensor_shape=_measurement_array(b_tensor_shape, n, "b_tensor_shape"),
            pulse_duration=_measurement_array(pulse_duration, n, "pulse_duration"),
            diffusion_time=_measurement_array(diffusion_time, n, "diffusion_time"),
            gradient_strength=_measurement_array(gradient_strength, n, "gradient_strength"),
            echo_time=_measurement_array(echo_time, n, "echo_time"),
            inversion_time=_measurement_array(inversion_time, n, "inversion_time"),
            repetition_time=_measurement_array(repetition_time, n, "repetition_time"),
        )

    @classmethod
    def from_pgse(
        cls,
        b_values: ArrayLike,
        *,
        b_unit: BValueUnit,
        pulse_duration: ArrayLike,
        diffusion_time: ArrayLike,
        gradients: ArrayLike | None = None,
        normalize_gradients: bool = True,
        **kwargs: ArrayLike,
    ) -> "Acquisition":
        """Construct a rectangular-PGSE acquisition and derive G in T/m."""
        b = _convert_b_values(b_values, b_unit)
        delta = _measurement_array(pulse_duration, b.size, "pulse_duration")
        big_delta = _measurement_array(diffusion_time, b.size, "diffusion_time")
        assert delta is not None
        assert big_delta is not None
        gradient_strength = pgse_gradient_strength(b, delta, big_delta)
        return cls.from_b_values(
            b,
            b_unit="ms/um^2",
            gradients=gradients,
            normalize_gradients=normalize_gradients,
            pulse_duration=delta,
            diffusion_time=big_delta,
            gradient_strength=gradient_strength,
            **kwargs,
        )

    @classmethod
    def from_fsl(
        cls,
        bval_path: str | Path,
        bvec_path: str | Path | None = None,
        *,
        bval_unit: BValueUnit = "s/mm^2",
        b0_threshold: float | None = 0.05,
        rectangular_pgse: bool = False,
        **kwargs: ArrayLike,
    ) -> "Acquisition":
        """Read FSL b-values/b-vectors and convert to canonical units.

        FSL b-values conventionally use s/mm², hence that explicit default.
        Set ``bval_unit="ms/um^2"`` for a non-standard file already converted
        to this package's canonical units. ``b0_threshold`` is always in
        canonical ms/µm² and is applied after conversion; set it to ``None``
        to preserve small non-zero b-values.
        """
        b_values = np.loadtxt(Path(bval_path), dtype=float)
        if b0_threshold is not None:
            if b0_threshold < 0:
                raise ValueError("b0_threshold must be non-negative or None.")
            canonical_b = _convert_b_values(b_values, bval_unit)
            canonical_b[canonical_b <= b0_threshold] = 0.0
            b_values = canonical_b
            bval_unit = "ms/um^2"
        gradients = None if bvec_path is None else np.loadtxt(Path(bvec_path), dtype=float)
        if gradients is not None and gradients.ndim == 1 and gradients.size == 3:
            gradients = gradients.reshape(3, 1)
        if gradients is not None and gradients.ndim == 2 and gradients.shape[0] == 3:
            gradients = gradients.T
        constructor = cls.from_pgse if rectangular_pgse else cls.from_b_values
        return constructor(
            b_values,
            b_unit=bval_unit,
            gradients=gradients,
            **kwargs,
        )

    @property
    def n_measurements(self) -> int:
        return int(self.b_values.size)

    def summary(self, *, b_tolerance: float = 0.1) -> str:
        """Return a human-readable overview of the acquisition protocol.

        Parameters
        ----------
        b_tolerance:
            Shell-merging tolerance in canonical b-value units (ms/um^2).
            Measurements whose b-values fall in the same tolerance bin are
            reported as one shell.
        """
        if not np.isfinite(b_tolerance) or b_tolerance <= 0:
            raise ValueError("b_tolerance must be finite and greater than zero.")

        def _format_measurement_range(
            label: str,
            values: NDArray[np.float64] | None,
            unit: str,
        ) -> str:
            if values is None:
                return f"- {label}: n/a"
            if np.allclose(values, values[0]):
                return f"- {label} ({unit}): {values[0]:.3f}"
            return (
                f"- {label} ({unit}): {np.min(values):.3f} to "
                f"{np.max(values):.3f}"
            )

        order = np.argsort(self.b_values)
        inverse = np.empty(self.n_measurements, dtype=int)
        shell_centres: list[float] = []
        shell_counts: list[int] = []
        current_shell = -1
        for index in order:
            b_value = float(self.b_values[index])
            if current_shell < 0:
                current_shell = 0
                shell_centres.append(b_value)
                shell_counts.append(1)
                inverse[index] = current_shell
                continue

            if abs(b_value - shell_centres[current_shell]) <= b_tolerance:
                shell_counts[current_shell] += 1
                count = shell_counts[current_shell]
                shell_centres[current_shell] += (b_value - shell_centres[current_shell]) / count
                inverse[index] = current_shell
            else:
                current_shell += 1
                shell_centres.append(b_value)
                shell_counts.append(1)
                inverse[index] = current_shell

        shell_b = np.asarray(shell_centres, dtype=float)
        lines = [
            "Acquisition summary",
            f"- measurements: {self.n_measurements}",
            (
                "- b-value shells (ms/um^2): "
                f"{shell_b.size} (tolerance: {b_tolerance:.3f})"
            ),
        ]

        gradient_norms: NDArray[np.float64] | None
        if self.gradients is None:
            gradient_norms = None
        else:
            gradient_norms = np.linalg.norm(self.gradients, axis=1)

        for shell_position, shell_value in enumerate(shell_b):
            shell_mask = inverse == shell_position
            shell_count = int(np.sum(shell_mask))
            if gradient_norms is None:
                lines.append(
                    f"  - b={shell_value:.3f}: {shell_count} measurements"
                )
            else:
                non_zero_dirs = int(np.sum(shell_mask & (gradient_norms > 1e-8)))
                lines.append(
                    f"  - b={shell_value:.3f}: {shell_count} measurements, "
                    f"{non_zero_dirs} non-zero directions"
                )

        if self.gradients is None:
            lines.append("- gradient directions: n/a")
        else:
            non_zero_mask = gradient_norms > 1e-8
            non_zero_count = int(np.sum(non_zero_mask))
            if non_zero_count == 0:
                unique_non_zero_directions = 0
            else:
                unique_non_zero_directions = int(
                    np.unique(np.round(self.gradients[non_zero_mask], 6), axis=0).shape[0]
                )
            lines.append(
                "- gradient directions: "
                f"{unique_non_zero_directions} unique non-zero across "
                f"{non_zero_count} diffusion-weighted measurements"
            )

        if self.b_tensor_shape is None:
            lines.append("- b-tensor shape: n/a")
        elif np.allclose(self.b_tensor_shape, self.b_tensor_shape[0]):
            lines.append(f"- b-tensor shape: {self.b_tensor_shape[0]:.3f}")
        else:
            lines.append(
                f"- b-tensor shape: {np.min(self.b_tensor_shape):.3f} to "
                f"{np.max(self.b_tensor_shape):.3f}"
            )

        lines.extend(
            [
                _format_measurement_range("pulse duration", self.pulse_duration, "ms"),
                _format_measurement_range("diffusion time", self.diffusion_time, "ms"),
                _format_measurement_range(
                    "gradient strength",
                    self.gradient_strength,
                    "T/m",
                ),
                _format_measurement_range("echo time", self.echo_time, "ms"),
                _format_measurement_range(
                    "inversion time",
                    self.inversion_time,
                    "ms",
                ),
                _format_measurement_range(
                    "repetition time",
                    self.repetition_time,
                    "ms",
                ),
            ]
        )
        return "\n".join(lines)

    def print_summary(self, *, b_tolerance: float = 0.1) -> None:
        """Print :meth:`summary` for quick inspection in scripts and notebooks."""
        print("----------------------------------------------------------------")
        print(self.summary(b_tolerance=b_tolerance))
        print("----------------------------------------------------------------")
