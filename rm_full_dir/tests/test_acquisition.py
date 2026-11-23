import numpy as np
import pytest

from dmri_models import Acquisition, pgse_gradient_strength


def test_converts_standard_fsl_b_value_units() -> None:
    acquisition = Acquisition.from_b_values(
        [0, 1000, 2500],
        b_unit="s/mm^2",
    )
    np.testing.assert_allclose(acquisition.b_values, [0, 1.0, 2.5])


def test_canonical_input_is_not_changed_based_on_magnitude() -> None:
    acquisition = Acquisition.from_b_values(
        [0, 1000],
        b_unit="ms/um^2",
    )
    np.testing.assert_allclose(acquisition.b_values, [0, 1000])


def test_scalar_measurement_fields_are_broadcast() -> None:
    acquisition = Acquisition.from_b_values(
        [0, 1, 2],
        b_unit="ms/um^2",
        echo_time=80,
    )
    np.testing.assert_allclose(acquisition.echo_time, [80, 80, 80])


def test_mismatched_measurement_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="echo_time has 2 values"):
        Acquisition.from_b_values(
            [0, 1, 2],
            b_unit="ms/um^2",
            echo_time=[50, 80],
        )


def test_fsl_reader_accepts_three_by_n_bvecs_and_normalizes(tmp_path) -> None:
    bval_path = tmp_path / "test.bval"
    bvec_path = tmp_path / "test.bvec"
    np.savetxt(bval_path, [[0, 1000]])
    np.savetxt(bvec_path, [[0, 2], [0, 0], [0, 0]])

    acquisition = Acquisition.from_fsl(bval_path, bvec_path)

    np.testing.assert_allclose(acquisition.b_values, [0, 1])
    np.testing.assert_allclose(acquisition.gradients, [[0, 0, 0], [1, 0, 0]])


def test_nonzero_b_value_requires_gradient_direction() -> None:
    with pytest.raises(ValueError, match="non-zero gradient"):
        Acquisition.from_b_values(
            [1000],
            b_unit="s/mm^2",
            gradients=[[0, 0, 0]],
        )


def test_direct_construction_is_validated_in_canonical_units() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Acquisition(b_values=np.array([-1.0]))


def test_generic_three_measurement_gradients_use_n_by_three_layout() -> None:
    gradients = np.eye(3)
    acquisition = Acquisition.from_b_values(
        [1, 1, 1],
        b_unit="ms/um^2",
        gradients=gradients,
    )
    np.testing.assert_allclose(acquisition.gradients, gradients)


def test_acquisition_arrays_are_read_only() -> None:
    acquisition = Acquisition.from_b_values([0, 1], b_unit="ms/um^2")
    with pytest.raises(ValueError, match="read-only"):
        acquisition.b_values[0] = 2


def test_known_pgse_gradient_has_correct_physical_scale() -> None:
    """1000 s/mm² at 30/50 ms requires about 19.7 mT/m."""
    acquisition = Acquisition.from_pgse(
        [1000],
        b_unit="s/mm^2",
        pulse_duration=30,
        diffusion_time=50,
    )
    assert acquisition.gradient_strength[0] == pytest.approx(
        0.01970103047295604,
        rel=1e-12,
    )


def test_pgse_gradient_round_trip_reconstructs_canonical_b_values() -> None:
    b_values = np.array([0.0, 0.5, 1.0, 2.0])
    delta = np.array([20.0, 25.0, 30.0, 35.0])
    big_delta = np.array([40.0, 45.0, 50.0, 60.0])
    gradient = pgse_gradient_strength(b_values, delta, big_delta)

    gamma_rad_per_ms_per_t = 2.6752218744e8 * 1e-3
    reconstructed = (
        gamma_rad_per_ms_per_t**2
        * (gradient * 1e-6) ** 2
        * delta**2
        * (big_delta - delta / 3.0)
    )
    np.testing.assert_allclose(reconstructed, b_values, rtol=1e-13, atol=1e-15)


def test_fsl_pgse_reader_converts_b_and_derives_gradient(tmp_path) -> None:
    bval_path = tmp_path / "test.bval"
    np.savetxt(bval_path, [[0, 1000, 2000]])
    acquisition = Acquisition.from_fsl(
        bval_path,
        rectangular_pgse=True,
        pulse_duration=30,
        diffusion_time=50,
    )
    np.testing.assert_allclose(acquisition.b_values, [0, 1, 2])
    np.testing.assert_allclose(
        acquisition.gradient_strength,
        [0, 0.01970103047295604, 0.02786146448758007],
        rtol=1e-12,
    )


def test_pgse_timing_order_is_validated() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        Acquisition.from_pgse(
            [1],
            b_unit="ms/um^2",
            pulse_duration=50,
            diffusion_time=30,
        )


def test_b_tensor_shape_range_is_validated() -> None:
    with pytest.raises(ValueError, match="between -0.5 and 1"):
        Acquisition.from_b_values(
            [1],
            b_unit="ms/um^2",
            b_tensor_shape=1.1,
        )


def test_fsl_b0_threshold_is_applied_before_gradient_validation(tmp_path) -> None:
    bval_path = tmp_path / "test.bval"
    bvec_path = tmp_path / "test.bvec"
    np.savetxt(bval_path, [[5, 1000]])
    np.savetxt(bvec_path, [[0, 1], [0, 0], [0, 0]])
    acquisition = Acquisition.from_fsl(bval_path, bvec_path)
    np.testing.assert_allclose(acquisition.b_values, [0, 1])


def test_fsl_b0_threshold_can_be_disabled(tmp_path) -> None:
    bval_path = tmp_path / "test.bval"
    bvec_path = tmp_path / "test.bvec"
    np.savetxt(bval_path, [[5]])
    np.savetxt(bvec_path, [[0], [0], [0]])
    with pytest.raises(ValueError, match="non-zero gradient"):
        Acquisition.from_fsl(
            bval_path,
            bvec_path,
            b0_threshold=None,
        )


def test_summary_reports_shells_directions_and_timing() -> None:
    acquisition = Acquisition.from_pgse(
        [0, 1000, 1000, 2000],
        b_unit="s/mm^2",
        pulse_duration=30,
        diffusion_time=50,
        gradients=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    text = acquisition.summary()

    assert "Acquisition summary" in text
    assert "measurements: 4" in text
    assert "b-value shells (ms/um^2): 3" in text
    assert "b=1.000: 2 measurements, 2 non-zero directions" in text
    assert "gradient directions: 3 unique non-zero across 3" in text
    assert "pulse duration (ms): 30.000" in text
    assert "diffusion time (ms): 50.000" in text
    assert "gradient strength (T/m):" in text


def test_summary_marks_missing_fields_as_not_available() -> None:
    acquisition = Acquisition.from_b_values([0, 1], b_unit="ms/um^2")
    text = acquisition.summary()

    assert "gradient directions: n/a" in text
    assert "b-tensor shape: n/a" in text
    assert "pulse duration: n/a" in text


def test_print_summary_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    acquisition = Acquisition.from_b_values([0, 1], b_unit="ms/um^2")
    acquisition.print_summary()
    captured = capsys.readouterr()
    assert "Acquisition summary" in captured.out


def test_summary_groups_close_b_values_with_default_tolerance() -> None:
    acquisition = Acquisition.from_b_values(
        [0.95, 1.02, 1.11],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    text = acquisition.summary()
    assert "b-value shells (ms/um^2): 2 (tolerance: 0.100)" in text
    assert "2 measurements, 2 non-zero directions" in text
    assert "1 measurements, 1 non-zero directions" in text


def test_summary_supports_custom_b_tolerance() -> None:
    acquisition = Acquisition.from_b_values(
        [0.95, 1.02, 1.11],
        b_unit="ms/um^2",
        gradients=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    text = acquisition.summary(b_tolerance=0.2)
    assert "b-value shells (ms/um^2): 1 (tolerance: 0.200)" in text
    assert "3 measurements, 3 non-zero directions" in text


def test_summary_rejects_non_positive_b_tolerance() -> None:
    acquisition = Acquisition.from_b_values([0, 1], b_unit="ms/um^2")
    with pytest.raises(ValueError, match="b_tolerance"):
        acquisition.summary(b_tolerance=0.0)
