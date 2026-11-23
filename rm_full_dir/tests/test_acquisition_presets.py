import numpy as np
import pytest

from dmri_models import acquisitions


@pytest.mark.parametrize(
    ("loader", "count", "delta", "big_delta"),
    [
        (acquisitions.hcp, 288, 10.6, 43.1),
        (acquisitions.uk_biobank, 105, 21.4, 45.5),
        (acquisitions.bigmac_highres, 136, 7.0, 13.0),
        (acquisitions.bigmac_lowres, 260, 14.0, 24.0),
        (acquisitions.sandi_full_acq, 170, 24.0, 39.0),
    ],
)
def test_packaged_acquisition_shapes_and_timing(
    loader,
    count,
    delta,
    big_delta,
) -> None:
    acquisition = loader()
    assert acquisition.n_measurements == count
    assert acquisition.gradients.shape == (count, 3)
    np.testing.assert_allclose(acquisition.pulse_duration, delta)
    np.testing.assert_allclose(acquisition.diffusion_time, big_delta)
    assert np.all(acquisition.b_values >= 0)
    assert np.all(acquisition.gradient_strength >= 0)


@pytest.mark.parametrize("loader", [acquisitions.hcp, acquisitions.uk_biobank])
def test_dive_acquisition_shells_are_converted_to_canonical_units(loader) -> None:
    acquisition = loader()
    assert acquisition.b_values.max() <= 3.011
    assert acquisition.b_values.min() == 0


def test_bigmac_shells_and_metadata() -> None:
    high = acquisitions.bigmac_highres()
    low = acquisitions.bigmac_lowres()
    assert high.b_values.max() == pytest.approx(4.0000079)
    assert low.b_values.max() == pytest.approx(4.000002681)
    np.testing.assert_allclose(high.echo_time, 25.4)
    np.testing.assert_allclose(low.repetition_time, 3500)


def test_named_loader_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unknown acquisition"):
        acquisitions.load("unknown")


def test_sandi_full_acq_has_five_non_zero_shells() -> None:
    acquisition = acquisitions.sandi_full_acq()
    shells = np.unique(np.round(acquisition.b_values, decimals=6))
    non_zero_shells = shells[shells > 0]
    np.testing.assert_allclose(non_zero_shells, [0.5, 1.0, 2.0, 3.0, 6.0])
    assert non_zero_shells.size == 5


def test_named_loader_supports_sandi_full_acq() -> None:
    loaded = acquisitions.load("sandi_full_acq")
    direct = acquisitions.sandi_full_acq()
    np.testing.assert_allclose(loaded.b_values, direct.b_values)
    np.testing.assert_allclose(loaded.gradients, direct.gradients)
