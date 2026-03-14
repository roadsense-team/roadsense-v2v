"""
Unit tests for sim-to-real replay helpers (Run 020 — decay-based braking signal).
"""

import pytest

from ml.scripts import validate_against_real_data


def test_decay_mode_resets_to_one_on_trigger():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=0.0,
        mode="decay",
    )
    assert value == pytest.approx(1.0)


def test_decay_mode_decays_without_trigger():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=1.0,
        mode="decay",
    )
    assert value == pytest.approx(0.95)

    # Two steps of decay
    value2 = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=value,
        mode="decay",
    )
    assert value2 == pytest.approx(0.95 * 0.95)


def test_decay_mode_retrigger_resets_to_one():
    """Trigger during decay resets back to 1.0."""
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=1.0,
        mode="decay",
    )
    assert value == pytest.approx(0.95)

    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=value,
        mode="decay",
    )
    assert value == pytest.approx(1.0)


def test_latched_mode_stays_at_one():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=0.0,
        mode="latched",
    )
    assert value == pytest.approx(1.0)

    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=value,
        mode="latched",
    )
    assert value == pytest.approx(1.0)  # sticky


def test_instant_mode_no_memory():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=0.0,
        mode="instant",
    )
    assert value == pytest.approx(1.0)

    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=False,
        current_decay=value,
        mode="instant",
    )
    assert value == pytest.approx(0.0)


def test_off_mode_forces_zero():
    value = validate_against_real_data._update_braking_received_decay(
        any_braking_this_step=True,
        current_decay=1.0,
        mode="off",
    )
    assert value == pytest.approx(0.0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown braking_received_mode"):
        validate_against_real_data._update_braking_received_decay(
            any_braking_this_step=False,
            current_decay=0.0,
            mode="bad-mode",
        )
