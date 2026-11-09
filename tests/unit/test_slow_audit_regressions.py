"""Regression contracts from the 2026-11-09 end-to-end Slow audit."""

from dataclasses import replace

from custom_components.better_thermostat.calibration import _compute_slow_balance
from custom_components.better_thermostat.utils.calibration.slow import (
    SlowCalibration,
    SlowContext,
)
from tests.unit.test_slow_dispatch import room


def ctx(t, room=19.6, supply=35.0):
    return SlowContext(
        t=t,
        target_temperature=20,
        room_temperature=room,
        outdoor_temperature=8.6,
        supply_temperature=supply,
        last_valve_percent=5,
    )


def test_boiler_ramp_does_not_postpone_the_second_cold_review():
    c = SlowCalibration()
    c.reset({"baseline": 5})
    out = c.step(ctx(0, supply=25.5))
    first = out.valve_percent
    for t in range(60, 1801, 60):
        c.observe_applied(t, [first])
        out = c.step(ctx(t, supply=25.5 + 15 * t / 1800))
    assert out.valve_percent > first
    response = out.diagnostics["response"]
    assert isinstance(response, dict)
    assert response["window_s"] == 1800


def test_no_stalled_state_between_point_two_and_point_five_deficit():
    c = SlowCalibration()
    c.reset({"baseline": 1})
    out = c.step(ctx(0, room=19.4))
    for t in range(60, 10801, 60):
        c.observe_applied(t, [out.valve_percent])
        out = c.step(ctx(t, room=19.4 if t <= 600 else 19.65))
    assert out.valve_percent > 11
    assert out.diagnostics["reason"] == "calibrate_open"


def test_room_trend_survives_supply_context_change():
    c = SlowCalibration()
    out = c.step(ctx(0, room=18, supply=30))
    for t in range(60, 10801, 60):
        c.observe_applied(t, [out.valve_percent])
        out = c.step(ctx(t, room=18 + t / 7200, supply=30 if t < 1800 else 40))
    assert out.diagnostics["recent_trend_k_per_h"] is not None


def test_unconfirmed_feedback_cannot_ratchet_a_stored_hint():
    c = SlowCalibration()
    c.reset(
        {
            "baseline": 5,
            "holding_points": [
                {
                    "opening": 20,
                    "target_temperature": 20,
                    "supply_temperature": 35,
                    "outdoor_temperature": 8.6,
                }
            ],
        }
    )
    first = c.step(ctx(0)).valve_percent
    for t in range(60, 301, 60):
        c.observe_applied(t, [5])
        assert c.step(ctx(t)).valve_percent == first


def test_warm_review_occurs_after_thirty_minutes_too():
    c = SlowCalibration()
    out = c.step(ctx(0, room=20.3))
    first = out.valve_percent
    for t in range(60, 1801, 60):
        out = c.step(ctx(t, room=20.3))
    assert out.valve_percent < first
    response = out.diagnostics["response"]
    assert isinstance(response, dict)
    assert response["window_s"] == 1800


def test_point_one_five_deficit_is_not_a_settled_holding_point():
    c = SlowCalibration()
    out = c.step(ctx(0, room=19.85))
    for t in range(60, 7201, 60):
        out = c.step(ctx(t, room=19.85))
    assert out.valve_percent > 6
    assert not c.export_state().get("holding_points")


def test_restored_half_integer_matches_the_integer_writer():
    c = SlowCalibration()
    c.reset({"baseline": 5.5})
    assert c.baseline == 6
    c.observe_applied(0, [6])
    out = c.step(ctx(0))
    first = out.valve_percent
    for t in range(60, 1801, 60):
        c.observe_applied(t, [first])
        out = c.step(ctx(t))
    assert out.valve_percent > first


async def test_group_mean_is_on_the_write_grid_and_does_not_deadlock(hass):
    bt = room(hass, count=2)
    trvs = list(bt.real_trvs.values())
    trvs[0].valve_position = 5
    trvs[1].valve_position = 6
    first = _compute_slow_balance(bt, trvs[0].entity_id)[0]
    assert first is not None and first == round(first)
    result = first
    for t in range(60, 3601, 60):
        for trv in trvs:
            trv.valve_position = result
        bt.clock.monotonic_value = t
        result = _compute_slow_balance(bt, trvs[0].entity_id)[0]
        assert result is not None
    assert result > first


async def test_fractional_cap_is_not_rounded_above_the_limit(hass):
    bt = room(hass)
    trv = next(iter(bt.real_trvs.values()))
    trv.valve_max_opening = 5.5
    assert _compute_slow_balance(bt, trv.entity_id)[0] == 5


def test_external_warming_with_zero_opening_does_not_create_more_heat():
    c = SlowCalibration()
    c.reset({"baseline": 0})
    # Establish a room at target, then warming despite zero commanded heat.
    out = c.step(ctx(0, room=20))
    for t in range(60, 1801, 60):
        out = c.step(ctx(t, room=20 + t / 1800 * 0.15))
    assert out.valve_percent == 0
    assert not c.export_state().get("holding_points")


def test_window_closed_and_target_changed_while_echo_pending_use_new_context():
    c = SlowCalibration()
    first = c.step(ctx(0)).valve_percent
    c.step(replace(ctx(60), window_open=True))
    c.observe_applied(120, [0])
    out = c.step(replace(ctx(120), target_temperature=19))
    assert out.valve_percent == 0
    assert out.diagnostics["recent_trend_k_per_h"] is None
    assert first > 0


def test_measurement_gaps_do_not_erase_a_failed_feedback_check():
    c = SlowCalibration()
    first = c.step(ctx(0)).valve_percent
    for t in (3600, 7200, 10800, 0):
        c.observe_applied(t, [5])
        out = c.step(ctx(t))
        assert out.valve_percent == first
        assert out.diagnostics["feedback_confirmed"] is False
