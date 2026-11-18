"""Observable contracts of the single slow room correction loop."""

from dataclasses import replace

import pytest

from tests.benchmark.adapters.base import BenchmarkContext
from tests.benchmark.adapters.slow_adapter import SlowCalibrationAdapter


def context(t=0, room=20.6, **kwargs):
    return BenchmarkContext(
        t=t,
        dt=60,
        target_temperature=21,
        room_temperature=room,
        raw_room_temperature=room,
        trv_temperature=30,
        outdoor_temperature=5,
        last_valve_percent=40,
        **kwargs,
    )


def drive(adapter, end, room=20.6, supply=35):
    result = None
    for t in range(0, end + 1, 60):
        result = adapter.step(context(t, room=room, supply_temperature=supply))
    assert result is not None
    return result


@pytest.mark.parametrize(
    "room,first,second",
    [(20.6, 43, 46), (20.85, 41, 42), (21.15, 39, 38), (21.4, 37, 34), (21, 40, 40)],
)
def test_immediate_review_then_exact_half_hour_cadence(room, first, second):
    a = SlowCalibrationAdapter()
    for t in range(0, 1800, 60):
        assert a.step(context(t, room=room)).valve_percent == first
    result = a.step(context(1800, room=room))
    assert result.valve_percent == second
    assert result.diagnostics["response"]["window_s"] == 1800
    assert result.diagnostics["next_review_in_s"] == 1800
    assert a.step(context(1860, room=room)).valve_percent == second


def test_explicit_observation_interval_is_honored():
    a = SlowCalibrationAdapter(hold_s=3600)
    for t in range(0, 3600, 60):
        assert a.step(context(t)).valve_percent == 43
    assert a.step(context(3600)).valve_percent == 46


@pytest.mark.parametrize("which", ["supply", "weather"])
def test_environment_drift_does_not_restart_correction(which):
    a = SlowCalibrationAdapter()
    result = None
    for t in range(0, 3601, 60):
        ctx = context(
            t, supply_temperature=27 + 14 * t / 3600 if which == "supply" else 35
        )
        if which == "weather":
            ctx = replace(ctx, outdoor_temperature=5 + 6 * t / 3600)
        result = a.step(ctx)
        if t in (1800, 3600):
            assert result.diagnostics["reason"] == "calibrate_open"
    assert result is not None
    assert result.valve_percent == 49


def test_warming_toward_target_does_not_receive_another_opening_step():
    a = SlowCalibrationAdapter()
    result = None
    for t in range(0, 1801, 60):
        result = a.step(context(t, room=20.6 + 0.2 * t / 1800, supply_temperature=35))
    assert result is not None
    assert result.valve_percent == 43
    assert result.diagnostics["reason"] == "accept"


def test_trend_can_trim_opening_before_current_temperature_crosses_target():
    a = SlowCalibrationAdapter()
    result = None
    for t in range(0, 1801, 60):
        result = a.step(context(t, room=20.6 + 0.35 * t / 1800, supply_temperature=35))
    assert result is not None
    assert result.valve_percent is not None
    assert result.valve_percent < 43
    assert result.diagnostics["reason"] == "calibrate_close"


def test_falling_temperature_is_corrected_before_point_two_deficit():
    a = SlowCalibrationAdapter()
    result = None
    for t in range(0, 1801, 60):
        result = a.step(context(t, room=21.05 - 0.1 * t / 1800, supply_temperature=35))
    assert result is not None
    assert result.valve_percent == 41


def test_direction_reversal_reduces_step_without_growing_it_back():
    a = SlowCalibrationAdapter()
    result = drive(a, 1800)
    assert result.valve_percent == 46
    for t in range(1860, 3601, 60):
        result = a.step(context(t, room=21.4))
    assert result.valve_percent == 44
    for t in range(3660, 5401, 60):
        result = a.step(context(t, room=21.4))
    assert result.valve_percent == 42
    for t in range(5460, 7201, 60):
        result = a.step(context(t, room=20.6))
    assert result.valve_percent == 43


def test_large_opening_is_not_replaced_by_old_startup_baseline_near_target():
    a = SlowCalibrationAdapter()
    result = drive(a, 7200, room=20.6)
    working = result.valve_percent
    for t in range(7260, 9001, 60):
        result = a.step(context(t, room=20.95))
    assert result.valve_percent >= working - 3


def test_window_close_restarts_observation_and_immediately_reassesses():
    a = SlowCalibrationAdapter()
    drive(a, 1200)
    assert a.step(context(1260, window_open=True)).valve_percent == 0
    assert a.step(context(1560, window_open=True)).valve_percent == 0
    result = a.step(context(1620, room=20.6))
    assert result.valve_percent == 46
    assert result.diagnostics["observation_age_s"] == 0
    assert result.diagnostics["recent_trend_k_per_h"] is None
    assert result.diagnostics["last_reset_reason"] == "window_closed"


@pytest.mark.parametrize("target,expected", [(20.0, 40), (22.0, 46)])
def test_target_change_immediately_reassesses_without_old_trend(target, expected):
    a = SlowCalibrationAdapter()
    drive(a, 1200)
    out = a.step(replace(context(1260), target_temperature=target))
    assert out.valve_percent == expected
    assert out.diagnostics["recent_trend_k_per_h"] is None
    assert out.diagnostics["observation_age_s"] == 0


def test_unavailable_heat_does_not_advance_or_learn():
    a = SlowCalibrationAdapter()
    assert drive(a, 7200, room=20, supply=19).valve_percent == 40
    assert a.export_state() == {"baseline": 40}
    out = a.step(context(7260, room=20, supply_temperature=35))
    assert out.valve_percent == 43
    assert out.diagnostics["observation_age_s"] == 0


def test_failed_feedback_cannot_staircase_opening_or_reuse_holding_hint():
    a = SlowCalibrationAdapter()
    a.reset(
        {
            "baseline": 40,
            "holding_points": [
                {
                    "opening": 80,
                    "target_temperature": 21,
                    "outdoor_temperature": 5,
                    "supply_temperature": 35,
                }
            ],
        }
    )
    assert a.step(context(supply_temperature=35)).valve_percent == 43
    for t in range(60, 7201, 60):
        a.controller.observe_applied(t, [40])
        assert a.step(context(t, supply_temperature=35)).valve_percent == 43
    assert "response" not in a.export_state()


def test_confirmation_starts_fresh_observation_without_extra_immediate_step():
    a = SlowCalibrationAdapter()
    a.step(context())
    for t in range(60, 601, 60):
        a.controller.observe_applied(t, [40])
        a.step(context(t))
    for t in range(660, 2460, 60):
        a.controller.observe_applied(t, [43])
        assert a.step(context(t)).valve_percent == 43
    a.controller.observe_applied(2460, [43])
    assert a.step(context(2460)).valve_percent == 46


def test_only_settled_near_target_windows_record_holding_points():
    a = SlowCalibrationAdapter()
    drive(a, 3600, room=21)
    state = a.export_state()
    assert len(state["holding_points"]) == 2
    assert state["response"]["observed_opening"] == 40
    b = SlowCalibrationAdapter()
    b.reset(state)
    assert b.export_state() == state
    assert b.step(context(room=21, supply_temperature=35)).valve_percent == 40
    assert b.controller._start == 0
    cold = SlowCalibrationAdapter()
    drive(cold, 7200, room=20.85)
    assert "holding_points" not in cold.export_state()


def test_warm_supply_drift_invalidates_settled_evidence_not_review_deadline():
    a = SlowCalibrationAdapter()
    out = None
    for t in range(0, 1801, 60):
        out = a.step(context(t, room=21, supply_temperature=27 + 14 * t / 1800))
    assert out is not None
    assert out.diagnostics["response"]["window_s"] == 1800
    assert not a.export_state().get("holding_points")


def test_supply_mean_is_time_weighted_and_drops_dhw_like_interruptions():
    a = SlowCalibrationAdapter()
    for t, v in [(0, 30), (600, 40), (900, 40)]:
        out = a.step(context(t, supply_temperature=v))
    assert out.diagnostics["supply_mean_temperature"] == pytest.approx(100 / 3)
    a.controller.interrupt()
    out = a.step(context(960, supply_temperature=36))
    assert out.diagnostics["supply_mean_temperature"] == 36


@pytest.mark.parametrize("room", [20, 20.6, 21, 21.7])
def test_weather_sunshine_does_not_change_control(room):
    a = SlowCalibrationAdapter()
    b = SlowCalibrationAdapter()
    for t in range(0, 7201, 60):
        ctx = context(t, room=room, supply_temperature=35)
        assert a.step(ctx) == b.step(replace(ctx, solar_intensity=1 if t % 600 else 0))


def test_time_gap_cannot_be_counted_as_measured_response():
    a = SlowCalibrationAdapter()
    a.step(context())
    out = a.step(context(10800))
    assert out.diagnostics["observation_age_s"] == 0
    assert "response" not in a.export_state()
    assert out.diagnostics["recent_trend_k_per_h"] is None


def test_clock_reset_discards_supply_mean_history():
    a = SlowCalibrationAdapter()
    a.step(context(supply_temperature=30))
    a.step(context(600, supply_temperature=40))
    out = a.step(context(0, supply_temperature=50))
    assert out.diagnostics["supply_mean_temperature"] == 50
