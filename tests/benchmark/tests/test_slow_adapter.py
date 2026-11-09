"""Behavioural checks for the experimental slow calibration path."""

from dataclasses import replace
from typing import Any

import pytest

from tests.benchmark.adapters.base import BenchmarkContext
from tests.benchmark.adapters.slow_adapter import SlowCalibrationAdapter


def context(t=0.0, room=20.6, **kwargs):
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


def test_hold_then_one_small_correction():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 10800, 60):
        assert adapter.step(context(t)).valve_percent == 40
    result = adapter.step(context(10800))
    assert result is not None
    assert result.valve_percent == 43
    assert result is not None
    assert result.diagnostics["reason"] == "calibrate_open"
    assert adapter.step(context(10860)).valve_percent == 43


def test_warming_room_does_not_receive_extra_heat():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 10801, 60):
        result = adapter.step(context(t, room=20.51 + 0.31 * t / 10800))
    assert result is not None
    assert result.valve_percent == 40


def test_recovery_is_sustained_rate_limited_and_not_persisted():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    assert adapter.step(context(room=20)).valve_percent == 40
    assert adapter.step(context(599, room=20)).valve_percent == 40
    assert adapter.step(context(600, room=20)).valve_percent == 50
    assert adapter.step(context(660, room=20)).valve_percent == 50
    assert adapter.step(context(1500, room=20)).valve_percent == 60
    assert adapter.export_state() == {"baseline": 40}
    assert adapter.step(context(1560, room=20.9)).valve_percent == 43
    assert adapter.step(context(1600)).valve_percent == 43


def test_warm_recovery_closes_and_restart_restores_baseline():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=21.7))
    assert adapter.step(context(600, room=21.7)).valve_percent == 0
    restored = SlowCalibrationAdapter(hold_s=10800)
    restored.reset(adapter.export_state())
    assert restored.step(context()).valve_percent == 40


def test_window_and_supply_change_restart_observation():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(supply_temperature=45))
    assert (
        adapter.step(
            context(10740, window_open=True, supply_temperature=45)
        ).valve_percent
        == 0
    )
    adapter.step(context(10800, supply_temperature=35))
    assert adapter.step(context(10860, supply_temperature=35)).valve_percent == 40
    for t in range(10920, 21601, 60):
        result = adapter.step(context(t, supply_temperature=35))
    assert result is not None
    assert result.valve_percent is not None
    assert result.valve_percent > 40
    assert result.diagnostics["response"]["window_s"] >= 1800


def test_unavailable_supply_does_not_train_baseline():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 21601, 60):
        adapter.step(context(t, room=20, supply_temperature=19))
    assert adapter.export_state() == {"baseline": 40}


def test_target_drop_does_not_reuse_old_observations():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context())
    adapter.step(context(10740))
    result = adapter.step(replace(context(10800), target_temperature=20.5))
    assert result is not None
    assert result.valve_percent == 40
    assert result is not None
    assert result.diagnostics["reason"] == "hold"


def test_target_change_discards_preceding_interval_from_mean():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context())
    # A new target begins after a long gap, with a brief warm reading.
    adapter.step(replace(context(100000, room=21.3), target_temperature=21.1))
    for t in range(100060, 110801, 60):
        result = adapter.step(replace(context(t, room=20.8), target_temperature=21.1))
    assert result is not None
    assert result is not None
    assert result.valve_percent == 41


def test_approaching_target_holds_recovery_instead_of_adding_heat():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=20))
    adapter.step(context(600, room=20))
    assert adapter.step(context(1500, room=20.6)).valve_percent == 50
    assert adapter.export_state() == {"baseline": 40}


def test_recovery_completion_learns_only_one_small_step():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=20))
    adapter.step(context(600, room=20))
    adapter.step(context(1500, room=20))
    adapter.step(context(2400, room=20))
    assert adapter.step(context(2460, room=20.9)).valve_percent == 43
    state = adapter.export_state()
    assert state["baseline"] == 43
    assert state["response"]["last_recovery_s"] == 1860
    restored = SlowCalibrationAdapter(hold_s=10800)
    restored.reset(state)
    assert restored.step(context()).valve_percent == 43


def test_changed_target_does_not_teach_recovery_direction():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=20))
    adapter.step(context(600, room=20))
    result = adapter.step(replace(context(660, room=20), target_temperature=20))
    assert result is not None
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_normal_window_records_response_at_applied_not_new_opening():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 10801, 60):
        adapter.step(context(t))
    state = adapter.export_state()
    assert state["baseline"] == 43
    assert state["response"]["observed_opening"] == 40
    assert state["response"]["trend_k_per_h"] == 0
    assert state["response"]["window_s"] == 10800
    restored = SlowCalibrationAdapter(hold_s=10800)
    restored.reset(state)
    assert restored.export_state() == state


def test_gradual_supply_drift_cannot_be_learned_as_valve_response():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 1801, 60):
        adapter.step(context(t, supply_temperature=45 - 12 * t / 1800))
    assert adapter.baseline == 40
    assert adapter.export_state() == {"baseline": 40}
    # Once the supply mean settles, a fresh response window may correct again.
    for t in range(1860, 9001, 60):
        adapter.step(context(t, supply_temperature=33))
    assert adapter.baseline > 40
    assert adapter.export_state()["response"]["window_s"] >= 1800


def test_gradual_weather_drift_restarts_normal_learning():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 10801, 60):
        result = adapter.step(
            replace(context(t), outdoor_temperature=5 - 6 * t / 10800)
        )
    assert result is not None
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_supply_sensor_loss_discards_incomplete_window():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(supply_temperature=45))
    result = adapter.step(context(10800, supply_temperature=None))
    assert result is not None
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_settled_holding_point_survives_restart_and_is_reused_once():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=21, supply_temperature=45))
    for t in range(60, 10801, 60):
        adapter.step(context(t, room=21, supply_temperature=45))
    state = adapter.export_state()
    assert state["holding_points"][0]["opening"] == 40
    # A later context has changed the normal baseline, not the learned point.
    state["baseline"] = 50
    restored = SlowCalibrationAdapter(hold_s=10800)
    restored.reset(state)
    assert restored.step(context(room=21, supply_temperature=45)).valve_percent == 47
    assert (
        restored.step(context(60, room=21, supply_temperature=45)).valve_percent == 47
    )
    assert state["holding_points"][0]["opening"] == 40


def test_holding_point_does_not_extrapolate_or_override_disturbances():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=21, supply_temperature=45))
    for t in range(60, 10801, 60):
        adapter.step(context(t, room=21, supply_temperature=45))
    state = adapter.export_state()
    state["baseline"] = 50
    variants = [
        context(room=21, supply_temperature=None),
        context(room=21, supply_temperature=49),
        replace(context(room=21, supply_temperature=45), outdoor_temperature=8),
        replace(context(room=21, supply_temperature=45), target_temperature=22),
    ]
    for ctx in variants:
        adapter.reset(state)
        assert adapter.step(ctx).valve_percent == 50
    adapter.reset(state)
    assert (
        adapter.step(
            context(room=21, supply_temperature=45, window_open=True)
        ).valve_percent
        == 0
    )


def test_correcting_or_trending_windows_are_not_holding_points():
    for final_room in (20.7, 21.19):
        adapter = SlowCalibrationAdapter(hold_s=10800)
        adapter.step(context(room=20.6, supply_temperature=45))
        adapter.step(context(10800, room=final_room, supply_temperature=45))
        assert "holding_points" not in adapter.export_state()


def test_context_return_reuses_point_but_recovery_never_teaches_one():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=21, supply_temperature=45))
    for t in range(60, 10801, 60):
        adapter.step(context(t, room=21, supply_temperature=45))
    for t in range(10860, 21661, 60):
        adapter.step(
            replace(context(t, room=21.7, supply_temperature=45), target_temperature=22)
        )
    before_return = adapter.baseline
    assert before_return is not None
    assert before_return > 40
    result = adapter.step(context(21720, room=21, supply_temperature=45))
    assert result is not None
    assert result.valve_percent == before_return - 3
    assert result is not None
    assert result.diagnostics["reason"] == "reuse_holding_point"
    adapter.step(context(21780, room=20, supply_temperature=45))
    adapter.step(context(22380, room=20, supply_temperature=45))
    assert len(adapter.export_state()["holding_points"]) == 1


def test_cold_recovery_waits_for_observed_warming_then_resumes_if_stalled():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 1501, 60):
        # Still >0.5 K cold, but warming fast enough to reach the band in an hour.
        result = adapter.step(context(t, room=20 + 0.4 * t / 1500))
    assert result is not None
    assert result.valve_percent == 50
    assert result is not None
    assert result.diagnostics["reason"] == "recovery_wait_response"
    assert result is not None
    assert result.diagnostics["recent_trend_k_per_h"] > 0.9
    assert adapter.export_state() == {"baseline": 40}
    for t in range(1560, 3361, 60):
        result = adapter.step(context(t, room=20.4))
    assert result is not None
    assert result.valve_percent is not None
    assert result.valve_percent > 50  # no endless wait after warming stops


def test_target_change_and_restart_discard_recent_response():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 1501, 60):
        adapter.step(context(t, room=20 + 0.4 * t / 1500))
    result = adapter.step(replace(context(1560, room=20.4), target_temperature=22))
    assert result is not None
    assert result.diagnostics["recent_trend_k_per_h"] is None
    restored = SlowCalibrationAdapter(hold_s=10800)
    restored.reset(adapter.export_state())
    assert restored.step(context(1620)).diagnostics["recent_trend_k_per_h"] is None


def test_weather_and_supply_changes_do_not_reopen_warm_recovery():
    result = None
    changes: dict[str, Any]
    for changes in ({"outdoor_temperature": 9}, {"supply_temperature": 39}):
        adapter = SlowCalibrationAdapter(hold_s=10800)
        adapter.step(context(room=21.7, supply_temperature=45))
        assert (
            adapter.step(context(600, room=21.7, supply_temperature=45)).valve_percent
            == 0
        )
        ctx = replace(context(660, room=21.45, supply_temperature=45), **changes)
        result = adapter.step(ctx)
        assert result is not None
        assert result.valve_percent == 0
        assert result is not None
        assert result.diagnostics["reason"] == "recovery_warm"
        assert (
            adapter.step(replace(ctx, t=720, room_temperature=21.3)).valve_percent == 0
        )
        # A sustained new heating level (not one boiler-cycle sample) must
        # invalidate recovery evidence without cancelling the closed command.
        for t in range(780, 4260, 60):
            assert adapter.step(replace(ctx, t=t)).valve_percent == 0
        result = adapter.step(replace(ctx, t=4260, room_temperature=21.1))
        assert result is not None
        assert result.valve_percent == 40
        assert adapter.export_state() == {"baseline": 40}


def test_environment_drift_preserves_cold_command_without_learning_it():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(room=20, supply_temperature=45))
    assert (
        adapter.step(context(600, room=20, supply_temperature=45)).valve_percent == 50
    )
    ctx = replace(context(660, room=20.6, supply_temperature=45), outdoor_temperature=9)
    assert adapter.step(ctx).valve_percent == 50
    assert adapter.step(replace(ctx, t=720, room_temperature=20.9)).valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_target_and_disturbances_still_cancel_warm_recovery():
    result = None
    changes: dict[str, Any]
    variants: list[tuple[dict[str, Any], int]] = [
        ({"target_temperature": 22}, 40),
        ({"window_open": True}, 0),
        ({"supply_temperature": 19}, 40),
    ]
    for changes, expected in variants:
        adapter = SlowCalibrationAdapter(hold_s=10800)
        adapter.step(context(room=21.7, supply_temperature=45))
        adapter.step(context(600, room=21.7, supply_temperature=45))
        result = adapter.step(
            replace(context(660, room=21.45, supply_temperature=45), **changes)
        )
        assert result is not None
        assert result.valve_percent == expected
        assert result is not None
        assert result.diagnostics["reason"] != "recovery_warm"


def test_boiler_cycles_allow_normal_calibration_from_zero():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    results = []
    for t in range(0, 18001, 60):
        # 30-minute burner cycle, like the live WC pilot; never a heat outage.
        supply = 27 + 14 * ((t % 1800) / 1800)
        results.append(
            adapter.step(
                replace(
                    context(t, room=20.65, supply_temperature=supply),
                    last_valve_percent=0,
                )
            )
        )
    assert any(r.diagnostics["reason"] == "calibrate_open" for r in results)
    corrections = [r for r in results if r.diagnostics["reason"] == "calibrate_open"]
    assert len(corrections) > 1
    assert all(r.diagnostics["response"]["window_s"] >= 1800 for r in corrections)
    assert adapter.baseline == 3 * len(corrections)
    assert results[-1].diagnostics["recent_trend_k_per_h"] == 0


def test_environment_resets_do_not_starve_comfort_timer():
    result = None
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t in range(0, 601, 60):
        result = adapter.step(
            replace(context(t, room=20), outdoor_temperature=5 + t / 60)
        )
    assert result is not None
    assert result.valve_percent == 50
    assert result.diagnostics["reason"] == "recovery_cold"


def test_supply_mean_is_time_weighted_and_expires_old_values():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    for t, supply in [(0, 30), (600, 40), (900, 40)]:
        result = adapter.step(context(t, supply_temperature=supply))
    assert abs(result.diagnostics["supply_mean_temperature"] - 100 / 3) < 1e-9
    for t in range(1500, 4201, 300):
        result = adapter.step(context(t, supply_temperature=40))
    assert result.diagnostics["supply_mean_temperature"] == 40
    result = adapter.step(context(4260, supply_temperature=19))
    assert result.diagnostics["reason"] == "disturbance"
    result = adapter.step(context(4320, supply_temperature=None))
    assert result.diagnostics["supply_mean_temperature"] is None


def test_clock_reset_discards_supply_mean_history():
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.step(context(0, supply_temperature=30))
    adapter.step(context(600, supply_temperature=40))
    adapter.step(context(0, supply_temperature=50))
    result = adapter.step(context(60, supply_temperature=50))
    assert result.diagnostics["supply_mean_temperature"] == 50


def test_boiler_cycles_allow_one_recovery_baseline_step():
    """Normal cycling must not keep a successfully warmed WC at zero baseline."""
    adapter = SlowCalibrationAdapter(hold_s=10800)
    adapter.reset({"baseline": 0})
    result = None
    for t in range(0, 7201, 60):
        supply = 28 if t % 1200 < 600 else 40
        room = 21.0 if t < 3600 else 20.0 if t < 7140 else 20.9
        result = adapter.step(context(t, room=room, supply_temperature=supply))
    assert result is not None
    assert adapter.baseline == 3
    assert result.valve_percent == 3
    assert "last_recovery_s" in adapter.export_state()["response"]


def test_sustained_supply_drift_during_recovery_invalidates_learning():
    """A changed heating level cannot teach a valve-only baseline correction."""
    adapter = SlowCalibrationAdapter(hold_s=10800)
    result = None
    for t in range(0, 7201, 60):
        room = 21.0 if t < 3600 else 20.0 if t < 7200 else 20.9
        supply = 35 if t < 4200 else 45
        result = adapter.step(context(t, room=room, supply_temperature=supply))
    assert result is not None
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


@pytest.mark.parametrize(
    "room,expected", [(20.6, 43), (20.75, 41), (21.25, 39), (21.4, 37), (21.0, 40)]
)
def test_default_hour_window_coarse_and_fine_steps(room, expected):
    adapter = SlowCalibrationAdapter()
    for t in range(0, 3600, 60):
        assert adapter.step(context(t, room=room)).valve_percent == 40
    result = adapter.step(context(3600, room=room))
    assert result.valve_percent == expected
    assert result.diagnostics["hold_window_s"] == 3600
    assert result.diagnostics["response"]["window_s"] == 3600
    assert adapter.step(context(3660, room=room)).valve_percent == expected


def test_coarse_direction_then_fine_adjustment_survives_restart():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 3601, 60):
        adapter.step(context(t, room=20.6))
    assert adapter.baseline == 43
    restored = SlowCalibrationAdapter()
    restored.reset(adapter.export_state())
    result = None
    for t in range(0, 3601, 60):
        result = restored.step(context(t, room=20.75))
    assert result is not None
    assert result.valve_percent == 44
    assert result.diagnostics["response"]["calibration_step_percent"] == 1


def test_unapplied_single_point_step_cannot_train_next_window():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 3601, 60):
        adapter.step(context(t, room=20.75))
    assert adapter.baseline == 41
    for t in range(3660, 10801, 60):
        adapter.controller.observe_applied(t, [40])
        adapter.step(context(t, room=20.75))
    assert adapter.baseline == 41
    assert adapter.export_state()["response"]["observed_opening"] == 40


@pytest.mark.parametrize("room", [20.0, 20.65, 21.0, 21.7])
def test_weather_sunshine_does_not_change_room_control_or_learning(room):
    """Unknown room exposure cannot block comfort or discard measured learning."""
    shaded = SlowCalibrationAdapter()
    weather_sunny = SlowCalibrationAdapter()
    actual = None
    for t in range(0, 7201, 60):
        ctx = context(t, room=room, supply_temperature=35)
        expected = shaded.step(ctx)
        # Both continuous sunshine and changing reports leave room inputs intact.
        solar = 1.0 if t < 3600 or t % 600 < 300 else 0.0
        actual = weather_sunny.step(replace(ctx, solar_intensity=solar))
        assert actual == expected
    assert weather_sunny.export_state() == shaded.export_state()
    assert actual is not None
    assert actual.valve_percent is not None
    if room == 20.0:
        assert actual.valve_percent > 40
    elif room == 21.7:
        assert actual.valve_percent == 0
    elif room == 20.65:
        assert weather_sunny.baseline is not None
        assert weather_sunny.baseline > 40
    else:
        assert weather_sunny.export_state()["holding_points"]


def test_sunshine_does_not_reopen_warm_room_during_recovery():
    """A weather change must not undo the room's active overheat protection."""
    adapter = SlowCalibrationAdapter()
    adapter.step(context(room=21.7, supply_temperature=35))
    assert (
        adapter.step(context(600, room=21.7, supply_temperature=35)).valve_percent == 0
    )
    result = adapter.step(
        context(660, room=21.45, supply_temperature=35, solar_intensity=1)
    )
    assert result.valve_percent == 0
    assert result.diagnostics["reason"] == "recovery_warm"


@pytest.mark.parametrize("baseline", [1, 5])
def test_persistent_deficit_probes_past_ineffective_small_openings(baseline):
    """A cold, stagnant room must not spend hours adding single points."""
    adapter = SlowCalibrationAdapter()
    adapter.reset({"baseline": baseline})
    result = None
    for t in range(0, 1800, 60):
        result = adapter.step(context(t, room=20.7, supply_temperature=35))
        assert result.valve_percent == baseline
    result = adapter.step(context(1800, room=20.7, supply_temperature=35))
    assert result.valve_percent == baseline + 3
    assert result.diagnostics["response"]["window_s"] == 1800
    assert "holding_points" not in adapter.export_state()
    # No rapid ratcheting on the next sensor update.
    assert (
        adapter.step(context(1860, room=20.7, supply_temperature=35)).valve_percent
        == baseline + 3
    )


def test_warming_response_defers_probe_and_keeps_fine_corrections():
    """A responding radiator gets time instead of another coarse increase."""
    adapter = SlowCalibrationAdapter()
    result = None
    for t in range(0, 3601, 60):
        # Still below band, but already warming: only one fine step at full window.
        result = adapter.step(
            context(t, room=20.67 + 0.06 * t / 3600, supply_temperature=35)
        )
        if t < 3600:
            assert result.valve_percent == 40
    assert result is not None
    assert result.valve_percent == 41


def test_adequately_warming_room_needs_no_probe():
    """Measured convergence suppresses an increase even before reaching target."""
    adapter = SlowCalibrationAdapter()
    for t in range(0, 3601, 60):
        result = adapter.step(
            context(t, room=20.6 + 0.15 * t / 3600, supply_temperature=35)
        )
        assert result.valve_percent == 40


def test_probe_requires_a_confirmed_held_opening():
    """Failed actuator writes cannot justify repeated opening increments."""
    adapter = SlowCalibrationAdapter()
    for t in range(0, 1801, 60):
        adapter.step(context(t, room=20.7, supply_temperature=35))
    assert adapter.baseline == 43
    for t in range(1860, 7201, 60):
        adapter.controller.observe_applied(t, [40])
        adapter.step(context(t, room=20.7, supply_temperature=35))
    assert adapter.baseline == 43


def test_probe_observation_restarts_after_interruption():
    """DHW, missing observations and restarts cannot fill a response window."""
    adapter = SlowCalibrationAdapter()
    for t in range(0, 1741, 60):
        adapter.step(context(t, room=20.7, supply_temperature=35))
    adapter.controller.interrupt()
    for t in range(1800, 3600, 60):
        assert (
            adapter.step(context(t, room=20.7, supply_temperature=35)).valve_percent
            == 40
        )
    assert (
        adapter.step(context(3600, room=20.7, supply_temperature=35)).valve_percent
        == 43
    )


@pytest.mark.parametrize("supply", [None, 19.0])
def test_no_accelerated_probe_without_available_heat(supply):
    """Missing or cold supply is not evidence that a small opening is ineffective."""
    adapter = SlowCalibrationAdapter()
    for t in range(0, 3541, 60):
        result = adapter.step(context(t, room=20.7, supply_temperature=supply))
        assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_recent_warming_after_cooling_prevents_extra_probe():
    """A delayed response must stop escalation even when the full-window trend is flat."""
    adapter = SlowCalibrationAdapter()
    for t in range(0, 3600, 60):
        room = 20.8 - 0.1 * t / 1800 if t <= 1800 else 20.7 + 0.15 * (t - 1800) / 1800
        result = adapter.step(context(t, room=room, supply_temperature=35))
        # The first probe is allowed while cooling. The delayed warming that
        # follows must stop a second probe, not be mistaken for continued stall.
        assert result.valve_percent is not None
        assert result.valve_percent <= 43
    assert (
        adapter.step(context(3600, room=20.85, supply_temperature=35)).valve_percent
        == 43
    )


def test_cold_limit_is_not_an_accepted_holding_band():
    """A persistent 0.15 K deficit needs correction before the 0.2 K limit."""
    adapter = SlowCalibrationAdapter()
    result = None
    for t in range(0, 1801, 60):
        result = adapter.step(context(t, room=20.85, supply_temperature=35))
    assert result is not None
    assert result.valve_percent == 41
    assert "holding_points" not in adapter.export_state()


def test_at_cold_limit_does_not_wait_for_half_degree_emergency():
    """At the comfort boundary a stalled room gets a bounded coarse correction."""
    adapter = SlowCalibrationAdapter()
    for t in range(0, 1800, 60):
        assert (
            adapter.step(context(t, room=20.8, supply_temperature=35)).valve_percent
            == 40
        )
    assert (
        adapter.step(context(1800, room=20.8, supply_temperature=35)).valve_percent
        == 43
    )


def test_falling_room_anticipates_the_cold_limit():
    """Crossing 0.2 K within the next half hour should trigger before it happens."""
    adapter = SlowCalibrationAdapter()
    result = None
    for t in range(0, 1801, 60):
        result = adapter.step(
            context(t, room=20.9 - 0.05 * t / 1800, supply_temperature=35)
        )
    assert result is not None
    assert result.valve_percent == 43
