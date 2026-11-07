"""Behavioural checks for the experimental slow calibration path."""

from dataclasses import replace

from tests.benchmark.adapters.base import BenchmarkContext
from tests.benchmark.adapters.slow_adapter import SlowCalibrationAdapter


def context(t=0.0, room=20.7, **kwargs):
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
    adapter = SlowCalibrationAdapter()
    for t in range(0, 10800, 60):
        assert adapter.step(context(t)).valve_percent == 40
    result = adapter.step(context(10800))
    assert result.valve_percent == 43
    assert result.diagnostics["reason"] == "calibrate_open"
    assert adapter.step(context(10860)).valve_percent == 43


def test_warming_room_does_not_receive_extra_heat():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 10801, 60):
        result = adapter.step(context(t, room=20.51 + 0.29 * t / 10800))
    assert result.valve_percent == 40


def test_recovery_is_sustained_rate_limited_and_not_persisted():
    adapter = SlowCalibrationAdapter()
    assert adapter.step(context(room=20)).valve_percent == 40
    assert adapter.step(context(599, room=20)).valve_percent == 40
    assert adapter.step(context(600, room=20)).valve_percent == 50
    assert adapter.step(context(660, room=20)).valve_percent == 50
    assert adapter.step(context(1500, room=20)).valve_percent == 60
    assert adapter.export_state() == {"baseline": 40}
    assert adapter.step(context(1560, room=20.9)).valve_percent == 43
    assert adapter.step(context(1600)).valve_percent == 43


def test_warm_recovery_closes_and_restart_restores_baseline():
    adapter = SlowCalibrationAdapter()
    adapter.step(context(room=21.7))
    assert adapter.step(context(600, room=21.7)).valve_percent == 0
    restored = SlowCalibrationAdapter()
    restored.reset(adapter.export_state())
    assert restored.step(context()).valve_percent == 40


def test_window_and_supply_change_restart_observation():
    adapter = SlowCalibrationAdapter()
    adapter.step(context(supply_temperature=45))
    assert (
        adapter.step(
            context(10740, window_open=True, supply_temperature=45)
        ).valve_percent
        == 0
    )
    adapter.step(context(10800, supply_temperature=35))
    assert adapter.step(context(10860, supply_temperature=35)).valve_percent == 40
    assert adapter.step(context(21600, supply_temperature=35)).valve_percent == 43


def test_solar_and_unavailable_supply_do_not_train_baseline():
    for disturbance in ({"solar_intensity": 0.5}, {"supply_temperature": 19}):
        adapter = SlowCalibrationAdapter()
        for t in range(0, 21601, 60):
            adapter.step(context(t, room=20, **disturbance))
        assert adapter.export_state() == {"baseline": 40}


def test_target_drop_does_not_reuse_old_observations():
    adapter = SlowCalibrationAdapter()
    adapter.step(context())
    adapter.step(context(10740))
    result = adapter.step(replace(context(10800), target_temperature=20.5))
    assert result.valve_percent == 40
    assert result.diagnostics["reason"] == "hold"


def test_target_change_discards_preceding_interval_from_mean():
    adapter = SlowCalibrationAdapter()
    adapter.step(context())
    # A new target begins after a long gap, with a brief warm reading.
    adapter.step(replace(context(100000, room=21.3), target_temperature=21.1))
    for t in range(100060, 110801, 60):
        result = adapter.step(replace(context(t, room=20.8), target_temperature=21.1))
    assert result.valve_percent == 43


def test_approaching_target_holds_recovery_instead_of_adding_heat():
    adapter = SlowCalibrationAdapter()
    adapter.step(context(room=20))
    adapter.step(context(600, room=20))
    assert adapter.step(context(1500, room=20.6)).valve_percent == 50
    assert adapter.export_state() == {"baseline": 40}


def test_recovery_completion_learns_only_one_small_step():
    adapter = SlowCalibrationAdapter()
    adapter.step(context(room=20))
    adapter.step(context(600, room=20))
    adapter.step(context(1500, room=20))
    adapter.step(context(2400, room=20))
    assert adapter.step(context(2460, room=20.9)).valve_percent == 43
    state = adapter.export_state()
    assert state["baseline"] == 43
    assert state["response"]["last_recovery_s"] == 1860
    restored = SlowCalibrationAdapter()
    restored.reset(state)
    assert restored.step(context()).valve_percent == 43


def test_changed_target_does_not_teach_recovery_direction():
    adapter = SlowCalibrationAdapter()
    adapter.step(context(room=20))
    adapter.step(context(600, room=20))
    result = adapter.step(replace(context(660, room=20), target_temperature=20))
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_normal_window_records_response_at_applied_not_new_opening():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 10801, 60):
        adapter.step(context(t))
    state = adapter.export_state()
    assert state["baseline"] == 43
    assert state["response"]["observed_opening"] == 40
    assert state["response"]["trend_k_per_h"] == 0
    assert state["response"]["window_s"] == 10800
    restored = SlowCalibrationAdapter()
    restored.reset(state)
    assert restored.export_state() == state


def test_gradual_supply_drift_cannot_be_learned_as_valve_response():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 10801, 60):
        result = adapter.step(context(t, supply_temperature=45 - 6 * t / 10800))
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}
    # A complete stable window is required after the final reset.
    for t in range(10860, 21601, 60):
        result = adapter.step(context(t, supply_temperature=39))
    assert result.valve_percent == 43


def test_gradual_weather_drift_restarts_normal_learning():
    adapter = SlowCalibrationAdapter()
    for t in range(0, 10801, 60):
        result = adapter.step(
            replace(context(t), outdoor_temperature=5 - 6 * t / 10800)
        )
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}


def test_supply_sensor_loss_discards_incomplete_window():
    adapter = SlowCalibrationAdapter()
    adapter.step(context(supply_temperature=45))
    result = adapter.step(context(10800, supply_temperature=None))
    assert result.valve_percent == 40
    assert adapter.export_state() == {"baseline": 40}
