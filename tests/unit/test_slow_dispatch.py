"""Slow room calibration must use the existing direct-valve and Store paths."""

from dataclasses import replace

from homeassistant.components.climate.const import HVACMode
import pytest

from custom_components.better_thermostat.calibration import _compute_slow_balance
from custom_components.better_thermostat.utils.calibration.slow import (
    SlowCalibration,
    SlowContext,
)
from custom_components.better_thermostat.utils.const import (
    CalibrationMode,
    CalibrationOutput,
)
from custom_components.better_thermostat.utils.state_manager import (
    StateManager,
    _deserialize,
    _serialize,
)
from tests.factories import make_bt


def room(hass, count=1):
    """A room with verified direct-valve feedback, not a mocked controller."""
    bt = make_bt(
        trv_ids=tuple(f"climate.trv_{i}" for i in range(count)), room_temperature=20.6
    )
    bt.unique_id = "slow_test"
    bt.state_mgr = StateManager(hass, "slow_test")
    bt.hass = hass
    bt.mpc_v2_supply_sensor_entity_id = None
    bt.mpc_v2_dhw_sensor_entity_id = None
    bt.door_open = False
    bt.contact_open = False
    for i, trv in enumerate(bt.real_trvs.values()):
        trv.advanced = {
            "calibration_mode": CalibrationMode.SLOW_CALIBRATION,
            "calibration": CalibrationOutput.DIRECT_VALVE_BASED,
        }
        trv.valve_position_entity = f"number.valve_{i}"
        trv.valve_position_writable = True
        trv.valve_position = 40
        trv.current_temperature = 25 + i * 5
    return bt


async def test_room_holds_then_makes_one_small_shared_step(hass):
    """One room decision, not one increment for each radiator."""
    bt = room(hass, 2)
    results: list[float | None] = []
    for t in range(0, 3601, 300):
        bt.clock.monotonic_value = t
        results = [_compute_slow_balance(bt, eid)[0] for eid in bt.real_trvs]
        # A new output is confirmed by all radiators on the next cycle.
        assert results[0] == results[1]
        assert results[0] is not None
        assert results[0] <= 49
        for trv in bt.real_trvs.values():
            trv.valve_position = results[0]
    assert results[0] is not None
    assert results[0] > 43


@pytest.mark.parametrize("gate", ["off", "window", "missing", "mixed", "no_valve"])
async def test_interrupted_or_incompatible_room_does_not_learn(hass, gate):
    """Safety/configuration interruptions cannot count toward a holding window."""
    bt = room(hass, 2)
    eid = "climate.trv_0"
    _compute_slow_balance(bt, eid)
    bt.clock.monotonic_value = 300
    if gate == "off":
        bt.bt_hvac_mode = HVACMode.OFF
    elif gate == "window":
        bt.window_open = True
        bt.contact_open = True
    elif gate == "missing":
        bt.room_temperature = None
    elif gate == "mixed":
        bt.real_trvs["climate.trv_1"].advanced["calibration_mode"] = (
            CalibrationMode.PID_CALIBRATION
        )
    else:
        bt.real_trvs[eid].valve_position_writable = False
    assert _compute_slow_balance(bt, eid) == (None, False)
    assert bt.real_trvs[eid].calibration_balance is None
    assert bt.state_mgr.get_slow_live("slow_test:group:slow")._start is None


async def test_capped_room_does_not_learn_unapplied_opening(hass):
    """The lowest actuator cap bounds the held group experiment."""
    bt = room(hass, 2)
    bt.real_trvs["climate.trv_1"].valve_max_opening = 30
    assert _compute_slow_balance(bt, "climate.trv_0") == (30, True)
    assert _compute_slow_balance(bt, "climate.trv_1") == (30, True)


async def test_missing_state_manager_does_not_issue_a_valve_command(hass):
    """An uninitialized store must not create an untracked live controller."""
    bt = room(hass)
    bt.state_mgr = None
    assert _compute_slow_balance(bt, "climate.trv_0") == (None, False)


@pytest.mark.parametrize("hot", [False, True])
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("interruption", ["dhw", "missing_supply"])
async def test_unavailable_heat_freezes_learning_but_preserves_warm_protection(
    hass, hot, learned, interruption
):
    """DHW diversion and missing configured supply cannot count as room trials."""
    bt = room(hass)
    if learned:
        controller = bt.state_mgr.get_slow_live("slow_test:group:slow")
        controller.reset({"baseline": 35.0})
    if interruption == "dhw":
        bt.mpc_v2_dhw_sensor_entity_id = "binary_sensor.dhw"
        hass.states.async_set("binary_sensor.dhw", "on")
    else:
        bt.mpc_v2_supply_sensor_entity_id = "sensor.missing_supply"
    if hot:
        bt.room_temperature = 21.7
    expected = 0.0 if hot else 35.0 if learned else 40.0
    assert _compute_slow_balance(bt, "climate.trv_0") == (expected, True)
    controller = bt.state_mgr.get_slow_live("slow_test:group:slow")
    assert controller._start is None
    assert controller.baseline == (35.0 if learned else None)
    assert bt.real_trvs["climate.trv_0"].calibration_balance["debug"]["reason"] == (
        "heat_unavailable"
    )


async def test_new_cap_clamps_existing_learning_and_later_calibration(hass):
    """Neither a saved baseline nor a new calibration step may exceed the cap."""
    bt = room(hass)
    eid = "climate.trv_0"
    _compute_slow_balance(bt, eid)
    bt.real_trvs[eid].valve_max_opening = 30
    bt.real_trvs[eid].valve_position = 30
    for t in range(0, 3601, 300):
        bt.clock.monotonic_value = t
        assert _compute_slow_balance(bt, eid) == (30, True)
    controller = bt.state_mgr.get_slow_live("slow_test:group:slow")
    assert controller.baseline == 30
    assert controller._start is None
    assert bt.real_trvs[eid].calibration_balance["debug"]["reason"] == "cap_hold"


async def test_failed_write_does_not_become_learning_evidence(hass):
    """Repeated recommendations without matching feedback cannot teach a baseline."""
    bt = room(hass)
    trv = bt.real_trvs["climate.trv_0"]
    for t in range(0, 3601, 300):
        bt.clock.monotonic_value = t
        _compute_slow_balance(bt, trv.entity_id)
    assert bt.state_mgr.state.slow["slow_test:group:slow"]["baseline"] == 43
    for t in range(3900, 11101, 300):
        bt.clock.monotonic_value = t
        _compute_slow_balance(bt, trv.entity_id)
    assert bt.state_mgr.state.slow["slow_test:group:slow"]["baseline"] == 43


def test_store_roundtrip_restores_working_opening_without_timer(hass):
    manager = StateManager(hass, "slow_test")
    controller = manager.get_slow_live("room:group:slow")
    ctx = SlowContext(
        t=0,
        target_temperature=21,
        room_temperature=20,
        outdoor_temperature=5,
        last_valve_percent=37,
    )
    assert controller.step(ctx).valve_percent == 40
    manager.set_slow_live("room:group:slow", controller)
    loaded = _deserialize(_serialize(manager.state))
    restored = SlowCalibration()
    restored.reset(loaded.slow["room:group:slow"])
    assert restored.baseline == 40
    assert restored._start is None
    assert restored.step(replace(ctx, room_temperature=21)).valve_percent == 40


def test_corrupt_stored_learning_is_not_reused():
    """Invalid baseline/holding points cannot produce a non-finite valve intent."""
    restored = SlowCalibration()
    restored.reset(
        {"baseline": float("nan"), "holding_points": [None, {"opening": float("inf")}]}
    )
    assert restored.export_state() == {}


def test_long_measurement_gap_discards_observation():
    """A three-hour outage is not a three-hour controlled experiment."""
    controller = SlowCalibration()
    ctx = SlowContext(
        t=0,
        target_temperature=21,
        room_temperature=20.6,
        outdoor_temperature=5,
        last_valve_percent=40,
    )
    controller.step(ctx)
    out = controller.step(replace(ctx, t=10800))
    assert out.diagnostics["observation_age_s"] == 0
    assert out.diagnostics["recent_trend_k_per_h"] is None


def test_solar_disturbance_does_not_disable_warm_protection():
    """Weather sunshine leaves sustained room-level overheat protection active."""
    controller = SlowCalibration()
    ctx = SlowContext(
        t=0,
        target_temperature=21,
        room_temperature=21.7,
        outdoor_temperature=5,
        last_valve_percent=40,
        solar_intensity=0.8,
    )
    controller.step(ctx)
    assert controller.step(replace(ctx, t=600)).valve_percent == 37
    assert controller.baseline == 37


def test_warming_trend_closes_before_significant_overshoot():
    """Sustained measured warming predicts the half-K comfort boundary."""
    result = None
    controller = SlowCalibration()
    for t in range(0, 1801, 60):
        result = controller.step(
            SlowContext(
                t=t,
                target_temperature=21,
                room_temperature=21.05 + t / 3600,
                outdoor_temperature=5,
                last_valve_percent=40,
            )
        )
    assert result is not None
    assert result.valve_percent < 40
    assert controller.baseline == result.valve_percent


async def test_unconfirmed_close_does_not_teach_holding_point(hass):
    bt = room(hass)
    bt.room_temperature = 21.7
    eid = "climate.trv_0"
    assert _compute_slow_balance(bt, eid)[0] == 37
    for t in range(300, 7201, 300):
        bt.clock.monotonic_value = t
        bt.room_temperature = 21.0
        assert _compute_slow_balance(bt, eid)[0] == 37
    assert not bt.state_mgr.state.slow["slow_test:group:slow"].get("holding_points")


async def test_slow_explanation_is_published_as_entity_attribute(hass):
    """The house trial needs baseline/reason/trend without reading Python state."""
    import json

    from custom_components.better_thermostat.utils.telemetry import (
        collect_balance_attrs,
    )

    bt = room(hass)
    bt.temperature_slope = None
    _compute_slow_balance(bt, "climate.trv_0")
    value = collect_balance_attrs(bt)["slow_calibration"]
    assert isinstance(value, str)
    debug = json.loads(value)["climate.trv_0"]
    assert debug["baseline"] == 43
    assert debug["reason"] == "calibrate_open"
    assert debug["next_review_in_s"] == 1800


async def test_dhw_hot_supply_never_enters_or_survives_in_heating_mean(hass):
    """DHW interrupts the mean as well as the learning timer."""
    bt = room(hass)
    bt.mpc_v2_dhw_sensor_entity_id = "sensor.dhw_pump"
    bt.mpc_v2_supply_sensor_entity_id = "sensor.supply"
    eid = "climate.trv_0"
    for t, dhw, supply in [
        (0, "0", 35),
        (300, "0", 35),
        (600, "1", 65),
        (900, "1", 70),
        (1200, "0", 36),
    ]:
        bt.clock.monotonic_value = t
        hass.states.async_set("sensor.dhw_pump", dhw)
        hass.states.async_set("sensor.supply", supply)
        _compute_slow_balance(bt, eid)
        debug = bt.real_trvs[eid].calibration_balance["debug"]
        if dhw == "1":
            assert debug["reason"] == "heat_unavailable"
            assert not bt.state_mgr.get_slow_live(
                "slow_test:group:slow"
            )._supply_samples
    assert debug["supply_mean_temperature"] == 36
