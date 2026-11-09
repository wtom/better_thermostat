"""The experimental slow mode must reach real adapter writes and obey OFF."""

from unittest.mock import patch

from homeassistant.components.climate import HVACMode
import pytest

from custom_components.better_thermostat.core.clock import FakeClock
from custom_components.better_thermostat.utils.calibration.slow import SlowContext
from custom_components.better_thermostat.utils.scheduler import request_control_cycle

from .conftest import (
    BT_ENTITY,
    WRITE_BUDGET,
    make_entry,
    set_room_sensor,
    setup_entry,
    wait_for,
    wait_for_startup,
)
from .device_profiles import SLOW_VALVE_TRV


@pytest.mark.parametrize("fake_trv", [SLOW_VALVE_TRV], indirect=True)
async def test_slow_recovery_write_and_off_reach_device(hass, fake_trv):
    """A recovery recommendation must open the number entity; OFF closes it."""
    set_room_sensor(hass, 19.0)
    entry = make_entry(fake_trv.profile)
    await setup_entry(hass, entry)
    bt = await wait_for_startup(hass, entry)
    bt.clock = FakeClock(monotonic_value=bt.clock.monotonic())
    # Start a new observation on the test clock without changing its baseline.
    assert bt.state_mgr is not None
    controller = bt.state_mgr.get_slow_live(f"{bt.unique_id}:group:slow")
    controller.interrupt()
    valve = fake_trv.valve_number
    with patch(WRITE_BUDGET, 0.0):
        await hass.services.async_call(
            "climate",
            "set_temperature",
            {"entity_id": BT_ENTITY, "temperature": 22},
            blocking=True,
        )
        request_control_cycle(bt)
        assert await wait_for(
            hass,
            lambda: (
                controller._previous is not None
                and controller._previous.target_temperature == 22
            ),
        )
        bt.clock.advance(600)
        request_control_cycle(bt)
        assert await wait_for(
            hass, lambda: valve.set_value_calls and valve.set_value_calls[-1] > 0
        ), (
            valve.set_value_calls,
            vars(controller),
            bt.real_trvs[fake_trv.entity_id].calibration_balance,
        )
        opening = valve.set_value_calls[-1]
        assert opening <= 10
        assert float(hass.states.get(valve.entity_id).state) == opening
        await hass.services.async_call(
            "climate",
            "set_hvac_mode",
            {"entity_id": BT_ENTITY, "hvac_mode": HVACMode.OFF},
            blocking=True,
        )
        assert await wait_for(hass, lambda: valve.set_value_calls[-1] == 0)
        assert float(hass.states.get(valve.entity_id).state) == 0


@pytest.mark.parametrize("fake_trv", [SLOW_VALVE_TRV], indirect=True)
async def test_slow_learning_survives_store_flush_and_entry_reload(
    hass, hass_storage, fake_trv
):
    """A fresh entity restores learned points, not an interrupted holding timer."""
    set_room_sensor(hass, 21.0)
    entry = make_entry(fake_trv.profile)
    await setup_entry(hass, entry)
    bt = await wait_for_startup(hass, entry)
    await hass.services.async_call(
        "climate",
        "set_hvac_mode",
        {"entity_id": BT_ENTITY, "hvac_mode": HVACMode.OFF},
        blocking=True,
    )
    assert bt.state_mgr is not None
    key = f"{bt.unique_id}:group:slow"
    controller = bt.state_mgr.get_slow_live(key)
    controller.reset({"baseline": 37.0})
    for t in range(0, 3601, 300):
        controller.step(
            SlowContext(
                t=t,
                target_temperature=21.0,
                room_temperature=21.0,
                outdoor_temperature=5.0,
                supply_temperature=35.0,
                last_valve_percent=37.0,
            )
        )
    learned = controller.export_state()
    assert learned["baseline"] == 37.0
    assert len(learned["holding_points"]) == 1
    assert learned["response"]["window_s"] == 3600
    bt.state_mgr.set_slow_live(key, controller)
    bt.schedule_save_state()
    # Unload must flush the pending delayed save; the next setup owns a new
    # manager and controller, so in-memory state cannot make this test pass.
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    stored = hass_storage[f"better_thermostat_{entry.entry_id}_state"]["data"]
    assert stored["slow"][key] == learned
    assert await hass.config_entries.async_setup(entry.entry_id)
    restored_bt = await wait_for_startup(hass, entry)
    assert restored_bt is not bt
    assert restored_bt.state_mgr is not None
    restored = restored_bt.state_mgr.get_slow_live(key)
    assert restored is not controller
    assert restored.export_state() == learned
    assert restored._start is None
    assert restored._recovery is None
