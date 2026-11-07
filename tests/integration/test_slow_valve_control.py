"""The experimental slow mode must reach real adapter writes and obey OFF."""

from unittest.mock import patch

from homeassistant.components.climate import HVACMode
import pytest

from custom_components.better_thermostat.core.clock import FakeClock
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
