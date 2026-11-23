"""Optional entries of BetterThermostat.extra_state_attributes.

``next_valve_maintenance`` and ``valve_method`` are published only when the
entity actually holds the data behind them: a scheduled maintenance run, and
at least one TRV that has recorded how its valve was last written. A schedule
stamp that is not a timestamp surfaces as an error rather than as a silently
missing attribute.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

from homeassistant.components.climate.const import HVACAction, HVACMode
from homeassistant.core import State
import pytest

from custom_components.better_thermostat.climate import BetterThermostat
from custom_components.better_thermostat.trv import Trv
from tests.factories import make_state_attributes_bt


def _extra_state_attributes(bt: BetterThermostat) -> dict[str, object]:
    """Read the property off the stand-in through the class's own getter."""
    getter = BetterThermostat.extra_state_attributes.fget
    assert getter is not None
    return getter(bt)


class TestNextValveMaintenance:
    """The schedule stamp is published as ISO8601 when it is a timestamp."""

    def test_absent_without_a_schedule(self):
        """No schedule means no key."""
        bt = make_state_attributes_bt(next_valve_maintenance=None)
        attrs = _extra_state_attributes(bt)
        assert "next_valve_maintenance" not in attrs

    def test_timestamp_is_published_as_iso8601(self):
        """A datetime schedule is rendered in ISO8601."""
        due = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)
        bt = make_state_attributes_bt(next_valve_maintenance=due)
        attrs = _extra_state_attributes(bt)
        assert attrs["next_valve_maintenance"] == due.isoformat()

    def test_a_schedule_that_is_not_a_timestamp_surfaces(self):
        """A stamp without ``isoformat`` raises rather than dropping the key."""
        bt = make_state_attributes_bt(next_valve_maintenance="2026-01-08T12:00:00")
        with pytest.raises(AttributeError):
            _extra_state_attributes(bt)


class TestValveMethod:
    """The per-TRV valve-method summary is published when a TRV reports one."""

    def test_absent_without_reported_methods(self):
        """No TRV reported a method, so no key."""
        bt = make_state_attributes_bt(
            real_trvs={"climate.trv": Trv(entity_id="climate.trv")}
        )
        attrs = _extra_state_attributes(bt)
        assert "valve_method" not in attrs

    def test_reported_methods_are_summarized(self):
        """Each TRV that reports a method appears in the summary."""
        trv = Trv(entity_id="climate.trv", last_valve_method="adapter")
        bt = make_state_attributes_bt(real_trvs={"climate.trv": trv})
        attrs = _extra_state_attributes(bt)
        assert attrs["valve_method"] == {"climate.trv": "adapter"}

    def test_only_reporting_trvs_appear(self):
        """A TRV without a recorded method is left out of the summary."""
        reporting = Trv(entity_id="climate.a", last_valve_method="adapter")
        silent = Trv(entity_id="climate.b")
        bt = make_state_attributes_bt(
            real_trvs={"climate.a": reporting, "climate.b": silent}
        )
        attrs = _extra_state_attributes(bt)
        assert attrs["valve_method"] == {"climate.a": "adapter"}


@pytest.mark.parametrize(
    ("openings", "mode", "contact_open", "bump_running", "reports_heating"),
    [
        (("65", "67"), HVACMode.HEAT, False, False, True),
        (("0", "67"), HVACMode.HEAT, False, False, True),
        (("0", "0"), HVACMode.HEAT, False, False, False),
        (("unavailable", "unknown"), HVACMode.HEAT, False, False, False),
        (("10", "0"), HVACMode.HEAT, False, True, False),
        (("65", "67"), HVACMode.HEAT, True, False, False),
        (("65", "67"), HVACMode.OFF, False, False, False),
    ],
)
def test_publish_direct_valve_activity_without_changing_control_action(
    openings, mode, contact_open, bump_running, reports_heating
):
    """An idle MPC controller can still have open valves at the setpoint."""
    trvs = {}
    states = {}
    for index, opening in enumerate(openings):
        valve_entity = f"number.valve_{index}"
        trv = Trv(
            entity_id=f"climate.trv_{index}",
            valve_position_entity=valve_entity,
            last_valve_percent=0,
        )
        if bump_running:
            trv.extra["_trvzb_valve_bump_task"] = Mock(done=Mock(return_value=False))
        trvs[trv.entity_id] = trv
        states[valve_entity] = State(valve_entity, opening)

    bt = make_state_attributes_bt(
        real_trvs=trvs,
        hvac_action=HVACAction.IDLE,
        hvac_mode=mode,
        bt_hvac_mode=mode,
        contact_open=contact_open,
        in_maintenance=False,
    )
    bt.hass.states.get.side_effect = states.get
    bt._has_open_direct_valve_while_idle = (
        BetterThermostat._has_open_direct_valve_while_idle.__get__(bt)
    )

    assert (_extra_state_attributes(bt).get("hvac_action") == HVACAction.HEATING) is (
        reports_heating
    )
    assert bt.hvac_action == HVACAction.IDLE
