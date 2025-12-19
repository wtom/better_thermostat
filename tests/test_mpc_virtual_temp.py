import pytest
from unittest.mock import patch

from custom_components.better_thermostat.utils.calibration import mpc


def _make_params():
    params = mpc.MpcParams()
    params.mpc_adapt = False
    params.mpc_thermal_gain = 0.06
    params.mpc_loss_coeff = 0.01
    params.mpc_gain_min = 0.01
    params.mpc_gain_max = 0.2
    params.mpc_loss_min = 0.002
    params.mpc_loss_max = 0.03
    return params


def test_virtual_temp_anchors_on_sensor_change_only():
    key = "t:climate.x:t20.0"
    mpc._MPC_STATES.pop(key, None)

    params = _make_params()

    inp = mpc.MpcInput(
        key=key,
        target_temp_C=21.0,
        current_temp_C=20.0,
        bt_name="BT",
        entity_id="climate.x",
    )

    # Seed state with a known u (50%) so prediction is deterministic.
    state = mpc._MPC_STATES.setdefault(key, mpc._MpcState())
    state.last_percent = 50.0

    # First call: sensor initializes anchor at now=100
    with patch("custom_components.better_thermostat.utils.calibration.mpc.monotonic", side_effect=[100.0]):
        out1 = mpc.compute_mpc(inp, params)
        assert out1 is not None
        assert state.last_sensor_temp == 20.0
        assert state.last_sensor_ts == 100.0
        assert state.virtual_base_temp == 20.0
        assert state.virtual_base_ts == 100.0

    # Second call: sensor unchanged, time passes to now=160 -> virtual_temp should move from base by dt
    # predicted_dT = (gain*u - loss) * dt_min = (0.06*0.5 - 0.01) * (60/60) = 0.02
    with patch("custom_components.better_thermostat.utils.calibration.mpc.monotonic", side_effect=[160.0]):
        out2 = mpc.compute_mpc(inp, params)
        assert out2 is not None
        assert state.last_sensor_ts == 100.0
        assert state.virtual_base_ts == 100.0
        assert state.virtual_temp == pytest.approx(20.02, rel=0, abs=1e-6)

    # Third call: sensor changes to 20.2 at now=220 -> anchor resets
    inp2 = mpc.MpcInput(
        key=key,
        target_temp_C=21.0,
        current_temp_C=20.2,
        bt_name="BT",
        entity_id="climate.x",
    )
    with patch("custom_components.better_thermostat.utils.calibration.mpc.monotonic", side_effect=[220.0]):
        out3 = mpc.compute_mpc(inp2, params)
        assert out3 is not None
        assert state.last_sensor_temp == 20.2
        assert state.last_sensor_ts == 220.0
        assert state.virtual_base_temp == 20.2
        assert state.virtual_base_ts == 220.0
