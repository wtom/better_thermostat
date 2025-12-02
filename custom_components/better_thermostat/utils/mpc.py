"""Lightweight MPC helper independent from balance logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, Mapping, Optional, Tuple


_LOGGER = logging.getLogger(__name__)


# MPC operates on fixed 5-minute steps and a 12-step horizon.
MPC_STEP_SECONDS = 300.0
MPC_HORIZON_STEPS = 12
PHASE_MIN_DURATION_S = 300.0
PHASE_PERCENT_THRESHOLD = 1.0


@dataclass
class MpcParams:
    """Configuration for the predictive controller."""

    cap_max_K: float = 0.8
    percent_hysteresis_pts: float = 0.5
    min_update_interval_s: float = 60.0
    mpc_thermal_gain: float = 0.02
    mpc_loss_coeff: float = 0.015
    mpc_control_penalty: float = 0.00002
    mpc_change_penalty: float = 0.01
    mpc_adapt: bool = True
    mpc_gain_min: float = 0.005
    mpc_gain_max: float = 0.5
    mpc_loss_min: float = 0.0
    mpc_loss_max: float = 0.05
    mpc_adapt_alpha: float = 0.1
    deadzone_threshold_pct: float = 20.0
    deadzone_temp_delta_K: float = 0.2
    deadzone_time_s: float = 300.0
    deadzone_hits_required: int = 3
    deadzone_raise_pct: float = 2.0
    deadzone_decay_pct: float = 1.0
    deadzone_room_delta_guard_K: float = 1.5
    deadzone_room_temp_delta_guard_K: float = 0.2
    deadzone_room_slope_guard_K_per_min: float = 0.01
    deadzone_phase_min_s: float = 120.0
    deadzone_score_increment: float = 1.0
    deadzone_score_decay: float = 0.5
    deadzone_score_cap: float = 10.0
    hold_tolerance_K: float = 0.2
    gain_phase_timeout_s: float = 1800.0
    idle_phase_timeout_s: float = 1800.0


@dataclass
class MpcInput:
    key: str
    target_temp_C: Optional[float]
    current_temp_C: Optional[float]
    trv_temp_C: Optional[float] = None
    tolerance_K: float = 0.0
    temp_slope_K_per_min: Optional[float] = None
    window_open: bool = False
    heating_allowed: bool = True
    bt_name: Optional[str] = None
    entity_id: Optional[str] = None


@dataclass
class MpcOutput:
    valve_percent: int
    flow_cap_K: float
    setpoint_eff_C: Optional[float]
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _MpcState:
    last_percent: Optional[float] = None
    last_update_ts: float = 0.0
    last_target_C: Optional[float] = None
    ema_slope: Optional[float] = None
    gain_est: Optional[float] = None
    loss_est: Optional[float] = None
    last_temp: Optional[float] = None
    last_time: float = 0.0
    last_trv_temp: Optional[float] = None
    last_trv_temp_ts: float = 0.0
    dead_zone_hits: int = 0
    dead_zone_score: float = 0.0
    min_effective_percent: Optional[float] = None
    heat_phase_start_temp: Optional[float] = None
    heat_phase_start_ts: float = 0.0
    heat_phase_percent: Optional[float] = None
    idle_phase_start_temp: Optional[float] = None
    idle_phase_start_ts: float = 0.0
    idle_phase_target: Optional[float] = None
    last_dead_zone_room_temp: Optional[float] = None
    last_dead_zone_room_ts: float = 0.0
    last_dead_zone_room_delta: Optional[float] = None


_MPC_STATES: Dict[str, _MpcState] = {}

_STATE_EXPORT_FIELDS = (
    "last_percent",
    "last_target_C",
    "ema_slope",
    "gain_est",
    "loss_est",
    "last_temp",
    "last_trv_temp",
    "min_effective_percent",
    "dead_zone_hits",
    "dead_zone_score",
)


def _serialize_state(state: _MpcState) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for attr in _STATE_EXPORT_FIELDS:
        value = getattr(state, attr, None)
        if value is None:
            continue
        payload[attr] = value
    return payload


def export_mpc_state_map(prefix: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return a serializable mapping of MPC states, optionally filtered by key prefix."""

    exported: Dict[str, Dict[str, Any]] = {}
    for key, state in _MPC_STATES.items():
        if prefix is not None and not key.startswith(prefix):
            continue
        payload = _serialize_state(state)
        if payload:
            exported[key] = payload
    return exported


def import_mpc_state_map(state_map: Mapping[str, Mapping[str, Any]]) -> None:
    """Hydrate MPC states from a previously exported mapping."""

    for key, payload in state_map.items():
        if not isinstance(payload, Mapping):
            continue
        state = _MPC_STATES.setdefault(key, _MpcState())
        for attr in _STATE_EXPORT_FIELDS:
            if attr not in payload:
                continue
            value = payload[attr]
            if value is None:
                setattr(state, attr, None)
                continue
            try:
                if attr == "dead_zone_hits":
                    coerced = int(value)
                else:
                    coerced = float(value)
            except (TypeError, ValueError):
                continue
            setattr(state, attr, coerced)


def _split_mpc_key(key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        uid, entity, bucket = key.split(":", 2)
        return uid, entity, bucket
    except ValueError:
        return None, None, None


def _seed_state_from_siblings(key: str, state: _MpcState) -> None:
    if state.min_effective_percent is not None:
        return
    uid, entity, _ = _split_mpc_key(key)
    if not uid or not entity:
        return
    for other_key, other_state in _MPC_STATES.items():
        if other_key == key:
            continue
        ouid, oentity, _ = _split_mpc_key(other_key)
        if ouid == uid and oentity == entity:
            if other_state.min_effective_percent is not None:
                state.min_effective_percent = other_state.min_effective_percent
                return


def build_mpc_key(bt, entity_id: str) -> str:
    """Return a stable key for MPC state tracking."""

    try:
        target = bt.bt_target_temp
        bucket = (
            f"t{round(float(target) * 2.0) / 2.0:.1f}"
            if isinstance(target, (int, float))
            else "tunknown"
        )
    except (TypeError, ValueError):
        bucket = "tunknown"

    uid = getattr(bt, "unique_id", None) or getattr(bt, "_unique_id", "bt")
    return f"{uid}:{entity_id}:{bucket}"


def _round_for_debug(value: Any, digits: int = 3) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _log_phase_event(
    name: str, entity: str, phase: str, event: str, **details: Any
) -> None:
    parts = []
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, bool):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={_round_for_debug(value, 3)}")
    suffix = f" ({' '.join(parts)})" if parts else ""
    _LOGGER.debug(
        "better_thermostat %s: MPC %s phase %s %s%s", name, entity, phase, event, suffix
    )


def _reset_heat_phase(state: _MpcState) -> None:
    state.heat_phase_start_temp = None
    state.heat_phase_start_ts = 0.0
    state.heat_phase_percent = None


def _reset_idle_phase(state: _MpcState) -> None:
    state.idle_phase_start_temp = None
    state.idle_phase_start_ts = 0.0
    state.idle_phase_target = None


def _blend_estimate(current: Optional[float], candidate: float, alpha: float) -> float:
    base = current if current is not None else candidate
    return (1.0 - alpha) * base + alpha * candidate


def _apply_gain_candidate(
    state: _MpcState, params: MpcParams, candidate: float
) -> None:
    alpha = min(max(params.mpc_adapt_alpha, 0.0), 1.0)
    blended = _blend_estimate(state.gain_est, candidate, alpha)
    state.gain_est = max(params.mpc_gain_min, min(params.mpc_gain_max, blended))


def _penalize_gain_estimate(
    state: _MpcState, params: MpcParams, *, penalty_scale: float = 1.0
) -> None:
    if state.gain_est is None:
        state.gain_est = params.mpc_thermal_gain
    alpha = min(max(params.mpc_adapt_alpha, 0.0), 1.0)
    alpha = min(1.0, alpha * max(penalty_scale, 0.0))
    target = max(params.mpc_gain_min, min(params.mpc_gain_max, params.mpc_gain_min))
    state.gain_est = (1.0 - alpha) * state.gain_est + alpha * target
    state.gain_est = max(params.mpc_gain_min, min(params.mpc_gain_max, state.gain_est))


def _apply_loss_candidate(
    state: _MpcState, params: MpcParams, candidate: float
) -> None:
    alpha = min(max(params.mpc_adapt_alpha, 0.0), 1.0)
    blended = _blend_estimate(state.loss_est, candidate, alpha)
    state.loss_est = max(params.mpc_loss_min, min(params.mpc_loss_max, blended))


def _penalize_loss_estimate(state: _MpcState, params: MpcParams) -> None:
    if state.loss_est is None:
        state.loss_est = params.mpc_loss_coeff
    shrink = max(0.0, 1.0 - min(max(params.mpc_adapt_alpha, 0.0), 1.0))
    state.loss_est *= shrink
    state.loss_est = max(params.mpc_loss_min, min(params.mpc_loss_max, state.loss_est))


def _start_heating_phase(
    state: _MpcState, inp: MpcInput, now: float, percent: float
) -> None:
    if inp.current_temp_C is None:
        _reset_heat_phase(state)
        return
    state.heat_phase_start_temp = float(inp.current_temp_C)
    state.heat_phase_start_ts = now
    state.heat_phase_percent = float(percent)


def _start_idle_phase(state: _MpcState, inp: MpcInput, now: float) -> None:
    if inp.current_temp_C is None:
        _reset_idle_phase(state)
        return
    state.idle_phase_start_temp = float(inp.current_temp_C)
    state.idle_phase_start_ts = now
    state.idle_phase_target = inp.target_temp_C


def _finalize_heating_phase(
    state: _MpcState, inp: MpcInput, params: MpcParams, now: float
) -> Dict[str, Any]:
    name = inp.bt_name or "BT"
    entity = inp.entity_id or "unknown"
    result: Dict[str, Any] = {}
    if not params.mpc_adapt:
        _reset_heat_phase(state)
        return result
    if (
        state.heat_phase_start_temp is None
        or state.heat_phase_start_ts <= 0.0
        or state.heat_phase_percent is None
    ):
        _reset_heat_phase(state)
        _log_phase_event(name, entity, "heating", "skip", reason="missing_context")
        return result
    temp_now = inp.current_temp_C
    if temp_now is None:
        _reset_heat_phase(state)
        _log_phase_event(name, entity, "heating", "skip", reason="missing_temp")
        return result
    duration = now - state.heat_phase_start_ts
    temp_gain = float(temp_now) - float(state.heat_phase_start_temp)
    percent_used = max(0.0, float(state.heat_phase_percent))
    result.update(
        {
            "gain_duration_s": duration,
            "gain_temp_delta": temp_gain,
            "gain_percent_used": percent_used,
        }
    )
    if duration < PHASE_MIN_DURATION_S or percent_used <= 0.0:
        result["gain_phase_skipped"] = True
        _reset_heat_phase(state)
        _log_phase_event(
            name,
            entity,
            "heating",
            "skip",
            duration_s=duration,
            percent=percent_used,
            reason="min_duration",
        )
        return result
    duration_min = duration / 60.0
    if duration_min <= 0.0:
        _reset_heat_phase(state)
        _log_phase_event(name, entity, "heating", "skip", reason="invalid_duration")
        return result
    if temp_gain > 0.0:
        gain_candidate = (temp_gain / duration_min) / (percent_used / 100.0)
        _apply_gain_candidate(state, params, gain_candidate)
        result["gain_candidate"] = gain_candidate
        result["gain_phase_skipped"] = False
        _log_phase_event(
            name,
            entity,
            "heating",
            "finalize",
            duration_s=duration,
            temp_delta=temp_gain,
            percent=percent_used,
            candidate=gain_candidate,
        )
    else:
        duration_scale = 0.0
        if PHASE_MIN_DURATION_S > 0.0:
            duration_scale = min(1.5, max(0.0, duration / PHASE_MIN_DURATION_S - 1.0))
        percent_scale = min(2.0, max(0.0, percent_used / 20.0))
        penalty_scale = 1.0 + duration_scale + percent_scale
        _penalize_gain_estimate(state, params, penalty_scale=penalty_scale)
        result["gain_candidate"] = None
        result["gain_phase_skipped"] = False
        result["gain_penalty_scale"] = penalty_scale
        _log_phase_event(
            name,
            entity,
            "heating",
            "finalize",
            duration_s=duration,
            temp_delta=temp_gain,
            percent=percent_used,
            penalty=penalty_scale,
            candidate=None,
        )
    _reset_heat_phase(state)
    return result


def _finalize_idle_phase(
    state: _MpcState, inp: MpcInput, params: MpcParams, now: float
) -> Dict[str, Any]:
    name = inp.bt_name or "BT"
    entity = inp.entity_id or "unknown"
    result: Dict[str, Any] = {}
    if not params.mpc_adapt:
        _reset_idle_phase(state)
        return result
    if state.idle_phase_start_temp is None or state.idle_phase_start_ts <= 0.0:
        _reset_idle_phase(state)
        _log_phase_event(name, entity, "idle", "skip", reason="missing_context")
        return result
    temp_now = inp.current_temp_C
    if temp_now is None:
        _reset_idle_phase(state)
        _log_phase_event(name, entity, "idle", "skip", reason="missing_temp")
        return result
    duration = now - state.idle_phase_start_ts
    temp_drop = float(state.idle_phase_start_temp) - float(temp_now)
    result.update({"loss_duration_s": duration, "loss_temp_delta": temp_drop})
    if duration < PHASE_MIN_DURATION_S:
        result["loss_phase_skipped"] = True
        _reset_idle_phase(state)
        _log_phase_event(
            name, entity, "idle", "skip", duration_s=duration, reason="min_duration"
        )
        return result
    duration_min = duration / 60.0
    if duration_min <= 0.0:
        _reset_idle_phase(state)
        _log_phase_event(name, entity, "idle", "skip", reason="invalid_duration")
        return result
    if temp_drop > 0.0:
        error_ref = None
        if (
            state.idle_phase_target is not None
            and state.idle_phase_start_temp is not None
        ):
            error_ref = float(state.idle_phase_target) - float(
                state.idle_phase_start_temp
            )
        denom = max(0.2, abs(error_ref) if error_ref is not None else temp_drop)
        loss_candidate = (temp_drop / duration_min) / denom
        _apply_loss_candidate(state, params, loss_candidate)
        result["loss_candidate"] = loss_candidate
        result["loss_phase_skipped"] = False
        _log_phase_event(
            name,
            entity,
            "idle",
            "finalize",
            duration_s=duration,
            temp_delta=-temp_drop,
            candidate=loss_candidate,
        )
    else:
        _penalize_loss_estimate(state, params)
        result["loss_candidate"] = None
        result["loss_phase_skipped"] = False
        _log_phase_event(
            name,
            entity,
            "idle",
            "finalize",
            duration_s=duration,
            temp_delta=-temp_drop,
            candidate=None,
        )
    _reset_idle_phase(state)
    return result


def _update_phase_tracking(
    *,
    state: _MpcState,
    inp: MpcInput,
    params: MpcParams,
    now: float,
    prev_percent: float,
    new_percent: float,
) -> Dict[str, Any]:
    if not params.mpc_adapt:
        return {}

    phase_debug: Dict[str, Any] = {}
    name = inp.bt_name or "BT"
    entity = inp.entity_id or "unknown"
    prev = max(0.0, float(prev_percent))
    new = max(0.0, float(new_percent))
    percent_changed = abs(new - prev) >= PHASE_PERCENT_THRESHOLD

    if prev <= 0.0 and new > 0.0:
        phase_debug.update(_finalize_idle_phase(state, inp, params, now))
        _start_heating_phase(state, inp, now, new)
        _log_phase_event(
            name, entity, "heating", "start", percent=new, temp=inp.current_temp_C
        )
    elif prev > 0.0 and new <= 0.0:
        phase_debug.update(_finalize_heating_phase(state, inp, params, now))
        _start_idle_phase(state, inp, now)
        _log_phase_event(name, entity, "idle", "start", temp=inp.current_temp_C)
    elif prev > 0.0 and new > 0.0 and percent_changed:
        phase_debug.update(_finalize_heating_phase(state, inp, params, now))
        _start_heating_phase(state, inp, now, new)
        _log_phase_event(
            name, entity, "heating", "restart", percent=new, temp=inp.current_temp_C
        )
    else:
        if new > 0.0 and state.heat_phase_start_temp is None:
            _start_heating_phase(state, inp, now, new)
            _log_phase_event(
                name, entity, "heating", "start", percent=new, temp=inp.current_temp_C
            )
        if (
            new <= 0.0
            and prev <= 0.0
            and state.idle_phase_start_temp is None
            and new == 0.0
        ):
            _start_idle_phase(state, inp, now)
            _log_phase_event(name, entity, "idle", "start", temp=inp.current_temp_C)

    heat_timeout = max(params.gain_phase_timeout_s, 0.0)
    if (
        heat_timeout > 0.0
        and state.heat_phase_start_ts > 0.0
        and state.heat_phase_start_temp is not None
        and new > 0.0
    ):
        heat_age = now - state.heat_phase_start_ts
        if heat_age >= heat_timeout:
            phase_debug.update(_finalize_heating_phase(state, inp, params, now))
            phase_debug["heat_phase_timeout"] = heat_age
            _log_phase_event(
                name,
                entity,
                "heating",
                "timeout",
                age_s=heat_age,
                timeout_s=heat_timeout,
            )
            _start_heating_phase(state, inp, now, new)

    idle_timeout = max(params.idle_phase_timeout_s, 0.0)
    if (
        idle_timeout > 0.0
        and state.idle_phase_start_ts > 0.0
        and state.idle_phase_start_temp is not None
        and new <= 0.0
    ):
        idle_age = now - state.idle_phase_start_ts
        if idle_age >= idle_timeout:
            phase_debug.update(_finalize_idle_phase(state, inp, params, now))
            phase_debug["idle_phase_timeout"] = idle_age
            _log_phase_event(
                name, entity, "idle", "timeout", age_s=idle_age, timeout_s=idle_timeout
            )
            _start_idle_phase(state, inp, now)

    return phase_debug


def compute_mpc(inp: MpcInput, params: MpcParams) -> Optional[MpcOutput]:
    """Run the predictive controller and emit a valve recommendation."""

    now = monotonic()
    state = _MPC_STATES.setdefault(inp.key, _MpcState())
    _seed_state_from_siblings(inp.key, state)

    extra_debug: Dict[str, Any] = {}
    name = inp.bt_name or "BT"
    entity = inp.entity_id or "unknown"

    _LOGGER.debug(
        "better_thermostat %s: MPC input (%s) target=%s current=%s trv=%s slope=%s window_open=%s allowed=%s last_percent=%s key=%s",
        name,
        entity,
        _round_for_debug(inp.target_temp_C, 3),
        _round_for_debug(inp.current_temp_C, 3),
        _round_for_debug(inp.trv_temp_C, 3),
        _round_for_debug(inp.temp_slope_K_per_min, 4),
        inp.window_open,
        inp.heating_allowed,
        _round_for_debug(state.last_percent, 2),
        inp.key,
    )

    initial_delta_t: Optional[float] = None

    if not inp.heating_allowed or inp.window_open:
        percent = 0.0
        delta_t = None
        _LOGGER.debug(
            "better_thermostat %s: MPC skip heating (%s) window_open=%s heating_allowed=%s",
            name,
            entity,
            inp.window_open,
            inp.heating_allowed,
        )
    else:
        if inp.target_temp_C is None or inp.current_temp_C is None:
            percent = state.last_percent if state.last_percent is not None else 0.0
            delta_t = None
            _LOGGER.debug(
                "better_thermostat %s: MPC missing temps (%s) reusing last_percent=%s",
                name,
                entity,
                _round_for_debug(percent, 2),
            )
        else:
            delta_t = inp.target_temp_C - inp.current_temp_C
            initial_delta_t = delta_t
            percent, mpc_debug = _compute_predictive_percent(
                inp, params, state, now, delta_t
            )
            extra_debug = mpc_debug
            _LOGGER.debug(
                "better_thermostat %s: MPC raw output (%s) percent=%s delta_T=%s debug=%s",
                name,
                entity,
                _round_for_debug(percent, 2),
                _round_for_debug(delta_t, 3),
                mpc_debug,
            )

    percent = max(0.0, min(100.0, percent))
    prev_percent = state.last_percent

    percent_out, debug, delta_t = _post_process_percent(
        inp=inp,
        params=params,
        state=state,
        now=now,
        raw_percent=percent,
        delta_t=delta_t,
    )

    debug.update(extra_debug)

    flow_cap = params.cap_max_K * (1.0 - (percent_out / 100.0))
    setpoint_eff = None
    if inp.trv_temp_C is not None and delta_t is not None and delta_t <= 0.0:
        setpoint_eff = inp.trv_temp_C - flow_cap

    debug.update(
        {
            "percent_out": percent_out,
            "flow_cap_K": _round_for_debug(flow_cap, 3),
            "setpoint_eff_C": (
                _round_for_debug(setpoint_eff, 3) if setpoint_eff is not None else None
            ),
        }
    )

    summary_delta = delta_t if delta_t is not None else initial_delta_t
    min_eff = state.min_effective_percent
    summary_gain = extra_debug.get("mpc_gain")
    summary_loss = extra_debug.get("mpc_loss")
    summary_horizon = extra_debug.get("mpc_horizon")
    summary_eval = extra_debug.get("mpc_eval_count")
    summary_cost = extra_debug.get("mpc_cost")

    _LOGGER.debug(
        "better_thermostat %s: mpc calibration for %s: e0=%sK gain=%s loss=%s horizon=%s | raw=%s%% out=%s%% min_eff=%s%% last=%s%% dead_hits=%s eval=%s cost=%s flow_cap=%sK",
        name,
        entity,
        _round_for_debug(summary_delta, 3),
        _round_for_debug(summary_gain, 4),
        _round_for_debug(summary_loss, 4),
        summary_horizon,
        _round_for_debug(percent, 2),
        percent_out,
        _round_for_debug(min_eff, 2) if min_eff is not None else None,
        _round_for_debug(prev_percent, 2),
        state.dead_zone_hits,
        summary_eval,
        _round_for_debug(summary_cost, 6),
        _round_for_debug(flow_cap, 3),
    )

    return MpcOutput(
        valve_percent=percent_out,
        flow_cap_K=round(flow_cap, 3),
        setpoint_eff_C=round(setpoint_eff, 3) if setpoint_eff is not None else None,
        debug=debug,
    )


def _resolve_gain_loss(
    state: _MpcState, params: MpcParams
) -> Tuple[float, float, float, float]:
    raw_gain = state.gain_est if state.gain_est is not None else params.mpc_thermal_gain
    raw_loss = state.loss_est if state.loss_est is not None else params.mpc_loss_coeff

    raw_gain = max(params.mpc_gain_min, min(params.mpc_gain_max, raw_gain))
    raw_loss = max(params.mpc_loss_min, min(params.mpc_loss_max, raw_loss))

    step_minutes = MPC_STEP_SECONDS / 60.0
    gain_step = max(0.0, float(raw_gain) * step_minutes)
    loss_step = max(0.0, float(raw_loss) * step_minutes)

    if loss_step > 0.9:
        loss_step = 0.9

    return float(raw_gain), float(raw_loss), gain_step, loss_step


def _compute_hold_percent(state: _MpcState, params: MpcParams) -> Optional[float]:
    _, _, gain_step, loss_step = _resolve_gain_loss(state, params)
    if gain_step <= 0.0 or loss_step <= 0.0:
        return None
    hold_percent = (loss_step / gain_step) * 100.0
    if hold_percent <= 0.0:
        return None
    return min(100.0, hold_percent)


def _compute_predictive_percent(
    inp: MpcInput, params: MpcParams, state: _MpcState, now: float, delta_t: float
) -> Tuple[float, Dict[str, Any]]:
    """Core MPC minimisation routine."""

    # delta_t is target - current (passed in by caller)
    error_now = delta_t

    # step scaling: interpret params as per-minute or per-step? use per-minute * step_minutes
    step_minutes = MPC_STEP_SECONDS / 60.0
    raw_gain, raw_loss, gain_step, loss_step = _resolve_gain_loss(state, params)

    # prepare MPC search
    horizon = MPC_HORIZON_STEPS
    control_pen = max(0.0, float(params.mpc_control_penalty))
    change_pen = max(0.0, float(params.mpc_change_penalty))
    last_percent = state.last_percent if state.last_percent is not None else None

    best_percent = 0.0
    best_cost = None
    eval_count = 0

    # iterate candidates (coarse grid ok). keep candidate in 0..100
    for candidate in range(0, 101, 2):
        u = candidate / 100.0
        future_error = error_now if error_now is not None else 0.0
        cost = 0.0
        # simulate horizon with per-step dynamics:
        for _ in range(horizon):
            # passive drift (cooling toward setpoint) reduces |error|
            future_error = future_error * (1.0 - loss_step)
            # heating effect (reduces error when positive)
            future_error = future_error - gain_step * u
            cost += future_error * future_error
            eval_count += 1

        # scale penalties to [0..1] space
        cost += control_pen * (u * u)  # use normalized u
        if last_percent is not None:
            # normalize change to 0..1
            cost += change_pen * abs((candidate - last_percent) / 100.0)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_percent = float(candidate)

    # save last measurements
    state.last_temp = inp.current_temp_C
    state.last_time = now

    mpc_debug = {
        "mpc_gain": _round_for_debug(raw_gain, 4),
        "mpc_loss": _round_for_debug(raw_loss, 4),
        "mpc_horizon": horizon,
        "mpc_eval_count": eval_count,
        "mpc_step_minutes": _round_for_debug(step_minutes, 3),
        "mpc_gain_step": _round_for_debug(gain_step, 6),
        "mpc_loss_step": _round_for_debug(loss_step, 6),
    }

    if best_cost is not None:
        mpc_debug["mpc_cost"] = _round_for_debug(best_cost, 6)

    if last_percent is not None:
        mpc_debug["mpc_last_percent"] = _round_for_debug(last_percent, 2)

    return best_percent, mpc_debug


def _apply_dead_zone_detection(
    *,
    inp: MpcInput,
    params: MpcParams,
    state: _MpcState,
    now: float,
    percent_out: int,
    delta_t: Optional[float],
    name: str,
    entity: str,
    min_clamp_active: bool,
) -> tuple[int, Optional[float], Optional[float], Dict[str, Any]]:
    """Update dead-zone tracking and min-effective clamps."""

    temp_delta: Optional[float] = None
    time_delta: Optional[float] = None
    dead_debug: Dict[str, Any] = {}

    if inp.trv_temp_C is None:
        state.last_trv_temp = None
        state.last_trv_temp_ts = 0.0
        state.dead_zone_hits = 0
        state.dead_zone_score = 0.0
        return percent_out, temp_delta, time_delta, dead_debug

    if state.last_trv_temp is None or state.last_trv_temp_ts == 0.0:
        state.last_trv_temp = inp.trv_temp_C
        state.last_trv_temp_ts = now
        if inp.current_temp_C is not None:
            state.last_dead_zone_room_temp = inp.current_temp_C
            state.last_dead_zone_room_ts = now
        return percent_out, temp_delta, time_delta, dead_debug

    temp_delta = inp.trv_temp_C - state.last_trv_temp
    time_delta = now - state.last_trv_temp_ts
    eval_after = max(params.deadzone_time_s, 1.0)

    if inp.current_temp_C is not None and state.last_dead_zone_room_temp is None:
        state.last_dead_zone_room_temp = inp.current_temp_C
        state.last_dead_zone_room_ts = now

    if time_delta >= eval_after:
        room_delta = None
        if inp.current_temp_C is not None and inp.trv_temp_C is not None:
            try:
                room_delta = float(inp.trv_temp_C) - float(inp.current_temp_C)
            except (TypeError, ValueError):
                room_delta = None

        room_temp_delta = None
        room_temp_dt = 0.0
        if (
            inp.current_temp_C is not None
            and state.last_dead_zone_room_temp is not None
            and state.last_dead_zone_room_ts > 0.0
        ):
            room_temp_delta = inp.current_temp_C - state.last_dead_zone_room_temp
            room_temp_dt = now - state.last_dead_zone_room_ts
        state.last_dead_zone_room_delta = room_temp_delta

        tol = max(inp.tolerance_K, 0.0)
        needs_heat = delta_t is not None and delta_t > tol
        small_command = percent_out > 0 and (
            percent_out <= params.deadzone_threshold_pct or min_clamp_active
        )
        weak_response = temp_delta is None or temp_delta <= params.deadzone_temp_delta_K
        room_hot_guard = max(params.deadzone_room_delta_guard_K, 0.0)
        trv_not_hot = room_delta is None or room_delta <= room_hot_guard

        room_flat_guard = max(params.deadzone_room_temp_delta_guard_K, 0.0)
        room_flat = room_temp_delta is None or abs(room_temp_delta) <= room_flat_guard

        slope_value = state.ema_slope
        if slope_value is None:
            slope_value = inp.temp_slope_K_per_min
        slope_guard = params.deadzone_room_slope_guard_K_per_min
        slope_ok = slope_value is None or slope_value <= slope_guard

        phase_ready = False
        phase_age = 0.0
        if state.heat_phase_start_ts > 0.0 and state.heat_phase_start_temp is not None:
            phase_age = now - state.heat_phase_start_ts
            phase_ready = phase_age >= max(params.deadzone_phase_min_s, 0.0)

        observation_ready = (
            phase_ready
            and small_command
            and needs_heat
            and weak_response
            and trv_not_hot
            and room_flat
            and slope_ok
        )

        if observation_ready:
            score_inc = max(params.deadzone_score_increment, 0.0)
            state.dead_zone_score = min(
                params.deadzone_score_cap, state.dead_zone_score + score_inc
            )
            state.dead_zone_hits = int(round(state.dead_zone_score))
            _LOGGER.debug(
                "better_thermostat %s: MPC dead-zone observation (%s) hits=%s/%s temp_delta=%s room_delta=%s room_temp_delta=%s slope=%s command=%s%%",
                name,
                entity,
                state.dead_zone_hits,
                params.deadzone_hits_required,
                _round_for_debug(temp_delta, 3),
                _round_for_debug(room_delta, 3),
                _round_for_debug(room_temp_delta, 3),
                _round_for_debug(slope_value, 4),
                percent_out,
            )
            if (
                params.deadzone_hits_required > 0
                and state.dead_zone_score >= params.deadzone_hits_required
            ):
                proposed = percent_out + params.deadzone_raise_pct
                current_min = state.min_effective_percent or 0.0
                state.min_effective_percent = min(100.0, max(current_min, proposed))
                state.dead_zone_score = 0.0
                state.dead_zone_hits = 0
                _LOGGER.debug(
                    "better_thermostat %s: MPC dead-zone raise (%s) proposed=%s new_min=%s",
                    name,
                    entity,
                    _round_for_debug(proposed, 2),
                    _round_for_debug(state.min_effective_percent, 2),
                )
        else:
            prev_hits = state.dead_zone_hits
            heating_detected = False
            decay_reason = None
            if temp_delta is not None and temp_delta > params.deadzone_temp_delta_K:
                heating_detected = True
                decay_reason = "trv_delta"
            elif room_delta is not None and room_delta > params.deadzone_temp_delta_K:
                heating_detected = True
                decay_reason = "room_delta"

            if not heating_detected:
                decay = max(params.deadzone_score_decay, 0.0)
                state.dead_zone_score = max(0.0, state.dead_zone_score - decay)
                state.dead_zone_hits = int(round(state.dead_zone_score))

            if state.min_effective_percent is not None and heating_detected:
                new_min = state.min_effective_percent - params.deadzone_decay_pct
                state.min_effective_percent = new_min if new_min > 0.0 else None
                _LOGGER.debug(
                    "better_thermostat %s: MPC dead-zone decay (%s) reason=%s temp_delta=%s room_delta=%s new_min=%s",
                    name,
                    entity,
                    decay_reason,
                    _round_for_debug(temp_delta, 3),
                    _round_for_debug(room_delta, 3),
                    _round_for_debug(state.min_effective_percent, 2),
                )
                state.dead_zone_score = 0.0
                state.dead_zone_hits = 0
            state.dead_zone_hits = 0
            if prev_hits:
                _LOGGER.debug(
                    "better_thermostat %s: MPC dead-zone reset (%s) prev_hits=%s temp_delta=%s",
                    name,
                    entity,
                    prev_hits,
                    _round_for_debug(temp_delta, 3),
                )

        state.last_trv_temp = inp.trv_temp_C
        state.last_trv_temp_ts = now
        if inp.current_temp_C is not None:
            state.last_dead_zone_room_temp = inp.current_temp_C
            state.last_dead_zone_room_ts = now

        dead_debug.update(
            {
                "dead_zone_score": _round_for_debug(state.dead_zone_score, 2),
                "dead_zone_phase_ready": phase_ready,
                "dead_zone_phase_age_s": _round_for_debug(phase_age, 1),
                "dead_zone_room_temp_delta": _round_for_debug(room_temp_delta, 3),
                "dead_zone_room_temp_dt_s": _round_for_debug(room_temp_dt, 1),
                "dead_zone_room_flat": room_flat,
                "dead_zone_slope": _round_for_debug(slope_value, 4),
                "dead_zone_slope_flat": slope_ok,
                "dead_zone_trv_not_hot": trv_not_hot,
            }
        )

    return percent_out, temp_delta, time_delta, dead_debug


def _post_process_percent(
    inp: MpcInput,
    params: MpcParams,
    state: _MpcState,
    now: float,
    raw_percent: float,
    delta_t: Optional[float],
) -> tuple[int, Dict[str, Any], Optional[float]]:
    """Apply smoothing, hysteresis, dead-zone detection, and produce debug info."""

    smooth = raw_percent
    target_changed = False
    name = inp.bt_name or "BT"
    entity = inp.entity_id or "unknown"
    prev_percent = state.last_percent if state.last_percent is not None else 0.0

    if inp.target_temp_C is not None:
        prev_target = state.last_target_C
        if prev_target is not None:
            try:
                target_changed = (
                    abs(float(inp.target_temp_C) - float(prev_target)) >= 0.1
                )
            except (TypeError, ValueError):
                target_changed = False
        state.last_target_C = inp.target_temp_C

    too_soon = (now - state.last_update_ts) < params.min_update_interval_s
    if target_changed:
        too_soon = False

    if inp.target_temp_C is not None and inp.current_temp_C is not None:
        try:
            if delta_t is None:
                delta_t = inp.target_temp_C - inp.current_temp_C
        except (TypeError, ValueError):
            delta_t = None

    hold_percent: Optional[float] = None
    hold_applied = False
    hold_tol = max(params.hold_tolerance_K, 0.0)
    if (
        hold_tol > 0.0
        and delta_t is not None
        and delta_t <= 0.0
        and abs(delta_t) <= hold_tol
    ):
        hold_percent = _compute_hold_percent(state, params)
        if hold_percent is not None and hold_percent > 0.0:
            last_heat = state.heat_phase_percent
            if last_heat is None and state.last_percent is not None:
                last_heat = max(0.0, state.last_percent)
            if last_heat is not None and last_heat > 0.0:
                hold_percent = min(hold_percent, last_heat)
            if hold_percent > 0.0 and smooth < hold_percent:
                _LOGGER.debug(
                    "better_thermostat %s: MPC hold compensation (%s) delta_T=%s hold=%s raw=%s",
                    name,
                    entity,
                    _round_for_debug(delta_t, 3),
                    _round_for_debug(hold_percent, 2),
                    _round_for_debug(smooth, 2),
                )
                smooth = hold_percent
                hold_applied = True

    min_clamp_allowed = not hold_applied
    min_clamp_used = False
    min_eff = state.min_effective_percent
    if (
        min_clamp_allowed
        and min_eff is not None
        and min_eff > 0.0
        and smooth > 0.0
        and smooth < min_eff
    ):
        smooth = min_eff
        min_clamp_used = True
        _LOGGER.debug(
            "better_thermostat %s: MPC clamp smooth (%s) to min_effective=%s",
            name,
            entity,
            _round_for_debug(min_eff, 2),
        )

    last_percent = state.last_percent
    if last_percent is not None:
        change = abs(smooth - last_percent)
        if (change < params.percent_hysteresis_pts and not target_changed) or too_soon:
            percent_out = int(round(last_percent))
        else:
            percent_out = int(round(smooth))
            state.last_percent = smooth
            state.last_update_ts = now
    else:
        percent_out = int(round(smooth))
        state.last_percent = smooth
        state.last_update_ts = now

    min_eff = state.min_effective_percent
    if (
        min_clamp_allowed
        and min_eff is not None
        and min_eff > 0.0
        and percent_out > 0
        and percent_out < min_eff
    ):
        percent_out = int(round(min_eff))
        state.last_percent = float(percent_out)
        state.last_update_ts = now
        min_clamp_used = True
        _LOGGER.debug(
            "better_thermostat %s: MPC clamp percent_out (%s) to min_effective=%s",
            name,
            entity,
            _round_for_debug(min_eff, 2),
        )

    min_clamp_for_dead_zone = min_clamp_used
    percent_out, temp_delta, time_delta, dead_debug = _apply_dead_zone_detection(
        inp=inp,
        params=params,
        state=state,
        now=now,
        percent_out=percent_out,
        delta_t=delta_t,
        name=name,
        entity=entity,
        min_clamp_active=min_clamp_for_dead_zone,
    )

    min_eff = state.min_effective_percent
    if (
        min_clamp_allowed
        and min_eff is not None
        and min_eff > 0.0
        and percent_out > 0
        and percent_out < min_eff
    ):
        percent_out = int(round(min_eff))
        state.last_percent = float(percent_out)
        state.last_update_ts = now
        min_clamp_used = True
        _LOGGER.debug(
            "better_thermostat %s: MPC clamp percent_out (%s) to min_effective=%s",
            name,
            entity,
            _round_for_debug(min_eff, 2),
        )

    phase_debug = _update_phase_tracking(
        state=state,
        inp=inp,
        params=params,
        now=now,
        prev_percent=float(prev_percent),
        new_percent=float(percent_out),
    )

    trv_room_delta = None
    if inp.trv_temp_C is not None and inp.current_temp_C is not None:
        try:
            trv_room_delta = float(inp.trv_temp_C) - float(inp.current_temp_C)
        except (TypeError, ValueError):
            trv_room_delta = None

    debug: Dict[str, Any] = {
        "raw_percent": _round_for_debug(raw_percent, 2),
        "smooth_percent": _round_for_debug(smooth, 2),
        "too_soon": too_soon,
        "target_changed": target_changed,
        "delta_T": _round_for_debug(delta_t, 3),
        "min_effective_percent": (
            _round_for_debug(state.min_effective_percent, 2)
            if state.min_effective_percent is not None
            else None
        ),
        "dead_zone_hits": state.dead_zone_hits,
        "trv_temp_delta": _round_for_debug(temp_delta, 3),
        "trv_time_delta_s": _round_for_debug(time_delta, 1),
        "trv_room_delta": _round_for_debug(trv_room_delta, 3),
        "min_clamp_active": min_clamp_used,
        "heating_phase_active": state.heat_phase_start_temp is not None,
        "idle_phase_active": state.idle_phase_start_temp is not None,
        "hold_percent": (
            _round_for_debug(hold_percent, 2) if hold_percent is not None else None
        ),
        "hold_active": hold_applied,
    }

    for key, value in dead_debug.items():
        debug[key] = value

    for key, value in phase_debug.items():
        debug[key] = _round_for_debug(value, 4) if isinstance(value, float) else value

    if inp.temp_slope_K_per_min is not None:
        if state.ema_slope is None:
            state.ema_slope = inp.temp_slope_K_per_min
        else:
            state.ema_slope = 0.6 * state.ema_slope + 0.4 * inp.temp_slope_K_per_min
        debug["slope_ema"] = _round_for_debug(state.ema_slope, 4)

    return percent_out, debug, delta_t
