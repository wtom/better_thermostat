"""Model Predictive Controller (MPC) for TRV heating valves.

This module implements a senior-level, physically grounded MPC that predicts
room temperature forward in discrete steps, evaluates valve opening candidates,
applies realistic rate/hold/deadzone constraints, and adapts model gain/loss
online.

Units and conventions:
- Temperatures: degrees Celsius (°C)
- Time: seconds; conversions to minutes are explicit
- Valve opening: percent in [0, 100]; internal `u` in [0.0, 1.0]
- Gains/Losses: °C per minute; converted to per-step using MPC step seconds

Core physical model per step:
    T_next = T + lag_alpha * ((T + heating - passive_loss) - T)
Where:
    heating      = gain_step * u         (°C per step)
    passive_loss = loss_step             (°C per step)
    gain_step    = gain_C_per_min * (step_s / 60)
    loss_step    = loss_C_per_min * (step_s / 60)
    lag_alpha    = 1 - exp(-step_s / tau)

Search strategy:
- Coarse grid: 0, 10, 20, …, 100
- Fine  grid: ±10% around best coarse in 2% steps

Constraints:
- du_max      (rate-of-change constraint relative to last_u)
- hold_pct    (minimum heating needed to avoid falling short)
- deadzone    (robust minimum opening raised/decayed by observations)

Learning (adaptation):
- gain/loss learned via EMA with bounds, using observed error changes
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


# ---------- Configuration constants (no magic numbers in code paths) ----------
MPC_STEP_SECONDS = 300.0  # step_s
MPC_HORIZON_STEPS = 12  # prediction steps
_STATE_FILE_PATH = ".storage/better_thermostat_mpc_states"
_STATE_AUTO_SAVE_INTERVAL_S = 60.0


# ---------- Dataclasses for inputs, params and internal state ----------


@dataclass
class MpcParams:
    """Tunables for the MPC and adaptation.

    All values are interpreted explicitly in the algorithm; keep units consistent.
    """

    # Physical model
    # New fields (this implementation); kept alongside legacy to avoid breakage
    gain_C_per_min: float = 0.012  # °C/min at u=100%
    loss_C_per_min: float = 0.005  # °C/min passive cooling
    tau_seconds: float = 1800.0  # dominant lag time constant (s)

    # Cost weights
    control_penalty: float = 0.0  # penalize large u
    change_penalty: float = 0.01  # penalize |u - last_u|

    # Rate limit and hysteresis
    du_max_pct: float = 20.0  # max change per decision (%)
    percent_hysteresis_pts: float = 0.5  # to avoid chattering
    min_update_interval_s: float = 60.0  # minimum time between updates

    # Hold computation
    hold_tolerance_K: float = 0.2  # if predicted within tolerance, no hold

    # Deadzone detector
    deadzone_initial_min_pct: float = 8.0
    deadzone_max_pct: float = 25.0
    deadzone_raise_pct: float = 2.0
    deadzone_decay_pct: float = 1.0
    deadzone_hits_required: int = 3
    deadzone_time_s: float = 180.0
    deadzone_temp_delta_K: float = 0.2  # |ΔT| threshold for a deadzone hit
    deadzone_delta_u_pct: float = 2.0  # |Δu| threshold for a deadzone hit

    # Adaptation
    adapt_alpha: float = 0.1  # EMA alpha for gain/loss updates
    gain_min_C_per_min: float = 0.0006  # bounds for gain estimate
    gain_max_C_per_min: float = 0.3
    loss_min_C_per_min: float = 0.0001  # bounds for loss estimate
    loss_max_C_per_min: float = 0.1


@dataclass
class MpcInput:
    # Legacy field names
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
    # New aliases for clarity (not required by legacy callers)

    @property
    def setpoint_T(self) -> Optional[float]:
        return self.target_temp_C

    @property
    def current_T(self) -> Optional[float]:
        return self.current_temp_C

    last_u_pct: Optional[float] = None


@dataclass
class MpcState:
    # Adapted estimates (°C/min)
    gain_est_C_per_min: Optional[float] = None
    loss_est_C_per_min: Optional[float] = None

    # Deadzone tracking
    deadzone_min_pct: float = 8.0
    deadzone_counter: int = 0
    last_decision_ts: float = 0.0
    last_temperature_ts: float = 0.0
    last_temperature: Optional[float] = None
    last_u_pct: Optional[float] = None


# ---------- Persistent state registry ----------

_MPC_STATE_REGISTRY: Dict[str, MpcState] = {}
_MPC_META: Dict[str, float | bool] = {"loaded": False, "last_save": 0.0}


def get_state(key: str) -> MpcState:
    """Get or create the MPC state for a given key from the registry."""
    if not _MPC_META["loaded"]:
        # Attempt lazy load once from default path
        load_state_from_file(_STATE_FILE_PATH)
        _MPC_META["loaded"] = True
    state = _MPC_STATE_REGISTRY.get(key)
    if state is None:
        state = MpcState()
        _MPC_STATE_REGISTRY[key] = state
    return state


def set_state(key: str, state: MpcState) -> None:
    """Set/replace the MPC state for a given key in the registry."""
    _MPC_STATE_REGISTRY[key] = state


def list_state_keys(prefix: Optional[str] = None) -> List[str]:
    """List registered state keys, optionally filtered by prefix."""
    if prefix is None:
        return list(_MPC_STATE_REGISTRY.keys())
    return [k for k in _MPC_STATE_REGISTRY.keys() if k.startswith(prefix)]


def remove_state(key: str) -> bool:
    """Remove a state entry. Returns True if it existed."""
    return _MPC_STATE_REGISTRY.pop(key, None) is not None


def export_state_map(prefix: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Export a serializable mapping of states for persistence."""
    result: Dict[str, Dict[str, Any]] = {}
    for key, st in _MPC_STATE_REGISTRY.items():
        if prefix is not None and not key.startswith(prefix):
            continue
        payload = {
            "gain_est_C_per_min": st.gain_est_C_per_min,
            "loss_est_C_per_min": st.loss_est_C_per_min,
            "deadzone_min_pct": st.deadzone_min_pct,
            "deadzone_counter": st.deadzone_counter,
            "last_decision_ts": st.last_decision_ts,
            "last_temperature_ts": st.last_temperature_ts,
            "last_temperature": st.last_temperature,
            "last_u_pct": st.last_u_pct,
        }
        result[key] = payload
    return result


def import_state_map(state_map: Dict[str, Dict[str, Any]]) -> None:
    """Import states from a previously exported mapping."""
    for key, payload in state_map.items():
        st = _MPC_STATE_REGISTRY.get(key)
        if st is None:
            st = MpcState()
            _MPC_STATE_REGISTRY[key] = st
        st.gain_est_C_per_min = _coerce_float(payload.get("gain_est_C_per_min"))
        st.loss_est_C_per_min = _coerce_float(payload.get("loss_est_C_per_min"))
        st.deadzone_min_pct = float(
            payload.get("deadzone_min_pct", st.deadzone_min_pct)
        )
        st.deadzone_counter = int(payload.get("deadzone_counter", st.deadzone_counter))
        st.last_decision_ts = float(
            payload.get("last_decision_ts", st.last_decision_ts)
        )
        st.last_temperature_ts = float(
            payload.get("last_temperature_ts", st.last_temperature_ts)
        )
        st.last_temperature = _coerce_float(payload.get("last_temperature"))
        st.last_u_pct = _coerce_float(payload.get("last_u_pct"))


def _coerce_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def save_state_to_file(file_path: str, prefix: Optional[str] = None) -> None:
    """Persist the registry to a JSON file.

    The caller is responsible for choosing a stable file path (e.g. within
    Home Assistant's config directory)."""
    import json

    data = export_state_map(prefix)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def load_state_from_file(file_path: str) -> None:
    """Load the registry from a JSON file if it exists."""
    import json

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            import_state_map(data)
    except FileNotFoundError:
        return
    except (OSError, ValueError, TypeError):
        # Robust against corrupted or unreadable files: ignore and continue
        return


# ---------- Utility helpers ----------


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def pct_to_u(percent: float) -> float:
    return clamp(percent / 100.0, 0.0, 1.0)


def u_to_pct(u: float) -> float:
    return clamp(u * 100.0, 0.0, 100.0)


# ---------- Core physical step prediction ----------


def predict_step(
    T: float,
    u_pct: float,
    gain_C_per_min: float,
    loss_C_per_min: float,
    tau_seconds: float,
    step_s: float,
) -> float:
    """One-step temperature prediction using lagged physical model.

    T_next = T + lag_alpha * ((T + heating - passive_loss) - T)
    heating = gain_step * u
    passive_loss = loss_step
    """
    u = pct_to_u(u_pct)
    gain_step = gain_C_per_min * (step_s / 60.0)
    loss_step = loss_C_per_min * (step_s / 60.0)
    lag_alpha = 1.0 - math.exp(-step_s / max(tau_seconds, 1e-6))
    heating = gain_step * u
    passive_loss = loss_step
    T_pred_target = T + heating - passive_loss
    return T + lag_alpha * (T_pred_target - T)


# ---------- MPC simulation over horizon ----------


def simulate_mpc(
    current_T: float,
    setpoint_T: float,
    last_u_pct: Optional[float],
    gain_C_per_min: float,
    loss_C_per_min: float,
    tau_seconds: float,
    step_s: float,
    horizon_steps: int,
    control_penalty: float,
    change_penalty: float,
    candidate_u_pct: float,
) -> Tuple[float, List[float]]:
    """Simulate forward for a candidate and compute cost.

    Returns (cost, errors_per_step).
    cost = Σ(e^2) + control_pen * u^2 + change_pen * |u - last_u|
    """
    T = current_T
    errors: List[float] = []
    for _ in range(horizon_steps):
        T = predict_step(
            T, candidate_u_pct, gain_C_per_min, loss_C_per_min, tau_seconds, step_s
        )
        e = setpoint_T - T
        errors.append(e)
    u = pct_to_u(candidate_u_pct)
    last_u = pct_to_u(last_u_pct or 0.0)
    cost = (
        sum(e * e for e in errors)
        + control_penalty * (u * u)
        + change_penalty * abs(u - last_u)
    )
    return cost, errors


# ---------- Hold computation ----------


def compute_hold(
    current_T: float,
    setpoint_T: float,
    gain_C_per_min: float,
    loss_C_per_min: float,
    tau_seconds: float,
    step_s: float,
    hold_tolerance_K: float,
) -> float:
    """Compute minimum opening to avoid dropping below setpoint.

    - Predict `T_noheat` with u=0 one step ahead.
    - If `T_noheat >= setpoint - hold_tol`: return 0.
    - Else: required heating per step = deficit / lag_alpha
      and hold_pct = required_heating / gain_step.
    """
    T_noheat = predict_step(
        current_T, 0.0, gain_C_per_min, loss_C_per_min, tau_seconds, step_s
    )
    if T_noheat >= (setpoint_T - hold_tolerance_K):
        return 0.0

    gain_step = gain_C_per_min * (step_s / 60.0)
    lag_alpha = 1.0 - math.exp(-step_s / max(tau_seconds, 1e-6))
    deficit = setpoint_T - T_noheat
    required_heating = deficit / max(lag_alpha, 1e-6)
    hold_u = required_heating / max(gain_step, 1e-9)
    return u_to_pct(hold_u)


# ---------- Deadzone detection & adjustment ----------


def detect_deadzone(
    state: MpcState,
    params: MpcParams,
    now_s: float,
    current_T: float,
    last_T: Optional[float],
    u_pct: float,
    last_u_pct: Optional[float],
) -> Tuple[float, int, Dict[str, Any]]:
    """Update deadzone_min_pct based on weak response observations.

    Hit conditions:
    - u < deadzone_min_pct
    - |ΔT| < deadzone_temp_delta_K
    - |Δu| < deadzone_delta_u_pct
    - elapsed time > deadzone_time_s
    """
    debug: Dict[str, Any] = {}

    elapsed = (
        now_s - state.last_temperature_ts if state.last_temperature_ts > 0 else 0.0
    )
    delta_T = None
    if last_T is not None:
        delta_T = abs(current_T - last_T)
    delta_u = None
    if last_u_pct is not None:
        delta_u = abs(u_pct - last_u_pct)

    hit = False
    if (
        u_pct < state.deadzone_min_pct
        and delta_T is not None
        and delta_T < params.deadzone_temp_delta_K
        and (delta_u is None or delta_u < params.deadzone_delta_u_pct)
        and elapsed >= params.deadzone_time_s
    ):
        state.deadzone_counter += 1
        hit = True
    else:
        # decay when many observations without hits
        if state.deadzone_counter > 0 and elapsed >= params.deadzone_time_s:
            state.deadzone_counter = max(0, state.deadzone_counter - 1)

    if state.deadzone_counter >= params.deadzone_hits_required:
        state.deadzone_min_pct = clamp(
            state.deadzone_min_pct + params.deadzone_raise_pct,
            0.0,
            params.deadzone_max_pct,
        )
        state.deadzone_counter = 0
    else:
        # gentle decay towards 0
        if elapsed >= params.deadzone_time_s and not hit:
            state.deadzone_min_pct = clamp(
                state.deadzone_min_pct - params.deadzone_decay_pct,
                0.0,
                params.deadzone_max_pct,
            )

    debug.update(
        {
            "deadzone_min": state.deadzone_min_pct,
            "deadzone_counter": state.deadzone_counter,
            "deadzone_elapsed_s": elapsed,
            "deadzone_delta_T": delta_T,
            "deadzone_delta_u": delta_u,
            "deadzone_hit": hit,
        }
    )
    return state.deadzone_min_pct, state.deadzone_counter, debug


# ---------- Adaptation (EMA updates) ----------


def adapt_gain_loss(
    state: MpcState,
    params: MpcParams,
    current_T: float,
    setpoint_T: float,
    last_T: Optional[float],
) -> Dict[str, Any]:
    """Update gain/loss estimates using EMA based on observed error change.

    Heuristic but stable approach:
    - If u_last > 0 and error decreased, blend gain upward.
    - If u_last == 0 and error drifted, blend loss accordingly.
    Bounds are enforced.
    """
    debug: Dict[str, Any] = {}
    alpha = clamp(params.adapt_alpha, 0.0, 1.0)

    # Initialize estimates if absent
    if state.gain_est_C_per_min is None:
        state.gain_est_C_per_min = params.gain_C_per_min
    if state.loss_est_C_per_min is None:
        state.loss_est_C_per_min = params.loss_C_per_min

    e_now = setpoint_T - current_T
    e_prev = None
    if last_T is not None:
        e_prev = setpoint_T - last_T

    # Learn gain from heating efficacy when last_u > 0
    if state.last_u_pct is not None and state.last_u_pct > 0 and e_prev is not None:
        if abs(e_prev) > 0 and abs(e_now) < abs(e_prev):
            # observed improvement -> candidate gain higher
            # proportional candidate based on fractional error reduction
            reduction = clamp(
                (abs(e_prev) - abs(e_now)) / max(abs(e_prev), 1e-6), 0.0, 1.0
            )
            candidate_gain = state.gain_est_C_per_min * (1.0 + 0.5 * reduction)
            blended_gain = (
                1.0 - alpha
            ) * state.gain_est_C_per_min + alpha * candidate_gain
            state.gain_est_C_per_min = clamp(
                blended_gain, params.gain_min_C_per_min, params.gain_max_C_per_min
            )

    # Learn loss from passive drift when last_u == 0
    if state.last_u_pct is not None and state.last_u_pct == 0 and e_prev is not None:
        drift = abs(e_now) - abs(e_prev)
        if drift > 0:
            # temperature moved away from setpoint without heating -> increase loss
            candidate_loss = state.loss_est_C_per_min * (
                1.0 + 0.5 * clamp(drift / max(abs(e_prev), 1e-6), 0.0, 1.0)
            )
            blended_loss = (
                1.0 - alpha
            ) * state.loss_est_C_per_min + alpha * candidate_loss
            state.loss_est_C_per_min = clamp(
                blended_loss, params.loss_min_C_per_min, params.loss_max_C_per_min
            )

    debug.update(
        {
            "gain_C_per_min": state.gain_est_C_per_min,
            "loss_C_per_min": state.loss_est_C_per_min,
        }
    )
    return debug


# ---------- Main percent computation (compat signature) ----------


def compute_mpc_percent(
    inp: MpcInput, params: MpcParams, state: Optional[MpcState], now_s: float
) -> Tuple[int, Dict[str, Any]]:
    """Compute valve percent and debug info.

    Maintains compatibility: returns (percent, debug_dict).
    """
    debug: Dict[str, Any] = {}

    # Resolve or allocate state from registry when not provided
    if state is None:
        state = get_state(inp.key)

    # Validate inputs
    if not inp.heating_allowed or inp.window_open:
        percent_out = 0
        debug.update(
            {"heating_allowed": inp.heating_allowed, "window_open": inp.window_open}
        )
        # Update state markers
        state.last_u_pct = float(percent_out)
        state.last_decision_ts = now_s
        state.last_temperature_ts = now_s
        state.last_temperature = inp.current_temp_C
        return percent_out, debug

    if inp.target_temp_C is None or inp.current_temp_C is None:
        # No temperatures -> keep last or 0
        percent_out = int(round(state.last_u_pct or 0.0))
        debug.update({"missing_temps": True})
        state.last_u_pct = float(percent_out)
        state.last_decision_ts = now_s
        return percent_out, debug

    setpoint_T = float(inp.target_temp_C)
    current_T = float(inp.current_temp_C)
    last_u_pct = float(state.last_u_pct or 0.0)

    # Initialize adaptation estimates if absent
    if state.gain_est_C_per_min is None:
        state.gain_est_C_per_min = params.gain_C_per_min
    if state.loss_est_C_per_min is None:
        state.loss_est_C_per_min = params.loss_C_per_min

    # Potentially adapt model from last observation
    debug.update(
        adapt_gain_loss(state, params, current_T, setpoint_T, state.last_temperature)
    )

    # Effective model parameters for this decision
    # Safe param getters to avoid static analysis false negatives
    def getp(name: str, default: float) -> float:
        return float(getattr(params, name, default))

    gain_C_per_min = float(state.gain_est_C_per_min or getp("gain_C_per_min", 0.012))
    loss_C_per_min = float(state.loss_est_C_per_min or getp("loss_C_per_min", 0.005))
    tau_seconds = getp("tau_seconds", 1800.0)
    step_s = float(MPC_STEP_SECONDS)
    horizon = int(MPC_HORIZON_STEPS)
    control_pen = getp("control_penalty", 0.0)
    change_pen = getp("change_penalty", 0.01)

    # Candidate selection: coarse
    coarse_candidates = list(range(0, 101, 10))
    best_u_coarse = 0.0
    best_cost_coarse = None
    for cand in coarse_candidates:
        cost, _ = simulate_mpc(
            current_T,
            setpoint_T,
            last_u_pct,
            gain_C_per_min,
            loss_C_per_min,
            tau_seconds,
            step_s,
            horizon,
            control_pen,
            change_pen,
            float(cand),
        )
        if best_cost_coarse is None or cost < best_cost_coarse:
            best_cost_coarse = cost
            best_u_coarse = float(cand)

    # Candidate selection: fine around coarse
    lo = int(clamp(best_u_coarse - 10, 0, 100))
    hi = int(clamp(best_u_coarse + 10, 0, 100))
    fine_candidates = list(range(lo, hi + 1, 2))
    best_u_fine = best_u_coarse
    best_cost_fine = best_cost_coarse
    # We don't expose per-step errors in debug here; keep local only
    for cand in fine_candidates:
        cost, _errors = simulate_mpc(
            current_T,
            setpoint_T,
            last_u_pct,
            gain_C_per_min,
            loss_C_per_min,
            tau_seconds,
            step_s,
            horizon,
            control_pen,
            change_pen,
            float(cand),
        )
        if best_cost_fine is None or cost < best_cost_fine:
            best_cost_fine = cost
            best_u_fine = float(cand)

    # Rate limit (du_max)
    du_max = clamp(getp("du_max_pct", 20.0), 0.0, 100.0)
    u_after_du = clamp(best_u_fine, last_u_pct - du_max, last_u_pct + du_max)

    # Hold level
    hold_pct = compute_hold(
        current_T,
        setpoint_T,
        gain_C_per_min,
        loss_C_per_min,
        tau_seconds,
        step_s,
        getp("hold_tolerance_K", 0.2),
    )
    u_after_hold = max(u_after_du, hold_pct)

    # Deadzone
    dz_min_pct, dz_counter, dz_debug = detect_deadzone(
        state,
        params,
        now_s,
        current_T,
        state.last_temperature,
        u_after_hold,
        last_u_pct,
    )
    u_after_deadzone = max(u_after_hold, dz_min_pct)

    # Hysteresis / update interval
    too_soon = (now_s - state.last_decision_ts) < getp("min_update_interval_s", 60.0)
    target_changed = False
    # We don't store previous setpoint in this minimal module, keep flag false
    if (
        too_soon
        and abs(u_after_deadzone - (state.last_u_pct or 0.0))
        < params.percent_hysteresis_pts
    ):
        percent_out = int(round(state.last_u_pct or 0.0))
    else:
        percent_out = int(round(u_after_deadzone))

    # Update state snapshots
    state.last_u_pct = float(percent_out)
    state.last_decision_ts = now_s
    state.last_temperature_ts = now_s
    state.last_temperature = current_T

    # Auto-save registry periodically to default path
    if (now_s - float(_MPC_META["last_save"])) >= _STATE_AUTO_SAVE_INTERVAL_S:
        try:
            save_state_to_file(_STATE_FILE_PATH)
            _MPC_META["last_save"] = now_s
        except (OSError, ValueError, TypeError):
            # Non-fatal: skip save if filesystem not ready
            pass

    # Debug enrichment (Level B ~60 keys consolidated)
    gain_step_C = gain_C_per_min * (step_s / 60.0)
    loss_step_C = loss_C_per_min * (step_s / 60.0)
    lag_alpha = 1.0 - math.exp(-step_s / max(tau_seconds, 1e-6))
    debug.update(
        {
            # Model parameters
            "gain_C_per_min": gain_C_per_min,
            "loss_C_per_min": loss_C_per_min,
            "gain_step_C": gain_step_C,
            "loss_step_C": loss_step_C,
            "lag_alpha": lag_alpha,
            # State
            "current_T": current_T,
            "setpoint_T": setpoint_T,
            "error": setpoint_T - current_T,
            "last_u": last_u_pct,
            "deadzone_min": dz_min_pct,
            "deadzone_counter": dz_counter,
            "hold_pct": hold_pct,
            # Candidate evaluation
            "coarse_candidates": coarse_candidates,
            "fine_candidates": fine_candidates,
            "best_u_coarse": best_u_coarse,
            "best_u_fine": best_u_fine,
            "cost_coarse": best_cost_coarse,
            "cost_fine": best_cost_fine,
            # Final decisions
            "u_final": percent_out,
            "u_after_du_max": u_after_du,
            "u_after_hold": u_after_hold,
            "u_after_deadzone": u_after_deadzone,
            "mpc_cost_final": best_cost_fine,
            # Control flow
            "too_soon": too_soon,
            "target_changed": target_changed,
        }
    )

    # Merge deadzone debug
    debug.update(dz_debug)

    return percent_out, debug


# Optional convenience: compute full output including a simple debug wrapper
def compute_mpc_output(
    inp: MpcInput, params: MpcParams, state: MpcState, now_s: float
) -> Tuple[int, Dict[str, Any]]:
    """Convenience wrapper returning (valve_percent, debug_dict)."""
    return compute_mpc_percent(inp, params, state, now_s)
