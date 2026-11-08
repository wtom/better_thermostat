"""Experimental slow room calibration, independent of MPC and Home Assistant.

The benchmark and the live dispatcher use this exact controller. Only accepted
baselines and measured holding points survive restarts, never active timers.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import TypedDict


class SlowState(TypedDict, total=False):
    """Validated restart-safe learning, without monotonic timers."""

    baseline: float
    response: dict[str, float | None]
    holding_points: list[dict[str, float]]


@dataclass(frozen=True)
class SlowContext:
    """Measured room inputs; no plant-internal truth or device access."""

    t: float
    target_temperature: float
    room_temperature: float
    outdoor_temperature: float
    last_valve_percent: float = 0.0
    window_open: bool = False
    solar_intensity: float = 0.0
    supply_temperature: float | None = None


@dataclass(frozen=True)
class SlowOutput:
    """Valve recommendation and explanation, not proof of actuation."""

    valve_percent: float
    diagnostics: dict[str, object] = field(default_factory=dict)


class SlowCalibration:
    """Hold a room opening for hours, with separate exceptional recovery."""

    name = "slow"
    family = "valve"

    def __init__(self, hold_s: float = 10800.0, step_percent: float = 3.0) -> None:
        self.hold_s = hold_s
        self.step_percent = step_percent
        self.reset()

    def reset(self, prior: Mapping[str, object] | None = None) -> None:
        """Restore the normal baseline, never a transient recovery command."""
        value = prior.get("baseline") if prior else None
        self.baseline = (
            float(value)
            if isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 100
            else None
        )
        response = prior.get("response") if prior else None
        self._response: dict[str, float | None] = (
            {
                key: value
                for key, value in response.items()
                if value is None
                or isinstance(value, (int, float))
                and math.isfinite(value)
            }
            if isinstance(response, dict)
            else {}
        )
        points = prior.get("holding_points") if prior else None
        self._holding_points: list[dict[str, float]] = (
            [
                dict(point)
                for point in points[-16:]
                if isinstance(point, dict)
                and all(
                    isinstance(point.get(key), (int, float))
                    and math.isfinite(point[key])
                    for key in (
                        "opening",
                        "target_temperature",
                        "outdoor_temperature",
                        "supply_temperature",
                    )
                )
                and 0 <= point["opening"] <= 100
            ]
            if isinstance(points, list)
            else []
        )
        self._supply_samples: deque[tuple[float, float]] = deque()
        self._recent_temperatures: deque[tuple[float, float]] = deque()
        self._recovery_context: SlowContext | None = None
        self._recovery_start: float | None = None
        self._recovery_context_changed = False
        self._start: float | None = None
        self._window_context: SlowContext | None = None
        self._temperature = 0.0
        self._sum_error = 0.0
        self._elapsed = 0.0
        self._previous: SlowContext | None = None
        self._exception_since: float | None = None
        self._exception_direction: str | None = None
        self._recovery: str | None = None
        self._command: float | None = None
        self._last_recovery_step = float("-inf")

    def interrupt(self) -> None:
        """Discard unobserved/externally actuated intervals, retaining learning."""
        self.reset(self.export_state())

    def observe_applied(self, now: float, openings: Sequence[float | None]) -> None:
        """Reject learning from unconfirmed commands, including recovery.

        Allow one minute for the existing write budget and device echo before
        rejecting a recovery intervention. Never cancel its comfort command.
        """
        expected = self._command if self._recovery else self.baseline
        if expected is None or all(
            value is not None and abs(value - expected) <= 1.0 for value in openings
        ):
            return
        if self._recovery:
            if now - self._last_recovery_step >= 60:
                self._recovery_context_changed = True
        else:
            self.interrupt()

    def _begin(self, ctx: SlowContext) -> None:
        self._start = ctx.t
        self._window_context = ctx
        self._temperature = ctx.room_temperature
        self._sum_error = 0.0
        self._elapsed = 0.0

    def _mean_supply(self, now: float, value: float | None) -> float | None:
        """Time-weighted trailing hour, using the last reported value per interval."""
        samples = self._supply_samples
        if value is None:
            samples.clear()
            return None
        samples.append((now, value))
        cutoff = now - 3600.0
        while len(samples) > 1 and samples[1][0] <= cutoff:
            samples.popleft()
        duration = now - max(cutoff, samples[0][0])
        if duration <= 0:
            return value
        return (
            sum(
                (end[0] - max(cutoff, start[0])) * start[1]
                for start, end in zip(samples, list(samples)[1:], strict=False)
            )
            / duration
        )

    def step(self, ctx: SlowContext) -> SlowOutput:
        """Compute a held opening without access to plant-internal truth."""
        if self.baseline is None:
            self.baseline = max(0.0, min(100.0, ctx.last_valve_percent))
        if self._previous is not None and (
            ctx.t < self._previous.t or ctx.t - self._previous.t > 900
        ):
            self.interrupt()
        # Detect real heat loss from the raw reading, never hide it in the mean.
        raw_supply = ctx.supply_temperature
        unavailable = raw_supply is not None and raw_supply <= ctx.room_temperature
        if (
            unavailable
            or ctx.window_open
            or (self._previous and self._previous.window_open)
        ):
            self._supply_samples.clear()
        ctx = replace(ctx, supply_temperature=self._mean_supply(ctx.t, raw_supply))
        error = ctx.target_temperature - ctx.room_temperature
        previous = self._previous
        changed = previous is not None and (
            ctx.target_temperature != previous.target_temperature
            or ctx.window_open != previous.window_open
            or abs(ctx.solar_intensity - previous.solar_intensity) >= 0.2
            or (
                ctx.supply_temperature is not None
                and previous.supply_temperature is not None
                and abs(ctx.supply_temperature - previous.supply_temperature) >= 3.0
            )
        )
        # Compare against the start of the held experiment as well as the
        # preceding sample: small successive changes can otherwise hide a large
        # supply/weather drift and teach the wrong valve response.
        # Recovery resets the normal window every tick. Keep its own anchor so
        # cumulative changes in the heating mean still invalidate learning,
        # while individual burner cycles do not.
        anchor = self._recovery_context if self._recovery else self._window_context
        if anchor is not None:
            changed = changed or (
                abs(ctx.outdoor_temperature - anchor.outdoor_temperature) >= 3.0
                or (ctx.supply_temperature is None)
                != (anchor.supply_temperature is None)
                or (
                    ctx.supply_temperature is not None
                    and anchor.supply_temperature is not None
                    and abs(ctx.supply_temperature - anchor.supply_temperature) >= 3.0
                )
            )
        disturbed = ctx.window_open or unavailable or ctx.solar_intensity >= 0.2
        self._previous = ctx
        # Weather/supply drift invalidates learning, not an active comfort
        # command. Target changes and disturbances still cancel that command.
        cancel_recovery = disturbed or (
            previous is not None
            and ctx.target_temperature != previous.target_temperature
        )
        if changed and self._recovery:
            self._recovery_context_changed = True
        if self._start is None or changed or disturbed:
            self._begin(ctx)
            if cancel_recovery:
                self._exception_since = None
            if cancel_recovery or self._recovery is None:
                self._recovery = None
                self._command = None
        if changed or disturbed:
            self._recent_temperatures.clear()
        self._recent_temperatures.append((ctx.t, ctx.room_temperature))
        while self._recent_temperatures[0][0] < ctx.t - 1800.0:
            self._recent_temperatures.popleft()
        recent_trend = None
        samples = self._recent_temperatures
        if len(samples) >= 3 and ctx.t - samples[0][0] >= 900.0:
            # Measured trend guard, not an identified thermal time constant.
            mean_t = sum(t - ctx.t for t, _ in samples) / len(samples)
            mean_room = sum(room for _, room in samples) / len(samples)
            denominator = sum((t - ctx.t - mean_t) ** 2 for t, _ in samples)
            recent_trend = (
                3600
                * sum((t - ctx.t - mean_t) * (room - mean_room) for t, room in samples)
                / denominator
            )
        # Reuse only measured settled points, never recovery commands or an
        # extrapolated weather model. Bound each context transition to one normal
        # step so a stale point cannot bypass the slow-calibration contract.
        reused = False
        if (previous is None or changed) and not disturbed:
            matches = [
                point
                for point in self._holding_points
                if ctx.supply_temperature is not None
                and abs(point["target_temperature"] - ctx.target_temperature) <= 0.1
                and abs(point["outdoor_temperature"] - ctx.outdoor_temperature) <= 1.0
                and abs(point["supply_temperature"] - ctx.supply_temperature) <= 2.0
            ]
            if matches:
                point = matches[-1]
                delta = point["opening"] - self.baseline
                self.baseline += max(-self.step_percent, min(self.step_percent, delta))
                reused = True
        reason = "reuse_holding_point" if reused else "hold"
        command = self.baseline
        if disturbed:
            # A window closes the simulated actuator. Unavailable heat freezes
            # learning, not the requested opening (hot water may return later).
            command = 0.0 if ctx.window_open or error <= -0.5 else self.baseline
            reason = "disturbance"
        else:
            cold = error >= 0.5
            warm = error <= -0.5 or (
                recent_trend is not None
                and recent_trend >= 0.1
                and error < -0.2
                and error - recent_trend * 0.5 <= -0.5
            )
            direction = "cold" if cold else "warm" if warm else None
            if self._recovery and (
                (self._recovery == "cold" and error <= 0.2)
                or (self._recovery == "warm" and error >= -0.2)
            ):
                # A completed comfort intervention is directional evidence,
                # not a measurement of the steady opening. Move only one normal
                # calibration step; never persist the transient recovery value.
                if not self._recovery_context_changed:
                    delta = (
                        self.step_percent
                        if self._recovery == "cold"
                        else -self.step_percent
                    )
                    self.baseline = max(0.0, min(100.0, self.baseline + delta))
                    command = self.baseline
                    assert self._recovery_start is not None
                    self._response["last_recovery_s"] = ctx.t - self._recovery_start
                self._recovery = None
                self._command = None
                self._begin(ctx)
            if direction is None:
                self._exception_since = None
            elif (
                direction != self._exception_direction or self._exception_since is None
            ):
                self._exception_since = ctx.t
            self._exception_direction = direction
            if (
                direction
                and self._exception_since is not None
                and (ctx.t - self._exception_since >= 600.0)
            ):
                if self._recovery != direction:
                    self._last_recovery_step = float("-inf")
                    self._recovery_start = ctx.t
                    self._recovery_context = ctx
                    self._recovery_context_changed = False
                self._recovery = direction
            if self._recovery:
                approaching = (
                    self._recovery == "cold"
                    and self._command is not None
                    and recent_trend is not None
                    and recent_trend > 0.0
                    and error - recent_trend <= 0.2
                )
                if (
                    not approaching
                    and direction == self._recovery
                    and ctx.t - self._last_recovery_step >= 900.0
                ):
                    opening = (
                        self._command if self._command is not None else self.baseline
                    )
                    self._command = (
                        min(100.0, opening + 10.0) if self._recovery == "cold" else 0.0
                    )
                    self._last_recovery_step = ctx.t
                command = self._command
                reason = (
                    "recovery_wait_response"
                    if approaching
                    else f"recovery_{self._recovery}"
                )
                self._begin(ctx)  # recovery is never baseline-learning evidence
            else:
                assert self._start is not None
                # Do not carry discarded pre-reset time into this window.
                dt = max(0.0, ctx.t - max(previous.t, self._start)) if previous else 0.0
                self._sum_error += error * dt
                self._elapsed += dt
                age = ctx.t - self._start
                if age >= self.hold_s and self._elapsed > 0.0:
                    mean_error = self._sum_error / self._elapsed
                    trend = (ctx.room_temperature - self._temperature) / (age / 3600)
                    self._response = {
                        **self._response,
                        "observed_opening": self.baseline,
                        "mean_error": mean_error,
                        "trend_k_per_h": trend,
                        "window_s": age,
                        "supply_temperature": ctx.supply_temperature,
                        "outdoor_temperature": ctx.outdoor_temperature,
                        "target_temperature": ctx.target_temperature,
                    }
                    # Correcting windows describe response, not equilibrium.
                    if (
                        abs(mean_error) <= 0.2
                        and abs(error) <= 0.2
                        and abs(trend) <= 0.05
                        and ctx.supply_temperature is not None
                    ):
                        self._holding_points.append(
                            {
                                "opening": self.baseline,
                                "target_temperature": ctx.target_temperature,
                                "outdoor_temperature": ctx.outdoor_temperature,
                                "supply_temperature": ctx.supply_temperature,
                            }
                        )
                        self._holding_points = self._holding_points[-16:]
                    # Project one hour ahead: do not add heat to an adequately
                    # warming room, nor remove it while it is cooling to target.
                    projected_error = error - trend
                    if mean_error > 0.2 and projected_error > 0.2:
                        self.baseline = min(100.0, self.baseline + self.step_percent)
                        reason = "calibrate_open"
                    elif mean_error < -0.2 and projected_error < -0.2:
                        self.baseline = max(0.0, self.baseline - self.step_percent)
                        reason = "calibrate_close"
                    else:
                        reason = "accept"
                    command = self.baseline
                    self._begin(ctx)
        assert command is not None
        return SlowOutput(
            valve_percent=command,
            diagnostics={
                "reason": reason,
                "baseline": self.baseline,
                "supply_mean_temperature": ctx.supply_temperature,
                "recent_trend_k_per_h": recent_trend,
                "response": dict(self._response),
                "holding_point_count": len(self._holding_points),
            },
        )

    def export_state(self) -> SlowState:
        """Persist only the normal room opening, not temporary recovery."""
        if self.baseline is None:
            return {}
        state: SlowState = {"baseline": self.baseline}
        if self._response:
            state["response"] = dict(self._response)
        if self._holding_points:
            state["holding_points"] = [dict(point) for point in self._holding_points]
        return state
