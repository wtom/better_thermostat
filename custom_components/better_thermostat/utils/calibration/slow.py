"""One slow room feedback loop, shared by Home Assistant and the benchmark.

The baseline is a working opening, not a claim of thermal equilibrium. Only
observed settled windows are holding points. Neither timers nor trends survive
interruptions; normal boiler cycles do not interrupt room feedback.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import TypedDict


class SlowState(TypedDict, total=False):
    """Restart-safe working opening and observations, without monotonic timers."""

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
    """Observe one working opening, then open, hold or close it every 30 minutes."""

    name = "slow"
    family = "valve"

    def __init__(self, hold_s: float = 1800.0, step_percent: float = 3.0) -> None:
        self.hold_s = hold_s
        self.step_percent = step_percent
        self.reset()

    def reset(self, prior: Mapping[str, object] | None = None) -> None:
        """Restore the working opening on the same integer grid as the writer."""
        value = prior.get("baseline") if prior else None
        self.baseline = (
            float(round(value))
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
        self._start: float | None = None
        self._window_context: SlowContext | None = None
        self._context_stable = True
        self._sum_error = 0.0
        self._elapsed = 0.0
        self._previous: SlowContext | None = None
        self._feedback_confirmed = True
        self._reevaluate = True
        self._search_step = self.step_percent
        self._direction = 0
        self._reset_reason = "startup"

    def interrupt(self) -> None:
        """Discard interrupted evidence, retaining the working opening and history."""
        confirmed = self._feedback_confirmed
        self.reset(self.export_state())
        # An observation gap is not proof that the device accepted the last write.
        self._feedback_confirmed = confirmed
        self._reset_reason = "interruption"

    def observe_applied(self, now: float, openings: Sequence[float | None]) -> None:
        """Do not advance the search while a command is still unconfirmed.

        Feedback never reapplies a saved hint or re-arms an immediate correction.
        Once the command is confirmed, observe its effect for a fresh interval.
        """
        confirmed = (
            self.baseline is None
            or bool(openings)
            and all(
                value is not None and abs(value - self.baseline) < 0.5
                for value in openings
            )
        )
        if not confirmed or not self._feedback_confirmed:
            self._start = None
            self._sum_error = self._elapsed = 0.0
            self._recent_temperatures.clear()
            self._reset_reason = (
                "feedback_wait" if not confirmed else "feedback_confirmed"
            )
        self._feedback_confirmed = confirmed

    def _begin(self, ctx: SlowContext) -> None:
        self._start = ctx.t
        self._window_context = ctx
        self._context_stable = True
        self._sum_error = self._elapsed = 0.0

    def _mean_supply(self, now: float, value: float | None) -> float | None:
        """Time-weighted heating mean for diagnostics/holding evidence, not a timer gate."""
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
        """One decision path for startup/events and periodic room corrections."""
        if self.baseline is None:
            self.baseline = float(round(max(0.0, min(100.0, ctx.last_valve_percent))))
        if self._previous is not None and (
            ctx.t < self._previous.t or ctx.t - self._previous.t > 900
        ):
            self.interrupt()
            self._reset_reason = "measurement_gap"
        previous = self._previous
        event = previous is not None and (
            ctx.target_temperature != previous.target_temperature
            or ctx.window_open != previous.window_open
        )
        unavailable = (
            ctx.supply_temperature is not None
            and ctx.supply_temperature <= ctx.room_temperature
        )
        disturbed = ctx.window_open or unavailable
        if event or disturbed:
            self._start = None
            self._recent_temperatures.clear()
            self._reevaluate = True
            self._direction = 0
            self._search_step = self.step_percent
            self._reset_reason = (
                "window_open"
                if ctx.window_open
                else "heat_unavailable"
                if unavailable
                else "target_change"
                if previous and ctx.target_temperature != previous.target_temperature
                else "window_closed"
            )
        if disturbed or (previous and previous.window_open):
            self._supply_samples.clear()
        ctx = replace(
            ctx, supply_temperature=self._mean_supply(ctx.t, ctx.supply_temperature)
        )
        self._previous = ctx
        if self._start is None:
            self._begin(ctx)
        anchor = self._window_context
        if anchor is not None:
            self._context_stable = self._context_stable and (
                abs(ctx.outdoor_temperature - anchor.outdoor_temperature) < 3.0
                and (ctx.supply_temperature is None)
                == (anchor.supply_temperature is None)
                and (
                    ctx.supply_temperature is None
                    or anchor.supply_temperature is None
                    or abs(ctx.supply_temperature - anchor.supply_temperature) < 3.0
                )
            )
        error = ctx.target_temperature - ctx.room_temperature
        samples = self._recent_temperatures
        if not disturbed and self._feedback_confirmed:
            samples.append((ctx.t, ctx.room_temperature))
            while samples[0][0] < ctx.t - min(self.hold_s, 1800.0):
                samples.popleft()
        recent_trend = None
        if len(samples) >= 3 and ctx.t - samples[0][0] >= 900.0:
            mean_t = sum(t - ctx.t for t, _ in samples) / len(samples)
            mean_room = sum(room for _, room in samples) / len(samples)
            denominator = sum((t - ctx.t - mean_t) ** 2 for t, _ in samples)
            recent_trend = (
                3600
                * sum((t - ctx.t - mean_t) * (room - mean_room) for t, room in samples)
                / denominator
            )
        assert self._start is not None
        age = ctx.t - self._start
        command = self.baseline
        reason = "hold"
        if disturbed:
            command = 0.0 if ctx.window_open or error < -0.05 else self.baseline
            reason = "disturbance"
        elif not self._feedback_confirmed:
            # Reissue the working opening, never staircase on a missing echo.
            # A new lower target must not reopen a valve already closed by OFF.
            if self._reevaluate and error < -0.05:
                self.baseline = command = 0.0
                self._reevaluate = False
            reason = "wait_feedback"
        else:
            dt = max(0.0, ctx.t - max(previous.t, self._start)) if previous else 0.0
            self._sum_error += error * dt
            self._elapsed += dt
            if self._reevaluate or age >= self.hold_s:
                immediate = self._reevaluate
                self._reevaluate = False
                mean_error = self._sum_error / self._elapsed if self._elapsed else error
                # Only forecast the next control interval. No learned time constant
                # is assumed; a measured trend prevents adding heat already on its way.
                projected = (
                    error - (recent_trend or 0.0) * min(self.hold_s, 1800.0) / 3600
                )
                direction = 1 if projected > 0.1 else -1 if projected < -0.1 else 0
                correction = 0.0
                if direction:
                    if self._direction and direction != self._direction:
                        self._search_step = max(1.0, self._search_step / 2)
                    correction = (
                        1.0 if abs(error) <= 0.2 else max(1.0, round(self._search_step))
                    )
                    self._direction = direction
                if not immediate:
                    self._response = {
                        "observed_opening": self.baseline,
                        "mean_error": mean_error,
                        "trend_k_per_h": recent_trend,
                        "window_s": age,
                        "supply_temperature": ctx.supply_temperature,
                        "outdoor_temperature": ctx.outdoor_temperature,
                        "target_temperature": ctx.target_temperature,
                        "calibration_step_percent": correction,
                    }
                    if (
                        direction == 0
                        and self._context_stable
                        and abs(mean_error) <= 0.05
                        and abs(error) <= 0.05
                        and recent_trend is not None
                        and abs(recent_trend) <= 0.05
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
                self.baseline = max(
                    0.0, min(100.0, self.baseline + direction * correction)
                )
                command = self.baseline
                reason = (
                    "calibrate_open"
                    if direction > 0
                    else "calibrate_close"
                    if direction < 0
                    else "accept"
                )
                self._begin(ctx)
        return SlowOutput(
            command,
            {
                "reason": reason,
                "hold_window_s": self.hold_s,
                "baseline": self.baseline,
                "supply_mean_temperature": ctx.supply_temperature,
                "recent_trend_k_per_h": recent_trend,
                "observation_age_s": ctx.t - self._start,
                "next_review_in_s": max(0.0, self.hold_s - (ctx.t - self._start)),
                "last_reset_reason": self._reset_reason,
                "feedback_confirmed": self._feedback_confirmed,
                "response": dict(self._response),
                "holding_point_count": len(self._holding_points),
            },
        )

    def export_state(self) -> SlowState:
        """Persist the working opening, never old timers or unobserved trends."""
        if self.baseline is None:
            return {}
        state: SlowState = {"baseline": self.baseline}
        if self._response:
            state["response"] = dict(self._response)
        if self._holding_points:
            state["holding_points"] = [dict(point) for point in self._holding_points]
        return state
