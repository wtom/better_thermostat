"""Experimental sample-and-hold room calibration; simulation only.

No production selection or device writes. All limits are provisional. Only the
normal baseline and settled holding points survive restart; observation and recovery windows start fresh.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from .base import BenchmarkContext, BenchmarkOutput, ControllerFamily


class SlowCalibrationAdapter:
    """Hold a room opening for hours, with separate exceptional recovery."""

    name = "slow"
    family: ControllerFamily = "valve"

    def __init__(self, hold_s: float = 10800.0, step_percent: float = 3.0) -> None:
        self.hold_s = hold_s
        self.step_percent = step_percent
        self.reset()

    def reset(self, prior: dict[str, Any] | None = None) -> None:
        """Restore the normal baseline, never a transient recovery command."""
        self.baseline = float(prior["baseline"]) if prior else None
        self._response = dict(prior.get("response", {})) if prior else {}
        self._holding_points = (
            [dict(point) for point in prior.get("holding_points", [])] if prior else []
        )
        self._recent_temperatures: deque[tuple[float, float]] = deque()
        self._recovery_start: float | None = None
        self._start: float | None = None
        self._window_context: BenchmarkContext | None = None
        self._temperature = 0.0
        self._sum_error = 0.0
        self._elapsed = 0.0
        self._previous: BenchmarkContext | None = None
        self._exception_since: float | None = None
        self._recovery: str | None = None
        self._command: float | None = None
        self._last_recovery_step = float("-inf")

    def _begin(self, ctx: BenchmarkContext) -> None:
        self._start = ctx.t
        self._window_context = ctx
        self._temperature = ctx.room_temperature
        self._sum_error = 0.0
        self._elapsed = 0.0

    def step(self, ctx: BenchmarkContext) -> BenchmarkOutput:
        """Compute a held opening without access to plant-internal truth."""
        if self.baseline is None:
            self.baseline = max(0.0, min(100.0, ctx.last_valve_percent))
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
        anchor = self._window_context
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
        unavailable = (
            ctx.supply_temperature is not None
            and ctx.supply_temperature <= ctx.room_temperature
        )
        disturbed = ctx.window_open or unavailable or ctx.solar_intensity >= 0.2
        self._previous = ctx
        if self._start is None or changed or disturbed:
            self._begin(ctx)
            self._exception_since = None
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
            command = 0.0 if ctx.window_open else self.baseline
            reason = "disturbance"
        else:
            cold = error >= 0.5
            warm = error <= -0.5
            direction = "cold" if cold else "warm" if warm else None
            if self._recovery and (
                (self._recovery == "cold" and error <= 0.2)
                or (self._recovery == "warm" and error >= -0.2)
                or changed
            ):
                # A completed comfort intervention is directional evidence,
                # not a measurement of the steady opening. Move only one normal
                # calibration step; never persist the transient recovery value.
                if not changed:
                    delta = (
                        self.step_percent
                        if self._recovery == "cold"
                        else -self.step_percent
                    )
                    self.baseline = max(0.0, min(100.0, self.baseline + delta))
                    command = self.baseline
                    self._response["last_recovery_s"] = ctx.t - self._recovery_start
                self._recovery = None
                self._command = None
                self._begin(ctx)
            if direction is None:
                self._exception_since = None
            elif previous is None or (
                (previous.target_temperature - previous.room_temperature >= 0.5) != cold
                or (previous.target_temperature - previous.room_temperature <= -0.5)
                != warm
            ):
                self._exception_since = ctx.t
            elif self._exception_since is None:
                self._exception_since = ctx.t
            if (
                direction
                and self._exception_since is not None
                and (ctx.t - self._exception_since >= 600.0)
            ):
                if self._recovery != direction:
                    self._last_recovery_step = float("-inf")
                    self._recovery_start = ctx.t
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
        return BenchmarkOutput(
            valve_percent=command,
            diagnostics={
                "reason": reason,
                "baseline": self.baseline,
                "recent_trend_k_per_h": recent_trend,
                "response": dict(self._response),
                "holding_point_count": len(self._holding_points),
            },
        )

    def export_state(self) -> dict[str, Any]:
        """Persist only the normal room opening, not temporary recovery."""
        if self.baseline is None:
            return {}
        state = {"baseline": self.baseline}
        if self._response:
            state["response"] = dict(self._response)
        if self._holding_points:
            state["holding_points"] = [dict(point) for point in self._holding_points]
        return state
