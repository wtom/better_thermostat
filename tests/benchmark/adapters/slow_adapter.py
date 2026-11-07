"""Experimental sample-and-hold room calibration; simulation only.

No production selection or device writes. All limits are provisional. Only the
normal baseline survives restart; observation and recovery windows start fresh.
"""

from __future__ import annotations

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
        self._start: float | None = None
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
        reason = "hold"
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
                self._recovery = direction
            if self._recovery:
                if ctx.t - self._last_recovery_step >= 900.0:
                    opening = (
                        self._command if self._command is not None else self.baseline
                    )
                    self._command = (
                        min(100.0, opening + 10.0) if self._recovery == "cold" else 0.0
                    )
                    self._last_recovery_step = ctx.t
                command = self._command
                reason = f"recovery_{self._recovery}"
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
            diagnostics={"reason": reason, "baseline": self.baseline},
        )

    def export_state(self) -> dict[str, Any]:
        """Persist only the normal room opening, not temporary recovery."""
        return {"baseline": self.baseline} if self.baseline is not None else {}
