"""Benchmark wrapper around the production slow calibration controller."""

from typing import Any

from custom_components.better_thermostat.utils.calibration.slow import (
    SlowCalibration,
    SlowContext,
)

from .base import BenchmarkContext, BenchmarkOutput, ControllerFamily


class SlowCalibrationAdapter:
    """Use the exact production algorithm with benchmark measurements."""

    name = "slow"
    family: ControllerFamily = "valve"

    def __init__(self, hold_s: float = 1800.0, step_percent: float = 3.0) -> None:
        self.controller = SlowCalibration(hold_s, step_percent)

    @property
    def baseline(self) -> float | None:
        return self.controller.baseline

    def reset(self, prior: dict[str, Any] | None = None) -> None:
        self.controller.reset(prior)

    def export_state(self) -> dict[str, Any]:
        return dict(self.controller.export_state())

    def step(self, ctx: BenchmarkContext) -> BenchmarkOutput:
        """Translate observable inputs, never plant-internal truth."""
        result = self.controller.step(
            SlowContext(
                t=ctx.t,
                target_temperature=ctx.target_temperature,
                room_temperature=ctx.room_temperature,
                outdoor_temperature=ctx.outdoor_temperature,
                last_valve_percent=ctx.last_valve_percent,
                window_open=ctx.window_open,
                solar_intensity=ctx.solar_intensity,
                supply_temperature=ctx.supply_temperature,
            )
        )
        return BenchmarkOutput(
            valve_percent=result.valve_percent, diagnostics=result.diagnostics
        )
