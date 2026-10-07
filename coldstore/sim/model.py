"""Deterministic cold room pull-down model for resumable runs.

Every step is a pure function of the confirmed state, the recorded inputs
and the chosen dt. Re-executing a step with the inputs captured at the
time must land on exactly the same result, which is what makes replay and
breakpoint resume safe.
"""

from __future__ import annotations

from dataclasses import dataclass

COOLING = "cooling"
HOLDING = "holding"

INPUT_KEYS = ("ambient", "door", "pull", "leak", "target", "band", "door_heat")


@dataclass(frozen=True)
class Inputs:
    """The external conditions a single step was executed with."""

    ambient: float
    door: int
    pull: float
    leak: float
    target: float
    band: float
    door_heat: float

    def as_dict(self) -> dict[str, object]:
        return {
            "ambient": self.ambient,
            "door": self.door,
            "pull": self.pull,
            "leak": self.leak,
            "target": self.target,
            "band": self.band,
            "door_heat": self.door_heat,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Inputs":
        return cls(
            ambient=float(payload.get("ambient", 20.0)),
            door=1 if payload.get("door", 0) in (1, "1", True) else 0,
            pull=float(payload.get("pull", 0.01)),
            leak=float(payload.get("leak", 0.0005)),
            target=float(payload.get("target", -18.0)),
            band=float(payload.get("band", 0.5)),
            door_heat=float(payload.get("door_heat", 0.02)),
        )

    def merged(self, overrides: dict[str, object] | None) -> "Inputs":
        """Apply ad-hoc overrides for the next unconfirmed step."""

        if not overrides:
            return self
        unknown = sorted(set(overrides) - set(INPUT_KEYS))
        if unknown:
            raise ValueError(f"unknown input fields: {', '.join(unknown)}")
        base = self.as_dict()
        base.update(overrides)
        return Inputs.from_dict(base)


@dataclass(frozen=True)
class State:
    """Confirmed simulation state carried between steps."""

    time: float
    temp: float
    mode: str

    def as_dict(self) -> dict[str, object]:
        return {"time": self.time, "temp": self.temp, "mode": self.mode}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "State":
        return cls(
            time=float(payload.get("time", 0.0)),
            temp=float(payload.get("temp", 0.0)),
            mode=str(payload.get("mode", HOLDING)),
        )


def initial_state(start_temp: float, target: float) -> State:
    mode = COOLING if start_temp > target else HOLDING
    return State(time=0.0, temp=round(start_temp, 4), mode=mode)


def model_step(state: State, inputs: Inputs, dt: float) -> State:
    """Advance the room by dt seconds; pure and rounded for stable replay."""

    mode = state.mode
    if mode == COOLING and state.temp <= inputs.target:
        mode = HOLDING
    elif mode == HOLDING and state.temp >= inputs.target + inputs.band:
        mode = COOLING
    heat = inputs.leak * (inputs.ambient - state.temp)
    if inputs.door:
        heat += inputs.door_heat
    cool = inputs.pull if mode == COOLING else 0.0
    temp = round(state.temp + (heat - cool) * dt, 4)
    return State(time=round(state.time + dt, 4), temp=temp, mode=mode)


def explain(state: State, inputs: Inputs, dt: float, nxt: State, dt_rule: str) -> dict[str, object]:
    """The recorded basis for a step: why this mode, this dt, this outcome."""

    if nxt.mode == COOLING:
        why = f"temp {state.temp:.4f} above target {inputs.target:.4f}, cooling toward the setpoint"
    else:
        why = f"temp {state.temp:.4f} within band of target {inputs.target:.4f}, holding off"
    delta = round(nxt.temp - state.temp, 4)
    return {
        "mode": nxt.mode,
        "why": why,
        "dt_rule": dt_rule,
        "delta": delta,
        "rate": round(delta / dt, 6) if dt > 0 else 0.0,
        "door": inputs.door,
    }
