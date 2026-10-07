"""Step sizing rules for resumable runs.

Standing rule: fixed steps by default. Adaptive sizing is allowed only
inside declared bounds, and every chosen dt is recorded on the checkpoint
so a replay with the recorded inputs reproduces it exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Inputs, State, model_step

FIXED = "fixed"
ADAPTIVE = "adaptive"

DEFAULT_STEP = 60.0
DEFAULT_MIN_STEP = 1.0
DEFAULT_MAX_STEP = 300.0
DEFAULT_MAX_DELTA = 0.5
STEP_CEILING = 3600.0


@dataclass(frozen=True)
class Policy:
    """How the dt for each step is chosen."""

    mode: str
    step: float
    min_step: float
    max_step: float
    max_delta: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "step": self.step,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "max_delta": self.max_delta,
        }

    @classmethod
    def build(
        cls, mode: str, step: float, min_step: float, max_step: float, max_delta: float
    ) -> "Policy":
        if mode not in (FIXED, ADAPTIVE):
            raise ValueError("policy must be fixed or adaptive")
        if step <= 0 or step > STEP_CEILING:
            raise ValueError(f"step must be within 0..{STEP_CEILING}")
        if min_step <= 0:
            raise ValueError("min_step must be positive")
        if max_step > STEP_CEILING:
            raise ValueError(f"max_step must be within 0..{STEP_CEILING}")
        if mode == ADAPTIVE and not min_step <= step <= max_step:
            raise ValueError("adaptive policy requires min_step <= step <= max_step")
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")
        return cls(mode=mode, step=step, min_step=min_step, max_step=max_step, max_delta=max_delta)


def choose_dt(policy: Policy, base_dt: float, state: State, inputs: Inputs) -> tuple[float, State, str]:
    """Pick the dt for one step and the resulting state.

    Returns (dt, next_state, rule) where rule explains the choice:
    "fixed", "base", "refined" or "grown". Pure and deterministic, so a
    replay derives the same dt from the same state and inputs.
    """

    if policy.mode == FIXED:
        return policy.step, model_step(state, inputs, policy.step), FIXED
    dt = min(max(base_dt, policy.min_step), policy.max_step)
    nxt = model_step(state, inputs, dt)
    rule = "base"
    while abs(nxt.temp - state.temp) > policy.max_delta and dt > policy.min_step:
        dt = max(policy.min_step, round(dt / 2, 4))
        nxt = model_step(state, inputs, dt)
        rule = "refined"
    if rule == "base" and abs(nxt.temp - state.temp) < policy.max_delta / 4 and dt < policy.max_step:
        dt = min(policy.max_step, round(dt * 2, 4))
        nxt = model_step(state, inputs, dt)
        rule = "grown"
        while abs(nxt.temp - state.temp) > policy.max_delta and dt > policy.min_step:
            dt = max(policy.min_step, round(dt / 2, 4))
            nxt = model_step(state, inputs, dt)
            rule = "refined"
    return dt, nxt, rule
