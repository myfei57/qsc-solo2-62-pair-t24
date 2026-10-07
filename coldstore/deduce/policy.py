"""The rules a deduction run pins down before its first step.

Step size, resume cadence and confirmation mode are decided up front and
frozen into the run config: ``fixed`` keeps one constant step, ``adaptive``
refines the step as the simulated temperature closes in on its target.
Interruption recovery either retries the unconfirmed step in place or rolls
the frontier back one step. Every rule is deterministic so a replay from the
recorded inputs reproduces the recorded outputs exactly.
"""

from __future__ import annotations

from coldstore.temp import validate_temp

FIXED = "fixed"
ADAPTIVE = "adaptive"
STEP_POLICIES = (FIXED, ADAPTIVE)

RETRY = "retry"
ROLLBACK = "rollback"
RESUME_POLICIES = (RETRY, ROLLBACK)

AUTO = "auto"
MANUAL = "manual"
CONFIRM_MODES = (AUTO, MANUAL)

DEFAULT_STEPS = 5
MAX_STEPS = 1000
DEFAULT_START_TEMP = -18.0
DEFAULT_STEP_SIZE = 0.4
DEFAULT_MIN_STEP = 0.1
DEFAULT_MAX_STEP = 1.0
MAX_STEP_SIZE = 5.0
DEFAULT_CLOCK_SECONDS = 60
NEAR_GAP = 1.0
FAR_GAP = 4.0
DEFAULT_ADVANCE_CAP = 100
MAX_ADVANCE_CAP = 1000
DEFAULT_ADVANCE_SECONDS = 2.0
MAX_ADVANCE_SECONDS = 30.0

RUN_ID_ALPHABET = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
MAX_RUN_ID = 64


def validate_run_id(run_id: str) -> None:
    """Keep run ids short and safe to embed in store keys."""

    if not run_id or len(run_id) > MAX_RUN_ID:
        raise ValueError(f"run id must be 1..{MAX_RUN_ID} characters")
    if any(ch not in RUN_ID_ALPHABET for ch in run_id):
        raise ValueError("run id may use letters, digits, dot, dash and underscore")


def validate_config(config: "object") -> None:
    """Reject a run config that breaks the agreed rules."""

    if not 1 <= config.steps <= MAX_STEPS:
        raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
    if config.policy not in STEP_POLICIES:
        raise ValueError(f"policy must be one of {', '.join(STEP_POLICIES)}")
    if config.resume not in RESUME_POLICIES:
        raise ValueError(f"resume must be one of {', '.join(RESUME_POLICIES)}")
    if config.confirm not in CONFIRM_MODES:
        raise ValueError(f"confirm must be one of {', '.join(CONFIRM_MODES)}")
    for name in ("step_size", "min_step", "max_step"):
        value = getattr(config, name)
        if not 0 < value <= MAX_STEP_SIZE:
            raise ValueError(f"{name} must be in (0, {MAX_STEP_SIZE}]")
    if not config.min_step <= config.step_size <= config.max_step:
        raise ValueError("min_step <= step_size <= max_step is required")
    validate_temp(config.start_temp)
    if config.target_temp is not None:
        validate_temp(config.target_temp)
    if config.policy == ADAPTIVE and config.target_temp is None:
        raise ValueError("adaptive policy requires target_temp")
    if config.clock_seconds <= 0:
        raise ValueError("clock_seconds must be positive")
    if not 1 <= config.advance_cap <= MAX_ADVANCE_CAP:
        raise ValueError(f"advance_cap must be between 1 and {MAX_ADVANCE_CAP}")
    if not 0 < config.advance_seconds <= MAX_ADVANCE_SECONDS:
        raise ValueError(f"advance_seconds must be in (0, {MAX_ADVANCE_SECONDS}]")


def choose_step(config: "object", temp: float) -> tuple[float, str]:
    """Return the signed step for ``temp`` and the reason it was chosen.

    Without a target the fixed policy ramps upward exactly like the legacy
    simulator. With a target both policies walk towards it and clamp on
    arrival; the adaptive policy additionally narrows the step as the gap
    shrinks through the far/mid/near bands.
    """

    target = config.target_temp
    if target is None:
        step = round(config.step_size, 4)
        return step, f"policy=fixed ramp step={step:+.4f} (no target)"
    gap = round(target - temp, 4)
    if gap == 0:
        return 0.0, f"at target {target:.4f}; holding"
    direction = 1.0 if gap > 0 else -1.0
    distance = abs(gap)
    if config.policy == ADAPTIVE:
        if distance <= NEAR_GAP:
            magnitude, band = config.min_step, "near"
        elif distance <= FAR_GAP:
            magnitude, band = config.step_size, "mid"
        else:
            magnitude, band = config.max_step, "far"
        reason = f"policy=adaptive gap={distance:.4f} band={band} step={magnitude:.4f}"
    else:
        magnitude = config.step_size
        reason = f"policy=fixed step={magnitude:.4f} toward target={target:.4f}"
    magnitude = min(magnitude, distance)
    return round(direction * magnitude, 4), reason
