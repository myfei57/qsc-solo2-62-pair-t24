"""Persisted shapes for deduction runs: config, header and step records."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass

RUNNING = "running"
COMPLETED = "completed"
ABORTED = "aborted"
RUN_STATUSES = (RUNNING, COMPLETED, ABORTED)

PENDING = "pending"
CONFIRMED = "confirmed"
SUPERSEDED = "superseded"
RECORD_STATES = (PENDING, CONFIRMED, SUPERSEDED)

DEFAULT_CONTEXT: dict[str, object] = {
    "running": 0,
    "total_stage": 0,
    "defrosting": 0,
    "due": 0,
    "latched": False,
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id(prefix: str = "run") -> str:
    """Generate a run id that sorts by creation time."""

    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def digest_payload(payload: dict[str, object]) -> str:
    """Stable short digest over a JSON-able payload."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def record_digest(
    run_id: str,
    index: int,
    attempt: int,
    step_input: dict[str, object],
    output: dict[str, object],
) -> str:
    """Digest everything a confirmed step stands on."""

    return digest_payload(
        {
            "run": run_id,
            "index": index,
            "attempt": attempt,
            "input": step_input,
            "output": output,
        }
    )


@dataclass(frozen=True)
class RunConfig:
    """The rules one run agreed to; frozen once the run starts."""

    run_id: str
    steps: int
    policy: str
    step_size: float
    min_step: float
    max_step: float
    start_temp: float
    target_temp: float | None
    clock_seconds: int
    start_at: int
    resume: str
    confirm: str
    advance_cap: int
    advance_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "steps": self.steps,
            "policy": self.policy,
            "step_size": self.step_size,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "start_temp": self.start_temp,
            "target_temp": self.target_temp,
            "clock_seconds": self.clock_seconds,
            "start_at": self.start_at,
            "resume": self.resume,
            "confirm": self.confirm,
            "advance_cap": self.advance_cap,
            "advance_seconds": self.advance_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RunConfig":
        target = payload.get("target_temp")
        return cls(
            run_id=str(payload.get("run_id", "")),
            steps=int(payload.get("steps", 5)),
            policy=str(payload.get("policy", "fixed")),
            step_size=float(payload.get("step_size", 0.4)),
            min_step=float(payload.get("min_step", 0.1)),
            max_step=float(payload.get("max_step", 1.0)),
            start_temp=float(payload.get("start_temp", -18.0)),
            target_temp=None if target is None else float(target),
            clock_seconds=int(payload.get("clock_seconds", 60)),
            start_at=int(payload.get("start_at", 0)),
            resume=str(payload.get("resume", "retry")),
            confirm=str(payload.get("confirm", "auto")),
            advance_cap=int(payload.get("advance_cap", 100)),
            advance_seconds=float(payload.get("advance_seconds", 2.0)),
        )

    def fingerprint(self) -> str:
        """Identity used to make repeated starts idempotent."""

        return digest_payload(self.as_dict())


@dataclass
class RunHeader:
    """The frontier of a run: how far confirmation has advanced."""

    config: RunConfig
    status: str = RUNNING
    cursor: int = -1
    temp: float = 0.0
    at: int = 0
    confirmed: int = 0
    superseded: int = 0
    recoveries: int = 0
    elapsed_ms: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def create(cls, config: RunConfig) -> "RunHeader":
        now = now_iso()
        return cls(
            config=config,
            temp=config.start_temp,
            at=config.start_at,
            created_at=now,
            updated_at=now,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.config.run_id,
            "status": self.status,
            "cursor": self.cursor,
            "temp": self.temp,
            "at": self.at,
            "confirmed": self.confirmed,
            "superseded": self.superseded,
            "recoveries": self.recoveries,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RunHeader":
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("run header is missing its config")
        return cls(
            config=RunConfig.from_dict(config),
            status=str(payload.get("status", RUNNING)),
            cursor=int(payload.get("cursor", -1)),
            temp=float(payload.get("temp", 0.0)),
            at=int(payload.get("at", 0)),
            confirmed=int(payload.get("confirmed", 0)),
            superseded=int(payload.get("superseded", 0)),
            recoveries=int(payload.get("recoveries", 0)),
            elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )


@dataclass
class StepRecord:
    """One journaled step: inputs, basis, outputs and confirmation state."""

    run_id: str
    index: int
    attempt: int
    state: str
    input: dict[str, object]
    context: dict[str, object]
    output: dict[str, object]
    basis: str
    digest: str
    recorded_at: str
    confirmed_at: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run_id,
            "index": self.index,
            "attempt": self.attempt,
            "state": self.state,
            "input": self.input,
            "context": self.context,
            "output": self.output,
            "basis": self.basis,
            "digest": self.digest,
            "recorded_at": self.recorded_at,
            "confirmed_at": self.confirmed_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "StepRecord":
        return cls(
            run_id=str(payload.get("run", "")),
            index=int(payload.get("index", 0)),
            attempt=int(payload.get("attempt", 1)),
            state=str(payload.get("state", PENDING)),
            input=dict(payload.get("input") or {}),
            context=dict(payload.get("context") or {}),
            output=dict(payload.get("output") or {}),
            basis=str(payload.get("basis", "")),
            digest=str(payload.get("digest", "")),
            recorded_at=str(payload.get("recorded_at", "")),
            confirmed_at=str(payload.get("confirmed_at", "")),
            note=str(payload.get("note", "")),
        )
