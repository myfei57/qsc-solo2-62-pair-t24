"""Deterministic simulation ticks.

The legacy one-shot endpoint now runs on the resumable deduction engine:
each call creates a fresh auto-confirmed run, advances it to completion and
reports the confirmed steps in the historical tick shape. The journal it
leaves behind is listed by ``GET /deduce`` like any other run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from coldstore.deduce import AUTO, CONFIRMED, FIXED, RETRY, RunConfig, new_run_id
from coldstore.ns import control_line

from .deduce import probe
from .http import Request, Response, json_response
from .runtime import Runtime

DEFAULT_TICKS = 5
MAX_TICKS = 100
START_TEMP = -18.0
TEMP_STEP = 0.4
CLOCK_SECONDS = 60
ADVANCE_ATTEMPTS = 5


@dataclass(frozen=True)
class Tick:
    """One simulated control step."""

    index: int
    temp: float
    load: float
    high: bool
    running: int
    total_stage: int
    defrosting: int
    due: int
    latched: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "temp": self.temp,
            "load": self.load,
            "high": self.high,
            "running": self.running,
            "total_stage": self.total_stage,
            "defrosting": self.defrosting,
            "due": self.due,
            "latched": self.latched,
        }


@dataclass(frozen=True)
class SimulationReport:
    """The full sequence plus closing counters."""

    ticks: list[Tick] = field(default_factory=list)
    final_audit_count: int = 0
    last_stage: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "ticks": [tick.as_dict() for tick in self.ticks],
            "final_audit_count": self.final_audit_count,
            "last_stage": self.last_stage,
        }


def run_simulation(rt: Runtime, request: Request) -> Response:
    """Run a bounded deduction and report it in the legacy tick shape."""

    ticks = request.int_field("ticks", DEFAULT_TICKS)
    if ticks <= 0:
        ticks = DEFAULT_TICKS
    if ticks > MAX_TICKS:
        ticks = MAX_TICKS

    config = RunConfig(
        run_id=new_run_id("simulate"),
        steps=ticks,
        policy=FIXED,
        step_size=TEMP_STEP,
        min_step=TEMP_STEP,
        max_step=TEMP_STEP,
        start_temp=START_TEMP,
        target_temp=None,
        clock_seconds=CLOCK_SECONDS,
        start_at=int(time.time()),
        resume=RETRY,
        confirm=AUTO,
        advance_cap=MAX_TICKS,
        advance_seconds=30.0,
    )
    engine = rt.deduce
    engine.start(config)
    for _ in range(ADVANCE_ATTEMPTS):
        result = engine.advance(
            config.run_id, max_steps=ticks, max_seconds=30.0, probe=probe(rt)
        )
        if result["completed"]:
            break

    steps = [
        Tick(
            index=record.index,
            temp=float(record.input["temp"]),
            load=float(record.output["load"]),
            high=bool(record.output["high"]),
            running=int(record.context["running"]),
            total_stage=int(record.context["total_stage"]),
            defrosting=int(record.context["defrosting"]),
            due=int(record.context["due"]),
            latched=bool(record.context["latched"]),
        )
        for record in engine.steps(config.run_id)
        if record.state == CONFIRMED
    ]

    last_stage = control_line().last()
    report = SimulationReport(
        ticks=steps,
        final_audit_count=rt.auditor.count(),
        last_stage="" if last_stage is None else last_stage.value,
    )
    return json_response(report.as_dict())
