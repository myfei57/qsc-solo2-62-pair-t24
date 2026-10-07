"""Console endpoints that drive resumable deduction runs."""

from __future__ import annotations

import time

from coldstore.deduce import (
    AUTO,
    FIXED,
    RETRY,
    RunConfig,
    new_run_id,
)
from coldstore.deduce.policy import (
    DEFAULT_ADVANCE_CAP,
    DEFAULT_ADVANCE_SECONDS,
    DEFAULT_CLOCK_SECONDS,
    DEFAULT_MAX_STEP,
    DEFAULT_MIN_STEP,
    DEFAULT_START_TEMP,
    DEFAULT_STEP_SIZE,
    DEFAULT_STEPS,
)

from .http import Request, Response, json_response
from .runtime import Runtime


def probe(rt: Runtime):
    """Snapshot the control plane counters one step observes."""

    def collect() -> dict[str, object]:
        return {
            "running": rt.compressors.running_count(),
            "total_stage": rt.compressors.total_stage(),
            "defrosting": len(rt.defrost.defrosting()),
            "due": len(rt.scheduler.due(rt.rooms)),
            "latched": rt.ammonia.latched(),
        }

    return collect


def config_from_request(request: Request) -> RunConfig:
    """Pin the run's rules from the request; defaults match the legacy ticks."""

    target = None
    if "target_temp" in request.payload and request.payload["target_temp"] is not None:
        target = request.float_field("target_temp")
    return RunConfig(
        run_id=request.str_field("run") or new_run_id(),
        steps=request.int_field("steps", DEFAULT_STEPS),
        policy=request.str_field("policy", FIXED),
        step_size=request.float_field("step_size", DEFAULT_STEP_SIZE),
        min_step=request.float_field("min_step", DEFAULT_MIN_STEP),
        max_step=request.float_field("max_step", DEFAULT_MAX_STEP),
        start_temp=request.float_field("start_temp", DEFAULT_START_TEMP),
        target_temp=target,
        clock_seconds=request.int_field("clock_seconds", DEFAULT_CLOCK_SECONDS),
        start_at=request.int_field("start_at", int(time.time())),
        resume=request.str_field("resume", RETRY),
        confirm=request.str_field("confirm", AUTO),
        advance_cap=request.int_field("advance_cap", DEFAULT_ADVANCE_CAP),
        advance_seconds=request.float_field("advance_seconds", DEFAULT_ADVANCE_SECONDS),
    )


def _run_id(request: Request) -> str:
    run_id = request.str_field("run")
    if not run_id:
        raise ValueError("run is required")
    return run_id


def start_run(rt: Runtime, request: Request) -> Response:
    config = config_from_request(request)
    header, created = rt.deduce.start(config)
    return json_response(
        {
            "run": config.run_id,
            "created": created,
            "status": header.status,
            "cursor": header.cursor,
            "config": config.as_dict(),
        }
    )


def advance_run(rt: Runtime, request: Request) -> Response:
    max_steps = request.int_field("max_steps", 0) or None
    max_seconds = request.float_field("max_seconds", 0.0) or None
    result = rt.deduce.advance(
        _run_id(request), max_steps=max_steps, max_seconds=max_seconds, probe=probe(rt)
    )
    return json_response(result)


def confirm_step(rt: Runtime, request: Request) -> Response:
    header, record = rt.deduce.confirm(_run_id(request))
    return json_response(
        {
            "run": header.config.run_id,
            "confirmed": record.index,
            "cursor": header.cursor,
            "status": header.status,
        }
    )


def rewind_run(rt: Runtime, request: Request) -> Response:
    header, dropped = rt.deduce.rewind(_run_id(request), request.int_field("step"))
    return json_response(
        {
            "run": header.config.run_id,
            "cursor": header.cursor,
            "status": header.status,
            "superseded": dropped,
        }
    )


def abort_run(rt: Runtime, request: Request) -> Response:
    header = rt.deduce.abort(_run_id(request))
    return json_response({"run": header.config.run_id, "status": header.status})


def list_runs(rt: Runtime, request: Request) -> Response:
    runs = rt.deduce.runs()
    return json_response({"runs": runs, "count": len(runs)})


def run_status(rt: Runtime, request: Request) -> Response:
    return json_response(rt.deduce.status(_run_id(request)))


def run_steps(rt: Runtime, request: Request) -> Response:
    run_id = _run_id(request)
    records = rt.deduce.steps(run_id)
    return json_response(
        {
            "run": run_id,
            "steps": [record.as_dict() for record in records],
            "count": len(records),
        }
    )


def replay_run(rt: Runtime, request: Request) -> Response:
    return json_response(rt.deduce.replay(_run_id(request)))
