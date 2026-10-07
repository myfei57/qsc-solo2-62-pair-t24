"""HTTP handlers for the resumable simulation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from coldstore.sim import engine

from .http import Request, RequestError, Response, json_response

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from .server import Server


def _run_id(request: Request) -> str:
    run_id = request.str_field("id")
    if not run_id:
        values = request.query.get("id")
        if values:
            run_id = values[0]
    if not run_id:
        raise RequestError(400, "id is required")
    return run_id


def _inputs(request: Request) -> dict[str, object] | None:
    raw = request.payload.get("inputs")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RequestError(400, "inputs must be an object")
    return raw


def create(server: "Server", request: Request) -> Response:
    config = {key: value for key, value in request.payload.items() if key != "id"}
    record, created = engine.create_run(server.runtime.store, request.str_field("id"), config)
    payload = record.as_dict()
    payload["created"] = created
    return json_response(payload)


def listing(server: "Server", request: Request) -> Response:
    return json_response(engine.list_runs(server.runtime.store))


def show(server: "Server", request: Request) -> Response:
    return json_response(engine.show(server.runtime.store, _run_id(request)))


def checkpoint(server: "Server", request: Request) -> Response:
    index = request.int_field("index", -1)
    if index < 0:
        values = request.query.get("index")
        if values:
            try:
                index = int(values[0])
            except ValueError:
                index = -1
    if index < 0:
        raise RequestError(400, "index is required")
    return json_response(engine.get_checkpoint(server.runtime.store, _run_id(request), index))


def advance(server: "Server", request: Request) -> Response:
    budget = request.payload.get("budget_ms")
    limit = request.payload.get("max_steps")
    return json_response(
        engine.advance(
            server.runtime.store,
            _run_id(request),
            budget_ms=None if budget is None else request.int_field("budget_ms"),
            max_steps=None if limit is None else request.int_field("max_steps"),
            inputs=_inputs(request),
        )
    )


def step(server: "Server", request: Request) -> Response:
    return json_response(engine.step_once(server.runtime.store, _run_id(request), inputs=_inputs(request)))


def recover(server: "Server", request: Request) -> Response:
    return json_response(engine.recover(server.runtime.store, _run_id(request)))


def rewind(server: "Server", request: Request) -> Response:
    return json_response(engine.rewind(server.runtime.store, _run_id(request), request.int_field("to")))


def replay(server: "Server", request: Request) -> Response:
    frm = request.int_field("from", 0)
    raw_to = request.payload.get("to")
    to = None if raw_to is None else request.int_field("to")
    return json_response(engine.replay(server.runtime.store, _run_id(request), frm=frm, to=to))


def compare(server: "Server", request: Request) -> Response:
    report = engine.journal.load_compare(server.runtime.store, _run_id(request))
    return json_response({"compare": report})


def cancel(server: "Server", request: Request) -> Response:
    return json_response(engine.cancel(server.runtime.store, _run_id(request)))
