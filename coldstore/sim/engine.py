"""Resumable simulation engine.

Cadence rules:

- a step is prepared, persisted as a checkpoint, and only then confirmed
  by moving the cursor; an unconfirmed step never counts
- after an interruption the run resumes in place: a dangling checkpoint
  is committed when it verifies, otherwise it is discarded and redone
- rolling back happens only when chain verification fails (recover) or
  when the operator explicitly rewinds to a chosen checkpoint
- replay is read-only: it re-executes with the recorded inputs and diffs
  the recomputation against the journal, field by field
"""

from __future__ import annotations

import json
import threading
import time

from coldstore.audit import Auditor
from coldstore.store.store import Store

from . import journal
from .journal import Checkpoint, CompareReport, RunRecord
from .model import Inputs, State, explain, initial_state, model_step  # noqa: F401
from .policy import ADAPTIVE, FIXED, Policy, choose_dt  # noqa: F401

MAX_STEPS = 1000
MAX_BUDGET_MS = 5000
DEFAULT_BUDGET_MS = 250
MAX_STEPS_PER_CALL = 200
CHECKPOINT_TAIL = 5

AUDIT_RUN = "sim-run"
AUDIT_STEP = "sim-step"
AUDIT_COMPARE = "sim-compare"

DEFAULTS: dict[str, object] = {
    "steps": 10,
    "policy": FIXED,
    "step": 60.0,
    "min_step": 1.0,
    "max_step": 300.0,
    "max_delta": 0.5,
    "start_temp": -5.0,
    "target": -18.0,
    "band": 0.5,
    "ambient": 20.0,
    "pull": 0.01,
    "leak": 0.0005,
    "door_heat": 0.02,
    "budget_ms": DEFAULT_BUDGET_MS,
}

_LOCK = threading.RLock()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _audit(store: Store, kind: str, detail: str) -> None:
    Auditor(store).record(kind, detail)


def _num(name: str, value: object, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number") from None
    if not low <= number <= high:
        raise ValueError(f"{name} must be within {low}..{high}")
    return number


def _int(name: str, value: object, low: int, high: int) -> int:
    return int(_num(name, value, low, high))


def normalize_config(raw: dict[str, object]) -> dict[str, object]:
    """Validate and complete a run config; the result is immutable per run."""

    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(unknown)}")
    merged = {**DEFAULTS, **raw}
    policy = Policy.build(
        str(merged["policy"]),
        _num("step", merged["step"], 0.0001, 3600.0),
        _num("min_step", merged["min_step"], 0.0001, 3600.0),
        _num("max_step", merged["max_step"], 0.0001, 3600.0),
        _num("max_delta", merged["max_delta"], 0.0001, 100.0),
    )
    return {
        "steps": _int("steps", merged["steps"], 1, MAX_STEPS),
        "policy": policy.mode,
        "step": policy.step,
        "min_step": policy.min_step,
        "max_step": policy.max_step,
        "max_delta": policy.max_delta,
        "start_temp": _num("start_temp", merged["start_temp"], -100.0, 100.0),
        "target": _num("target", merged["target"], -100.0, 100.0),
        "band": _num("band", merged["band"], 0.0001, 50.0),
        "ambient": _num("ambient", merged["ambient"], -100.0, 100.0),
        "pull": _num("pull", merged["pull"], 0.0, 10.0),
        "leak": _num("leak", merged["leak"], 0.0, 1.0),
        "door_heat": _num("door_heat", merged["door_heat"], 0.0, 10.0),
        "budget_ms": _int("budget_ms", merged["budget_ms"], 1, MAX_BUDGET_MS),
    }


def _validate_run_id(run_id: str) -> None:
    if not run_id or len(run_id) > 64:
        raise ValueError("run id must be 1..64 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in run_id):
        raise ValueError("run id may only contain letters, digits, dash, dot and underscore")


def must_load(store: Store, run_id: str) -> RunRecord:
    record = journal.load_record(store, run_id)
    if record is None:
        raise ValueError(f"unknown run {run_id}")
    return record


def create_run(store: Store, run_id: str, raw_config: dict[str, object]) -> tuple[RunRecord, bool]:
    """Create a run; identical id plus identical config returns the original."""

    config = normalize_config(raw_config)
    if not run_id:
        run_id = f"run-{journal.genesis_hash(config)[:12]}"
    _validate_run_id(run_id)
    with _LOCK:
        existing = journal.load_record(store, run_id)
        if existing is not None:
            if existing.config != config:
                raise ValueError(f"run {run_id} already exists with a different config")
            return existing, False
        now = _now()
        record = RunRecord(
            id=run_id,
            created_at=now,
            updated_at=now,
            status=journal.RUNNING,
            config=config,
            cursor=0,
            stats={"prepared": 0, "confirmed": 0, "redone": 0, "rolled_back": 0, "replay_runs": 0},
            final=None,
        )
        journal.save_record(store, record)
        journal.register_run(store, run_id)
    _audit(store, AUDIT_RUN, f"run={run_id} event=created steps={config['steps']} policy={config['policy']}")
    return record, True


def list_runs(store: Store) -> dict[str, object]:
    summaries = []
    for run_id in journal.run_ids(store):
        record = journal.load_record(store, run_id)
        if record is not None:
            summaries.append(record.summary())
    return {"runs": summaries, "count": len(summaries)}


def _policy_of(record: RunRecord) -> Policy:
    config = record.config
    return Policy.build(
        str(config["policy"]),
        float(config["step"]),
        float(config["min_step"]),
        float(config["max_step"]),
        float(config["max_delta"]),
    )


def _compute(
    record: RunRecord,
    index: int,
    state: State,
    prev_hash: str,
    base_dt: float,
    inputs: Inputs,
) -> Checkpoint:
    """The deterministic heart: derive one checkpoint from its antecedents."""

    config = record.config
    dt, nxt, rule = choose_dt(_policy_of(record), base_dt, state, inputs)
    next_dt = float(config["step"]) if config["policy"] == FIXED else dt
    rationale = explain(state, inputs, dt, nxt, rule)
    content = {
        "run": record.id,
        "index": index,
        "dt": dt,
        "next_dt": next_dt,
        "inputs": inputs.as_dict(),
        "state": nxt.as_dict(),
        "rationale": rationale,
        "prev_hash": prev_hash,
    }
    return Checkpoint(
        run=record.id,
        index=index,
        dt=dt,
        next_dt=next_dt,
        inputs=content["inputs"],
        state=content["state"],
        rationale=rationale,
        prev_hash=prev_hash,
        hash=journal.checkpoint_hash(prev_hash, content),
        recorded_at=_now(),
    )


def _context(record: RunRecord, store: Store, index: int) -> tuple[State, str, float]:
    """Confirmed antecedents for step index: state, chain hash, base dt."""

    if index <= 0:
        config = record.config
        return (
            initial_state(float(config["start_temp"]), float(config["target"])),
            journal.genesis_hash(record.config),
            float(config["step"]),
        )
    prev = journal.load_checkpoint(store, record.id, index - 1)
    if prev is None:
        raise ValueError(f"run {record.id} is missing checkpoint {index - 1}")
    return State.from_dict(prev.state), prev.hash, prev.next_dt


def _prepare(store: Store, record: RunRecord, overrides: dict[str, object] | None) -> Checkpoint:
    """Build the checkpoint for the step at the cursor, without confirming."""

    state, prev_hash, base_dt = _context(record, store, record.cursor)
    inputs = Inputs.from_dict(record.config).merged(overrides)
    record.stats["prepared"] = record.stats.get("prepared", 0) + 1
    return _compute(record, record.cursor, state, prev_hash, base_dt, inputs)


def _expected_prev_hash(store: Store, record: RunRecord, index: int) -> str:
    if index <= 0:
        return journal.genesis_hash(record.config)
    prev = journal.load_checkpoint(store, record.id, index - 1)
    return prev.hash if prev is not None else ""


def _ckpt_valid(store: Store, record: RunRecord, ckpt: Checkpoint) -> bool:
    if ckpt.prev_hash != _expected_prev_hash(store, record, ckpt.index):
        return False
    return ckpt.hash == journal.checkpoint_hash(ckpt.prev_hash, ckpt.content())


def _commit(store: Store, record: RunRecord, ckpt: Checkpoint) -> None:
    """Confirm a persisted checkpoint by moving the cursor past it."""

    record.cursor = ckpt.index + 1
    record.stats["confirmed"] = record.stats.get("confirmed", 0) + 1
    record.final = ckpt.state
    record.updated_at = _now()
    if record.cursor >= int(record.config["steps"]):
        record.status = journal.COMPLETED
    journal.save_record(store, record)
    _audit(
        store,
        AUDIT_STEP,
        f"run={record.id} index={ckpt.index} dt={ckpt.dt} "
        f"temp={ckpt.state['temp']} mode={ckpt.state['mode']} hash={ckpt.hash[:12]}",
    )
    if record.status == journal.COMPLETED:
        _audit(store, AUDIT_RUN, f"run={record.id} event=completed steps={record.cursor}")


def _confirm_next(
    store: Store, record: RunRecord, overrides: dict[str, object] | None
) -> tuple[Checkpoint, str]:
    """Confirm exactly one step at the cursor, healing a dangling one first."""

    index = record.cursor
    existing = journal.load_checkpoint(store, record.id, index)
    if existing is not None:
        if _ckpt_valid(store, record, existing):
            _commit(store, record, existing)
            return existing, "committed"
        journal.delete_checkpoint(store, record.id, index)
        record.stats["redone"] = record.stats.get("redone", 0) + 1
        ckpt = _prepare(store, record, overrides)
        journal.save_checkpoint(store, ckpt)
        _commit(store, record, ckpt)
        return ckpt, "redone"
    ckpt = _prepare(store, record, overrides)
    journal.save_checkpoint(store, ckpt)
    _commit(store, record, ckpt)
    return ckpt, "executed"


def advance(
    store: Store,
    run_id: str,
    budget_ms: int | None = None,
    max_steps: int | None = None,
    inputs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Confirm steps until the budget, the call limit or the end of the run.

    Always confirms at least one step when work remains, so a tiny budget
    cannot livelock the run. Overrides apply only to the first step that
    is actually prepared during this call; a dangling checkpoint keeps its
    recorded inputs.
    """

    with _LOCK:
        record = must_load(store, run_id)
        if record.status == journal.ABORTED:
            raise ValueError(f"run {run_id} is aborted")
        config = record.config
        budget = _int("budget_ms", budget_ms if budget_ms is not None else config["budget_ms"], 0, MAX_BUDGET_MS)
        limit = _int("max_steps", max_steps if max_steps is not None else MAX_STEPS_PER_CALL, 1, MAX_STEPS_PER_CALL)
        started = time.monotonic()
        counts = {"executed": 0, "committed": 0, "redone": 0}
        last: Checkpoint | None = None
        pending = inputs
        total = int(config["steps"])
        while record.cursor < total and sum(counts.values()) < limit:
            ckpt, how = _confirm_next(store, record, pending)
            pending = None
            counts[how] += 1
            last = ckpt
            if (time.monotonic() - started) * 1000 >= budget:
                break
        return {
            "run": record.id,
            "status": record.status,
            "cursor": record.cursor,
            "steps": total,
            "executed": counts["executed"],
            "committed": counts["committed"],
            "redone": counts["redone"],
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "budget_ms": budget,
            "remaining": total - record.cursor,
            "last": last.as_dict() if last is not None else None,
        }


def step_once(store: Store, run_id: str, inputs: dict[str, object] | None = None) -> dict[str, object]:
    """Confirm exactly one step; the explicit form of the advance cadence."""

    return advance(store, run_id, budget_ms=MAX_BUDGET_MS, max_steps=1, inputs=inputs)


def recover(store: Store, run_id: str) -> dict[str, object]:
    """Verify the chain and repair the cursor after an interruption.

    Resume in place by default: a dangling checkpoint at the cursor is
    committed when it verifies, discarded when it does not. Roll back to
    the most recent valid checkpoint only when the chain itself is broken.
    """

    with _LOCK:
        record = must_load(store, run_id)
        rolled_back = 0
        redone = 0
        committed = False
        bad = journal.verify_chain(store, record)
        if bad >= 0:
            rolled_back = record.cursor - bad
            for index in range(bad, record.cursor + 1):
                journal.delete_checkpoint(store, record.id, index)
            record.cursor = bad
            record.status = journal.RUNNING
            record.stats["rolled_back"] = record.stats.get("rolled_back", 0) + rolled_back
            prev = journal.load_checkpoint(store, record.id, bad - 1) if bad > 0 else None
            record.final = prev.state if prev is not None else None
        dangling = journal.load_checkpoint(store, record.id, record.cursor)
        if dangling is not None:
            if _ckpt_valid(store, record, dangling):
                _commit(store, record, dangling)
                committed = True
            else:
                journal.delete_checkpoint(store, record.id, record.cursor)
                record.stats["redone"] = record.stats.get("redone", 0) + 1
                redone += 1
        if record.cursor >= int(record.config["steps"]) and bad < 0:
            record.status = journal.COMPLETED
        record.updated_at = _now()
        journal.save_record(store, record)
    _audit(
        store,
        AUDIT_RUN,
        f"run={run_id} event=recovered cursor={record.cursor} "
        f"rolled_back={rolled_back} redone={redone} committed_dangling={1 if committed else 0}",
    )
    return {
        "run": run_id,
        "status": record.status,
        "cursor": record.cursor,
        "chain_ok": bad < 0,
        "rolled_back": rolled_back,
        "redone": redone,
        "committed_dangling": committed,
    }


def rewind(store: Store, run_id: str, to_step: int) -> dict[str, object]:
    """Roll the cursor back to a chosen valid checkpoint and resume from it."""

    with _LOCK:
        record = must_load(store, run_id)
        if record.status == journal.ABORTED:
            raise ValueError(f"run {run_id} is aborted")
        to = _int("to", to_step, 0, record.cursor)
        if to == record.cursor:
            return {"run": run_id, "status": record.status, "cursor": record.cursor, "rewound": 0}
        if to > 0:
            prev = journal.load_checkpoint(store, record.id, to - 1)
            if prev is None or not _ckpt_valid(store, record, prev):
                raise ValueError(f"checkpoint chain not valid at {to}; run recover first")
        rewound = record.cursor - to
        for index in range(to, record.cursor + 1):
            journal.delete_checkpoint(store, record.id, index)
        record.stats["rolled_back"] = record.stats.get("rolled_back", 0) + rewound
        record.cursor = to
        record.status = journal.RUNNING
        prev = journal.load_checkpoint(store, record.id, to - 1) if to > 0 else None
        record.final = prev.state if prev is not None else None
        record.updated_at = _now()
        journal.save_record(store, record)
    _audit(store, AUDIT_RUN, f"run={run_id} event=rewound to={to} rewound={rewound}")
    return {"run": run_id, "status": record.status, "cursor": record.cursor, "rewound": rewound}


def _short(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 160 else f"{text[:157]}..."


def replay(store: Store, run_id: str, frm: int = 0, to: int | None = None) -> dict[str, object]:
    """Re-execute confirmed steps with their recorded inputs and diff.

    The recomputation chain is the source of truth: every checkpoint whose
    stored content diverges from what its recorded inputs imply is
    reported as a mismatch. The journal is never mutated.
    """

    with _LOCK:
        record = must_load(store, run_id)
        cursor = record.cursor
        start = _int("from", frm, 0, cursor)
        end = cursor if to is None else _int("to", to, start, cursor)
        state, prev_hash, base_dt = _context(record, store, start)
        mismatches: list[dict[str, object]] = []
        checked = 0
        for index in range(start, end):
            stored = journal.load_checkpoint(store, record.id, index)
            if stored is None:
                mismatches.append(
                    {"index": index, "field": "checkpoint", "expected": "present", "actual": "missing"}
                )
                break
            inputs = Inputs.from_dict(stored.inputs)
            expected = _compute(record, index, state, prev_hash, base_dt, inputs)
            for field_name in ("dt", "next_dt", "inputs", "state", "rationale", "prev_hash"):
                want = expected.content()[field_name]
                got = stored.content()[field_name]
                if want != got:
                    mismatches.append(
                        {
                            "index": index,
                            "field": field_name,
                            "expected": _short(want),
                            "actual": _short(got),
                        }
                    )
            if expected.hash != stored.hash:
                mismatches.append(
                    {
                        "index": index,
                        "field": "hash",
                        "expected": expected.hash[:16],
                        "actual": stored.hash[:16],
                    }
                )
            checked += 1
            state = State.from_dict(expected.state)
            prev_hash = expected.hash
            base_dt = expected.next_dt
        report = CompareReport(run=run_id, at=_now(), frm=start, to=end, checked=checked, mismatches=mismatches)
        journal.save_compare(store, report)
        record.stats["replay_runs"] = record.stats.get("replay_runs", 0) + 1
        record.updated_at = _now()
        journal.save_record(store, record)
    _audit(
        store,
        AUDIT_COMPARE,
        f"run={run_id} checked={checked} mismatches={len(mismatches)} ok={1 if not mismatches else 0}",
    )
    return report.as_dict()


def cancel(store: Store, run_id: str) -> dict[str, object]:
    """Abort a run; aborting twice is a no-op, aborting a finished run is not."""

    with _LOCK:
        record = must_load(store, run_id)
        if record.status == journal.COMPLETED:
            raise ValueError(f"run {run_id} is already completed")
        if record.status != journal.ABORTED:
            record.status = journal.ABORTED
            record.updated_at = _now()
            journal.save_record(store, record)
            _audit(store, AUDIT_RUN, f"run={run_id} event=cancelled cursor={record.cursor}")
    return record.as_dict()


def show(store: Store, run_id: str) -> dict[str, object]:
    """The control plane view of one run: record, chain health, tail, compare."""

    record = must_load(store, run_id)
    tail = []
    for index in range(max(0, record.cursor - CHECKPOINT_TAIL), record.cursor):
        ckpt = journal.load_checkpoint(store, run_id, index)
        if ckpt is not None:
            tail.append(ckpt.as_dict())
    return {
        "run": record.as_dict(),
        "chain_ok": journal.verify_chain(store, record) < 0,
        "checkpoints": tail,
        "compare": journal.load_compare(store, run_id),
    }


def get_checkpoint(store: Store, run_id: str, index: int) -> dict[str, object]:
    must_load(store, run_id)
    ckpt = journal.load_checkpoint(store, run_id, index)
    if ckpt is None:
        raise ValueError(f"no checkpoint {index} for run {run_id}")
    return ckpt.as_dict()
