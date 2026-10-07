"""Persistence for runs: records, checkpoints and comparison reports.

Layout inside the shared store document:

sim:runs                        list of run ids
sim:run:<id>                    the run record (config, cursor, status)
sim:run:<id>:ckpt:<index>       one confirmed checkpoint per step
sim:run:<id>:compare            the latest replay comparison

Each checkpoint is written under its own key with an atomic put, so a
crash can leave at most one dangling checkpoint at the cursor position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from coldstore.store.store import Store

RUNS_KEY = "sim:runs"
RUNNING = "running"
COMPLETED = "completed"
ABORTED = "aborted"


def _run_key(run_id: str) -> str:
    return f"sim:run:{run_id}"


def _ckpt_key(run_id: str, index: int) -> str:
    return f"sim:run:{run_id}:ckpt:{index:06d}"


def _compare_key(run_id: str) -> str:
    return f"sim:run:{run_id}:compare"


def _canonical(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def genesis_hash(config: dict[str, object]) -> str:
    """Content address of the run config; anchors the checkpoint chain."""

    return hashlib.sha256(f"coldstore-sim:{_canonical(config)}".encode("utf-8")).hexdigest()


def checkpoint_hash(prev_hash: str, content: dict[str, object]) -> str:
    return hashlib.sha256(f"{prev_hash}|{_canonical(content)}".encode("utf-8")).hexdigest()


@dataclass
class RunRecord:
    """The mutable head of a run: config, cursor and lifecycle status."""

    id: str
    created_at: str
    updated_at: str
    status: str
    config: dict[str, object]
    cursor: int
    stats: dict[str, int]
    final: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "config": self.config,
            "cursor": self.cursor,
            "stats": dict(self.stats),
            "final": self.final,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RunRecord":
        return cls(
            id=str(payload.get("id", "")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            status=str(payload.get("status", RUNNING)),
            config=dict(payload.get("config", {})),
            cursor=int(payload.get("cursor", 0)),
            stats={str(k): int(v) for k, v in dict(payload.get("stats", {})).items()},
            final=payload.get("final") if isinstance(payload.get("final"), dict) else None,
        )

    def summary(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "cursor": self.cursor,
            "steps": self.config.get("steps", 0),
            "policy": self.config.get("policy", ""),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class Checkpoint:
    """One confirmed step: inputs, basis, outcome and the chain hash."""

    run: str
    index: int
    dt: float
    next_dt: float
    inputs: dict[str, object]
    state: dict[str, object]
    rationale: dict[str, object]
    prev_hash: str
    hash: str
    recorded_at: str

    def content(self) -> dict[str, object]:
        """The hashed payload: everything except the hash and wall clock."""

        return {
            "run": self.run,
            "index": self.index,
            "dt": self.dt,
            "next_dt": self.next_dt,
            "inputs": self.inputs,
            "state": self.state,
            "rationale": self.rationale,
            "prev_hash": self.prev_hash,
        }

    def as_dict(self) -> dict[str, object]:
        payload = self.content()
        payload["hash"] = self.hash
        payload["recorded_at"] = self.recorded_at
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Checkpoint":
        return cls(
            run=str(payload["run"]),
            index=int(payload["index"]),
            dt=float(payload["dt"]),
            next_dt=float(payload["next_dt"]),
            inputs=dict(payload["inputs"]),
            state=dict(payload["state"]),
            rationale=dict(payload["rationale"]),
            prev_hash=str(payload["prev_hash"]),
            hash=str(payload["hash"]),
            recorded_at=str(payload.get("recorded_at", "")),
        )


@dataclass(frozen=True)
class CompareReport:
    """The outcome of replaying recorded inputs against the journal."""

    run: str
    at: str
    frm: int
    to: int
    checked: int
    mismatches: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run,
            "at": self.at,
            "from": self.frm,
            "to": self.to,
            "checked": self.checked,
            "mismatches": self.mismatches,
            "ok": not self.mismatches,
        }


def run_ids(store: Store) -> list[str]:
    raw, present = store.get(RUNS_KEY)
    if not present:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(items, list):
        return []
    return [str(item) for item in items]


def register_run(store: Store, run_id: str) -> None:
    ids = run_ids(store)
    if run_id not in ids:
        ids.append(run_id)
        store.put(RUNS_KEY, json.dumps(ids, ensure_ascii=False))


def save_record(store: Store, record: RunRecord) -> None:
    store.put(_run_key(record.id), json.dumps(record.as_dict(), ensure_ascii=False))


def load_record(store: Store, run_id: str) -> RunRecord | None:
    raw, present = store.get(_run_key(run_id))
    if not present:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return RunRecord.from_dict(payload)
    except (TypeError, ValueError, KeyError):
        return None


def save_checkpoint(store: Store, ckpt: Checkpoint) -> None:
    store.put(_ckpt_key(ckpt.run, ckpt.index), json.dumps(ckpt.as_dict(), ensure_ascii=False))


def load_checkpoint(store: Store, run_id: str, index: int) -> Checkpoint | None:
    raw, present = store.get(_ckpt_key(run_id, index))
    if not present:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return Checkpoint.from_dict(payload)
    except (TypeError, ValueError, KeyError):
        return None


def delete_checkpoint(store: Store, run_id: str, index: int) -> None:
    store.delete(_ckpt_key(run_id, index))


def save_compare(store: Store, report: CompareReport) -> None:
    store.put(_compare_key(report.run), json.dumps(report.as_dict(), ensure_ascii=False))


def load_compare(store: Store, run_id: str) -> dict[str, object] | None:
    raw, present = store.get(_compare_key(run_id))
    if not present:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def verify_chain(store: Store, record: RunRecord) -> int:
    """Return the first invalid checkpoint index below the cursor, or -1."""

    prev_hash = genesis_hash(record.config)
    for index in range(record.cursor):
        ckpt = load_checkpoint(store, record.id, index)
        if ckpt is None:
            return index
        if ckpt.prev_hash != prev_hash:
            return index
        if ckpt.hash != checkpoint_hash(prev_hash, ckpt.content()):
            return index
        prev_hash = ckpt.hash
    return -1
