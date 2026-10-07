"""Resumable deduction engine.

A run advances one checkpoint at a time: every step is first persisted as
pending with its inputs and basis, and the frontier only moves once the step
is confirmed. A crash therefore leaves at most one unconfirmed record, which
the configured resume policy either retries in place or rolls back one step.
Re-executing a step recomputes its outputs from the recorded inputs, so
repeating any call is idempotent.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Callable

from coldstore.load import LoadCalculator
from coldstore.load.calc import HIGH_LOAD, compute_load
from coldstore.store.commands import append_command, load_commands
from coldstore.store.store import Store
from coldstore.temp import Collector

from .policy import (
    AUTO,
    MANUAL,
    MAX_ADVANCE_CAP,
    MAX_ADVANCE_SECONDS,
    RETRY,
    choose_step,
    validate_config,
    validate_run_id,
)
from .records import (
    ABORTED,
    COMPLETED,
    CONFIRMED,
    DEFAULT_CONTEXT,
    PENDING,
    RUNNING,
    SUPERSEDED,
    RunConfig,
    RunHeader,
    StepRecord,
    now_iso,
    record_digest,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from coldstore.audit import Auditor

RUNS_KEY = "deduce:runs"
RUN_PREFIX = "deduce:run:"
MAX_MISMATCHES = 20

Probe = Callable[[], dict[str, object]]


def evaluate(
    config: RunConfig, step_input: dict[str, object]
) -> tuple[dict[str, object], str, float]:
    """Derive outputs and a human readable basis from one recorded input.

    Pure function of the config and the input, which is what makes replay and
    in-place retries deterministic. Returns the output mapping, the basis
    text and the step the policy would choose for the recorded temperature.
    """

    temp = float(step_input["temp"])
    baseline = float(step_input["baseline"])
    coefficient = float(step_input["coefficient"])
    window = float(step_input["window"])
    step = round(float(step_input["step"]), 4)
    expected, reason = choose_step(config, temp)
    next_temp = round(temp + step, 4)
    load = compute_load(temp, baseline, coefficient)
    high = load >= HIGH_LOAD
    rate = step / config.clock_seconds if config.clock_seconds > 0 else 0.0
    within = abs(rate) <= window
    output: dict[str, object] = {
        "load": load,
        "high": high,
        "rate": rate,
        "within": within,
        "next_temp": next_temp,
    }
    basis = (
        f"{reason}; load=({temp:.4f}-({baseline:.4f}))*{coefficient:.4f}={load:.4f}"
        f" vs {HIGH_LOAD:.4f} -> {'high' if high else 'normal'};"
        f" rate={rate:.6f} {'within' if within else 'outside'} window={window:.4f};"
        f" next={next_temp:.4f}"
    )
    return output, basis, expected


class Engine:
    """Journals deduction runs into the store and resumes them safely."""

    def __init__(self, store: Store, auditor: "Auditor | None" = None) -> None:
        self._store = store
        self._auditor = auditor
        self._lock = threading.RLock()

    # -- persistence -----------------------------------------------------

    def _meta_key(self, run_id: str) -> str:
        return f"{RUN_PREFIX}{run_id}:meta"

    def _steps_key(self, run_id: str) -> str:
        return f"{RUN_PREFIX}{run_id}:steps"

    def _replay_key(self, run_id: str) -> str:
        return f"{RUN_PREFIX}{run_id}:replay"

    def _save_header(self, header: RunHeader) -> None:
        header.updated_at = now_iso()
        self._store.put(
            self._meta_key(header.config.run_id),
            json.dumps(header.as_dict(), ensure_ascii=False),
        )

    def _load_header(self, run_id: str) -> RunHeader | None:
        raw, present = self._store.get(self._meta_key(run_id))
        if not present:
            return None
        try:
            return RunHeader.from_dict(json.loads(raw))
        except (ValueError, TypeError, KeyError):
            return None

    def _require(self, run_id: str) -> RunHeader:
        header = self._load_header(run_id)
        if header is None:
            raise ValueError(f"unknown deduction run {run_id}")
        return header

    def _load_records(self, run_id: str) -> list[StepRecord]:
        records: list[StepRecord] = []
        for item in load_commands(self._store, self._steps_key(run_id)):
            try:
                payload = json.loads(item)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            records.append(StepRecord.from_dict(payload))
        return records

    def _save_records(self, run_id: str, records: list[StepRecord]) -> None:
        packed = [json.dumps(record.as_dict(), ensure_ascii=False) for record in records]
        self._store.put(self._steps_key(run_id), json.dumps(packed, ensure_ascii=False))

    def _audit(self, kind: str, detail: str) -> None:
        if self._auditor is not None:
            self._auditor.record(kind, detail)

    # -- run lifecycle ---------------------------------------------------

    def start(self, config: RunConfig) -> tuple[RunHeader, bool]:
        """Create a run; restarting with the same config returns the original."""

        validate_run_id(config.run_id)
        validate_config(config)
        with self._lock:
            existing = self._load_header(config.run_id)
            if existing is not None:
                if existing.config.fingerprint() != config.fingerprint():
                    raise ValueError(
                        f"run {config.run_id} already exists with a different config"
                    )
                return existing, False
            header = RunHeader.create(config)
            self._save_header(header)
            append_command(self._store, RUNS_KEY, config.run_id)
            self._audit(
                "deduce-start",
                f"run={config.run_id} steps={config.steps} policy={config.policy}"
                f" resume={config.resume} confirm={config.confirm}",
            )
            return header, True

    def advance(
        self,
        run_id: str,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        probe: Probe | None = None,
    ) -> dict[str, object]:
        """Move the frontier forward inside one step and time budget."""

        with self._lock:
            header = self._require(run_id)
            if header.status == ABORTED:
                raise ValueError(f"run {run_id} is aborted")
            config = header.config
            records = self._load_records(run_id)
            lap = time.monotonic()
            recovered = self._recover(header, records)
            recover_ms = (time.monotonic() - lap) * 1000.0
            cap = max(1, min(max_steps or config.advance_cap, MAX_ADVANCE_CAP))
            budget = max(
                0.0,
                min(
                    max_seconds if max_seconds is not None else config.advance_seconds,
                    MAX_ADVANCE_SECONDS,
                ),
            )
            started = time.monotonic()
            executed = 0
            confirmed = 0
            awaiting = -1
            exhausted = ""
            while header.status != COMPLETED:
                if executed >= cap:
                    exhausted = "steps"
                    break
                if time.monotonic() - started >= budget:
                    exhausted = "time"
                    break
                pending = self._pending(records)
                if pending is not None:
                    awaiting = pending.index
                    break
                step_started = time.monotonic()
                record = self._execute_pending(header, records, probe)
                executed += 1
                if config.confirm == MANUAL:
                    awaiting = record.index
                    header.elapsed_ms += (time.monotonic() - step_started) * 1000.0
                    break
                self._confirm(header, record, records)
                confirmed += 1
                header.elapsed_ms += (time.monotonic() - step_started) * 1000.0
            header.elapsed_ms += recover_ms
            elapsed = (time.monotonic() - started) * 1000.0 + recover_ms
            if recovered or executed:
                self._save_header(header)
            return {
                "run": run_id,
                "status": header.status,
                "cursor": header.cursor,
                "executed": executed,
                "confirmed": confirmed,
                "awaiting": awaiting,
                "remaining": config.steps - 1 - header.cursor,
                "exhausted": exhausted,
                "recovered": recovered,
                "elapsed_ms": round(elapsed, 3),
                "total_elapsed_ms": round(header.elapsed_ms, 3),
                "completed": header.status == COMPLETED,
            }

    def confirm(self, run_id: str) -> tuple[RunHeader, StepRecord]:
        """Confirm the pending step of a manual run; nothing else advances it."""

        with self._lock:
            header = self._require(run_id)
            if header.status == ABORTED:
                raise ValueError(f"run {run_id} is aborted")
            if header.status == COMPLETED:
                raise ValueError(f"run {run_id} is already complete")
            if header.config.confirm != MANUAL:
                raise ValueError(f"run {run_id} confirms steps automatically")
            records = self._load_records(run_id)
            self._recover(header, records)
            pending = self._pending(records)
            if pending is None:
                raise ValueError(f"run {run_id} has no pending step to confirm")
            lap = time.monotonic()
            self._confirm(header, pending, records)
            header.elapsed_ms += (time.monotonic() - lap) * 1000.0
            self._save_header(header)
            self._audit("deduce-confirm", f"run={run_id} step={pending.index}")
            return header, pending

    def rewind(self, run_id: str, to_cursor: int) -> tuple[RunHeader, int]:
        """Roll the frontier back to ``to_cursor`` so the run resumes from there.

        Later records are kept in the journal as superseded, never deleted.
        """

        with self._lock:
            header = self._require(run_id)
            if header.status == ABORTED:
                raise ValueError(f"run {run_id} is aborted")
            if to_cursor < -1 or to_cursor > header.cursor:
                raise ValueError(f"step must be between -1 and {header.cursor}")
            records = self._load_records(run_id)
            dropped = 0
            for record in records:
                if record.index > to_cursor and record.state != SUPERSEDED:
                    self._supersede(header, record, "rewind")
                    dropped += 1
            self._frontier(header, records, to_cursor)
            if dropped and header.status == COMPLETED:
                header.status = RUNNING
            if dropped:
                self._save_records(run_id, records)
                self._save_header(header)
                self._audit(
                    "deduce-rewind", f"run={run_id} to={to_cursor} superseded={dropped}"
                )
            return header, dropped

    def abort(self, run_id: str) -> RunHeader:
        """Stop a run for good while keeping its journal for audit."""

        with self._lock:
            header = self._require(run_id)
            if header.status == COMPLETED:
                raise ValueError(f"run {run_id} is already complete")
            if header.status != ABORTED:
                header.status = ABORTED
                self._save_header(header)
                self._audit("deduce-abort", f"run={run_id} cursor={header.cursor}")
            return header

    # -- inspection ------------------------------------------------------

    def run_ids(self) -> list[str]:
        """Every known run id, healing index entries lost to a crash."""

        ids = list(load_commands(self._store, RUNS_KEY))
        for key in self._store.keys():
            if key.startswith(RUN_PREFIX) and key.endswith(":meta"):
                run_id = key[len(RUN_PREFIX) : -len(":meta")]
                if run_id not in ids:
                    ids.append(run_id)
        return ids

    def runs(self) -> list[dict[str, object]]:
        with self._lock:
            summaries: list[dict[str, object]] = []
            for run_id in self.run_ids():
                header = self._load_header(run_id)
                if header is None:
                    continue
                summaries.append(
                    {
                        "run": run_id,
                        "status": header.status,
                        "cursor": header.cursor,
                        "steps": header.config.steps,
                        "confirmed": header.confirmed,
                        "created_at": header.created_at,
                        "elapsed_ms": round(header.elapsed_ms, 3),
                    }
                )
            return summaries

    def steps(self, run_id: str) -> list[StepRecord]:
        with self._lock:
            self._require(run_id)
            return self._load_records(run_id)

    def status(self, run_id: str) -> dict[str, object]:
        with self._lock:
            header = self._require(run_id)
            records = self._load_records(run_id)
            pending = next((r for r in records if r.state == PENDING), None)
            confirmed = [r for r in records if r.state == CONFIRMED]
            payload = header.as_dict()
            payload["pending"] = pending.as_dict() if pending else None
            payload["recent"] = [record.as_dict() for record in confirmed[-5:]]
            payload["journal"] = {
                "records": len(records),
                "confirmed": len(confirmed),
                "superseded": sum(1 for r in records if r.state == SUPERSEDED),
            }
            raw, present = self._store.get(self._replay_key(run_id))
            replay: dict[str, object] | None = None
            if present:
                try:
                    loaded = json.loads(raw)
                    replay = loaded if isinstance(loaded, dict) else None
                except ValueError:
                    replay = None
            payload["replay"] = replay
            return payload

    def replay(self, run_id: str) -> dict[str, object]:
        """Recompute every confirmed step from its recorded input and compare.

        The report is persisted and audited so the comparison itself leaves a
        trail; it never mutates the journal.
        """

        with self._lock:
            header = self._require(run_id)
            records = self._load_records(run_id)
            confirmed = [r for r in records if r.state == CONFIRMED]
            mismatches: list[dict[str, object]] = []
            for record in confirmed:
                output, _basis, expected = evaluate(header.config, record.input)
                if expected != round(float(record.input["step"]), 4):
                    mismatches.append(
                        {
                            "index": record.index,
                            "attempt": record.attempt,
                            "kind": "step",
                            "recorded": record.input["step"],
                            "replayed": expected,
                        }
                    )
                for field in ("load", "high", "rate", "within", "next_temp"):
                    if output[field] != record.output.get(field):
                        mismatches.append(
                            {
                                "index": record.index,
                                "attempt": record.attempt,
                                "kind": "output",
                                "field": field,
                                "recorded": record.output.get(field),
                                "replayed": output[field],
                            }
                        )
                digest = record_digest(
                    run_id, record.index, record.attempt, record.input, record.output
                )
                if digest != record.digest:
                    mismatches.append(
                        {
                            "index": record.index,
                            "attempt": record.attempt,
                            "kind": "digest",
                            "recorded": record.digest,
                            "replayed": digest,
                        }
                    )
            drifted = {(m["index"], m["attempt"]) for m in mismatches}
            matched = len(confirmed) - len(drifted)
            report: dict[str, object] = {
                "run": run_id,
                "at": now_iso(),
                "steps": len(confirmed),
                "superseded": sum(1 for r in records if r.state == SUPERSEDED),
                "matched": matched,
                "mismatch_count": len(mismatches),
                "mismatches": mismatches[:MAX_MISMATCHES],
                "ok": not mismatches,
            }
            self._store.put(self._replay_key(run_id), json.dumps(report, ensure_ascii=False))
            self._audit("deduce-replay", f"run={run_id} matched={matched}/{len(confirmed)}")
            return report

    # -- step execution --------------------------------------------------

    def _pending(self, records: list[StepRecord]) -> StepRecord | None:
        pending = [r for r in records if r.state == PENDING]
        return min(pending, key=lambda r: r.index) if pending else None

    def _execute_pending(
        self, header: RunHeader, records: list[StepRecord], probe: Probe | None
    ) -> StepRecord:
        """Compute one step and persist it as pending before anything confirms it."""

        config = header.config
        index = header.cursor + 1
        attempt = 1 + max((r.attempt for r in records if r.index == index), default=0)
        step, _reason = choose_step(config, header.temp)
        calculator = LoadCalculator(self._store)
        collector = Collector(self._store)
        step_input: dict[str, object] = {
            "temp": round(header.temp, 4),
            "at": header.at,
            "step": step,
            "baseline": calculator.baseline(),
            "coefficient": calculator.coefficient(),
            "window": collector.window(),
        }
        output, basis, _expected = evaluate(config, step_input)
        context = dict(DEFAULT_CONTEXT)
        if probe is not None:
            context.update(probe())
        record = StepRecord(
            run_id=config.run_id,
            index=index,
            attempt=attempt,
            state=PENDING,
            input=step_input,
            context=context,
            output=output,
            basis=basis,
            digest=record_digest(config.run_id, index, attempt, step_input, output),
            recorded_at=now_iso(),
        )
        records.append(record)
        self._save_records(config.run_id, records)
        return record

    def _confirm(
        self, header: RunHeader, record: StepRecord, records: list[StepRecord]
    ) -> None:
        """Mark a step confirmed, then move the frontier past it.

        The journal is saved before the header so a crash in between heals
        forward on the next resume instead of losing a confirmed step.
        """

        record.state = CONFIRMED
        record.confirmed_at = now_iso()
        self._save_records(header.config.run_id, records)
        header.cursor = record.index
        header.temp = float(record.output["next_temp"])
        header.at = int(record.input["at"]) + header.config.clock_seconds
        header.confirmed += 1
        if header.cursor == header.config.steps - 1:
            header.status = COMPLETED
        self._save_header(header)
        self._audit(
            "deduce-step",
            f"run={header.config.run_id} step={record.index} attempt={record.attempt}"
            f" temp={float(record.input['temp']):.4f} load={float(record.output['load']):.4f}",
        )
        if header.status == COMPLETED:
            self._audit(
                "deduce-complete",
                f"run={header.config.run_id} steps={header.config.steps}"
                f" elapsed_ms={header.elapsed_ms:.1f}",
            )

    # -- recovery --------------------------------------------------------

    def _recover(self, header: RunHeader, records: list[StepRecord]) -> list[str]:
        """Reconcile the frontier with the journal after an interruption.

        A confirmed record ahead of the cursor means the crash happened
        between the journal write and the header write: heal the cursor. A
        pending record in an auto-confirm run means the crash happened mid
        step: retry it in place or roll back one step, per the run's policy.
        Manual runs keep their pending step for the operator to confirm.
        """

        config = header.config
        actions: list[str] = []
        last = max((r.index for r in records if r.state == CONFIRMED), default=-1)
        if header.cursor != last:
            self._frontier(header, records, last)
            actions.append(f"heal:cursor={last}")
        pending = sorted((r for r in records if r.state == PENDING), key=lambda r: r.index)
        if pending and config.confirm == AUTO:
            if config.resume == RETRY:
                for record in pending:
                    self._reexecute(header, record)
                    self._confirm(header, record, records)
                actions.append("retry:" + ",".join(str(r.index) for r in pending))
            else:
                for record in pending:
                    self._supersede(header, record, "rollback")
                lowest = pending[0].index
                for record in records:
                    if record.state == CONFIRMED and record.index >= lowest - 1:
                        self._supersede(header, record, "rollback")
                self._frontier(header, records, lowest - 2)
                actions.append(f"rollback:to={lowest - 2}")
        if actions:
            header.recoveries += len(actions)
            self._save_records(config.run_id, records)
            self._save_header(header)
            self._audit(
                "deduce-recover", f"run={config.run_id} actions={';'.join(actions)}"
            )
        return actions

    def _reexecute(self, header: RunHeader, record: StepRecord) -> None:
        """Recompute a pending step in place from its recorded input."""

        output, basis, _expected = evaluate(header.config, record.input)
        record.output = output
        record.basis = basis
        record.digest = record_digest(
            record.run_id, record.index, record.attempt, record.input, output
        )

    def _supersede(self, header: RunHeader, record: StepRecord, note: str) -> None:
        record.state = SUPERSEDED
        record.note = note
        header.superseded += 1

    def _frontier(
        self, header: RunHeader, records: list[StepRecord], to_cursor: int
    ) -> None:
        """Place the cursor and rebuild the simulated temperature and clock."""

        confirmed = {r.index: r for r in records if r.state == CONFIRMED}
        cursor = to_cursor
        while cursor >= 0 and cursor not in confirmed:
            cursor -= 1
        header.cursor = cursor
        if cursor >= 0:
            record = confirmed[cursor]
            header.temp = float(record.output["next_temp"])
            header.at = int(record.input["at"]) + header.config.clock_seconds
        else:
            header.temp = header.config.start_temp
            header.at = header.config.start_at
