"""Tests for the resumable deduction engine and its console routes."""

from __future__ import annotations

import json
import tempfile
import unittest

from coldstore.audit import Auditor
from coldstore.console.seed import seed_defaults
from coldstore.console.server import Server
from coldstore.deduce import (
    ADAPTIVE,
    AUTO,
    COMPLETED,
    CONFIRMED,
    MANUAL,
    PENDING,
    RETRY,
    ROLLBACK,
    RUNNING,
    SUPERSEDED,
    Engine,
    RunConfig,
)
from coldstore.store import Store


def make_config(run_id: str = "t1", **overrides) -> RunConfig:
    values = {
        "run_id": run_id,
        "steps": 5,
        "policy": "fixed",
        "step_size": 0.4,
        "min_step": 0.1,
        "max_step": 1.0,
        "start_temp": -18.0,
        "target_temp": None,
        "clock_seconds": 60,
        "start_at": 1000,
        "resume": RETRY,
        "confirm": AUTO,
        "advance_cap": 100,
        "advance_seconds": 2.0,
    }
    values.update(overrides)
    return RunConfig(**values)


class EngineCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store.open(f"{self._tmp.name}/state.json")
        self.auditor = Auditor(self.store)
        self.engine = Engine(self.store, self.auditor)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def kinds(self) -> list[str]:
        return [entry.kind for entry in self.auditor.entries()]

    def test_fixed_run_records_basis_and_completes(self) -> None:
        header, created = self.engine.start(make_config())
        self.assertTrue(created)
        self.assertEqual(header.cursor, -1)

        result = self.engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(result["completed"])
        self.assertEqual(result["confirmed"], 5)
        self.assertEqual(result["cursor"], 4)

        records = self.engine.steps("t1")
        self.assertEqual(len(records), 5)
        self.assertTrue(all(record.state == CONFIRMED for record in records))
        temps = [record.input["temp"] for record in records]
        self.assertEqual(temps, [-18.0, -17.6, -17.2, -16.8, -16.4])
        self.assertEqual(records[0].input["at"], 1000)
        self.assertEqual(records[1].input["at"], 1060)
        self.assertIn("policy=fixed", records[0].basis)
        self.assertIn("load=", records[0].basis)
        self.assertTrue(records[0].digest)
        self.assertEqual(records[0].output["next_temp"], -17.6)

        kinds = self.kinds()
        self.assertEqual(kinds.count("deduce-start"), 1)
        self.assertEqual(kinds.count("deduce-step"), 5)
        self.assertEqual(kinds.count("deduce-complete"), 1)

    def test_step_and_time_budgets_bound_each_advance(self) -> None:
        self.engine.start(make_config())
        first = self.engine.advance("t1", max_steps=2, max_seconds=10.0)
        self.assertEqual(first["executed"], 2)
        self.assertEqual(first["exhausted"], "steps")
        self.assertEqual(first["remaining"], 3)

        paused = self.engine.advance("t1", max_seconds=0.0)
        self.assertEqual(paused["executed"], 0)
        self.assertEqual(paused["exhausted"], "time")

        rest = self.engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(rest["completed"])
        self.assertGreaterEqual(rest["total_elapsed_ms"], rest["elapsed_ms"])

    def test_repeated_calls_are_idempotent(self) -> None:
        self.engine.start(make_config())
        header, created = self.engine.start(make_config())
        self.assertFalse(created)
        self.assertEqual(header.status, RUNNING)
        with self.assertRaises(ValueError):
            self.engine.start(make_config(steps=7))

        self.engine.advance("t1", max_steps=10, max_seconds=10.0)
        audits = self.auditor.count()
        again = self.engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertEqual(again["executed"], 0)
        self.assertTrue(again["completed"])
        self.assertEqual(self.auditor.count(), audits)
        self.assertEqual(len(self.engine.steps("t1")), 5)

    def test_manual_mode_waits_for_confirmation(self) -> None:
        self.engine.start(make_config(confirm=MANUAL, steps=2))
        first = self.engine.advance("t1")
        self.assertEqual(first["awaiting"], 0)
        self.assertEqual(first["confirmed"], 0)
        self.assertEqual(first["cursor"], -1)

        blocked = self.engine.advance("t1")
        self.assertEqual(blocked["executed"], 0)
        self.assertEqual(blocked["awaiting"], 0)

        header, record = self.engine.confirm("t1")
        self.assertEqual(record.index, 0)
        self.assertEqual(header.cursor, 0)
        self.assertEqual(header.status, RUNNING)

        second = self.engine.advance("t1")
        self.assertEqual(second["awaiting"], 1)
        header, _ = self.engine.confirm("t1")
        self.assertEqual(header.status, COMPLETED)
        with self.assertRaises(ValueError):
            self.engine.confirm("t1")
        self.assertIn("deduce-confirm", self.kinds())

    def test_retry_recovers_an_unconfirmed_step_in_place(self) -> None:
        engine = CrashBeforeConfirm(self.store, self.auditor)
        engine.start(make_config(steps=3, resume=RETRY))
        with self.assertRaises(RuntimeError):
            engine.advance("t1", max_steps=10, max_seconds=10.0)

        records = engine.steps("t1")
        self.assertEqual([r.state for r in records], [CONFIRMED, PENDING])

        resumed = engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(resumed["completed"])
        self.assertTrue(any(action.startswith("retry:1") for action in resumed["recovered"]))
        records = engine.steps("t1")
        self.assertEqual([r.state for r in records], [CONFIRMED] * 3)
        self.assertEqual([r.attempt for r in records], [1, 1, 1])
        self.assertEqual([r.input["temp"] for r in records], [-18.0, -17.6, -17.2])
        self.assertIn("deduce-recover", self.kinds())

    def test_rollback_steps_back_one_before_resuming(self) -> None:
        engine = CrashBeforeConfirm(self.store, self.auditor)
        engine.start(make_config(steps=3, resume=ROLLBACK))
        with self.assertRaises(RuntimeError):
            engine.advance("t1", max_steps=10, max_seconds=10.0)

        resumed = engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(resumed["completed"])
        self.assertTrue(any(action.startswith("rollback:") for action in resumed["recovered"]))

        records = engine.steps("t1")
        by_index: dict[int, list] = {}
        for record in records:
            by_index.setdefault(record.index, []).append(record)
        self.assertEqual(by_index[0][0].state, SUPERSEDED)
        self.assertEqual(by_index[0][0].note, "rollback")
        self.assertEqual(by_index[0][1].state, CONFIRMED)
        self.assertEqual(by_index[0][1].attempt, 2)
        self.assertEqual(by_index[1][0].state, SUPERSEDED)
        self.assertEqual(by_index[1][1].state, CONFIRMED)
        confirmed = [r for r in records if r.state == CONFIRMED]
        self.assertEqual([r.input["temp"] for r in confirmed], [-18.0, -17.6, -17.2])

        replay = engine.replay("t1")
        self.assertTrue(replay["ok"])
        self.assertEqual(replay["superseded"], 2)

    def test_confirmed_record_ahead_of_cursor_heals_forward(self) -> None:
        engine = CrashAfterRecordSave(self.store, self.auditor)
        engine.start(make_config(steps=3))
        with self.assertRaises(RuntimeError):
            engine.advance("t1", max_steps=10, max_seconds=10.0)

        records = engine.steps("t1")
        self.assertEqual(records[0].state, CONFIRMED)

        resumed = engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(resumed["completed"])
        self.assertTrue(any(action.startswith("heal:") for action in resumed["recovered"]))
        records = engine.steps("t1")
        self.assertEqual([r.attempt for r in records], [1, 1, 1])
        self.assertEqual(sum(1 for r in records if r.state == SUPERSEDED), 0)

    def test_rewind_resumes_from_any_checkpoint(self) -> None:
        self.engine.start(make_config())
        self.engine.advance("t1", max_steps=10, max_seconds=10.0)

        header, dropped = self.engine.rewind("t1", 1)
        self.assertEqual(dropped, 3)
        self.assertEqual(header.cursor, 1)
        self.assertEqual(header.status, RUNNING)
        self.assertEqual(header.temp, -17.2)

        result = self.engine.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(result["completed"])
        records = self.engine.steps("t1")
        confirmed = [r for r in records if r.state == CONFIRMED]
        self.assertEqual([r.input["temp"] for r in confirmed], [-18.0, -17.6, -17.2, -16.8, -16.4])
        self.assertEqual(sum(1 for r in records if r.state == SUPERSEDED), 3)
        self.assertTrue(self.engine.replay("t1")["ok"])

    def test_adaptive_policy_refines_steps_towards_target(self) -> None:
        self.engine.start(
            make_config(
                steps=20,
                policy=ADAPTIVE,
                start_temp=-23.0,
                target_temp=-17.0,
                step_size=0.4,
                min_step=0.1,
                max_step=1.0,
            )
        )
        result = self.engine.advance("t1", max_steps=25, max_seconds=10.0)
        self.assertTrue(result["completed"])
        records = self.engine.steps("t1")
        steps = [record.input["step"] for record in records]
        self.assertEqual(steps[0], 1.0)
        self.assertEqual(steps[1], 1.0)
        self.assertEqual(steps[2], 0.4)
        self.assertEqual(steps[10], 0.1)
        self.assertEqual(steps[-1], 0.0)
        self.assertEqual(records[-1].output["next_temp"], -17.0)
        self.assertIn("band=far", records[0].basis)
        self.assertIn("band=near", records[10].basis)
        self.assertIn("holding", records[-1].basis)

    def test_replay_detects_tampered_records(self) -> None:
        self.engine.start(make_config())
        self.engine.advance("t1", max_steps=10, max_seconds=10.0)

        key = "deduce:run:t1:steps"
        raw, present = self.store.get(key)
        self.assertTrue(present)
        packed = [json.loads(item) for item in json.loads(raw)]
        packed[1]["output"]["load"] = 999.0
        self.store.put(key, json.dumps([json.dumps(item) for item in packed]))

        report = self.engine.replay("t1")
        self.assertFalse(report["ok"])
        kinds = {mismatch["kind"] for mismatch in report["mismatches"]}
        self.assertIn("output", kinds)
        self.assertIn("digest", kinds)
        self.assertEqual(report["matched"], 4)

        status = self.engine.status("t1")
        self.assertIsNotNone(status["replay"])
        self.assertFalse(status["replay"]["ok"])
        self.assertIn("deduce-replay", self.kinds())

    def test_abort_is_terminal_and_keeps_the_journal(self) -> None:
        self.engine.start(make_config())
        self.engine.advance("t1", max_steps=2, max_seconds=10.0)
        header = self.engine.abort("t1")
        self.assertEqual(header.status, "aborted")
        header = self.engine.abort("t1")
        self.assertEqual(header.status, "aborted")
        with self.assertRaises(ValueError):
            self.engine.advance("t1")
        self.assertEqual(len(self.engine.steps("t1")), 2)
        self.assertIn("deduce-abort", self.kinds())

    def test_unknown_run_and_invalid_config_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.advance("ghost")
        with self.assertRaises(ValueError):
            self.engine.start(make_config(steps=0))
        with self.assertRaises(ValueError):
            self.engine.start(make_config(policy="adaptive"))
        with self.assertRaises(ValueError):
            self.engine.start(make_config(resume="rewind"))
        with self.assertRaises(ValueError):
            self.engine.start(make_config(run_id="bad id!"))

    def test_fresh_engine_resumes_from_the_store_after_restart(self) -> None:
        self.engine.start(make_config(steps=3))
        self.engine.advance("t1", max_steps=1, max_seconds=10.0)
        key = "deduce:run:t1:steps"
        packed = [json.loads(item) for item in json.loads(self.store.get(key)[0])]
        orphan = dict(packed[0])
        orphan["index"] = 1
        orphan["state"] = PENDING
        orphan["confirmed_at"] = ""
        orphan["input"] = dict(orphan["input"], temp=-17.6, at=1060)
        packed.append(orphan)
        self.store.put(key, json.dumps([json.dumps(item) for item in packed]))

        restarted = Engine(self.store, self.auditor)
        result = restarted.advance("t1", max_steps=10, max_seconds=10.0)
        self.assertTrue(result["completed"])
        self.assertTrue(any(action.startswith("retry:1") for action in result["recovered"]))
        records = restarted.steps("t1")
        self.assertEqual([r.state for r in records], [CONFIRMED] * 3)
        self.assertTrue(restarted.replay("t1")["ok"])


class CrashBeforeConfirm(Engine):
    """Fault injection: die after the pending record is persisted."""

    def __init__(self, store, auditor) -> None:
        super().__init__(store, auditor)
        self.crashed = False

    def _confirm(self, header, record, records) -> None:
        if not self.crashed and record.index == 1:
            self.crashed = True
            raise RuntimeError("simulated crash before confirm")
        super()._confirm(header, record, records)


class CrashAfterRecordSave(Engine):
    """Fault injection: die between the journal write and the header write."""

    def __init__(self, store, auditor) -> None:
        super().__init__(store, auditor)
        self.crashed = False

    def _save_header(self, header) -> None:
        if not self.crashed and header.cursor == 0:
            self.crashed = True
            raise RuntimeError("simulated crash after record save")
        super()._save_header(header)


class DeduceHttpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store.open(f"{self._tmp.name}/state.json")
        seed_defaults(self.store)
        self.server = Server(self.store)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def call(self, method: str, path: str, payload: dict | None = None):
        response = self.server.dispatch(method, path, payload)
        body = response.body.decode("utf-8")
        parsed = json.loads(body) if response.content_type.startswith("application/json") else body
        return response.status, parsed

    def test_run_lifecycle_over_http(self) -> None:
        status, body = self.call("POST", "/deduce/start", {"run": "r1", "steps": 3})
        self.assertEqual(status, 200)
        self.assertTrue(body["created"])
        self.assertEqual(body["config"]["policy"], "fixed")

        status, body = self.call("POST", "/deduce/start", {"run": "r1", "steps": 3})
        self.assertFalse(body["created"])
        status, error = self.call("POST", "/deduce/start", {"run": "r1", "steps": 9})
        self.assertEqual(status, 400)

        status, body = self.call("POST", "/deduce/advance", {"run": "r1", "max_steps": 2})
        self.assertEqual(body["executed"], 2)
        self.assertEqual(body["exhausted"], "steps")
        status, body = self.call("POST", "/deduce/advance", {"run": "r1"})
        self.assertTrue(body["completed"])

        status, body = self.call("GET", "/deduce")
        self.assertEqual(status, 200)
        self.assertIn("r1", [run["run"] for run in body["runs"]])

        status, body = self.call("POST", "/deduce/status", {"run": "r1"})
        self.assertEqual(body["cursor"], 2)
        self.assertEqual(body["journal"]["confirmed"], 3)
        self.assertIsNone(body["pending"])

        status, body = self.call("POST", "/deduce/steps", {"run": "r1"})
        self.assertEqual(body["count"], 3)
        first = body["steps"][0]
        self.assertEqual(first["input"]["temp"], -18.0)
        self.assertIn("policy=fixed", first["basis"])
        self.assertEqual(first["context"]["running"], 0)

        status, body = self.call("POST", "/deduce/replay", {"run": "r1"})
        self.assertTrue(body["ok"])
        self.assertEqual(body["matched"], 3)

        status, body = self.call("POST", "/deduce/rewind", {"run": "r1", "step": 0})
        self.assertEqual(body["superseded"], 2)
        self.assertEqual(body["status"], "running")
        status, body = self.call("POST", "/deduce/advance", {"run": "r1"})
        self.assertTrue(body["completed"])

        status, summary = self.call("GET", "/audit/summary")
        for kind in ("deduce-start", "deduce-step", "deduce-complete", "deduce-rewind", "deduce-replay"):
            self.assertIn(kind, summary["by_kind"])

    def test_manual_confirmation_over_http(self) -> None:
        self.call("POST", "/deduce/start", {"run": "m1", "steps": 2, "confirm": "manual"})
        status, body = self.call("POST", "/deduce/advance", {"run": "m1"})
        self.assertEqual(body["awaiting"], 0)
        status, body = self.call("POST", "/deduce/status", {"run": "m1"})
        self.assertEqual(body["pending"]["index"], 0)
        status, body = self.call("POST", "/deduce/confirm", {"run": "m1"})
        self.assertEqual(body["confirmed"], 0)
        status, body = self.call("POST", "/deduce/advance", {"run": "m1"})
        self.assertEqual(body["awaiting"], 1)
        status, body = self.call("POST", "/deduce/confirm", {"run": "m1"})
        self.assertEqual(body["status"], "completed")

    def test_abort_and_errors_over_http(self) -> None:
        self.call("POST", "/deduce/start", {"run": "r2", "steps": 3})
        status, body = self.call("POST", "/deduce/abort", {"run": "r2"})
        self.assertEqual(body["status"], "aborted")
        status, error = self.call("POST", "/deduce/advance", {"run": "r2"})
        self.assertEqual(status, 400)
        self.assertIn("aborted", error["error"])

        status, error = self.call("POST", "/deduce/status", {"run": "ghost"})
        self.assertEqual(status, 400)
        self.assertIn("unknown", error["error"])
        status, error = self.call("POST", "/deduce/advance", {})
        self.assertEqual(status, 400)
        self.assertIn("run is required", error["error"])
        status, error = self.call("POST", "/deduce/start", {"run": "bad", "policy": "adaptive"})
        self.assertEqual(status, 400)
        self.assertIn("target_temp", error["error"])

    def test_adaptive_run_over_http(self) -> None:
        status, body = self.call(
            "POST",
            "/deduce/start",
            {
                "run": "a1",
                "steps": 20,
                "policy": "adaptive",
                "start_temp": -23.0,
                "target_temp": -17.0,
                "step_size": 0.4,
                "min_step": 0.1,
                "max_step": 1.0,
            },
        )
        self.assertEqual(status, 200)
        status, body = self.call("POST", "/deduce/advance", {"run": "a1"})
        self.assertTrue(body["completed"])
        status, body = self.call("POST", "/deduce/steps", {"run": "a1"})
        steps = body["steps"]
        self.assertEqual(steps[0]["input"]["step"], 1.0)
        self.assertEqual(steps[-1]["output"]["next_temp"], -17.0)
        status, body = self.call("POST", "/deduce/replay", {"run": "a1"})
        self.assertTrue(body["ok"])

    def test_simulate_keeps_its_legacy_shape(self) -> None:
        status, body = self.call("POST", "/simulate", {"ticks": 3})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["ticks"]), 3)
        self.assertEqual([tick["temp"] for tick in body["ticks"]], [-18.0, -17.6, -17.2])
        self.assertEqual(body["last_stage"], "audit")
        status, body = self.call("GET", "/deduce")
        self.assertTrue(any(run["run"].startswith("simulate-") for run in body["runs"]))


if __name__ == "__main__":
    unittest.main()
