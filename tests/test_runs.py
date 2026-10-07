"""End to end tests for the resumable simulation runs."""

from __future__ import annotations

import json
import tempfile
import unittest

from coldstore.console.seed import seed_defaults
from coldstore.console.server import Server
from coldstore.sim import engine, journal
from coldstore.store import Store


class RunsCase(unittest.TestCase):
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

    def create(self, run_id: str = "", **config):
        payload: dict = {"id": run_id, **config}
        return self.call("POST", "/runs", payload)

    def forge_dangling(self, run_id: str, overrides: dict | None = None):
        """Simulate a crash: a checkpoint persisted but never confirmed."""

        record = engine.must_load(self.store, run_id)
        ckpt = engine._prepare(self.store, record, overrides)
        journal.save_checkpoint(self.store, ckpt)
        return ckpt

    def tamper(self, run_id: str, index: int, field: str, value) -> None:
        key = f"sim:run:{run_id}:ckpt:{index:06d}"
        raw, present = self.store.get(key)
        self.assertTrue(present)
        payload = json.loads(raw)
        payload[field] = value
        self.store.put(key, json.dumps(payload, ensure_ascii=False))

    def finish(self, run_id: str) -> dict:
        status, body = self.call("POST", "/runs/advance", {"id": run_id, "budget_ms": 5000})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "completed")
        return body

    def test_create_is_idempotent_and_validated(self) -> None:
        status, body = self.create("r1", steps=4)
        self.assertEqual(status, 200)
        self.assertTrue(body["created"])
        self.assertEqual(body["config"]["policy"], "fixed")
        self.assertEqual(body["cursor"], 0)

        status, again = self.create("r1", steps=4)
        self.assertEqual(status, 200)
        self.assertFalse(again["created"])

        status, error = self.create("r1", steps=5)
        self.assertEqual(status, 400)
        self.assertIn("different config", error["error"])

        status, generated = self.create("", steps=4)
        self.assertEqual(status, 200)
        self.assertTrue(generated["id"].startswith("run-"))
        status, duplicate = self.create("", steps=4)
        self.assertEqual(duplicate["id"], generated["id"])
        self.assertFalse(duplicate["created"])

        status, error = self.create("bad", policy="weird")
        self.assertEqual(status, 400)
        self.assertIn("policy", error["error"])
        status, error = self.create("bad", steps=0)
        self.assertEqual(status, 400)
        self.assertIn("steps", error["error"])
        status, error = self.create("bad", policy="adaptive", step=10, min_step=20)
        self.assertEqual(status, 400)
        self.assertIn("min_step", error["error"])
        status, error = self.create("bad", bogus=1)
        self.assertEqual(status, 400)
        self.assertIn("unknown config", error["error"])

    def test_step_confirmation_records_basis_and_inputs(self) -> None:
        self.create("r1", steps=3, step=60.0)
        for index in range(3):
            status, body = self.call("POST", "/runs/step", {"id": "r1"})
            self.assertEqual(status, 200)
            self.assertEqual(body["cursor"], index + 1)
            self.assertEqual(body["executed"], 1)
            last = body["last"]
            self.assertEqual(last["index"], index)
            self.assertEqual(last["dt"], 60.0)
            self.assertEqual(last["rationale"]["dt_rule"], "fixed")
            self.assertTrue(last["rationale"]["why"])
            self.assertIn("ambient", last["inputs"])
            self.assertEqual(len(last["hash"]), 64)
        self.assertEqual(body["status"], "completed")

        status, ckpt = self.call("GET", "/runs/checkpoint", {"id": "r1", "index": 1})
        self.assertEqual(status, 200)
        self.assertEqual(ckpt["index"], 1)
        self.assertEqual(ckpt["rationale"]["mode"], "cooling")

        status, audit = self.call("POST", "/audit/filter", {"kind": "sim-step"})
        self.assertEqual(len(audit["entries"]), 3)
        status, audit = self.call("POST", "/audit/filter", {"kind": "sim-run"})
        events = [entry["detail"] for entry in audit["entries"]]
        self.assertTrue(any("event=created" in detail for detail in events))
        self.assertTrue(any("event=completed" in detail for detail in events))

    def test_budget_bounds_each_advance_call(self) -> None:
        self.create("r1", steps=5)
        for expected in range(1, 6):
            status, body = self.call("POST", "/runs/advance", {"id": "r1", "budget_ms": 0})
            self.assertEqual(status, 200)
            self.assertEqual(body["executed"], 1)
            self.assertEqual(body["cursor"], expected)
        self.assertEqual(body["status"], "completed")

    def test_completed_run_advance_is_a_no_op(self) -> None:
        self.create("r1", steps=3)
        self.finish("r1")
        status, before = self.call("GET", "/audit/summary")
        status, body = self.call("POST", "/runs/advance", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["executed"], 0)
        self.assertEqual(body["cursor"], 3)
        status, after = self.call("GET", "/audit/summary")
        self.assertEqual(before["count"], after["count"])

    def test_dangling_checkpoint_commits_in_place(self) -> None:
        self.create("r1", steps=3)
        self.call("POST", "/runs/step", {"id": "r1"})
        self.forge_dangling("r1")

        status, body = self.call("POST", "/runs/recover", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertTrue(body["chain_ok"])
        self.assertTrue(body["committed_dangling"])
        self.assertEqual(body["rolled_back"], 0)
        self.assertEqual(body["cursor"], 2)

        status, show = self.call("GET", "/runs/show", {"id": "r1"})
        self.assertEqual(show["run"]["stats"]["redone"], 0)
        self.assertTrue(show["chain_ok"])

    def test_invalid_dangling_checkpoint_is_redone(self) -> None:
        self.create("r1", steps=2)
        self.forge_dangling("r1")
        self.tamper("r1", 0, "state", {"time": 60.0, "temp": -99.0, "mode": "cooling"})

        status, body = self.call("POST", "/runs/recover", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 0)
        self.assertEqual(body["redone"], 1)
        self.assertFalse(body["committed_dangling"])

        status, body = self.call("POST", "/runs/step", {"id": "r1"})
        self.assertEqual(body["cursor"], 1)
        self.assertEqual(body["executed"], 1)

    def test_broken_chain_rolls_back_to_last_valid_checkpoint(self) -> None:
        self.create("r1", steps=4)
        self.finish("r1")
        self.tamper("r1", 1, "state", {"time": 120.0, "temp": -99.0, "mode": "cooling"})

        status, body = self.call("POST", "/runs/recover", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertFalse(body["chain_ok"])
        self.assertEqual(body["cursor"], 1)
        self.assertEqual(body["rolled_back"], 3)

        self.finish("r1")
        status, report = self.call("POST", "/runs/replay", {"id": "r1"})
        self.assertTrue(report["ok"])
        self.assertEqual(report["checked"], 4)

    def test_recorded_inputs_win_over_retry_overrides(self) -> None:
        self.create("r1", steps=2)
        self.forge_dangling("r1", {"door": 1})

        status, body = self.call("POST", "/runs/step", {"id": "r1", "inputs": {"door": 0}})
        self.assertEqual(status, 200)
        self.assertEqual(body["committed"], 1)
        self.assertEqual(body["executed"], 0)

        status, ckpt = self.call("GET", "/runs/checkpoint", {"id": "r1", "index": 0})
        self.assertEqual(ckpt["inputs"]["door"], 1)

    def test_step_inputs_are_recorded_and_replayable(self) -> None:
        self.create("r1", steps=2)
        status, body = self.call("POST", "/runs/step", {"id": "r1", "inputs": {"door": 1, "ambient": 25.0}})
        self.assertEqual(body["executed"], 1)
        self.assertEqual(body["last"]["inputs"]["door"], 1)
        self.assertEqual(body["last"]["inputs"]["ambient"], 25.0)
        self.assertEqual(body["last"]["rationale"]["door"], 1)

        self.finish("r1")
        status, report = self.call("POST", "/runs/replay", {"id": "r1"})
        self.assertTrue(report["ok"])

    def test_replay_reports_tampering(self) -> None:
        self.create("r1", steps=3)
        self.finish("r1")
        self.tamper("r1", 1, "inputs", {"ambient": 99.0, "door": 0, "pull": 0.01,
                                        "leak": 0.0005, "target": -18.0, "band": 0.5,
                                        "door_heat": 0.02})

        status, report = self.call("POST", "/runs/replay", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertFalse(report["ok"])
        self.assertEqual(report["checked"], 3)
        self.assertTrue(any(item["index"] == 1 for item in report["mismatches"]))

        status, body = self.call("GET", "/runs/compare", {"id": "r1"})
        self.assertFalse(body["compare"]["ok"])
        status, audit = self.call("POST", "/audit/filter", {"kind": "sim-compare"})
        self.assertEqual(len(audit["entries"]), 1)
        self.assertIn("mismatches=", audit["entries"][0]["detail"])

    def test_adaptive_policy_is_bounded_and_replayable(self) -> None:
        self.create(
            "r1",
            steps=3,
            policy="adaptive",
            step=60.0,
            min_step=1.0,
            max_step=300.0,
            max_delta=0.5,
            pull=0.5,
        )
        status, body = self.call("POST", "/runs/step", {"id": "r1"})
        self.assertEqual(status, 200)
        last = body["last"]
        self.assertEqual(last["rationale"]["dt_rule"], "refined")
        self.assertGreaterEqual(last["dt"], 1.0)
        self.assertLess(abs(last["rationale"]["delta"]), 0.5 + 1e-9)

        self.finish("r1")
        status, report = self.call("POST", "/runs/replay", {"id": "r1"})
        self.assertTrue(report["ok"])

    def test_adaptive_policy_grows_cautiously(self) -> None:
        self.create(
            "r1",
            steps=2,
            policy="adaptive",
            step=60.0,
            min_step=1.0,
            max_step=300.0,
            max_delta=5.0,
            start_temp=-18.0,
            target=-18.0,
        )
        status, body = self.call("POST", "/runs/step", {"id": "r1"})
        self.assertEqual(body["last"]["rationale"]["dt_rule"], "grown")
        self.assertEqual(body["last"]["dt"], 120.0)
        self.assertEqual(body["last"]["next_dt"], 120.0)

    def test_rewind_resumes_from_a_chosen_checkpoint(self) -> None:
        self.create("r1", steps=4)
        self.finish("r1")

        status, body = self.call("POST", "/runs/rewind", {"id": "r1", "to": 2})
        self.assertEqual(status, 200)
        self.assertEqual(body["cursor"], 2)
        self.assertEqual(body["rewound"], 2)
        self.assertEqual(body["status"], "running")

        self.finish("r1")
        status, report = self.call("POST", "/runs/replay", {"id": "r1"})
        self.assertTrue(report["ok"])

        status, error = self.call("POST", "/runs/rewind", {"id": "r1", "to": 9})
        self.assertEqual(status, 400)
        status, body = self.call("POST", "/runs/rewind", {"id": "r1", "to": 4})
        self.assertEqual(body["rewound"], 0)

    def test_cancel_stops_the_run(self) -> None:
        self.create("r1", steps=5)
        status, body = self.call("POST", "/runs/cancel", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "aborted")

        status, error = self.call("POST", "/runs/advance", {"id": "r1"})
        self.assertEqual(status, 400)
        self.assertIn("aborted", error["error"])

        status, body = self.call("POST", "/runs/cancel", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "aborted")

        self.create("r2", steps=1)
        self.finish("r2")
        status, error = self.call("POST", "/runs/cancel", {"id": "r2"})
        self.assertEqual(status, 400)
        self.assertIn("completed", error["error"])

    def test_control_plane_lists_and_shows_runs(self) -> None:
        self.create("r1", steps=2)
        self.create("r2", steps=3, policy="adaptive")
        self.call("POST", "/runs/step", {"id": "r1"})

        status, listing = self.call("GET", "/runs")
        self.assertEqual(status, 200)
        self.assertEqual(listing["count"], 2)
        by_id = {run["id"]: run for run in listing["runs"]}
        self.assertEqual(by_id["r1"]["cursor"], 1)
        self.assertEqual(by_id["r2"]["policy"], "adaptive")

        status, show = self.call("GET", "/runs/show", {"id": "r1"})
        self.assertEqual(status, 200)
        self.assertTrue(show["chain_ok"])
        self.assertEqual(len(show["checkpoints"]), 1)
        self.assertIn("why", show["checkpoints"][0]["rationale"])
        self.assertIsNone(show["compare"])

        status, error = self.call("GET", "/runs/show", {"id": "ghost"})
        self.assertEqual(status, 400)
        self.assertIn("unknown run", error["error"])


if __name__ == "__main__":
    unittest.main()
