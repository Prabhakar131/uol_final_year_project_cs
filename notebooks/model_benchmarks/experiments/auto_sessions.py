"""Automated five-turn sessions through the running app, the way the browser plays them.

Drives the local server's own routes (start a labelled Test Mode run, read the
turn, evaluate, continue, final debrief), so every session is saved as a normal
run that can be opened in the app later. Each session always takes the turn's
best action and the evidence of one fixed support role (strong, partial or
weak), so every evidence role is graded in every turn across the three sessions.

Start the server first with next-turn generation on:
    CLOUDIR_SKIP_NEXT_TURN=0 python app.py
Then:
    .venv/bin/python notebooks/model_benchmarks/experiments/auto_sessions.py cost-management automated-security-response
Results: notebooks/model_benchmarks/results/experiments/auto_sessions_<date>.json (written after every turn).
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

BASE = "http://127.0.0.1:5000"
ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(os.environ['RESULTS_FILE']) if os.environ.get('RESULTS_FILE') else ROOT / "notebooks/model_benchmarks/results/experiments" / f"auto_sessions_{date.today():%Y%m%d}.json"
EXPECTED = {"strong": "Strong Support", "partial": "Partial Support", "weak": "Weak Support"}
JUSTIFICATION = "I chose this evidence because it shows what the selected action needs to check."


def call(method: str, path: str, body: dict | None = None, timeout: int = 3600) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(BASE + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {"error": str(exc)}


LIBC = ctypes.CDLL("/usr/lib/libSystem.B.dylib")


def footprint_gb(pid: int, peak: bool = False) -> float | None:
    """macOS physical footprint (what Activity Monitor shows), or its lifetime peak."""
    buffer = ctypes.create_string_buffer(512)
    # rusage_info_v2 ri_phys_footprint at offset 72; v4 ri_lifetime_max_phys_footprint at 240.
    if LIBC.proc_pid_rusage(pid, 4 if peak else 2, buffer) != 0:
        return None
    offset = 240 if peak else 72
    return int.from_bytes(buffer.raw[offset:offset + 8], "little") / 1e9


def pids(pattern: str) -> list[int]:
    return [int(p) for p in subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()]


def server_memory_gb() -> float | None:
    """Current footprint of the server process (the larger of the reloader pair)."""
    sizes = [size for size in (footprint_gb(pid) for pid in pids("app.py")) if size is not None]
    return round(max(sizes), 2) if sizes else None


class WorkerPeak:
    """Largest lifetime peak of any model worker process since the last reset."""

    def __init__(self):
        self.peak = 0.0
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while True:
            for pid in pids("cloudir.ai_models.model_worker"):
                self.peak = max(self.peak, footprint_gb(pid, peak=True) or 0.0)
            time.sleep(1)

    def take(self) -> float:
        peak, self.peak = self.peak, 0.0
        return round(peak, 1)


WORKERS = WorkerPeak()


def evaluation_trace(turn: int) -> dict:
    path = ROOT / "data/runtime/evaluations" / f"turn_{turn}_evaluation.json"
    return json.loads(path.read_text()) if path.exists() else {}


def generation_outcome(turn: int) -> dict | None:
    path = ROOT / "data/runtime/turns" / f"turn_{turn}" / "generation_diagnostics.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return {"outcome": d.get("outcome"), "attempts": d.get("attempts"),
            "anchor_repairs": d.get("anchor_repairs"), "validation_errors": d.get("validation_errors"),
            "review_scores": [r.get("quality_score") for r in d.get("quality_reviews") or []]}


def save(results: list) -> None:
    RESULTS.write_text(json.dumps(results, indent=1))


ROTATION = ("strong", "partial", "weak")


def play(scenario: str, path: str, results: list, choice: str = "best") -> dict:
    # A "rotate" path changes the evidence level each turn (strong, partial, weak, ...).
    role = path
    action_part = "" if choice == "best" else f"{choice} action · "
    label = f"AUTO TEST · {scenario} · {action_part}{path} evidence path · {date.today():%d %b}{os.environ.get('LABEL_SUFFIX', '')}"
    session = {"scenario": scenario, "path": path, "choice": choice, "label": label, "turns": [], "completed": False,
               "error": None, "started": time.strftime("%H:%M:%S")}
    results.append(session)
    status, payload = call("POST", "/api/runs/start", {"scenarioId": scenario, "label": label, "testMode": True})
    if status != 200:
        session["error"] = f"start failed ({status}): {payload.get('error')}"
        save(results)
        return session
    session["run_id"] = (payload.get("run") or {}).get("id")

    while True:
        state = payload["state"]
        turn_number = state.get("currentTurn") or state.get("current_turn")
        actions = payload["turn"]["actions"]
        evidence = payload["generatedEvidence"]
        # The API sends camelCase fields (choiceRole, supportRole), as the browser reads them.
        if path == "rotate":
            role = ROTATION[(turn_number - 1) % 3]
        action = next(a for a in actions if (a.get("choiceRole") or a.get("choice_role")) == choice)
        item = next(e for e in evidence if (e.get("supportRole") or e.get("support_role")) == role)
        before = {k: state.get(k) for k in ("containment", "visibility", "risk")}

        started = time.time()
        status, result = call("POST", "/api/evaluate", {
            "selectedAction": action, "selectedEvidence": item,
            "learnerJustification": {"mode": "typed", "prompt": "", "transcript": JUSTIFICATION,
                                     "source": "typed_justification"}})
        seconds = round(time.time() - started)
        trace = evaluation_trace(turn_number) if status == 200 else {}
        security = trace.get("security_output") or result.get("security_output") or {}
        coach = trace.get("coach_output") or result.get("coach_output") or {}
        record = {
            "turn": turn_number, "choice": choice, "evidence_role": role, "action": action.get("title"), "evidence_id": item.get("id"),
            "template": item.get("template"), "expected": EXPECTED[role], "verdict": security.get("verdict"),
            "correct": security.get("verdict") == EXPECTED[role], "whose_evidence": security.get("whose_evidence"),
            "key_fact": security.get("key_fact"), "reasoning": security.get("reasoning"),
            "coach_feedback": coach.get("feedback") or coach.get("coach_feedback") or coach,
            "state_before": before, "evaluate_seconds": seconds, "http_status": status,
            "error": None if status == 200 else result.get("error"),
            "next_turn_generation": generation_outcome(turn_number + 1),
            "server_memory_gb": server_memory_gb(),
            "worker_peak_gb": WORKERS.take(),
        }
        session["turns"].append(record)
        save(results)
        nxt = record["next_turn_generation"] or {}
        print(f"  {scenario} · {choice} action · {role} · turn {turn_number}: {record['verdict']} "
              f"({'ok' if record['correct'] else 'MISS'}) in {seconds}s | server {record['server_memory_gb'] or 0:.2f} GB, "
              f"worker peak {record['worker_peak_gb']} GB | next turn: {nxt.get('outcome')} "
              f"attempts {nxt.get('attempts')} repairs {nxt.get('anchor_repairs')} review {nxt.get('review_scores')}"
              + (f" | ERROR {record['error']}" if record["error"] else ""), flush=True)
        if status != 200:
            session["error"] = f"turn {turn_number} evaluation failed ({status}): {record['error']}"
            break

        status, payload = call("POST", "/api/continue", {})
        if status != 200:
            session["error"] = f"continue after turn {turn_number} failed ({status}): {payload.get('error')}"
            break
        record["state_after"] = {k: payload["state"].get(k) for k in ("containment", "visibility", "risk")}
        save(results)
        if payload.get("isComplete"):
            session["completed"] = True
            _, debrief = call("GET", "/api/final-debrief")
            session["final_debrief"] = debrief.get("debrief")
            break
        if not payload.get("turn"):
            _, payload = call("GET", "/api/state")

    call("POST", "/api/runs/leave", {"label": label, "note": "Automated session"})
    session["finished"] = time.strftime("%H:%M:%S")
    save(results)
    return session


def main(scenarios: list[str]) -> None:
    status, health = call("GET", "/api/health", timeout=10)
    if status != 200:
        sys.exit(f"The server is not reachable at {BASE}.")
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    for scenario in scenarios:
        for role in os.environ.get("ROLES", "strong,partial,weak").split(","):
            print(f"== {scenario} · {role} evidence path", flush=True)
            play(scenario, role, results, os.environ.get("CHOICE", "best"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["automated-security-response", "cost-management", "identity-management"])
