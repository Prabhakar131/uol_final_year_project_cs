"""Scores the vision model's screenshot readings against the facts each screenshot was drawn from.

Every generated screenshot is rendered from a known fact list (evidence_facts.json),
and every evaluation saves what the vision model read (vlm_output). For each
automated test run this compares the two:

  found     a drawn fact appears in the reading
  missed    a drawn fact is absent from the reading
  invented  the reading names an IP, principal, key ID, amount or date that the
            screenshot's facts do not contain

Reads saved run files only, so it is safe while the server runs.

    .venv/bin/python notebooks/model_benchmarks/experiments/extraction_accuracy.py [label filter]
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "data/runs"
RESULTS = ROOT / "notebooks/model_benchmarks/results/experiments" / f"extraction_accuracy_{date.today():%Y%m%d}.json"

IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
KEY_ID = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{8,}\b")
PRINCIPAL = re.compile(r"\b(?:user|role|assumed-role)/[\w.+=,@-]+")
AMOUNT = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
EVENT = re.compile(r"^[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+$")  # ConsoleLogin, PutEvents
# Only what the screenshot templates draw is scored: CloudTrail shows 4 related
# events, IAM activity 5 rows, and the CloudTrail risk_signal is not drawn.
ROW_LIMITS = {"related_events": 4, "activity_rows": 5}
NOT_DRAWN = {"risk_signal"}
PROSE_KEYS = {"summary", "risk_signal", "query", "policy_change", "access_key_status", "request_parameters"}


def squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def category(key: str, value: str) -> str:
    if IP.fullmatch(value):
        return "ip"
    if DATE.match(value):
        return "time"
    if KEY_ID.fullmatch(value):
        return "key_id"
    if value.startswith("arn:") or PRINCIPAL.search(value) or key in {"owner", "user", "principal", "resource"}:
        return "principal"
    if "$" in value or re.fullmatch(r"[+-]?\d+(?:\.\d+)?%", value):
        return "amount"
    if EVENT.fullmatch(value) or key in {"event_name", "finding_type"} or ":" in value and "/" in value:
        return "event"
    if key in PROSE_KEYS or len(value) > 60:
        return "prose"
    return "other"


def leaf_facts(facts: dict) -> list[tuple[str, str]]:
    """Every value drawn on the screenshot, table cells included, as (key, value)."""
    out = []
    for key, value in facts.items():
        if key in NOT_DRAWN:
            continue
        if isinstance(value, list):
            for row in value[:ROW_LIMITS.get(key)]:
                cells = row if isinstance(row, list) else list(row.values()) if isinstance(row, dict) else [row]
                for cell in cells:
                    if str(cell) not in ("", "Unknown", "-"):
                        out.append((key, str(cell)))
        elif isinstance(value, dict):
            out.extend(leaf_facts(value))
        elif value not in (None, ""):
            out.append((key, str(value)))
    return out


def is_found(kind: str, value: str, reading: str, reading_squashed: str, reading_digits: str) -> bool:
    if kind == "time":
        return digits(value)[:12] in reading_digits  # to the minute, any date format
    if kind == "principal":
        name = PRINCIPAL.search(value)
        return squash(name.group(0) if name else value) in reading_squashed
    if kind == "prose":
        words = [w for w in re.findall(r"[a-z0-9]{4,}", value.lower())]
        return bool(words) and sum(w in reading_squashed for w in words) / len(words) >= 0.6
    return squash(value) in reading_squashed


def invented(reading: str, facts_text: str) -> list[str]:
    facts_squashed, facts_digits = squash(facts_text), digits(facts_text)
    extra = []
    for pattern, kind in ((IP, "ip"), (KEY_ID, "key_id"), (PRINCIPAL, "principal"), (AMOUNT, "amount")):
        for hit in set(pattern.findall(reading)):
            if squash(hit) not in facts_squashed:
                extra.append(f"{kind}: {hit}")
    for hit in set(re.findall(r"\d{4}-\d{2}-\d{2}", reading)):
        if digits(hit) not in facts_digits:
            extra.append(f"date: {hit}")
    return extra


def evaluations(label_filter: str):
    """Latest saved evaluation per (run, turn) for runs whose label contains the filter."""
    for run_file in sorted(RUNS.glob("*/run.json")):
        run = json.loads(run_file.read_text())
        label = run.get("label") or run.get("name") or ""
        if label_filter not in label:
            continue
        latest = {}
        for cp in sorted(run_file.parent.glob("checkpoints/*"), key=lambda p: p.stat().st_mtime):
            for ev in cp.glob("runtime/evaluations/turn_*_evaluation.json"):
                latest[ev.name] = (cp, ev)
        for cp, ev in latest.values():
            yield label, cp, json.loads(ev.read_text())


def drawn_facts(checkpoint: Path, turn: int, evidence_id: str) -> dict | None:
    path = checkpoint / "runtime/turns" / f"turn_{turn}" / "evidence_facts.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    items = data if isinstance(data, list) else data.get("evidence", [])
    return next((e.get("facts") for e in items if e.get("id") == evidence_id), None)


def main(label_filter: str) -> None:
    cases = []
    for label, checkpoint, ev in evaluations(label_filter):
        evidence = ev.get("selected_evidence") or {}
        vlm = ev.get("vlm_output") or {}
        facts = drawn_facts(checkpoint, ev.get("turn"), evidence.get("id"))
        if not facts or not vlm:
            continue
        reading = json.dumps(vlm, ensure_ascii=False)
        rs, rd = squash(reading), digits(reading)
        scored = [{"key": k, "value": v, "kind": category(k, v), "found": False} for k, v in leaf_facts(facts)]
        for fact in scored:
            fact["found"] = is_found(fact["kind"], fact["value"], reading, rs, rd)
        cases.append({"run": label, "turn": ev.get("turn"), "template": evidence.get("template"),
                      "evidence_id": evidence.get("id"), "facts": scored,
                      "invented": invented(reading, json.dumps(facts, ensure_ascii=False))})

    def rate(facts):
        return f"{sum(f['found'] for f in facts)}/{len(facts)} ({100 * sum(f['found'] for f in facts) / max(len(facts), 1):.0f}%)"

    by_kind, by_template = defaultdict(list), defaultdict(list)
    for case in cases:
        for fact in case["facts"]:
            by_kind[fact["kind"]].append(fact)
            by_template[case["template"]].append(fact)
    all_facts = [f for c in cases for f in c["facts"]]
    with_invented = [c for c in cases if c["invented"]]

    print(f"Screenshots scored: {len(cases)} (runs matching '{label_filter}')")
    print(f"Facts found overall: {rate(all_facts)}")
    print("\nBy field type:")
    for kind in sorted(by_kind):
        print(f"  {kind:10} {rate(by_kind[kind])}")
    print("\nBy screenshot type:")
    for template in sorted(by_template, key=str):
        n = sum(1 for c in cases if c["template"] == template)
        print(f"  {str(template):13} {rate(by_template[template])}  ({n} screenshots)")
    print(f"\nScreenshots with invented details: {len(with_invented)}/{len(cases)}")
    for case in with_invented[:15]:
        print(f"  {case['template']:12} turn {case['turn']}: {', '.join(case['invented'][:4])}")

    RESULTS.write_text(json.dumps({
        "label_filter": label_filter, "screenshots": len(cases), "facts_found": rate(all_facts),
        "by_field_type": {k: rate(v) for k, v in by_kind.items()},
        "by_screenshot_type": {str(k): rate(v) for k, v in by_template.items()},
        "screenshots_with_invented_details": len(with_invented), "cases": cases}, indent=1))
    print(f"\nSaved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AUTO TEST")
