"""
Stage D — production type assignment over all 11,302 concepts.

  * Deterministic overrides (rubric type_overrides) are applied by concept_id
    and NEVER sent to the model.
  * Every other concept is typed by the model (type-v1.2.0, temp 0, 40/batch,
    judged independently).
  * A concept the model abstains on is HELD: its type is null and it carries a
    held_reason for manual review — it is never given a voted type.

Writes data/concepts/concept_type.csv, joined to concept at query time:
  concept_id, type, confidence, abstain, held_reason, source,
  prompt_version, model_version, rules_version

The model never edits labels, member forms, or concept_ids.

Run: python3 scripts/run_typing.py            (full run)
     python3 scripts/run_typing.py --dry-run  (build batch, no API)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

from typing_concepts import (  # noqa: E402
    BATCH_SIZE, EMIT_TOOL, HELD_REASONS, MODEL_VERSION, OVERRIDES,
    PROMPT_VERSION, RULES_VERSION, SYSTEM_PROMPT, TYPE_ENUM,
    build_user_prompt, parse_tool_result, validate,
)
from run_merge_batch import client, load_env  # noqa: E402

CONCEPT_CSV = ROOT / "data" / "concepts" / "concept.csv"
SURFACE_CSV = ROOT / "data" / "normalise" / "surface_form.csv"
OUT_CSV = ROOT / "data" / "concepts" / "concept_type.csv"
SUMMARY = ROOT / "data" / "concepts" / "type_summary.json"
MAX_TOKENS = 4096


def load_concepts() -> list[dict]:
    labels, descs = {}, {}
    for c in csv.DictReader(CONCEPT_CSV.open(encoding="utf-8")):
        labels[c["concept_id"]] = c["label"]
        descs[c["concept_id"]] = c["description"]
    forms = defaultdict(list)
    mentions = defaultdict(int)
    seen = set()
    for r in csv.DictReader(SURFACE_CSV.open(encoding="utf-8")):
        nf, cid = r["normalised_form"], r["concept_id"]
        if not cid or nf in seen:
            continue
        seen.add(nf)
        forms[cid].append(nf)
        mentions[cid] += int(r["count"])
    return [{"id": cid, "label": labels[cid], "description": descs[cid],
             "forms": sorted(forms[cid]), "mentions": mentions[cid]}
            for cid in labels]


def batches(items, n):
    for i in range(0, len(items), n):
        yield i // n, items[i:i + n]


RUNS = 3


def run_model(concepts: list[dict], tag: str) -> list[dict]:
    """Type `concepts` RUNS times at temp 0. Return a list of RUNS dicts,
    each concept_id -> assignment."""
    cl = client()
    reqs = []
    for run in range(1, RUNS + 1):
        for bidx, chunk in batches(concepts, BATCH_SIZE):
            reqs.append({
                "custom_id": f"{tag}-r{run}-b{bidx}",
                "params": {
                    "model": MODEL_VERSION, "max_tokens": MAX_TOKENS, "temperature": 0,
                    "system": SYSTEM_PROMPT, "tools": [EMIT_TOOL],
                    "tool_choice": {"type": "tool", "name": "emit_types"},
                    "messages": [{"role": "user", "content": build_user_prompt(chunk)}],
                },
            })
    batch = cl.messages.batches.create(requests=reqs)
    print(f"submitted {tag} batch {batch.id} with {len(reqs)} requests "
          f"({len(concepts)} concepts x {RUNS} runs)", flush=True)
    while True:
        batch = cl.messages.batches.retrieve(batch.id)
        c = batch.request_counts
        print(f"  status={batch.processing_status} succeeded={c.succeeded} "
              f"errored={c.errored} processing={c.processing}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(20)
    raw = {res.custom_id: (res.result.message.content
                           if res.result.type == "succeeded" else None)
           for res in cl.messages.batches.results(batch.id)}
    per_run = [dict() for _ in range(RUNS)]
    for run in range(1, RUNS + 1):
        for bidx, chunk in batches(concepts, BATCH_SIZE):
            res = parse_tool_result(raw.get(f"{tag}-r{run}-b{bidx}"))
            if res:
                per_run[run - 1].update(res.assignments)
    return per_run


def decide(runs: list[dict]) -> dict:
    """Combine RUNS assignments for one concept into a final decision.

    Majority vote decides the type; a 1/1/1 tie is HELD (type_tie). A concept
    abstained in a majority of runs is HELD with the majority held_reason.
    stable = all runs agreed on the emitted type.
    """
    types = [a["type"] for a in runs]
    abst = [a["abstain"] for a in runs]
    confs = [a.get("confidence") or 0.0 for a in runs]
    stable = len(set(types)) == 1
    variants = "" if stable else "/".join(str(t) for t in types)
    mean_conf = round(sum(confs) / len(confs), 3)

    if sum(1 for a in abst if a) >= 2:
        hr = Counter(runs[i].get("held_reason") for i in range(len(runs))
                     if abst[i] and runs[i].get("held_reason") in HELD_REASONS)
        reason = hr.most_common(1)[0][0] if hr else "type_tie"
        return {"type": "", "confidence": mean_conf, "abstain": True,
                "held_reason": reason, "stable": stable, "run_variants": variants,
                "source": "model_held"}

    top, cnt = Counter(types).most_common(1)[0]
    if cnt >= 2:
        return {"type": top, "confidence": mean_conf, "abstain": False,
                "held_reason": "", "stable": stable, "run_variants": variants,
                "source": "model"}
    # 1/1/1 — no majority
    return {"type": "", "confidence": mean_conf, "abstain": True,
            "held_reason": "type_tie", "stable": False, "run_variants": variants,
            "source": "model_tie"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_env()

    concepts = load_concepts()
    by_id = {c["id"]: c for c in concepts}
    to_model = [c for c in concepts if c["id"] not in OVERRIDES]
    print(f"concepts: {len(concepts)}  overrides: {len(OVERRIDES)}  "
          f"to model: {len(to_model)}  ({PROMPT_VERSION}/{RULES_VERSION})", flush=True)
    if args.dry_run:
        per = (len(to_model) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"would submit {per} x {RUNS} = {per*RUNS} requests")
        return

    per_run = run_model(to_model, "t1")

    def n_valid(cid):
        n = 0
        for pr in per_run:
            a = pr.get(cid)
            if a and a["type"] in TYPE_ENUM and not (
                    a["abstain"] and a.get("held_reason") not in HELD_REASONS):
                n += 1
        return n
    missing = [by_id[c["id"]] for c in to_model if n_valid(c["id"]) < RUNS]
    if missing:
        print(f"retrying {len(missing)} concepts lacking {RUNS} valid runs", flush=True)
        try:
            retry = run_model(missing, "t2")
            for i in range(RUNS):
                per_run[i].update(retry[i])
        except Exception as e:  # e.g. API usage limit — keep what we have
            print(f"retry skipped ({type(e).__name__}: {e}); "
                  f"concepts with <2 valid runs will be held as pending_retry", flush=True)

    # assemble rows via majority vote across the RUNS runs
    rows = []
    for c in concepts:
        cid = c["id"]
        if cid in OVERRIDES:
            rows.append({"concept_id": cid, "type": OVERRIDES[cid], "confidence": 1.0,
                         "abstain": False, "held_reason": "", "stable": True,
                         "run_variants": "", "source": "override"})
            continue
        got = [pr[cid] for pr in per_run if cid in pr and pr[cid]["type"] in TYPE_ENUM]
        if len(got) < 2:
            rows.append({"concept_id": cid, "type": "", "confidence": 0.0,
                         "abstain": True, "held_reason": "insufficient_evidence",
                         "stable": False,
                         "run_variants": "/".join(a["type"] for a in got),
                         "source": "model_unresolved"})
            continue
        d = decide(got)
        rows.append({"concept_id": cid, **d})

    rows.sort(key=lambda r: r["concept_id"])
    fields = ["concept_id", "type", "confidence", "abstain", "held_reason",
              "stable", "run_variants", "source",
              "prompt_version", "model_version", "rules_version"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "prompt_version": PROMPT_VERSION,
                        "model_version": MODEL_VERSION, "rules_version": RULES_VERSION})

    # validation
    problems = []
    ids = {r["concept_id"] for r in rows}
    if len(rows) != len(concepts) or ids != set(by_id):
        problems.append("row/concept id mismatch")
    for r in rows:
        if r["abstain"] and r["type"]:
            problems.append(f"held concept carries a type: {r['concept_id']}")
        if not r["abstain"] and r["type"] not in TYPE_ENUM:
            problems.append(f"non-held concept has no valid type: {r['concept_id']}")
    for cid, t in OVERRIDES.items():
        got = next((r for r in rows if r["concept_id"] == cid), None)
        if not got or got["type"] != t:
            problems.append(f"override not applied: {cid}")

    typed = Counter(r["type"] for r in rows if not r["abstain"])
    held = [r for r in rows if r["abstain"]]
    model_rows = [r for r in rows if r["source"] != "override"]
    stable = sum(1 for r in model_rows if r["stable"])
    summary = {
        "prompt_version": PROMPT_VERSION, "model_version": MODEL_VERSION,
        "rules_version": RULES_VERSION, "concepts": len(rows),
        "overrides": len(OVERRIDES), "held": len(held),
        "held_by_reason": dict(Counter(r["held_reason"] for r in held)),
        "type_distribution": dict(typed.most_common()),
        "stable": stable, "unstable": len(model_rows) - stable,
        "runs": RUNS, "problems": problems,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_CSV} ({len(rows)} rows)", flush=True)
    print("type distribution:", dict(typed.most_common()), flush=True)
    print(f"held: {len(held)}  by reason: {summary['held_by_reason']}", flush=True)
    print(f"stable (all {RUNS} agree): {stable}/{len(model_rows)} model concepts  "
          f"({len(model_rows)-stable} unstable)", flush=True)
    print(f"overrides applied: {len(OVERRIDES)}", flush=True)
    print("problems:", problems or "none", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
