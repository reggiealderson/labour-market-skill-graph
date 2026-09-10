"""
Stage F — category assignment over all TYPED concepts (held excluded).

Protocol (assign-v1.1.0): 3 runs, temperature 0, 40 concepts/batch, judged
independently, 2-of-3 majority. Assignment: one primary category, an optional
secondary when the concept genuinely covers two functions (no minimise-overlap
pressure); unassigned is allowed when nothing fits. The category order is
RANDOMISED per request (seeded by custom_id, so reproducible). The request
carries NO mention counts, occupation, or year. The model never edits labels,
forms, ids, or categories.yaml.

Writes data/concepts/concept_category.csv. Before that, writes two gate files
from a random 30-concept sample (excluding the first pilot's 80):
  data/gate/gate_blank.csv   concept_id, label, surface_forms
  data/gate/gate_model.csv   the model's assignments for those 30

  python3 scripts/run_stage_f.py --dry-run
  python3 scripts/run_stage_f.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

from assign_categories import (  # noqa: E402
    BATCH_SIZE, CATEGORY_ENUM, CATEGORY_LIST, EMIT_TOOL, MAX_CATEGORIES,
    MODEL_VERSION, PROMPT_VERSION, STAGE_E_VERSION, build_system_prompt,
    build_user_prompt, parse_tool_result,
)
from run_merge_batch import client, load_env  # noqa: E402

CONCEPT_CSV = ROOT / "data" / "concepts" / "concept.csv"
TYPE_CSV = ROOT / "data" / "concepts" / "concept_type.csv"
SURFACE_CSV = ROOT / "data" / "normalise" / "surface_form.csv"
OUT_CSV = ROOT / "data" / "concepts" / "concept_category.csv"
SUMMARY = ROOT / "data" / "concepts" / "category_summary.json"
PILOT_CSV = ROOT / "reports" / "stage_f_pilot.csv"
GATE_DIR = ROOT / "data" / "gate"
MAX_TOKENS = 4096
RUNS = 3
GATE_SEED = 20260831
GATE_N = 30


def load_typed_concepts() -> list[dict]:
    labels, descs = {}, {}
    for c in csv.DictReader(CONCEPT_CSV.open(encoding="utf-8")):
        labels[c["concept_id"]] = c["label"]
        descs[c["concept_id"]] = c["description"]
    typed = {}
    for r in csv.DictReader(TYPE_CSV.open(encoding="utf-8")):
        if r["abstain"].strip().lower() != "true":
            typed[r["concept_id"]] = r["type"]
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
    out = [{"id": cid, "label": labels[cid], "description": descs[cid],
            "forms": sorted(forms[cid]), "mentions": mentions[cid],
            "type": typed[cid]} for cid in typed]
    out.sort(key=lambda c: -c["mentions"])
    return out


def batches(items, n):
    for i in range(0, len(items), n):
        yield i // n, items[i:i + n]


def _system_for(custom_id: str) -> str:
    """Per-request randomised category order, seeded by custom_id (reproducible)."""
    seed = int(hashlib.sha256(custom_id.encode()).hexdigest()[:8], 16)
    order = CATEGORY_LIST[:]
    random.Random(seed).shuffle(order)
    return build_system_prompt(order)


def run_model(concepts: list[dict], tag: str) -> list[dict]:
    cl = client()
    reqs = []
    for run in range(1, RUNS + 1):
        for bidx, chunk in batches(concepts, BATCH_SIZE):
            cid = f"{tag}-r{run}-b{bidx}"
            reqs.append({
                "custom_id": cid,
                "params": {
                    "model": MODEL_VERSION, "max_tokens": MAX_TOKENS, "temperature": 0,
                    "system": _system_for(cid), "tools": [EMIT_TOOL],
                    "tool_choice": {"type": "tool", "name": "emit_assignments"},
                    "messages": [{"role": "user", "content": build_user_prompt(chunk)}],
                },
            })
    batch = cl.messages.batches.create(requests=reqs)
    print(f"submitted {tag} batch {batch.id}: {len(reqs)} requests "
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
    """2-of-3 majority over category sets, with an unassigned outcome."""
    keys = [("U",) if a["unassigned"] else tuple(sorted(a["categories"]))
            for a in runs]
    stable = len(set(keys)) == 1
    variants = "" if stable else " | ".join(
        "UNASSIGNED" if a["unassigned"] else "+".join(sorted(a["categories"]))
        for a in runs)
    if sum(1 for a in runs if a["unassigned"]) >= 2:
        return {"categories": [], "primary_category": "", "n_categories": 0,
                "unassigned": True, "stable": stable, "run_variants": variants,
                "source": "model"}
    assigned = [a for a in runs if not a["unassigned"]]
    votes = Counter()
    for a in assigned:
        votes.update(set(a["categories"]))
    accepted = sorted([c for c, n in votes.items() if n >= 2])
    prims = [a["primary"] for a in assigned if a["primary"] in CATEGORY_ENUM]
    primary = Counter(prims).most_common(1)[0][0] if prims else (
        assigned[0]["categories"][0] if assigned and assigned[0]["categories"]
        else "")
    if not accepted:
        accepted = [primary] if primary else (
            sorted(assigned[0]["categories"]) if assigned else [])
    ordered = ([primary] + [c for c in accepted if c != primary]) if primary \
        else accepted
    ordered = ordered[:MAX_CATEGORIES]
    return {"categories": ordered, "primary_category": primary,
            "n_categories": len(ordered), "unassigned": False,
            "stable": stable, "run_variants": variants, "source": "model"}


FIELDS = ["concept_id", "categories", "primary_category", "n_categories",
          "unassigned", "stable", "run_variants", "source", "prompt_version",
          "model_version", "stage_e_version"]


def row_out(r):
    return {"concept_id": r["concept_id"], "categories": "|".join(r["categories"]),
            "primary_category": r["primary_category"],
            "n_categories": r["n_categories"], "unassigned": r["unassigned"],
            "stable": r["stable"], "run_variants": r["run_variants"],
            "source": r["source"], "prompt_version": PROMPT_VERSION,
            "model_version": MODEL_VERSION, "stage_e_version": STAGE_E_VERSION}


def write_gate(rows_by_id, concepts_by_id):
    """30 random concepts, excluding the first pilot's 80. Written BEFORE the
    authoritative file. gate_model.csv is not printed anywhere."""
    pilot_ids = set()
    if PILOT_CSV.exists():
        pilot_ids = {r["concept_id"] for r in csv.DictReader(PILOT_CSV.open())}
    candidates = sorted(cid for cid in rows_by_id if cid not in pilot_ids)
    gate_ids = random.Random(GATE_SEED).sample(candidates, GATE_N)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    with (GATE_DIR / "gate_blank.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept_id", "label", "surface_forms"])
        for cid in gate_ids:
            c = concepts_by_id[cid]
            w.writerow([cid, c["label"], " | ".join(c["forms"])])
    with (GATE_DIR / "gate_model.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept_id", "primary_category", "categories", "unassigned"])
        for cid in gate_ids:
            r = rows_by_id[cid]
            w.writerow([cid, r["primary_category"], "|".join(r["categories"]),
                        r["unassigned"]])
    return len(gate_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    concepts = load_typed_concepts()
    by_id = {c["id"]: c for c in concepts}
    print(f"typed concepts: {len(concepts)}  categories: {len(CATEGORY_ENUM)}  "
          f"({PROMPT_VERSION}/taxonomy {STAGE_E_VERSION})", flush=True)
    if args.dry_run:
        per = (len(concepts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"would submit {per} x {RUNS} = {per*RUNS} requests")
        return

    load_env()
    per_run = run_model(concepts, "f2")

    rows = []
    unresolved = 0
    for c in concepts:
        cid = c["id"]
        got = [pr[cid] for pr in per_run if cid in pr]
        if len(got) < 2:
            unresolved += 1
            g = got[0] if got else {"primary": "", "categories": [],
                                    "unassigned": True}
            rows.append({"concept_id": cid, "categories": g["categories"],
                         "primary_category": g.get("primary", ""),
                         "n_categories": len(g["categories"]),
                         "unassigned": g.get("unassigned", True), "stable": False,
                         "run_variants": "", "source": "model_unresolved"})
            continue
        rows.append({"concept_id": cid, **decide(got)})

    rows_by_id = {r["concept_id"]: r for r in rows}

    # gate files FIRST (before the authoritative file)
    n_gate = write_gate(rows_by_id, by_id)

    # authoritative file
    rows.sort(key=lambda r: r["concept_id"])
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(row_out(r))

    # stats (state, do not judge)
    N = len(rows)
    n_un = sum(1 for r in rows if r["unassigned"])
    assigned = [r for r in rows if not r["unassigned"]]
    two_plus = sum(1 for r in rows if r["n_categories"] >= 2)
    mean_cat = sum(r["n_categories"] for r in rows) / N
    stable = sum(1 for r in rows if r["stable"])
    per_cat = Counter()
    for r in rows:
        per_cat.update(r["categories"])
    summary = {"prompt_version": PROMPT_VERSION, "model_version": MODEL_VERSION,
               "stage_e_version": STAGE_E_VERSION, "assigned_rows": N,
               "runs": RUNS, "stable": stable, "unassigned": n_un,
               "two_plus_categories": two_plus,
               "mean_categories_per_concept": round(mean_cat, 4),
               "unresolved_lt2_runs": unresolved,
               "count_per_category": dict(per_cat.most_common())}
    SUMMARY.write_text(json.dumps(summary, indent=2))

    print(f"\nwrote gate files ({n_gate} concepts) to {GATE_DIR.relative_to(ROOT)} "
          "[gate_model.csv not printed]", flush=True)
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({N} rows)", flush=True)
    print("--- REPORT (numbers only) ---", flush=True)
    print(f"mean categories per concept: {mean_cat:.4f}", flush=True)
    print(f"share with 2+ categories: {two_plus}/{N} = {two_plus/N:.2%}", flush=True)
    print(f"share unassigned: {n_un}/{N} = {n_un/N:.2%}", flush=True)
    print(f"stable (all {RUNS} agree): {stable}/{N} = {stable/N:.2%}", flush=True)
    print(f"unresolved (<2 valid runs): {unresolved}", flush=True)
    print("count per category:", flush=True)
    for cat in CATEGORY_ENUM:
        print(f"  {cat}: {per_cat.get(cat, 0)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
