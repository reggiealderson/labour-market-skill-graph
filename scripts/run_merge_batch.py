"""
Step 3 (semantic merge) — batch runner.

Sends multi-member candidate groups to Claude Haiku via the Message Batches
API, validates each partition, and writes the concept tables.

  data/concepts/concept.csv       concept_id,label,type,description,source,
                                  prompt_version,model_version   (type left empty)
  data/normalise/surface_form.csv concept_id, method, confidence filled in place
  reports/merges.csv              one row per merged group, largest first
  reports/merge_failures.csv      groups the model failed to resolve (not written)

Ledgers (idempotent, let pilot + rest runs compose):
  data/concepts/merge_ledger.jsonl   one record per processed group

Modes:
  --top N        process the N highest mention-count multi-member groups
  --rest         process every multi-member group not yet OK in the ledger
  --all          process every multi-member group
  --singletons   write each singleton form straight to concept.csv (no API)
  --retry-failed process only groups previously recorded as failed

  --dry-run      build the batch requests and print counts, do not call the API

Run: python3 scripts/run_merge_batch.py --top 200
Deterministic id assignment; the API step itself is not byte-reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from merge_concepts import (  # noqa: E402
    EMIT_TOOL, MODEL_VERSION, PROMPT_VERSION, SYSTEM_PROMPT,
    build_user_prompt, concept_id, parse_tool_result, validate, _label_key,
)

CONCEPTS_DIR = ROOT / "data" / "concepts"
NORM_DIR = ROOT / "data" / "normalise"
REPORTS_DIR = ROOT / "reports"
PAYLOADS = CONCEPTS_DIR / "payloads.jsonl"
LEDGER = CONCEPTS_DIR / "merge_ledger.jsonl"
CONCEPT_CSV = CONCEPTS_DIR / "concept.csv"
SURFACE_CSV = NORM_DIR / "surface_form.csv"
MERGES_CSV = REPORTS_DIR / "merges.csv"
FAILURES_CSV = REPORTS_DIR / "merge_failures.csv"
TOP_CSV = REPORTS_DIR / "top_concepts.csv"
SUMMARY_JSON = CONCEPTS_DIR / "summary.json"

MAX_TOKENS = 2048


# ── env / client ─────────────────────────────────────────────────────────────
def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        import os
        os.environ.setdefault(k.strip(), v.strip())


def client():
    import anthropic
    return anthropic.Anthropic()


# ── io helpers ───────────────────────────────────────────────────────────────
def load_payloads() -> list[dict]:
    return [json.loads(l) for l in PAYLOADS.read_text().splitlines() if l.strip()]


def load_ledger() -> dict[str, dict]:
    """group_id -> latest ledger record."""
    out: dict[str, dict] = {}
    if LEDGER.exists():
        for l in LEDGER.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["group_id"]] = r
    return out


def append_ledger(records: list[dict]) -> None:
    with LEDGER.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def select_groups(payloads: list[dict], args, ledger: dict) -> list[dict]:
    if args.top is not None:
        return payloads[: args.top]
    if args.all:
        return payloads
    if args.rest:
        return [p for p in payloads
                if ledger.get(p["group_id"], {}).get("status") != "ok"]
    if args.retry_failed:
        return [p for p in payloads
                if ledger.get(p["group_id"], {}).get("status") == "failed"]
    return []


# ── batch submit + poll ──────────────────────────────────────────────────────
def build_requests(groups: list[dict]) -> list[dict]:
    reqs = []
    for p in groups:
        reqs.append({
            "custom_id": f"grp-{p['group_id']}",
            "params": {
                "model": MODEL_VERSION,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "tools": [EMIT_TOOL],
                "tool_choice": {"type": "tool", "name": "emit_concepts"},
                "messages": [{"role": "user", "content": build_user_prompt(p)}],
            },
        })
    return reqs


def run_batch(groups: list[dict]) -> dict[str, list]:
    """Submit, poll to completion, return custom_id -> message content blocks."""
    cl = client()
    reqs = build_requests(groups)
    batch = cl.messages.batches.create(requests=reqs)
    print(f"submitted batch {batch.id} with {len(reqs)} requests", flush=True)

    while True:
        batch = cl.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"  status={batch.processing_status} "
              f"succeeded={counts.succeeded} errored={counts.errored} "
              f"processing={counts.processing}", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(20)

    results: dict[str, list] = {}
    for res in cl.messages.batches.results(batch.id):
        cid = res.custom_id
        if res.result.type == "succeeded":
            results[cid] = res.result.message.content
        else:
            results[cid] = None  # errored / expired / canceled
            print(f"  request {cid} did not succeed: {res.result.type}", flush=True)
    return results


# ── result -> ledger records ─────────────────────────────────────────────────
def process_results(groups: list[dict], results: dict[str, list]) -> list[dict]:
    records = []
    for p in groups:
        gid = p["group_id"]
        forms = [m["form"] for m in p["members"]]
        content = results.get(f"grp-{gid}")
        if content is None:
            records.append({"group_id": gid, "status": "failed",
                            "reason": "no successful response",
                            "total_mention_count": p["total_mention_count"],
                            "input_forms": forms})
            continue
        part = parse_tool_result(content)
        ok, reason = validate(part, forms)
        if not ok:
            records.append({"group_id": gid, "status": "failed", "reason": reason,
                            "total_mention_count": p["total_mention_count"],
                            "input_forms": forms})
            continue
        concepts = []
        for c in part.concepts:
            label = c["canonical_label"].strip()
            concepts.append({
                "concept_id": concept_id(label),
                "label": label,
                "member_forms": c["member_forms"],
                "confidence": float(c.get("confidence", 0.0)),
                "reason": (c.get("canonical_reason") or "").strip(),
                "merge_evidence": (c.get("merge_evidence") or "").strip(),
            })
        records.append({"group_id": gid, "status": "ok",
                        "total_mention_count": p["total_mention_count"],
                        "input_forms": forms, "concepts": concepts})
    return records


# ── singletons ───────────────────────────────────────────────────────────────
def singleton_records() -> list[dict]:
    """Each singleton form becomes its own concept, method=singleton."""
    payload_forms = set()
    for p in load_payloads():
        for m in p["members"]:
            payload_forms.add(m["form"])
    # all normalised forms
    counts: dict[str, int] = defaultdict(int)
    with SURFACE_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[r["normalised_form"]] += int(r["count"])
    records = []
    for form, cnt in counts.items():
        if form in payload_forms:
            continue
        records.append({
            "group_id": f"single:{form}", "status": "singleton",
            "total_mention_count": cnt, "input_forms": [form],
            "concepts": [{
                "concept_id": concept_id(form), "label": form,
                "member_forms": [form], "confidence": 1.0,
                "reason": "singleton — no candidate merge partners",
                "merge_evidence": "",
            }],
        })
    return records


def surface_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with SURFACE_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[r["normalised_form"]] += int(r["count"])
    return counts


# ── writers (rebuild outputs from the full ledger) ───────────────────────────
def write_outputs() -> dict:
    """Rebuild every output from the full ledger, with cross-group dedup.

    Two concepts that share a canonical label — or differ only by its
    normalisation (case / whitespace) — hash to the SAME concept_id and are
    therefore aggregated into one concept here. Singletons take part in the
    same aggregation, so a singleton form carrying a group's label collapses
    into that group's concept. The canonical label kept for a merged id is the
    variant with the largest mention weight (ties: shortest, then lexical).
    """
    ledger = load_ledger()
    ok = [r for r in ledger.values() if r["status"] in ("ok", "singleton")]
    failed = [r for r in ledger.values() if r["status"] == "failed"]
    counts = surface_counts()

    # Aggregate every emitted concept instance under its label-hash id.
    agg: dict[str, dict] = {}
    instances_before = 0
    for r in ok:
        method = "singleton" if r["status"] == "singleton" else "merge"
        is_group = r["status"] == "ok"
        for c in r["concepts"]:
            instances_before += 1
            cid = c["concept_id"]
            a = agg.setdefault(cid, {
                "labels": defaultdict(int), "forms": set(),
                "reason": {}, "assign": {}, "from_group": False,
            })
            weight = sum(counts.get(f, 0) for f in c["member_forms"])
            a["labels"][c["label"]] += weight
            a["reason"][c["label"]] = c.get("reason", "")
            if is_group:
                a["from_group"] = True
            for form in c["member_forms"]:
                a["forms"].add(form)
                a["assign"][form] = (method, c["confidence"])

    # Resolve one canonical label + row per concept id.
    concepts: dict[str, dict] = {}
    assign: dict[str, tuple[str, str, float]] = {}
    for cid, a in agg.items():
        label = sorted(a["labels"].items(),
                       key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
        concepts[cid] = {
            "concept_id": cid, "label": label, "type": "",
            "description": a["reason"].get(label, ""),
            "source": "step3-merge" if a["from_group"] else "step3-singleton",
            "prompt_version": PROMPT_VERSION, "model_version": MODEL_VERSION,
        }
        for form, (method, conf) in a["assign"].items():
            assign[form] = (cid, method, conf)
    collapsed = instances_before - len(concepts)

    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    with CONCEPT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "concept_id", "label", "type", "description", "source",
            "prompt_version", "model_version"])
        w.writeheader()
        for cid in sorted(concepts):
            w.writerow(concepts[cid])

    # surface_form.csv — assign concept_id/method/confidence by normalised_form.
    rows = list(csv.DictReader(SURFACE_CSV.open(encoding="utf-8")))
    fields = ["form", "normalised_form", "count", "concept_id",
              "method", "confidence", "config_version"]
    assigned = 0
    for row in rows:
        a = assign.get(row["normalised_form"])
        if a:
            row["concept_id"], row["method"], row["confidence"] = a[0], a[1], a[2]
            assigned += 1
    with SURFACE_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # reports/merges.csv — one row per merged group (exclude singletons), largest first.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    merged = [r for r in ok if r["status"] == "ok"]
    merged.sort(key=lambda r: -r["total_mention_count"])
    with MERGES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "total_mention_count", "n_input_forms",
                    "n_concepts", "input_forms", "resulting_concepts",
                    "confidence", "reason"])
        for r in merged:
            concs = r["concepts"]
            res = " || ".join(
                f"{c['label']} [{'; '.join(c['member_forms'])}]" for c in concs)
            conf = "; ".join(f"{c['label']}:{c['confidence']:.2f}" for c in concs)
            reason = " | ".join(
                (c["reason"] + (f" (merge evidence: {c['merge_evidence']})"
                                if c["merge_evidence"] else ""))
                for c in concs)
            w.writerow([r["group_id"], r["total_mention_count"],
                        len(r["input_forms"]), len(concs),
                        " | ".join(r["input_forms"]), res, conf, reason])

    # reports/merge_failures.csv
    failed.sort(key=lambda r: -r.get("total_mention_count", 0))
    with FAILURES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "total_mention_count", "input_forms", "reason"])
        for r in failed:
            w.writerow([r["group_id"], r.get("total_mention_count", 0),
                        " | ".join(r["input_forms"]), r.get("reason", "")])

    # Per-concept rollups for the reports.
    per: list[dict] = []
    for cid, a in agg.items():
        forms = sorted(a["forms"])
        per.append({
            "concept_id": cid, "label": concepts[cid]["label"],
            "source": concepts[cid]["source"], "n_forms": len(forms),
            "total_mentions": sum(counts.get(f, 0) for f in forms),
            "member_forms": forms,
        })
    from_group = sum(1 for p in per if p["source"] == "step3-merge")
    from_singleton = len(per) - from_group
    single_form = sum(1 for p in per if p["n_forms"] == 1)

    # reports/top_concepts.csv — top 100 by total mention count.
    per.sort(key=lambda p: (-p["total_mentions"], p["label"]))
    with TOP_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "concept_id", "label", "source",
                    "n_forms", "total_mentions", "member_forms"])
        for i, p in enumerate(per[:100], 1):
            w.writerow([i, p["concept_id"], p["label"], p["source"],
                        p["n_forms"], p["total_mentions"],
                        " | ".join(p["member_forms"])])

    stats = {
        "prompt_version": PROMPT_VERSION, "model_version": MODEL_VERSION,
        "concepts": len(concepts), "forms_assigned": assigned,
        "concepts_from_groups": from_group,
        "concepts_from_singletons": from_singleton,
        "concepts_single_form": single_form,
        "concept_instances_before_dedup": instances_before,
        "concepts_collapsed_by_dedup": collapsed,
        "merged_groups": len(merged), "failed_groups": len(failed),
    }
    SUMMARY_JSON.write_text(json.dumps(stats, indent=2))
    return stats


# ── validation of the full assignment ────────────────────────────────────────
def validate_full() -> list[str]:
    """Return a list of problems; empty means every form maps cleanly."""
    problems = []
    rows = list(csv.DictReader(SURFACE_CSV.open(encoding="utf-8")))
    by_form: dict[str, set] = defaultdict(set)
    total_forms = set()
    for row in rows:
        total_forms.add(row["normalised_form"])
        if row["concept_id"]:
            by_form[row["normalised_form"]].add(row["concept_id"])
    unassigned = total_forms - set(by_form)
    if unassigned:
        problems.append(f"{len(unassigned)} forms with no concept_id")
    multi = {f for f, cids in by_form.items() if len(cids) > 1}
    if multi:
        problems.append(f"{len(multi)} forms with >1 concept_id")
    # empty labels + shared canonical labels + id stability
    empties = 0
    label_key_to_ids: dict[str, set] = defaultdict(set)
    id_ok = 0
    if CONCEPT_CSV.exists():
        for c in csv.DictReader(CONCEPT_CSV.open(encoding="utf-8")):
            if not c["label"].strip():
                empties += 1
            label_key_to_ids[_label_key(c["label"])].add(c["concept_id"])
            if c["concept_id"] == concept_id(c["label"]):
                id_ok += 1
    if empties:
        problems.append(f"{empties} concepts with empty label")
    shared = {k: ids for k, ids in label_key_to_ids.items() if len(ids) > 1}
    if shared:
        problems.append(f"{len(shared)} canonical labels shared by >1 concept id")
    n_concepts = sum(len(v) for v in label_key_to_ids.values())
    if id_ok != n_concepts:
        problems.append(f"{n_concepts - id_ok} concept ids not label-hash derived")
    print(f"validation: {len(total_forms)} distinct forms, "
          f"{len(by_form)} assigned, {len(unassigned)} unassigned, "
          f"{len(multi)} multi-concept, {empties} empty-label, "
          f"{len(shared)} shared-label, {n_concepts - id_ok} non-stable-id",
          flush=True)
    return problems


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--top", type=int)
    g.add_argument("--rest", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--singletons", action="store_true")
    g.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env()

    if args.singletons:
        recs = singleton_records()
        print(f"singleton concepts to write: {len(recs)}", flush=True)
        if args.dry_run:
            return
        append_ledger(recs)
    else:
        payloads = load_payloads()
        ledger = load_ledger()
        groups = select_groups(payloads, args, ledger)
        print(f"selected {len(groups)} groups", flush=True)
        if not groups:
            print("nothing to do", flush=True)
            return
        if args.dry_run:
            print("sample user prompt for the top group:\n")
            print(build_user_prompt(groups[0])[:1200])
            return
        results = run_batch(groups)
        recs = process_results(groups, results)
        n_ok = sum(1 for r in recs if r["status"] == "ok")
        n_fail = sum(1 for r in recs if r["status"] == "failed")
        print(f"processed: {n_ok} ok, {n_fail} failed", flush=True)
        append_ledger(recs)

    stats = write_outputs()
    print(f"concepts={stats['concepts']} (groups={stats['concepts_from_groups']} "
          f"singletons={stats['concepts_from_singletons']}) "
          f"single_form={stats['concepts_single_form']} "
          f"collapsed_by_dedup={stats['concepts_collapsed_by_dedup']} "
          f"forms_assigned={stats['forms_assigned']} "
          f"merged_groups={stats['merged_groups']} failed_groups={stats['failed_groups']}",
          flush=True)
    problems = validate_full()
    if problems:
        print("VALIDATION PROBLEMS (expected until the full run completes):", flush=True)
        for p in problems:
            print("  - " + p, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
