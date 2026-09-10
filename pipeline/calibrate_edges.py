"""Edge calibration — the missing half of item 4's measurement.

The acronym calibration (24 pairs) proved the model unsafe in one direction on
PAIRS. Nobody has ever measured it on EDGES, which is what most of the review
sheet actually is. This runs the same blind test, on the human's 332 edge
verdicts recorded today (319 correct/rollup, 13 wrong/reject).

BLIND BY CONSTRUCTION: the human verdicts already exist in decisions.json, and
adjudicate.py's prompt has no access to that file — it only ever sees the
parent/child strings and mention counts, the same as edge_item() builds for the
production --edges run. Comparing afterwards is the same method item 4 already
used and documents in RESOLVER_STATE.md.

Uses the same SYSTEM prompt, TOOL schema and model/escalation policy as
production adjudication — this is not a separate, easier test.

Run: python3 pipeline/calibrate_edges.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adjudicate import (SYSTEM, TOOL, build_requests, edge_item, load_dotenv,
                        run_batch, run_standard)
from resolver_decisions import edge_key, load as load_decisions
from resolver_level1 import LEVEL1_VERSION
from resolver_policy import load as load_policy

import anthropic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "resolver"


def main():
    load_dotenv()
    policy = load_policy()
    decisions = load_decisions(level1_version=LEVEL1_VERSION)

    # Every human edge decision made today, both directions.
    human_edges = {k: v for k, v in decisions.edges.items() if v["by"] == "human"}
    print(f"{len(human_edges)} human edge decisions on file")

    # Mention counts survive in the prepass regardless of reject/rollup status.
    prepass = json.loads((OUT / "edge_prepass.json").read_text())
    by_key = {(e["parent"], e["child"]): e
             for e in prepass["obvious"] + prepass["remainder"]}

    items, skipped = [], []
    for i, key in enumerate(human_edges):
        parent, child = key.split(" || ")
        e = by_key.get((parent, child))
        if e is None:
            skipped.append(key)
            continue
        items.append(edge_item(i, e))

    if skipped:
        print(f"  {len(skipped)} human decisions have no mention record, "
              f"skipped: {skipped}")
    print(f"{len(items)} items going to the model, blind\n")

    model = policy.get("adjudication.model")
    escalation = policy.get("adjudication.escalation_model")
    size = policy.get("adjudication.items_per_request")
    batch_min = policy.get("adjudication.batch_api_min_items")

    reqs = build_requests(items, model, size)
    # ALWAYS standard, regardless of batch_api_min_items. This is an
    # interactive calibration run a person is waiting on, not a production
    # batch — exactly the case RESOLVER_STATE.md already warns about: sending
    # the 24-pair calibration to the batch queue once traded a fast answer for
    # an unbounded wait to save about a cent. 332 items on the standard API
    # costs cents and returns in minutes.
    send = run_standard
    print(f"model {model} (escalation {escalation} on uncertain)")
    print(f"{len(items)} items -> {len(reqs)} requests of {size}")
    print(f"transport: standard (forced — interactive calibration, not a "
          f"production batch; batch_api_min_items={batch_min} does not apply here)\n")

    client = anthropic.Anthropic()
    print("SONNET PASS")
    verdicts = send(client, reqs)

    unsure = [it for it in items if verdicts.get(it["id"], {}).get("verdict") == "uncertain"]
    if unsure:
        print(f"\nOPUS ESCALATION — {len(unsure)} uncertain")
        esc = build_requests(unsure, escalation, size)
        for item_id, v in send(client, esc).items():
            v["escalated"] = True
            verdicts[item_id] = v

    out = [{
        "id": it["id"], "kind": it["kind"], "key": list(it["key"]),
        "verdict": verdicts.get(it["id"], {}).get("verdict", "MISSING"),
        "reason": verdicts.get(it["id"], {}).get("reason", ""),
        "escalated": verdicts.get(it["id"], {}).get("escalated", False),
        "model": escalation if verdicts.get(it["id"], {}).get("escalated") else model,
    } for it in items]

    path = OUT / "edge_calibration_model_verdicts.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    missing = sum(1 for r in out if r["verdict"] == "MISSING")
    print(f"\n{len(out)} verdicts -> {path.name}"
          + (f"  ({missing} MISSING)" if missing else ""))


if __name__ == "__main__":
    main()
