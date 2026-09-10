"""Apply hand-adjudicated vendor corrections to concept_facet.csv.

Stage G, facet_vendor_ecosystem. The classifier over-attributed GPU/accelerator
concepts to nvidia (competitors' products and generic GPU work). These are the
taxonomy author's adjudicated corrections, applied here with by=human provenance
so a human decision is never recorded as a model one.

Each correction rewrites the concept's vendor row in place:
  - value  -> the corrected value
  - by     -> "human"
  - model  -> ""  (blank: the value is human-assigned, not model output)

Read-only inputs except the target CSV, which is backed up first.

Usage:  python3 scripts/apply_vendor_corrections.py
"""
import csv
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACET_CSV = ROOT / "taxonomy/concept_facet.csv"
CONCEPTS = ROOT / "data/concepts/concept_label_type.csv"
FACET_ID = "facet_vendor_ecosystem"

# label -> corrected value  (author adjudication, 2026-09-07)
CORRECTIONS = {
    # wrong vendor -> none
    "ROCm": "none",
    "vitis ai": "none",
    "QNN": "none",
    "arm ethos": "none",
    # wrong vendor -> openai (OpenAI's Triton language, not NVIDIA Triton)
    "Triton (CUDA programming)": "openai",
    # generic GPU -> none
    "distributed gpu training": "none",
    "gpu acceleration": "none",
    "gpu clusters": "none",
    "GPU computing": "none",
    "gpu deployment": "none",
    "gpu enabled machine learning": "none",
    "GPU kernels": "none",
    "GPU workloads": "none",
    "gpu-accelerated ai workloads": "none",
    "GPU-based training and inference": "none",
    "gpu/ai servers": "none",
    "multi-GPU inference": "none",
    "hardware-aware optimization": "none",
    "edge hardware optimization": "none",
    "high-performance inferencing": "none",
    "vector programming": "none",
    "guardrails frameworks": "none",
    # borderline -> none
    "InfiniBand": "none",
}


def main():
    # label -> concept_id
    label_to_id = {}
    with open(CONCEPTS, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            label_to_id[r["label"]] = r["concept_id"]

    # resolve corrections to concept_ids; report any label that does not resolve
    target = {}  # concept_id -> value
    unresolved = []
    for label, val in CORRECTIONS.items():
        cid = label_to_id.get(label)
        if cid is None:
            unresolved.append(label)
        else:
            target[cid] = val
    if unresolved:
        print("ERROR: these labels did not match a concept_id:")
        for l in unresolved:
            print(f"  - {l!r}")
        return 1

    # read facet file
    with open(FACET_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    # each target concept must have exactly one vendor row (the nvidia FP)
    vendor_rows_by_cid = {}
    for r in rows:
        if r["facet_id"] == FACET_ID:
            vendor_rows_by_cid.setdefault(r["concept_id"], []).append(r)
    problems = []
    for cid in target:
        vr = vendor_rows_by_cid.get(cid, [])
        if len(vr) != 1:
            problems.append((cid, [x["value"] for x in vr]))
        elif vr[0]["value"] != "nvidia":
            problems.append((cid, [vr[0]["value"]]))
    if problems:
        print("ERROR: unexpected vendor rows for these concepts "
              "(expected exactly one 'nvidia' row):")
        for cid, vals in problems:
            print(f"  - {cid}: {vals}")
        return 1

    # backup
    backup = ROOT / f"taxonomy/archive/concept_facet.pre_vendor_correction_{date.today()}.csv"
    shutil.copy2(FACET_CSV, backup)

    # apply in place
    applied = 0
    for r in rows:
        if r["facet_id"] == FACET_ID and r["concept_id"] in target:
            r["value"] = target[r["concept_id"]]
            r["by"] = "human"
            r["model"] = ""
            applied += 1

    with open(FACET_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"backup: {backup.relative_to(ROOT)}")
    print(f"applied {applied} corrections (by=human)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
