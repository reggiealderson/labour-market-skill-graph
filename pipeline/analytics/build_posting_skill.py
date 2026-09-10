"""Stage 1 — build the analytics backbone: posting + posting_skill.

Inputs
  data/extracted/<year>/<occ>.jsonl  ad metadata (id, job_title, company, ...)
  data/skills/<year>/<occ>.jsonl     per-ad extracted skills (raw surface form)
  data/normalise/surface_form.csv    form (lower) -> concept_id  [the bridge]
  data/concepts/concept.csv          concept_id -> label
  data/concepts/concept_type.csv     concept_id -> type (blank = held)
  data/concepts/concept_category.csv concept_id -> primary_category

Join key: skill.lower().strip()  (probe: 96-98% coverage; exact case ~40%).

Outputs (data/analytics/)
  posting.csv        one row per in-scope ad (the penetration denominator).
  posting_skill.csv  one row per (year, occupation, posting_id, concept_id);
                     type + primary_category + label joined on.
  build_coverage.json  per-cell coverage + unmatched surface forms.

Grain of posting_skill = (year, occupation, posting_id, concept_id), deduped:
a concept named twice in one ad is one row. requirement = strongest seen.
"""
import csv, json, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/analytics")
os.makedirs(OUT, exist_ok=True)

SCOPE = {
    "2021": ["data_analyst", "data_engineer", "data_scientist"],
    "2026": ["ai_engineer", "analytics_engineer", "data_analyst",
             "data_engineer", "data_scientist", "machine_learning_engineer"],
}
REQ_RANK = {"required": 3, "preferred": 2, "unclear": 1, "": 0}


def load_bridge():
    m = {}
    with open(os.path.join(ROOT, "data/normalise/surface_form.csv")) as f:
        for row in csv.DictReader(f):
            cid = (row.get("concept_id") or "").strip()
            if cid:
                m[row["form"].lower().strip()] = cid
    # curated analytics aliases (acronyms the normaliser missed); only add a
    # form the bridge does not already have, so we never override the pipeline.
    apath = os.path.join(ROOT, "config/analytics_aliases.csv")
    if os.path.exists(apath):
        added = 0
        with open(apath) as f:
            for row in csv.DictReader(f):
                key = row["form"].lower().strip()
                if key and key not in m:
                    m[key] = row["concept_id"].strip()
                    added += 1
        print(f"aliases added to bridge: {added}")
    return m


def load_map(path, key, val):
    m = {}
    with open(os.path.join(ROOT, path)) as f:
        for row in csv.DictReader(f):
            m[row[key]] = (row.get(val) or "").strip()
    return m


def main():
    bridge = load_bridge()
    label = load_map("data/concepts/concept.csv", "concept_id", "label")
    ctype = load_map("data/concepts/concept_type.csv", "concept_id", "type")
    ccat = load_map("data/concepts/concept_category.csv", "concept_id", "primary_category")

    posting_rows = []
    ps_rows = []
    coverage = {}

    for year, occs in SCOPE.items():
        for occ in occs:
            spath = os.path.join(ROOT, f"data/skills/{year}/{occ}.jsonl")
            epath = os.path.join(ROOT, f"data/extracted/{year}/{occ}.jsonl")
            if not os.path.exists(spath):
                continue

            meta = {}
            if os.path.exists(epath):
                with open(epath) as f:
                    for line in f:
                        r = json.loads(line)
                        meta[str(r["id"])] = r

            n_ment = n_match = 0
            unmatched = collections.Counter()
            meta_miss = 0

            with open(spath) as f:
                for line in f:
                    rec = json.loads(line)
                    pid = str(rec["id"])
                    # dedupe concepts within this ad; keep strongest requirement
                    best = {}
                    ment = 0
                    for s in rec.get("skills", []):
                        ment += 1
                        n_ment += 1
                        key = s.get("skill", "").lower().strip()
                        cid = bridge.get(key)
                        if not cid:
                            unmatched[key] += 1
                            continue
                        n_match += 1
                        req = s.get("requirement", "") or ""
                        cur = best.get(cid)
                        if cur is None or REQ_RANK.get(req, 0) > REQ_RANK.get(cur[0], 0):
                            best[cid] = (req, cur[1] + 1 if cur else 1)
                        else:
                            best[cid] = (cur[0], cur[1] + 1)

                    m = meta.get(pid)
                    if m is None:
                        meta_miss += 1
                    posting_rows.append({
                        "posting_id": pid, "year": year, "occupation": occ,
                        "status": rec.get("status", ""),
                        "n_skill_mentions": ment, "n_concepts_mapped": len(best),
                        "job_title": (m or {}).get("job_title", ""),
                        "company": (m or {}).get("company", ""),
                        "date": (m or {}).get("date", ""),
                        "location": (m or {}).get("location", ""),
                        "source_url": (m or {}).get("source_url", ""),
                    })
                    for cid, (req, cnt) in best.items():
                        ps_rows.append({
                            "year": year, "occupation": occ, "posting_id": pid,
                            "concept_id": cid, "concept_label": label.get(cid, ""),
                            "concept_type": ctype.get(cid, ""),
                            "primary_category": ccat.get(cid, ""),
                            "held": "1" if not ctype.get(cid, "") else "0",
                            "requirement": req, "n_mentions": cnt,
                        })

            coverage[f"{year}/{occ}"] = {
                "mentions": n_ment, "matched": n_match,
                "coverage_pct": round(100 * n_match / n_ment, 2) if n_ment else 0,
                "meta_miss": meta_miss,
                "top_unmatched": unmatched.most_common(15),
            }

    with open(os.path.join(OUT, "posting.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(posting_rows[0].keys()))
        w.writeheader(); w.writerows(posting_rows)
    with open(os.path.join(OUT, "posting_skill.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ps_rows[0].keys()))
        w.writeheader(); w.writerows(ps_rows)
    with open(os.path.join(OUT, "build_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)

    print(f"posting rows       : {len(posting_rows)}")
    print(f"posting_skill rows : {len(ps_rows)}")
    print(f"{'cell':32s} {'ads':>5s} {'cov%':>6s} {'meta_miss':>9s}")
    counts = collections.Counter((r["year"], r["occupation"]) for r in posting_rows)
    for cell, cov in coverage.items():
        y, o = cell.split("/")
        print(f"{cell:32s} {counts[(y, o)]:5d} {cov['coverage_pct']:6.1f} {cov['meta_miss']:9d}")


if __name__ == "__main__":
    main()
