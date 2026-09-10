"""
FROZEN 2026-08-27. DO NOT EXTEND. DO NOT ADD A CORE 6 ACRONYM BLOCK.

This script exists for one purpose: to reproduce the seven occupations
completed under the retired scope. The resolver replaces it for everything new.

Adding Core 6 acronym entries here would be configuration for a script that is
going to be deleted, and would create a THIRD map to keep in agreement with the
one in normalise_skills.py and the one in pipeline/acronyms.py.

It also cannot answer the resolver's design questions, because it reads one
occupation at a time. The resolver's value is that a decision made for one
occupation is reused by all the others; a single-occupation view shows none of
that.

  KNOWN DEFECT — the acronym map here disagrees with normalise_skills.py.
  BASE_ACRONYM_MAP has 9 entries plus per-occupation additions; that file has
  one, and they conflict on "gaap". Left in place deliberately: the completed
  output was produced with the divergence present. Full account in the header
  of normalise_skills.py.

────────────────────────────────────────────────────────────────────────────────

Generate high-impact fuzzy clusters for human review.

For a given occupation, runs normalisation steps 1-4 (code only),
then surfaces only clusters where at least one member skill has
penetration >= REVIEW_THRESHOLD in either year. These are the only
clusters worth reviewing manually — the rest can be trusted to the LLM.

Usage:
    python3 pipeline/generate_review_clusters.py --occupation data_scientist
    python3 pipeline/generate_review_clusters.py --occupation software_engineer

Output:
    - Prints clusters to stdout for in-conversation review
    - Saves full cluster list to data/skills/{occupation}_review_clusters.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from skills_io import load_skill_records  # noqa: E402

SKILLS_DIR       = Path("data/skills")
EXTRACTED_DIR    = Path("data/extracted")
YEARS            = ["2021", "2026"]
REVIEW_THRESHOLD = 0.05   # 5% penetration in either year triggers review
FUZZY_THRESHOLD  = 88
MIN_FREQ         = 2
MAX_LEN          = 60

# ── Steps 1-3 (same as normalise_skills.py) ───────────────────────────────────

BASE_ACRONYM_MAP = {
    "llm":   "large language models",
    "llms":  "large language models",
    "nlp":   "natural language processing",
    "genai": "generative ai",
    "rag":   "retrieval-augmented generation",
    "ml":    "machine learning",
    "eda":   "exploratory data analysis",
    "gcp":   "google cloud platform",
    "mcp":   "model context protocol",
}

# Occupation-specific acronym additions
OCC_ACRONYM_ADDITIONS = {
    "registered_nurse":  {"rn": "registered nurse", "ehr": "electronic health records",
                          "emr": "electronic medical records", "bls": "basic life support",
                          "acls": "advanced cardiac life support"},
    "financial_analyst": {"fp&a": "financial planning and analysis", "dcf": "discounted cash flow",
                          "irr": "internal rate of return", "npv": "net present value"},
    "accountant":        {"gaap": "generally accepted accounting principles",
                          "ifrs": "international financial reporting standards",
                          "cpa": "cpa", "erp": "erp"},
    "graphic_designer":  {"ui": "ui design", "ux": "ux design",
                          "ui/ux": "ui/ux design"},
}


def steps_1_2_3(s: str, acronym_map: dict) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s*\([^)]{1,60}\)\s*$", "", s).strip()
    return acronym_map.get(s, s)


def build_clusters(freq_normed: Counter) -> list[set]:
    candidates = [s for s, c in freq_normed.items() if c >= MIN_FREQ and len(s) <= MAX_LEN]
    first_word_groups: dict[str, list] = defaultdict(list)
    for s in candidates:
        first = s.split()[0] if s.split() else s
        first_word_groups[first].append(s)

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for group in first_word_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if fuzz.token_set_ratio(group[i], group[j]) >= FUZZY_THRESHOLD:
                    union(group[i], group[j])

    clusters: dict = defaultdict(set)
    for s in candidates:
        clusters[find(s)].add(s)

    return sorted(
        [m for m in clusters.values() if len(m) > 1],
        key=lambda m: sum(freq_normed[x] for x in m),
        reverse=True,
    )


def penetration_by_string(records: list[dict], acronym_map: dict) -> Counter:
    """Per-record deduped penetration count keyed on normed string."""
    counter: Counter = Counter()
    for r in records:
        seen: set = set()
        for skill in r["skills"]:
            key = steps_1_2_3(skill, acronym_map)
            if key not in seen:
                counter[key] += 1
                seen.add(key)
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occupation", required=True)
    parser.add_argument("--threshold", type=float, default=REVIEW_THRESHOLD)
    args = parser.parse_args()

    occ = args.occupation
    threshold = args.threshold

    acronym_map = {**BASE_ACRONYM_MAP, **OCC_ACRONYM_ADDITIONS.get(occ, {})}

    # Load raw extractions
    year_records: dict[str, list] = {}
    for year in YEARS:
        path = SKILLS_DIR / year / f"{occ}.jsonl"
        if not path.exists():
            print(f"Missing: {path} — skipping {year}")
            continue
        year_records[year] = load_skill_records(path)

    if not year_records:
        print("No data found.")
        return

    # Combined freq for clustering
    all_records = [r for recs in year_records.values() for r in recs]
    freq_raw: Counter = Counter()
    for r in all_records:
        for skill in r["skills"]:
            freq_raw[skill] += 1

    freq_normed: Counter = Counter()
    for skill, count in freq_raw.items():
        freq_normed[steps_1_2_3(skill, acronym_map)] += count

    # Build penetration counters per year
    ns = {year: len(recs) for year, recs in year_records.items()}
    pen = {year: penetration_by_string(recs, acronym_map) for year, recs in year_records.items()}

    def max_penetration(string: str) -> float:
        return max(pen[y].get(string, 0) / ns[y] for y in year_records)

    # Build clusters
    clusters = build_clusters(freq_normed)

    # Filter to high-impact only
    review_clusters = []
    for members in clusters:
        if any(max_penetration(m) >= threshold for m in members):
            review_clusters.append(members)

    print(f"\nOccupation: {occ}")
    print(f"Total fuzzy clusters: {len(clusters)}")
    print(f"Clusters with ≥{threshold*100:.0f}% penetration (need review): {len(review_clusters)}")
    print(f"Clusters below threshold (trust LLM): {len(clusters) - len(review_clusters)}")

    # Print review clusters
    years_available = list(year_records.keys())
    print(f"\n{'='*70}")
    print(f"  CLUSTERS FOR REVIEW  ({occ})")
    print(f"{'='*70}")
    for i, members in enumerate(review_clusters):
        items = sorted(members, key=lambda x: -freq_normed[x])
        total = sum(freq_normed[m] for m in members)
        pen_strs = []
        for y in years_available:
            n = ns[y]
            pen_strs.append(f"{y}: {pen[y].get(items[0], 0)/n*100:.1f}%")
        print(f"\nCluster {i+1}  [total mentions: {total}]  top-member penetration: {', '.join(pen_strs)}")
        for m in items:
            pen_detail = "  ".join(f"{y}={pen[y].get(m,0)/ns[y]*100:.1f}%" for y in years_available)
            print(f"  [{freq_normed[m]:4d}]  {pen_detail}  {m}")

    # Save to JSON for later reference
    out = {
        "occupation": occ,
        "threshold": threshold,
        "total_clusters": len(clusters),
        "review_cluster_count": len(review_clusters),
        "review_clusters": [
            {
                "cluster_rank": i + 1,
                "members": sorted(members, key=lambda x: -freq_normed[x]),
                "total_mentions": sum(freq_normed[m] for m in members),
                "member_penetration": {
                    m: {y: round(pen[y].get(m, 0) / ns[y], 4) for y in years_available}
                    for m in members
                },
            }
            for i, members in enumerate(review_clusters)
        ],
    }
    out_path = SKILLS_DIR / f"{occ}_review_clusters.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
