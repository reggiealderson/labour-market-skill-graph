"""
FROZEN 2026-08-27. DO NOT EXTEND. DO NOT FIX THE DEFECTS LISTED BELOW.

This script exists for one purpose: to reproduce the normalisation of the seven
occupations completed under the retired scope. It is NOT carried forward to the
Core 6 — the resolver replaces it.

The defects below are real and are deliberately left in place. That output was
produced with them present, so correcting them changes the result and destroys
reproducibility. They are recorded here so nobody re-derives them and fixes them.

  DEFECT 1 — acronym map divergence.
    ACRONYM_MAP here has ONE entry. generate_review_clusters.py carries a
    9-entry base map plus per-occupation additions. They also conflict outright:
    this file maps "gaap" -> "gaap" (no expansion) while the clusters script
    maps it -> "generally accepted accounting principles". Review decisions are
    keyed on the post-expansion string, so a decision made against a cluster
    whose string expanded differently here never matched and was silently lost.

  DEFECT 2 — build_full_norm_map discards decisions silently.
    A decision whose cluster index exceeds len(sorted_clusters) is skipped with
    a bare `continue` and no message. Measured against the archived decision
    files this is reachable, not theoretical: ai_engineer references cluster 53
    and data_scientist references cluster 205.

  DEFECT 3 — MANUAL_PATCHES "accounting" does not drop.
    The comment says "drop bare generic terms", but the entry is an identity
    mapping, which KEEPS the skill. Dropping requires None.

The correct home for all three is the resolver: ONE acronym map shared across
every occupation as its first deterministic level (seeded in
pipeline/acronyms.py), an explicit error on out-of-range decisions, and None
meaning drop.

────────────────────────────────────────────────────────────────────────────────

Skill normalisation — set OCCUPATION at top before running.

Steps:
  1. Case fold + strip whitespace
  2. Strip parenthetical expansions  e.g. "Large Language Models (LLMs)" → "large language models"
  3. Acronym expansion lookup table (ACRONYM_MAP — update per occupation)
  4. In-conversation cluster merge decisions (from normalisation_decisions_{OCCUPATION}.json)
  5. Manual patches for cross-cluster misses (MANUAL_PATCHES — update per occupation)

Input:  data/skills/{year}/{OCCUPATION}.jsonl
Output: data/skills/{year}/{OCCUPATION}_normed.jsonl  — same format, skills replaced
        data/skills/{OCCUPATION}_norm_map.json        — raw_string → canonical (or null if dropped)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from skills_io import load_skill_records  # noqa: E402

SKILLS_DIR  = Path("data/skills")
YEARS       = ["2021", "2026"]
OCCUPATION  = "accountant"

# ── Step 3: acronym map ────────────────────────────────────────────────────────

ACRONYM_MAP = {
    "gaap":  "gaap",
}

# ── Step 5: manual patches (cross-cluster misses + known issues) ───────────────

MANUAL_PATCHES = {
    # excel variants with different first words
    "ms excel":                                 "excel",
    # drop bare generic terms
    "accounting":                               "accounting",
}

# One-to-many patches: raw string → list of canonicals (e.g. "etl/elt" → both etl and elt)
MULTI_PATCHES: dict[str, list[str]] = {}

# Per-occupation spacing/compound-word fixes injected by run_normalise_all.py
EXTRA_MANUAL_PATCHES: dict[str, str] = {}

# ── Normalisation function ─────────────────────────────────────────────────────

def steps_1_2_3(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s*\([^)]{1,60}\)\s*$", "", s).strip()
    return ACRONYM_MAP.get(s, s)


def build_full_norm_map(llm_decisions: list, sorted_clusters: list, freq_normed: Counter) -> dict:
    """Returns raw_normed_string → canonical (or None if dropped)."""
    mapping: dict[str, str | None] = {}

    # LLM decisions
    for d in llm_decisions:
        idx = d["cluster"] - 1
        if idx >= len(sorted_clusters):
            continue
        for drop_str in d.get("drop", []):
            mapping[drop_str] = None
        for group in d.get("groups", []):
            canonical = group["canonical"].lower()
            for member in group.get("members", []):
                mapping[member] = canonical

    # Manual patches override
    mapping.update(MANUAL_PATCHES)
    mapping.update(EXTRA_MANUAL_PATCHES)

    return mapping


def full_normalize(raw: str, norm_map: dict) -> str | list[str] | None:
    s = steps_1_2_3(raw)
    if s in MULTI_PATCHES:
        return MULTI_PATCHES[s]
    if s in norm_map:
        return norm_map[s]
    return s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load raw extractions
    all_records: list[dict] = []
    year_records: dict[str, list] = {}
    for year in YEARS:
        path = SKILLS_DIR / year / f"{OCCUPATION}.jsonl"
        recs = load_skill_records(path)
        year_records[year] = recs
        all_records.extend(recs)

    # Build freq table after steps 1-3 (needed for cluster rebuild)
    freq_raw: Counter = Counter()
    for r in all_records:
        for skill in r["skills"]:
            freq_raw[skill] += 1

    freq_normed: Counter = Counter()
    for skill, count in freq_raw.items():
        freq_normed[steps_1_2_3(skill)] += count

    # Rebuild clusters (same logic as during adjudication)
    candidates = [s for s, c in freq_normed.items() if c >= 2 and len(s) <= 60]
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
                if fuzz.token_set_ratio(group[i], group[j]) >= 88:
                    union(group[i], group[j])

    clusters: dict = defaultdict(set)
    for s in candidates:
        clusters[find(s)].add(s)
    multi = {root: members for root, members in clusters.items() if len(members) > 1}

    def cluster_total(m):
        return sum(freq_normed[x] for x in m)

    sorted_clusters = sorted(multi.values(), key=cluster_total, reverse=True)

    # Load LLM decisions
    decisions_path = SKILLS_DIR / f"normalisation_decisions_{OCCUPATION}.json"
    llm_decisions = json.loads(decisions_path.read_text())

    # Build full normalisation map
    norm_map = build_full_norm_map(llm_decisions, sorted_clusters, freq_normed)

    # Save the map
    map_path = SKILLS_DIR / f"{OCCUPATION}_norm_map.json"
    map_path.write_text(json.dumps(norm_map, indent=2, ensure_ascii=False))
    print(f"Norm map saved: {len(norm_map)} entries → {map_path}")

    # Apply to each year and write normed output
    for year in YEARS:
        records = year_records[year]
        out_path = SKILLS_DIR / year / f"{OCCUPATION}_normed.jsonl"

        before_unique: set = set()
        after_unique: set = set()
        zero_before = zero_after = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                raw_skills = r["skills"]
                normed: list[str] = []
                seen: set = set()
                for skill in raw_skills:
                    before_unique.add(steps_1_2_3(skill))
                    canonical = full_normalize(skill, norm_map)
                    if canonical is None:
                        continue
                    targets = canonical if isinstance(canonical, list) else [canonical]
                    for c in targets:
                        if c not in seen:
                            normed.append(c)
                            after_unique.add(c)
                            seen.add(c)

                if not raw_skills:
                    zero_before += 1
                if not normed:
                    zero_after += 1

                out = {"id": r["id"], "skills": normed}
                if r.get("extraction_error"):
                    out["extraction_error"] = True
                f.write(json.dumps(out) + "\n")

        print(f"\n{year} {OCCUPATION}:")
        print(f"  Unique strings before normalisation: {len(before_unique)}")
        print(f"  Unique strings after  normalisation: {len(after_unique)}")
        print(f"  Records with 0 skills before: {zero_before}  after: {zero_after}")


if __name__ == "__main__":
    main()
