"""
SUPERSEDED 2026-08-27 by the resolver's level 4 (pipeline/resolver_level4.py).
DO NOT USE. Never run: no skill_groups_*.json config and no *_grouped.jsonl
output has ever existed, so this is dead code, not a reproduction dependency.

It cannot hold the rollup layer. The config is {canonical_skill: group_name} —
one group per skill — and group_postings does `group_map[s]`, returning a single
value. The rollup layer is a DAG: "azure sql" has two parents, "azure" and
"sql", because it really is both a cloud skill and a SQL skill. Loading a DAG
into this schema fails SILENTLY, keeping whichever parent was written last and
discarding the rest.

Rather than widen the schema on a path the resolver replaces, the resolver
carries DAG-capable family assignment itself. Kept for reference only.

────────────────────────────────────────────────────────────────────────────────

Stage 6 — Taxonomy grouping (occupation-specific, rerun-safe).

Reads:  data/skills/skill_groups_{occ}.json        — group config
        data/skills/{year}/{occ}_normed.jsonl       — normed postings

Writes: data/skills/{year}/{occ}_grouped.jsonl     — postings with skill_groups field

Config format (skill_groups_{occ}.json):
  {
    "version": 1,
    "description": "...",
    "groups": {
      "<canonical_skill>": "<group_name>",
      ...
    }
  }

Each output record adds a "skill_groups" field — the deduplicated, sorted list of
group names that apply to the posting (based on its normed canonical skills).
Skills not listed in the config are silently left ungrouped.

Rerun: just re-run — all grouped files are overwritten from normed + config.
Add a new group: update the config, re-run — all postings updated automatically.

Run from project root:
  python3 pipeline/group_skills.py
"""

import json
from pathlib import Path

# ── Set occupation here ────────────────────────────────────────────────────────
OCCUPATION = "data_scientist"

# ── Paths ──────────────────────────────────────────────────────────────────────
SKILLS_DIR  = Path("data/skills")
CONFIG_PATH = SKILLS_DIR / f"skill_groups_{OCCUPATION}.json"
YEARS       = ["2021", "2026"]


def load_config(path: Path) -> dict[str, str]:
    """Return {canonical_skill: group_name} from config file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            f"Create it first — see skill_groups_template.json for format."
        )
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg["groups"]


def group_postings(occ: str, year: str, group_map: dict[str, str]) -> list[dict]:
    """
    Read normed JSONL for (occ, year), add skill_groups field.
    Returns list of records with the original fields plus skill_groups.
    """
    in_path = SKILLS_DIR / year / f"{occ}_normed.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Normed JSONL not found: {in_path}")

    records = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        skills = record.get("skills", [])
        groups = sorted({group_map[s] for s in skills if s in group_map})
        record["skill_groups"] = groups
        records.append(record)
    return records


def write_grouped(records: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved → {out_path}  ({len(records):,} records)")


def summarise(records: list[dict], year: str):
    """Print brief stats for the grouped output."""
    total   = len(records)
    grouped = sum(1 for r in records if r["skill_groups"])
    n_grp   = sum(len(r["skill_groups"]) for r in records)
    all_groups = sorted({g for r in records for g in r["skill_groups"]})
    print(f"    {year}: {total:,} postings, "
          f"{grouped:,} with ≥1 group ({grouped/total*100:.0f}%), "
          f"{n_grp:,} group mentions across {len(all_groups)} distinct groups")
    for g in all_groups:
        n = sum(1 for r in records if g in r["skill_groups"])
        print(f"      {g:<40} {n:>4} postings ({n/total*100:.1f}%)")


def main():
    print(f"\nStage 6 — taxonomy grouping: {OCCUPATION}")
    print(f"Config: {CONFIG_PATH}")

    group_map = load_config(CONFIG_PATH)
    print(f"Groups loaded: {len(set(group_map.values()))} groups, "
          f"{len(group_map)} canonical skills mapped")

    for year in YEARS:
        records  = group_postings(OCCUPATION, year, group_map)
        out_path = SKILLS_DIR / year / f"{OCCUPATION}_grouped.jsonl"
        write_grouped(records, out_path)
        summarise(records, year)


if __name__ == "__main__":
    main()
