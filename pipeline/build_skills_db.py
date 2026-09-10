"""
Build (or rebuild) data/skills.db — a queryable SQLite store covering all five
skill-to-job-ad mappings across all occupations and years.

Five tables:
  raw_occurrences     — raw skill entity → job ad
  norm_map            — raw skill entity → canonical skill
  normed_occurrences  — canonical skill → job ad
  skill_groups        — canonical skill → group name  (Stage 6; skipped if no config)
  grouped_occurrences — group name → job ad           (Stage 6; skipped if no config)

Plus convenience views:
  v_penetration_raw    — penetration rate per (occ, year, raw_skill)
  v_penetration_normed — penetration rate per (occ, year, canonical_skill)
  v_penetration_groups — penetration rate per (occ, year, skill_group)

Run from project root:
  python3 pipeline/build_skills_db.py

Safe to re-run at any time — all tables are rebuilt from source artifacts.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from skills_io import load_skill_records  # noqa: E402

SKILLS_DIR = Path("data/skills")
DB_PATH    = Path("data/skills.db")
YEARS      = ["2021", "2026"]

OCCUPATIONS = [
    "data_scientist",
    "data_engineer",
    "data_analyst",
    "machine_learning_engineer",
    "ai_engineer",
    "analytics_engineer",
    "business_intelligence_analyst",
]


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_occurrences (
    occupation  TEXT NOT NULL,
    year        TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    raw_skill   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS norm_map (
    occupation      TEXT NOT NULL,
    raw_skill       TEXT NOT NULL,
    canonical_skill TEXT NOT NULL,
    PRIMARY KEY (occupation, raw_skill)
);

CREATE TABLE IF NOT EXISTS normed_occurrences (
    occupation      TEXT NOT NULL,
    year            TEXT NOT NULL,
    job_id          TEXT NOT NULL,
    canonical_skill TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_groups (
    occupation      TEXT NOT NULL,
    canonical_skill TEXT NOT NULL,
    skill_group     TEXT NOT NULL,
    PRIMARY KEY (occupation, canonical_skill)
);

CREATE TABLE IF NOT EXISTS grouped_occurrences (
    occupation  TEXT NOT NULL,
    year        TEXT NOT NULL,
    job_id      TEXT NOT NULL,
    skill_group TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_raw_occ_year  ON raw_occurrences(occupation, year);
CREATE INDEX IF NOT EXISTS idx_raw_skill     ON raw_occurrences(raw_skill);
CREATE INDEX IF NOT EXISTS idx_norm_occ      ON norm_map(occupation);
CREATE INDEX IF NOT EXISTS idx_normed_occ    ON normed_occurrences(occupation, year);
CREATE INDEX IF NOT EXISTS idx_normed_skill  ON normed_occurrences(canonical_skill);
CREATE INDEX IF NOT EXISTS idx_groups_occ    ON skill_groups(occupation);
CREATE INDEX IF NOT EXISTS idx_grouped_occ   ON grouped_occurrences(occupation, year);
"""

VIEWS = """
CREATE VIEW IF NOT EXISTS v_penetration_raw AS
    SELECT
        r.occupation,
        r.year,
        r.raw_skill                                          AS skill,
        COUNT(DISTINCT r.job_id)                             AS n_postings_skill,
        n.n_total,
        ROUND(1.0 * COUNT(DISTINCT r.job_id) / n.n_total, 4) AS penetration_rate
    FROM raw_occurrences r
    JOIN (
        SELECT occupation, year, COUNT(DISTINCT job_id) AS n_total
        FROM raw_occurrences GROUP BY occupation, year
    ) n ON n.occupation = r.occupation AND n.year = r.year
    GROUP BY r.occupation, r.year, r.raw_skill;

CREATE VIEW IF NOT EXISTS v_penetration_normed AS
    SELECT
        o.occupation,
        o.year,
        o.canonical_skill                                     AS skill,
        COUNT(DISTINCT o.job_id)                              AS n_postings_skill,
        n.n_total,
        ROUND(1.0 * COUNT(DISTINCT o.job_id) / n.n_total, 4) AS penetration_rate
    FROM normed_occurrences o
    JOIN (
        SELECT occupation, year, COUNT(DISTINCT job_id) AS n_total
        FROM normed_occurrences GROUP BY occupation, year
    ) n ON n.occupation = o.occupation AND n.year = o.year
    GROUP BY o.occupation, o.year, o.canonical_skill;

CREATE VIEW IF NOT EXISTS v_penetration_groups AS
    SELECT
        g.occupation,
        g.year,
        g.skill_group,
        COUNT(DISTINCT g.job_id)                              AS n_postings_group,
        n.n_total,
        ROUND(1.0 * COUNT(DISTINCT g.job_id) / n.n_total, 4) AS penetration_rate
    FROM grouped_occurrences g
    JOIN (
        SELECT occupation, year, COUNT(DISTINCT job_id) AS n_total
        FROM normed_occurrences GROUP BY occupation, year
    ) n ON n.occupation = g.occupation AND n.year = g.year
    GROUP BY g.occupation, g.year, g.skill_group;
"""


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_raw_occurrences(occ: str) -> list[tuple]:
    rows = []
    for year in YEARS:
        path = SKILLS_DIR / year / f"{occ}.jsonl"
        if not path.exists():
            continue
        for record in load_skill_records(path):
            job_id = record["id"]
            for skill in record["skills"]:
                rows.append((occ, year, job_id, skill))
    return rows


def load_norm_map(occ: str) -> list[tuple]:
    path = SKILLS_DIR / f"{occ}_norm_map.json"
    if not path.exists():
        return []
    mapping = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for raw, canonical in mapping.items():
        # None means no change — raw_skill is its own canonical
        canonical_skill = canonical if canonical is not None else raw
        rows.append((occ, raw, canonical_skill))
    return rows


def load_normed_occurrences(occ: str) -> list[tuple]:
    rows = []
    for year in YEARS:
        path = SKILLS_DIR / year / f"{occ}_normed.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            job_id = record["id"]
            for skill in record.get("skills", []):
                rows.append((occ, year, job_id, skill))
    return rows


def load_skill_groups(occ: str) -> list[tuple]:
    path = SKILLS_DIR / f"skill_groups_{occ}.json"
    if not path.exists():
        return []
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return [(occ, skill, group) for skill, group in cfg["groups"].items()]


def load_grouped_occurrences(occ: str) -> list[tuple]:
    rows = []
    for year in YEARS:
        path = SKILLS_DIR / year / f"{occ}_grouped.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            job_id = record["id"]
            for group in record.get("skill_groups", []):
                rows.append((occ, year, job_id, group))
    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    print("Rebuilding data/skills.db …")

    # Drop all tables and views for a clean rebuild
    for view in ["v_penetration_raw", "v_penetration_normed", "v_penetration_groups"]:
        cur.execute(f"DROP VIEW IF EXISTS {view}")
    for table in ["raw_occurrences", "norm_map", "normed_occurrences",
                  "skill_groups", "grouped_occurrences"]:
        cur.execute(f"DROP TABLE IF EXISTS {table}")

    cur.executescript(SCHEMA)
    cur.executescript(VIEWS)

    totals = {
        "raw_occurrences":     0,
        "norm_map":            0,
        "normed_occurrences":  0,
        "skill_groups":        0,
        "grouped_occurrences": 0,
    }

    for occ in OCCUPATIONS:
        raw_rows     = load_raw_occurrences(occ)
        norm_rows    = load_norm_map(occ)
        normed_rows  = load_normed_occurrences(occ)
        group_rows   = load_skill_groups(occ)
        grouped_rows = load_grouped_occurrences(occ)

        cur.executemany(
            "INSERT INTO raw_occurrences VALUES (?,?,?,?)", raw_rows)
        cur.executemany(
            "INSERT INTO norm_map VALUES (?,?,?)", norm_rows)
        cur.executemany(
            "INSERT INTO normed_occurrences VALUES (?,?,?,?)", normed_rows)
        cur.executemany(
            "INSERT INTO skill_groups VALUES (?,?,?)", group_rows)
        cur.executemany(
            "INSERT INTO grouped_occurrences VALUES (?,?,?,?)", grouped_rows)

        totals["raw_occurrences"]     += len(raw_rows)
        totals["norm_map"]            += len(norm_rows)
        totals["normed_occurrences"]  += len(normed_rows)
        totals["skill_groups"]        += len(group_rows)
        totals["grouped_occurrences"] += len(grouped_rows)

        stage6 = f"  {len(group_rows)} groups, {len(grouped_rows)} grouped mentions" \
                 if group_rows else "  (Stage 6 not yet run)"
        print(f"  {occ:<24}  raw={len(raw_rows):>6}  normed={len(normed_rows):>6}{stage6}")

    conn.commit()
    conn.close()

    print(f"\nDone → {DB_PATH}")
    print(f"  raw_occurrences     {totals['raw_occurrences']:>8,} rows")
    print(f"  norm_map            {totals['norm_map']:>8,} rows")
    print(f"  normed_occurrences  {totals['normed_occurrences']:>8,} rows")
    print(f"  skill_groups        {totals['skill_groups']:>8,} rows")
    print(f"  grouped_occurrences {totals['grouped_occurrences']:>8,} rows")
    print()
    print("Example queries:")
    print("  sqlite3 data/skills.db")
    print("  -- Top 10 normed skills for data_scientist in 2026:")
    print("  SELECT skill, penetration_rate FROM v_penetration_normed")
    print("    WHERE occupation='data_scientist' AND year='2026'")
    print("    ORDER BY penetration_rate DESC LIMIT 10;")
    print()
    print("  -- Which job ads mention 'python' (raw)?")
    print("  SELECT DISTINCT job_id FROM raw_occurrences")
    print("    WHERE occupation='data_scientist' AND raw_skill LIKE '%ython%';")


if __name__ == "__main__":
    main()
