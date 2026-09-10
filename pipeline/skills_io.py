"""
Loader for the raw skills JSONL, tolerant of both output shapes.

v1 records held `skills` as a list of strings. v2 (skill_extraction_prompt.py)
holds a list of {skill, type, requirement, evidence} objects and adds `status`.

Everything downstream of extraction — clustering, normalisation, the skills DB —
only needs the skill names. This flattens v2 back to a list of strings and parks
the structured entries under `skills_detail`, so those stages keep working
unchanged while the richer fields stay available.

Rows with status != "ok" are dropped by default: an API error or a refusal is
not a zero-skill posting, and counting it as one biases penetration downwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def skill_names(record: dict[str, Any]) -> list[str]:
    """Skill names from a record in either the v1 or v2 shape."""
    out = []
    for s in record.get("skills", []) or []:
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, dict) and s.get("skill"):
            out.append(s["skill"])
    return out


def load_skill_records(
    path: Path,
    *,
    ok_only: bool = True,
) -> list[dict[str, Any]]:
    """Read a skills JSONL, returning records with `skills` as list[str].

    The original structured entries, if any, are kept under `skills_detail`.
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)

        # v1 rows have no `status`; treat them as ok.
        if ok_only and rec.get("status", "ok") != "ok":
            continue
        # v1 error marker
        if ok_only and rec.get("extraction_error"):
            continue

        raw = rec.get("skills", []) or []
        if raw and isinstance(raw[0], dict):
            rec["skills_detail"] = raw
        rec["skills"] = skill_names(rec)
        records.append(rec)
    return records
