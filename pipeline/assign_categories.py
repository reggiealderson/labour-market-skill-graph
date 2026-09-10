"""
Stage F (category assignment) — shared logic.

Builds the assignment prompt from taxonomy/categories.yaml, using ONLY each
category's definition, includes, and excludes. Defines the structured-output
tool, and parses/validates responses. The taxonomy is the single source of the
category enum; nothing here invents categories, boundaries, or groupings.

categories.yaml is read, never written. Nothing here calls the network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATS_PATH = ROOT / "taxonomy" / "categories.yaml"

PROMPT_VERSION = "assign-v1.1.0"
MODEL_VERSION = "claude-haiku-4-5-20251001"
BATCH_SIZE = 40
MAX_CATEGORIES = 2  # one primary + at most one secondary


def load_categories(path: Path = CATS_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CATS = load_categories()
STAGE_E_VERSION = str(CATS.get("version"))
CATEGORY_LIST = CATS["categories"]
CATEGORY_ENUM = [c["id"] for c in CATEGORY_LIST]


# ── Structured-output tool ────────────────────────────────────────────────────
EMIT_TOOL = {
    "name": "emit_assignments",
    "description": "Return a category assignment for every concept given.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "string"},
                        "unassigned": {
                            "type": "boolean",
                            "description": "true when NO category genuinely fits; "
                                           "then omit primary_category/categories."},
                        "primary_category": {
                            "type": "string", "enum": CATEGORY_ENUM,
                            "description": "The single best-fitting category id "
                                           "(omit only when unassigned=true)."},
                        "categories": {
                            "type": "array",
                            "items": {"type": "string", "enum": CATEGORY_ENUM},
                            "description": "Primary first, then AT MOST one "
                                           "secondary category when the concept "
                                           "genuinely covers a second function."},
                        "reason": {"type": "string",
                                   "description": "One short clause citing the "
                                                  "deciding definition/exclude, or "
                                                  "why nothing fits."},
                    },
                    "required": ["concept_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    },
}


def build_system_prompt(category_order: list[dict] | None = None) -> str:
    """Render the assignment prompt. `category_order` lets the caller present
    the categories in a per-request randomised order; defaults to file order."""
    cats = category_order if category_order is not None else CATEGORY_LIST
    lines = [
        "You assign each job-advertisement skill concept to the fixed categories "
        "below, using ONLY their definition, includes, and excludes. Prompt "
        "version " + PROMPT_VERSION + ", taxonomy version " + STAGE_E_VERSION
        + ".",
        "",
        "Judge every concept INDEPENDENTLY — its assignment must not depend on "
        "any other concept in the same request.",
        "",
        "Assign exactly ONE `primary_category`: the best fit. Add ONE secondary "
        "category to `categories` (after the primary) WHENEVER the concept "
        "genuinely covers two distinct functions — a secondary category is a "
        "correct outcome, not a failure. Use at most two categories total. "
        "`primary_category` must also appear in `categories`.",
        "",
        "If NO category genuinely fits, set `unassigned=true` and give a reason. "
        "Do NOT force a fit.",
        "",
        "Never invent, split, or rename a category; use only the ids listed. Use "
        "each category's EXCLUDES to route: if a concept matches the shape of an "
        "exclude line, it belongs in that line's target category. The includes "
        "are positive examples of what belongs.",
        "",
        "CATEGORIES:",
    ]
    for c in cats:
        lines.append("")
        lines.append(f"- id: {c['id']}")
        lines.append(f"  label: {c['label']}")
        lines.append(f"  definition: {c['definition']}")
        inc = c.get("includes") or []
        if inc:
            lines.append(f"  includes: {', '.join(str(x) for x in inc)}")
        for e in (c.get("excludes") or []):
            lines.append(f"  exclude: {e.get('concept_shape')} -> "
                         f"{e.get('goes_to')}")
        if c.get("notes"):
            lines.append(f"  note: {c['notes']}")
    lines.append("")
    lines.append("Return your answer only through the emit_assignments tool.")
    return "\n".join(lines)


def build_user_prompt(concepts: list[dict]) -> str:
    out = [f"Assign each of these {len(concepts)} concepts independently.", ""]
    for c in concepts:
        forms = ", ".join(c.get("forms", []))
        out.append(f"concept_id: {c['id']}")
        out.append(f"  label: {c['label']}")
        if c.get("description"):
            out.append(f"  description: {c['description']}")
        out.append(f"  member forms: {forms}")
        out.append(f"  type (Stage D): {c.get('type', '')}")
        out.append("")
    return "\n".join(out)


# ── Parsing + validation ──────────────────────────────────────────────────────
@dataclass
class AssignResult:
    assignments: dict  # concept_id -> {primary, categories:[...], unassigned, reason}


def parse_tool_result(message_content: list) -> AssignResult | None:
    for block in message_content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype != "tool_use":
            continue
        name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if name != "emit_assignments":
            continue
        data = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        if isinstance(data, str):
            data = json.loads(data)
        out = {}
        for a in data.get("assignments", []):
            unassigned = bool(a.get("unassigned", False))
            prim = a.get("primary_category")
            cats = [c for c in (a.get("categories") or []) if c in CATEGORY_ENUM]
            if prim in CATEGORY_ENUM and prim not in cats:
                cats = [prim] + cats
            # keep primary first, then at most one secondary (cap at MAX_CATEGORIES)
            if prim in CATEGORY_ENUM:
                cats = [prim] + [c for c in cats if c != prim]
            cats = cats[:MAX_CATEGORIES]
            if not cats:
                unassigned = True
            out[a["concept_id"]] = {
                "primary": prim if prim in CATEGORY_ENUM else "",
                "categories": [] if unassigned else cats,
                "unassigned": unassigned, "reason": a.get("reason", "")}
        return AssignResult(assignments=out)
    return None


def validate(result: AssignResult | None, concept_ids: list[str]) -> tuple[bool, str]:
    if not result or not result.assignments:
        return False, "no assignments returned"
    want, got = set(concept_ids), set(result.assignments)
    missing = want - got
    if missing:
        return False, f"missing assignment(s): {sorted(missing)[:5]}"
    extra = got - want
    if extra:
        return False, f"assignment(s) for unknown id(s): {sorted(extra)[:5]}"
    for cid, a in result.assignments.items():
        if a["unassigned"]:
            continue                       # a valid outcome
        if not a["categories"]:
            return False, f"{cid}: no valid category and not unassigned"
        if a["primary"] not in CATEGORY_ENUM:
            return False, f"{cid}: invalid primary_category {a['primary']!r}"
    return True, "ok"
