"""
Stage D (concept typing) — shared logic.

Builds the versioned type-assignment prompt from config/typing_rules.yaml,
defines the structured-output tool, and validates responses. The rubric is the
single source of the enum, definitions, precedence rules, and exemplars;
nothing here hardcodes them beyond the version stamp.

Deterministic overrides (rubric `type_overrides`) are applied by the runner,
never by the model — this module exposes them but the prompt never sees them.

Nothing here calls the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = ROOT / "config" / "typing_rules.yaml"

PROMPT_VERSION = "type-v1.3.0"
MODEL_VERSION = "claude-haiku-4-5-20251001"

BATCH_SIZE = 40  # concepts per model call


def load_rules(path: Path = RULES_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


RULES = load_rules()
RULES_VERSION = RULES["version"]
TYPE_ENUM = list(RULES["types"].keys())                       # the nine types — the only type values
HELD_REASONS = list((RULES.get("held_reasons") or {}).keys())  # why a held concept is held
OVERRIDES = {cid: v["type"] for cid, v in (RULES.get("type_overrides") or {}).items()}


# ── Structured-output tool ────────────────────────────────────────────────────
EMIT_TOOL = {
    "name": "emit_types",
    "description": "Return exactly one type assignment for every concept given.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept_id": {"type": "string"},
                        "type": {"type": "string", "enum": TYPE_ENUM},
                        "confidence": {"type": "number",
                                       "description": "0-1 confidence in the chosen type."},
                        "abstain": {"type": "boolean",
                                    "description": "true to HOLD the concept for review; still "
                                                   "return your single best type."},
                        "held_reason": {"type": "string", "enum": HELD_REASONS,
                                        "description": "Why the concept is held. Required when "
                                                       "abstain=true; omit when abstain=false."},
                        "reason": {"type": "string",
                                   "description": "One short clause citing the deciding rule."},
                    },
                    "required": ["concept_id", "type", "confidence", "abstain"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    },
}


def _build_system_prompt() -> str:
    r = RULES
    lines = [
        "You assign exactly ONE type to each job-advertisement skill concept, "
        "using only the closed rubric below. Prompt version " + PROMPT_VERSION
        + ", rubric " + RULES_VERSION + ".",
        "",
        "Judge every concept INDEPENDENTLY. A concept's type must not depend on "
        "any other concept in the same request — treat each as if it arrived "
        "alone. Return an assignment for every concept_id you are given, exactly "
        "once, and use only these nine types:",
        "",
    ]
    for name, body in r["types"].items():
        lines.append(f"- {name}: {body['definition']}")
    lines.append("")
    lines.append("HOLDING — set abstain=true to hold a concept you cannot place, and give a "
                 "held_reason from this set (still return your single best type):")
    for name, body in (r.get("held_reasons") or {}).items():
        lines.append(f"- {name}: {body['definition']}")
    lines.append("")
    lines.append("PRECEDENCE RULES for the boundary cases — apply these before deciding:")
    for p in r["precedence"]:
        lines.append(f"- {p['id']}: {' '.join(p['rule'].split())}")
    lines.append("")
    lines.append("ABSTAIN: " + " ".join(r["abstain_guidance"].split()))
    lines.append("")
    lines.append("WORKED EXEMPLARS (illustration only; the rules above decide):")
    for name, ex in r["exemplars"].items():
        lines.append(f"- {name}: {', '.join(ex)}")
    lines.append("")
    lines.append("Return your answer only through the emit_types tool.")
    return "\n".join(lines)


SYSTEM_PROMPT = _build_system_prompt()


def build_user_prompt(concepts: list[dict]) -> str:
    """Render a batch of concepts. Each concept dict: id, label, description,
    forms (list), mentions (int). Independent — no cross-references."""
    out = [f"Classify each of these {len(concepts)} concepts independently.", ""]
    for c in concepts:
        forms = ", ".join(c.get("forms", []))
        out.append(f"concept_id: {c['id']}")
        out.append(f"  label: {c['label']}")
        if c.get("description"):
            out.append(f"  description: {c['description']}")
        out.append(f"  member forms: {forms}")
        out.append(f"  total mentions: {c.get('mentions', 0)}")
        out.append("")
    return "\n".join(out)


# ── Parsing + validation ──────────────────────────────────────────────────────
@dataclass
class TypeResult:
    assignments: dict  # concept_id -> {type, confidence, abstain, reason}


def parse_tool_result(message_content: list) -> TypeResult | None:
    for block in message_content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype != "tool_use":
            continue
        name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if name != "emit_types":
            continue
        data = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        if isinstance(data, str):
            data = json.loads(data)
        out = {}
        for a in data.get("assignments", []):
            out[a["concept_id"]] = {
                "type": a.get("type"), "confidence": a.get("confidence"),
                "abstain": bool(a.get("abstain", False)),
                "held_reason": a.get("held_reason", ""),
                "reason": a.get("reason", ""),
            }
        return TypeResult(assignments=out)
    return None


def validate(result: TypeResult | None, concept_ids: list[str]) -> tuple[bool, str]:
    """Every input concept has exactly one assignment with a valid enum type."""
    if not result or not result.assignments:
        return False, "no assignments returned"
    want = set(concept_ids)
    got = set(result.assignments)
    missing = want - got
    if missing:
        return False, f"missing assignment(s): {sorted(missing)[:5]}"
    extra = got - want
    if extra:
        return False, f"assignment(s) for unknown id(s): {sorted(extra)[:5]}"
    for cid, a in result.assignments.items():
        if a["type"] not in TYPE_ENUM:
            return False, f"invalid type {a['type']!r} for {cid}"
        if a["abstain"] and a.get("held_reason") not in HELD_REASONS:
            return False, f"held concept {cid} missing valid held_reason"
    return True, "ok"
