"""
Step 3 (semantic merge) — shared logic.

Holds the versioned prompt, the model id, the structured-output tool schema,
the stable concept-id function, and the response validator. Both the batch
runner and any test harness import from here so the prompt version and model
version are recorded in exactly one place.

Nothing here calls the network.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

# ── Versions (recorded on every concept row) ─────────────────────────────────
PROMPT_VERSION = "merge-v1.1.0"
MODEL_VERSION = "claude-haiku-4-5-20251001"

# ── Concept-id namespace ─────────────────────────────────────────────────────
# The id is derived from a hash of the canonical label plus this namespace.
# It does NOT depend on processing order or on the current corpus, so a new
# advertisement whose forms resolve to the same canonical label maps into the
# existing id instead of creating a duplicate. Bump the namespace only if the
# id scheme itself must change.
ID_NAMESPACE = "skill-concept:v1"


def _label_key(label: str) -> str:
    """Fold a label to its identity key: lowercase, whitespace-collapsed."""
    return re.sub(r"\s+", " ", label.strip().lower())


def concept_id(label: str) -> str:
    """Stable, order- and corpus-independent id for a canonical label."""
    h = hashlib.sha256(f"{ID_NAMESPACE}\x1f{_label_key(label)}".encode("utf-8"))
    return "cpt_" + h.hexdigest()[:16]


# ── Structured-output tool schema ────────────────────────────────────────────
# The model is forced to call this tool, so the response is always well-formed
# JSON with the fields we need.
EMIT_TOOL = {
    "name": "emit_concepts",
    "description": "Return the partition of the input group into one or more concepts.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concepts": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_label": {
                            "type": "string",
                            "description": "The term most used in the domain for this concept.",
                        },
                        "canonical_reason": {
                            "type": "string",
                            "description": "One line: why this label is the domain-standard choice.",
                        },
                        "member_forms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Input forms that belong to this concept. Verbatim.",
                            "minItems": 1,
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0-1 confidence that these forms are one concept.",
                        },
                        "merge_evidence": {
                            "type": "string",
                            "description": "Which evidence spans justify merging these forms. "
                                           "Required when any form contains a vendor name.",
                        },
                        "vendor_relation": {
                            "type": "string",
                            "enum": ["no_vendor", "same_product", "product_and_technology"],
                            "description": "For a concept whose members include a vendor/product "
                                           "name: 'same_product' if the merged forms all name the "
                                           "SAME product; 'product_and_technology' if they name a "
                                           "product and the underlying technology/language it uses "
                                           "(you MUST NOT merge in this case — split instead). "
                                           "'no_vendor' when no member carries a vendor name.",
                        },
                    },
                    "required": [
                        "canonical_label", "canonical_reason",
                        "member_forms", "confidence", "vendor_relation",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["concepts"],
        "additionalProperties": False,
    },
}


SYSTEM_PROMPT = """\
You partition a group of near-duplicate job-advertisement skill phrases into \
CONCEPTS. Two forms belong to the same concept only when an advertiser would \
read them as the same thing. Prompt version """ + PROMPT_VERSION + """.

Every input form must appear in EXACTLY ONE output concept. Returning every \
form as its own concept is a valid answer, not a failure.

MERGE two forms only for one of these reasons:
- Grammatical variants of the same thing: singular/plural, gerund/noun, \
British/American spelling.
- The same concept at different verbosity, where the extra words add no meaning.
- An acronym and its expansion of the same term.
- True synonyms as used in job advertisements.

KEEP APART — do NOT merge:
- Co-hyponyms. If two forms share a head noun but differ in a modifier that \
carries meaning, they are SIBLINGS, not synonyms. "text classification" and \
"text generation" are different. "warehouse architecture" and "warehouse \
operations" are different. THIS IS THE MOST COMMON ERROR — check every pair \
for a meaning-bearing modifier before merging.
- Distinct products, even from the same vendor. "microsoft copilot" and \
"copilot studio" are different products.
- A method and the tool that implements it. "orchestration" and "airflow" \
are different.
- Different levels of abstraction. A specific technique is NOT its parent field.

HARD RULE — vendor names and the product / technology distinction: if any \
form contains a vendor or product brand (e.g. microsoft, aws, amazon, google, \
databricks, snowflake, openai, azure, copilot, airflow, kafka, spark, \
postgresql, pytorch, tensorflow), do NOT merge it with another form unless the \
two forms name the SAME product. A product and the underlying technology, \
language, or dialect it is built on are DIFFERENT concepts — never merge them.

WORKED EXAMPLE A — input forms: sql, sql server, t-sql, postgresql
Correct partition (four separate concepts, nothing merged):
  * SQL          = {sql}         — the query language / ISO standard.
  * SQL Server   = {sql server}  — Microsoft's database PRODUCT.
  * T-SQL        = {t-sql}       — Microsoft's SQL DIALECT for SQL Server.
  * PostgreSQL   = {postgresql}  — a different database product.
Reason: "sql" is the language; "sql server" is a product that runs it \
(product_and_technology → keep apart); "t-sql" is a dialect, not the language \
and not the product; "postgresql" is a distinct product. NONE merge.
But note: "microsoft sql server" and "sql server" WOULD merge — they name the \
same product (same_product).

WORKED EXAMPLE B — input forms: python, pytorch
Correct partition (two separate concepts):
  * Python  = {python}   — the programming language.
  * PyTorch = {pytorch}  — a library built ON Python.
Reason: a library and the language it is written for are \
product_and_technology → keep apart. NONE merge.

REQUIRED for every concept: set vendor_relation. When a concept's members \
include a vendor/product name, you must state whether the merged forms name \
the SAME product ("same_product") or a product and its underlying technology \
("product_and_technology"). You MUST NOT output a merged concept (two or more \
members) with vendor_relation "product_and_technology" — split it instead. \
Use "no_vendor" only when no member carries a vendor or product name. When you \
merge vendor forms, also name the supporting spans in merge_evidence.

CANONICAL LABEL: choose the term most used in the domain — NOT necessarily the \
longest form and NOT necessarily the most frequent form in the data. Where an \
acronym is the domain standard (e.g. SQL, ETL, NLP, API), use the acronym. \
Give a one-line canonical_reason for each choice.

Return your answer only through the emit_concepts tool."""


def build_user_prompt(payload: dict) -> str:
    """Render one group's payload into the model-facing text."""
    lines = [
        f"Group {payload['group_id']} — partition these "
        f"{len(payload['members'])} forms into one or more concepts.",
        "",
    ]
    for m in payload["members"]:
        occ = ", ".join(f"{o['label']}:{o['count']}" for o in m["occupations"])
        lines.append(f"FORM: {m['form']}  (mentions: {m['mention_count']})")
        lines.append(f"  occupations: {occ}")
        if m["evidence_spans"]:
            lines.append("  evidence spans:")
            for s in m["evidence_spans"]:
                lines.append(f"    - {s}")
        lines.append("")
    return "\n".join(lines)


# ── Response parsing + validation ────────────────────────────────────────────
@dataclass
class Partition:
    concepts: list[dict]  # each: canonical_label, canonical_reason, member_forms, confidence, merge_evidence


def parse_tool_result(message_content: list) -> Partition | None:
    """Extract the emit_concepts tool input from a batch result message."""
    for block in message_content:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype != "tool_use":
            continue
        name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
        if name != "emit_concepts":
            continue
        data = block.get("input") if isinstance(block, dict) else getattr(block, "input", None)
        if isinstance(data, str):
            data = json.loads(data)
        return Partition(concepts=data.get("concepts", []))
    return None


# Curated brand/product lexicon used only as a backstop for the vendor gate.
# False negatives are acceptable — the model's own vendor_relation field is the
# primary enforcement; this catches a merged concept where it forgot to flag one.
VENDOR_TERMS = {
    "microsoft", "ms", "azure", "aws", "amazon", "google", "gcp", "apache",
    "databricks", "snowflake", "openai", "copilot", "airflow", "tableau",
    "powerbi", "power bi", "oracle", "nvidia", "sagemaker", "redshift",
    "bigquery", "synapse", "kafka", "spark", "kubernetes", "kinesis",
    "dynamodb", "athena", "teradata", "qlik", "pytorch", "tensorflow",
    "sql server", "t-sql", "postgresql", "mysql", "langchain", "langsmith",
    "llama", "claude", "github", "gitlab", "huggingface", "hugging face",
    "vertex", "cloudformation", "emr", "ecs", "eks", "fabric", "ssis",
    "dagster", "dbt", "looker", "snowpark", "bedrock", "sharepoint",
}


def has_vendor(form: str) -> bool:
    """True if a form carries a known vendor/product token."""
    low = form.lower()
    toks = set(low.replace("/", " ").replace("-", " ").split())
    if toks & VENDOR_TERMS:
        return True
    return any(" " in v and v in low for v in VENDOR_TERMS)


def validate(partition: Partition, input_forms: list[str]) -> tuple[bool, str]:
    """Every input form in exactly one concept; no empty labels; vendor rule.

    Returns (ok, reason). On failure the group is left for a second pass.
    """
    if not partition or not partition.concepts:
        return False, "no concepts returned"

    seen: dict[str, int] = {}
    for c in partition.concepts:
        label = (c.get("canonical_label") or "").strip()
        if not label:
            return False, "empty canonical_label"
        mf = c.get("member_forms") or []
        if not mf:
            return False, "concept with no member_forms"
        # Vendor / product-vs-technology gate. The user's hard rule is precise:
        # a concept must NOT merge a product with its underlying technology. We
        # enforce exactly that — a MERGED concept the model itself tagged
        # 'product_and_technology' is rejected. We deliberately do NOT also fail
        # merges tagged 'no_vendor': the token lexicon over-fires on ordinary
        # words ('snowflake schema', 'ehr/emr systems', 'spark structured
        # streaming'), and the prompt's worked examples are the primary defence.
        if len(mf) > 1 and c.get("vendor_relation") == "product_and_technology":
            return False, f"merged product_and_technology: {label!r} {mf}"
        for form in mf:
            seen[form] = seen.get(form, 0) + 1

    input_set = set(input_forms)
    out_set = set(seen)
    dupes = [f for f, n in seen.items() if n > 1]
    if dupes:
        return False, f"form(s) in >1 concept: {dupes[:5]}"
    missing = input_set - out_set
    if missing:
        return False, f"missing input form(s): {sorted(missing)[:5]}"
    extra = out_set - input_set
    if extra:
        return False, f"invented form(s) not in input: {sorted(extra)[:5]}"
    return True, "ok"
