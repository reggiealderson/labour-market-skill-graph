"""
Seed for the resolver's first deterministic level. NOT imported by any script
that currently runs.

The resolver will carry ONE acronym map for all occupations. This module holds
the starting content for it, salvaged from the frozen Stage 2 scripts.

It is deliberately NOT wired into normalise_skills.py or
generate_review_clusters.py. Those two are frozen to reproduce the seven
completed occupations, and they produced that output with divergent maps — this
file maps "gaap" -> "generally accepted accounting principles" while
normalise_skills.py maps it -> "gaap". Making them agree now would change their
output and destroy reproducibility. The divergence is documented as a defect in
their headers instead.

Content below is generate_review_clusters.py's version, which is the one the
archived human review decisions were made against.

SCOPE: OCC_ACRONYM_ADDITIONS covers only registered_nurse, financial_analyst,
accountant and graphic_designer — all four RETIRED. None of the Core 6 has
entries. Do NOT add Core 6 entries here speculatively; the resolver's design
decides whether per-occupation additions survive at all. A single shared map
across occupations is the stated direction, which may mean this nesting goes
away entirely.
"""

from __future__ import annotations

# Applies to every occupation.
BASE_ACRONYM_MAP: dict[str, str] = {
    "ml":    "machine learning",
    "nlp":   "natural language processing",
    "llm":   "large language models",
    "llms":  "large language models",
    "genai": "generative ai",
    "rag":   "retrieval-augmented generation",
    "eda":   "exploratory data analysis",
    "gcp":   "google cloud platform",
    "mcp":   "model context protocol",
}

# Occupation-specific. Merged over BASE_ACRONYM_MAP.
OCC_ACRONYM_ADDITIONS: dict[str, dict[str, str]] = {
    # --- retired occupations, retained for reference only ---
    "registered_nurse": {
        "rn":   "registered nurse",
        "ehr":  "electronic health records",
        "emr":  "electronic medical records",
        "bls":  "basic life support",
        "acls": "advanced cardiac life support",
    },
    "financial_analyst": {
        "fp&a": "financial planning and analysis",
        "dcf":  "discounted cash flow",
        "irr":  "internal rate of return",
        "npv":  "net present value",
    },
    "accountant": {
        "gaap": "generally accepted accounting principles",
        "ifrs": "international financial reporting standards",
        "cpa":  "cpa",
        "erp":  "erp",
    },
    "graphic_designer": {
        "ui":    "ui design",
        "ux":    "ux design",
        "ui/ux": "ui/ux design",
    },
}


def get_acronym_map(occupation: str) -> dict[str, str]:
    """The expansion map for one occupation. Use this everywhere."""
    return {**BASE_ACRONYM_MAP, **OCC_ACRONYM_ADDITIONS.get(occupation, {})}


def expand(s: str, acronym_map: dict[str, str]) -> str:
    return acronym_map.get(s, s)
