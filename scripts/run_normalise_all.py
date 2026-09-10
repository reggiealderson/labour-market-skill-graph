"""Run normalise_skills.py for all 7 occupations."""
import subprocess
import sys
from pathlib import Path

OCCUPATIONS = [
    "data_scientist",
    "data_engineer",
    "data_analyst",
    "machine_learning_engineer",
    "ai_engineer",
    "analytics_engineer",
    "business_intelligence_analyst",
]

# etl/elt* variants in data_engineer all map to both etl and elt
_DE_ETL_ELT_MEMBERS = [
    "etl/elt", "etl/elt pipelines", "etl/elt processes", "etl/elt development",
    "etl/elt data pipelines", "etl/elt workflows", "etl/elt tools",
    "etl/elt pipeline development", "etl/elt design", "etl/elt orchestration",
    "etl/elt pipeline design", "etl / elt processes",
]

MULTI_PATCHES_PER_OCC = {
    "data_engineer": {m: ["etl", "elt"] for m in _DE_ETL_ELT_MEMBERS},
}

# Spacing/compound-word variants that the fuzzy clusterer misses (different first word)
EXTRA_MANUAL_PATCHES_PER_OCC = {
    "data_scientist": {
        "tensor flow":  "tensorflow",
        "powerbi":      "power bi",
        "map reduce":   "mapreduce",
    },
    "data_engineer": {
        "data bricks":  "databricks",
        "powerbi":      "power bi",
        "big query":    "bigquery",
        "map reduce":   "mapreduce",
        "sparksql":     "spark sql",
        "data flow":    "dataflow",
    },
    "data_analyst": {
        "powerbi":      "power bi",
        "power point":  "powerpoint",
        "pivottables":  "pivot tables",
    },
    "machine_learning_engineer": {
        "tensor flow":  "tensorflow",
        "ml ops":       "mlops",
        "huggingface":  "hugging face",
        "hugging face": "hugging face",  # override if decisions file chose huggingface
    },
    "ai_engineer": {
        "lang graph":   "langgraph",
        "huggingface":  "hugging face",
        "gen ai":       "generative ai",
        "genai":        "generative ai",
    },
    "analytics_engineer": {
        "powerbi":      "power bi",
        "big query":    "bigquery",
    },
    "business_intelligence_analyst": {
        "powerbi":      "power bi",
        "power point":  "powerpoint",
    },
}

script = Path("pipeline/normalise_skills.py")

for occ in OCCUPATIONS:
    print(f"\n{'='*60}")
    print(f"  {occ}")
    print(f"{'='*60}")
    src = script.read_text()
    patched = src.replace('OCCUPATION  = "accountant"', f'OCCUPATION  = "{occ}"')
    multi = MULTI_PATCHES_PER_OCC.get(occ, {})
    patched = patched.replace(
        "MULTI_PATCHES: dict[str, list[str]] = {}",
        f"MULTI_PATCHES: dict[str, list[str]] = {repr(multi)}"
    )
    extra = EXTRA_MANUAL_PATCHES_PER_OCC.get(occ, {})
    patched = patched.replace(
        "EXTRA_MANUAL_PATCHES: dict[str, str] = {}",
        f"EXTRA_MANUAL_PATCHES: dict[str, str] = {repr(extra)}"
    )
    tmp = Path(f"/tmp/normalise_{occ}.py")
    tmp.write_text(patched)
    result = subprocess.run([sys.executable, str(tmp)], capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {occ} failed with return code {result.returncode}")
        sys.exit(1)

print("\nAll 7 occupations normalised.")
