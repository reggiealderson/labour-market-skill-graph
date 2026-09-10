"""Classify concepts against a facet using the Anthropic API.

Reads a facet definition from facets.yaml (the value definitions serve as the
classification rubric) and a concept list from a CSV file. Sends batches to
Claude Haiku, collects classifications, and writes the results to
concept_facet.csv.

Only facets with assignment: classifier are supported.

Usage:
  python3 classify_facet.py <facet_id> <concept_csv> \
      --facets-yaml <path> --out <path> [--batch-size N] [--dry-run]

Requirements:
  - ANTHROPIC_API_KEY environment variable set.
  - pip install anthropic pyyaml

API-step record (for CLAUDE.md):
  model       : claude-haiku-4-5-20251001
  purpose     : Classify each concept against one facet, using the facet's
                value definitions as the rubric. One API call per batch of
                concepts.
  human_review: Required. Output is model-made (by=model) and must be
                reviewed before use. Borderline cases concentrate in
                ai_adjacent / core_ai boundary.
  provenance  : Every output row carries model= and by=model columns so
                model verdicts are never mistaken for human ones.

Exit 0 = success, 1 = error.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import yaml

try:
    import anthropic
except ImportError:
    print("ERROR: 'anthropic' package not installed. "
          "Run: pip install anthropic --break-system-packages")
    sys.exit(1)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096

OUTPUT_COLUMNS = ["concept_id", "facet_id", "value", "model", "by"]


# ---------------------------------------------------------------------------
# Load facet rubric
# ---------------------------------------------------------------------------

def load_facet(facet_id, facets_path):
    """Return the facet dict for facet_id, or exit with an error."""
    doc = yaml.safe_load(facets_path.read_text())
    for f in doc.get("facets", []):
        if f.get("id") == facet_id:
            if f.get("assignment", "pattern") != "classifier":
                print(f"ERROR: facet '{facet_id}' does not use "
                      "assignment: classifier")
                sys.exit(1)
            return f
    print(f"ERROR: facet '{facet_id}' not found in {facets_path}")
    sys.exit(1)


def build_rubric(facet):
    """Build the rubric text from value definitions."""
    lines = []
    for v in facet.get("values", []):
        slug = v["value"]
        defn = v.get("definition", "").strip()
        lines.append(f"- {slug}: {defn}")
    return "\n".join(lines)


def get_valid_slugs(facet):
    """Return the set of allowed value slugs for this facet."""
    return {v["value"] for v in facet.get("values", [])}


# ---------------------------------------------------------------------------
# Load concepts
# ---------------------------------------------------------------------------

def load_concepts(csv_path):
    """Return list of dicts with concept_id, label, type."""
    concepts = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            concepts.append({
                "concept_id": row["concept_id"].strip(),
                "label": row["label"].strip(),
                "type": (row.get("type") or "").strip(),
            })
    return concepts


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def make_batches(concepts, batch_size):
    """Yield successive batches of concepts."""
    for i in range(0, len(concepts), batch_size):
        yield concepts[i:i + batch_size]


# ---------------------------------------------------------------------------
# Prompt construction (adapts to exclusivity and unmatched)
# ---------------------------------------------------------------------------

def build_system_prompt(facet):
    """Build a system prompt that respects the facet's exclusivity and
    unmatched settings."""
    exclusivity = facet.get("exclusivity", "multi_valued")
    unmatched = facet.get("unmatched", "absent")

    if exclusivity == "exclusive":
        cardinality = (
            "Assign exactly ONE value to each concept. Choose the single "
            "best fit from the rubric."
        )
        json_format = '{"id": "<concept_id>", "value": "<value_slug>"}'
    else:
        cardinality = (
            "Assign ONE OR MORE values to each concept. A concept may match "
            "several values — include every value that applies."
        )
        json_format = ('{"id": "<concept_id>", "values": '
                       '["<slug1>", "<slug2>"]}'
                       '  (use a list even when only one value applies)')

    if unmatched == "absent":
        no_match = (
            'If no value fits a concept, return "none" as the value. '
            "Do not force a poor fit."
        )
    else:
        # declared: every concept must get a value; the fallback is applied
        # downstream, so the model should still pick the closest fit.
        no_match = (
            "Every concept must receive at least one value. If uncertain, "
            "choose the closest fit."
        )

    return f"""\
You are a concept classifier. {cardinality}

Return ONLY a JSON array, no other text.
Each element: {json_format}

Rules:
- Use ONLY the value slugs listed in the rubric.
- Classify every concept in the batch — do not skip any.
- Base your decision on the concept label, its Stage D type (if present), \
and the value definitions in the rubric.
- {no_match}
"""


def build_user_prompt(rubric_text, batch):
    """Build the user message for one batch."""
    lines = ["RUBRIC:", rubric_text, "", "CONCEPTS TO CLASSIFY:"]
    for c in batch:
        type_str = f" (type: {c['type']})" if c["type"] else ""
        lines.append(
            f"- id={c['concept_id']}  label=\"{c['label']}\"{type_str}"
        )
    lines.append("")
    lines.append("Respond with a JSON array only. No markdown fences, "
                 "no preamble.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _parse_response(text, batch_ids, valid_slugs, exclusivity):
    """Parse the JSON response, return list of (concept_id, value) tuples.

    For multi_valued facets, one concept may produce several tuples.
    Returns (pairs, seen_ids).
    """
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:text.rfind("```")]
        text = text.strip()

    results = json.loads(text)
    if not isinstance(results, list):
        raise ValueError("Response is not a JSON array")

    pairs = []
    seen = set()
    for item in results:
        cid = item["id"]
        if cid not in batch_ids:
            continue

        # Normalise: exclusive uses "value", multi uses "values"
        if exclusivity == "exclusive":
            raw_values = [item.get("value", "")]
        else:
            raw_values = item.get("values", [])
            if isinstance(raw_values, str):
                raw_values = [raw_values]

        real = [v for v in raw_values if v in valid_slugs]
        if real:
            # One or more real values fit. Record them and ignore any stray
            # "none" the model may have added alongside (a contradiction).
            for val in real:
                pairs.append((cid, val))
        elif "none" in raw_values:
            # "none" is a real decision: no value fit. Record it so the
            # concept counts as classified and resume never re-sends it.
            # Filter these out at analysis time: df[df.value != "none"].
            pairs.append((cid, "none"))
        # Warn on any token that is neither a valid slug nor "none".
        for val in raw_values:
            if val != "none" and val not in valid_slugs:
                print(f"  WARNING: concept {cid} got invalid value "
                      f"'{val}', skipping")
        seen.add(cid)

    return pairs, seen


def classify_batch(client, system_prompt, rubric_text, batch, valid_slugs,
                   exclusivity, max_retries=3):
    """Call the API and return a list of (concept_id, value) tuples."""
    user_prompt = build_user_prompt(rubric_text, batch)
    batch_ids = {c["concept_id"] for c in batch}

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
            pairs, seen = _parse_response(
                text, batch_ids, valid_slugs, exclusivity
            )

            # Check for missing concepts
            missing = batch_ids - seen
            if missing:
                print(f"  WARNING: {len(missing)} concept(s) missing from "
                      f"response (attempt {attempt})")
                if attempt < max_retries:
                    time.sleep(1)
                    continue

            return pairs

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  WARNING: parse error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            print(f"  ERROR: batch failed after {max_retries} attempts, "
                  f"skipping {len(batch)} concepts")
            return []

        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue

        except anthropic.APIError as e:
            print(f"  API error on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"  ERROR: batch failed after {max_retries} attempts")
            return []

    return []


# ---------------------------------------------------------------------------
# Safe multi-facet file I/O
# ---------------------------------------------------------------------------

def load_existing_file(out_path):
    """Read all existing rows from concept_facet.csv. Returns (rows, done)
    where rows is the full list of row-dicts and done is the set of
    concept_ids for the current facet (filled in by the caller)."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return [], set()
    rows = []
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows, set()


def write_output(out_path, existing_rows, new_pairs, facet_id):
    """Write concept_facet.csv, preserving other facets' rows.

    Steps:
      1. Keep every existing row whose facet_id != current facet_id.
      2. Keep existing rows for current facet_id (resume: already done).
      3. Append new_pairs for current facet_id.
    This is a full rewrite of the file, so no data is lost.
    """
    # Preserve all rows from other facets, plus existing rows from this facet
    preserved = [r for r in existing_rows]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in preserved:
            writer.writerow(row)
        for cid, val in new_pairs:
            writer.writerow({
                "concept_id": cid,
                "facet_id": facet_id,
                "value": val,
                "model": MODEL,
                "by": "model",
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Classify concepts against a facet using the "
                    "Anthropic API."
    )
    parser.add_argument("facet_id", help="Facet id from facets.yaml")
    parser.add_argument("concept_csv", help="Path to concept CSV file")
    parser.add_argument("--out", required=True,
                        help="Output CSV path for concept_facet.csv")
    parser.add_argument("--facets-yaml", required=True,
                        help="Path to facets.yaml")
    parser.add_argument("--workspace-id", default=None,
                        help="Anthropic workspace id (required for "
                             "identity-linked keys; or set "
                             "ANTHROPIC_WORKSPACE_ID)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Concepts per API call (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without calling the API")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Identity-linked keys require the workspace id as a header. Optional:
    # unused by workspace-scoped keys. From --workspace-id or the environment.
    workspace_id = args.workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")

    # Load facet
    facets_path = Path(args.facets_yaml)
    facet = load_facet(args.facet_id, facets_path)
    rubric_text = build_rubric(facet)
    valid_slugs = get_valid_slugs(facet)
    exclusivity = facet.get("exclusivity", "multi_valued")
    unmatched = facet.get("unmatched", "absent")
    system_prompt = build_system_prompt(facet)

    print(f"Facet: {args.facet_id}")
    print(f"Values: {sorted(valid_slugs)}")
    print(f"Exclusivity: {exclusivity}  |  Unmatched: {unmatched}")
    print(f"Model: {MODEL}  |  Provenance: by=model")
    print(f"Rubric:\n{rubric_text}\n")

    # Load concepts
    concepts = load_concepts(args.concept_csv)
    print(f"Loaded {len(concepts)} concepts from {args.concept_csv}")

    # Resume support: read existing file, find already-done concepts
    out_path = Path(args.out)
    existing_rows, _ = load_existing_file(out_path)
    already_done = {
        r["concept_id"] for r in existing_rows
        if r.get("facet_id") == args.facet_id
    }
    other_facet_rows = len([
        r for r in existing_rows if r.get("facet_id") != args.facet_id
    ])

    if already_done:
        before = len(concepts)
        concepts = [
            c for c in concepts if c["concept_id"] not in already_done
        ]
        print(f"Resuming: {before - len(concepts)} already classified, "
              f"{len(concepts)} remaining")
    if other_facet_rows:
        print(f"Preserving {other_facet_rows} rows from other facets")

    if not concepts:
        print("Nothing to classify.")
        return 0

    # Batch
    batches = list(make_batches(concepts, args.batch_size))
    print(f"Batches: {len(batches)} (batch size {args.batch_size})")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"Would send {len(batches)} API calls to {MODEL}")
        print(f"Would classify {len(concepts)} concepts")
        print(f"Would write results to {out_path}")
        print(f"\nSystem prompt:\n{system_prompt}")
        print(f"Sample user prompt (first 5 of batch 1):")
        print(build_user_prompt(rubric_text, batches[0][:5]))
        return 0

    # Classify
    client_kwargs = {"api_key": api_key}
    if workspace_id:
        client_kwargs["default_headers"] = {
            "anthropic-workspace-id": workspace_id
        }
        print(f"Workspace: {workspace_id}")
    client = anthropic.Anthropic(**client_kwargs)
    all_results = []
    t0 = time.time()

    for i, batch in enumerate(batches):
        pct = (i / len(batches)) * 100
        print(f"  Batch {i+1}/{len(batches)} ({pct:.0f}%) — "
              f"{len(batch)} concepts...", end=" ", flush=True)
        pairs = classify_batch(
            client, system_prompt, rubric_text, batch,
            valid_slugs, exclusivity,
        )
        all_results.extend(pairs)
        print(f"got {len(pairs)} results")

        # Brief pause between batches to stay under rate limits
        if i < len(batches) - 1:
            time.sleep(0.3)

    elapsed = time.time() - t0
    print(f"\nClassified {len(all_results)} concept-value pairs "
          f"in {elapsed:.1f}s")

    # Write output (safe: preserves other facets' rows)
    write_output(out_path, existing_rows, all_results, args.facet_id)

    total = len(already_done) + len({cid for cid, _ in all_results})
    print(f"Wrote to {out_path} ({total} concepts for {args.facet_id}, "
          f"{other_facet_rows} rows from other facets preserved)")

    # Report any gaps
    classified_ids = already_done | {cid for cid, _ in all_results}
    all_ids = {c["concept_id"] for c in load_concepts(args.concept_csv)}
    missing = all_ids - classified_ids
    if missing:
        print(f"WARNING: {len(missing)} concept(s) not classified — "
              f"re-run to retry")
    else:
        print("All concepts classified.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
