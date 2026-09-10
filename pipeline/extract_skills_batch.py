"""
LLM skill extraction using the Anthropic Batch API.

Submits all records for the configured occupation/year as a single batch,
polls until complete, then retrieves and saves results.

Checkpoint-safe:
  - Batch ID saved to state file immediately after submission — re-running
    the script will resume polling and retrieval rather than re-submitting.
  - Results written to the output JSONL and flushed after each record is
    retrieved — safe to interrupt during result retrieval.
  - Already-done IDs (from previous runs) are skipped before submission.

Input:  data/extracted/{YEAR}/{OCCUPATION}.jsonl
Output: data/skills/{YEAR}/{OCCUPATION}.jsonl
          {id, status, prompt_version, model, skills:[{skill,type,requirement,evidence}]}
State:  data/skills/{YEAR}/{OCCUPATION}_batch_state.json

Only rows with status == "ok" count as done, so re-running retries the
failures rather than cementing them as zero-skill postings.
"""

import anthropic
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from skill_extraction_prompt import (  # noqa: E402
    PROMPT_VERSION,
    build_request_params,
    parse_skill_response,
)

# ── Config ────────────────────────────────────────────────────────────────────

YEAR       = "2021"
OCCUPATION = "data_scientist"
MODEL      = "claude-haiku-4-5-20251001"

EXTRACTED_DIR = Path("data/extracted")
SKILLS_DIR    = Path("data/skills")
POLL_INTERVAL = 30   # seconds between status checks

# ── Load .env ─────────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Prompt ────────────────────────────────────────────────────────────────────
# Lives in pipeline/skill_extraction_prompt.py — see PROMPT_VERSION.

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_done_ids(out_path: Path) -> set[str]:
    """IDs already extracted SUCCESSFULLY.

    Only status == "ok" counts. Anything else — api_error, no_tool_use — is left
    pending so a re-run retries it instead of leaving a zero-skill row in place.
    """
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("status") == "ok":
                done.add(rec["id"])
    return done


def sanitize_custom_id(raw_id: str) -> str:
    """Replace chars not in [a-zA-Z0-9_-] with '_', truncate to 64."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id)
    return safe[:64]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir    = SKILLS_DIR / YEAR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / f"{OCCUPATION}.jsonl"
    state_path = out_dir / f"{OCCUPATION}_batch_state.json"

    client = anthropic.Anthropic()

    # Load source records
    in_path = EXTRACTED_DIR / YEAR / f"{OCCUPATION}.jsonl"
    records  = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    done_ids = load_done_ids(out_path)
    pending  = [r for r in records if r["id"] not in done_ids]

    print(f"Occupation : {YEAR}/{OCCUPATION}")
    print(f"Total      : {len(records)}  |  Already done: {len(done_ids)}  |  Pending: {len(pending)}")

    if not pending:
        print("All records already processed. Nothing to do.")
        return

    # ── Submit or resume ──────────────────────────────────────────────────────

    batch_id = None
    saved_id_map: dict[str, str] = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("status") != "results_saved":
            batch_id = state.get("batch_id")
            saved_id_map = state.get("id_map", {})
            print(f"Resuming existing batch: {batch_id}")

    # Build safe-id → original-id lookup (handles path-style IDs with slashes etc.)
    # Also build orig_to_safe so batch_requests uses the same resolved safe IDs.
    id_map: dict[str, str] = {}
    orig_to_safe: dict[str, str] = {}
    for rec in pending:
        safe = sanitize_custom_id(rec["id"])
        # Ensure uniqueness: append index suffix if collision
        if safe in id_map and id_map[safe] != rec["id"]:
            for i in range(1, 10000):
                candidate = sanitize_custom_id(rec["id"])[:59] + f"_{i:04d}"
                if candidate not in id_map:
                    safe = candidate
                    break
        id_map[safe] = rec["id"]
        orig_to_safe[rec["id"]] = safe

    if not batch_id:
        print(f"\nSubmitting batch of {len(pending)} requests to Anthropic Batch API...")
        batch_requests = [
            {
                "custom_id": orig_to_safe[rec["id"]],
                "params": build_request_params(rec["description"], MODEL),
            }
            for rec in pending
        ]
        batch = client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        state_path.write_text(json.dumps({
            "batch_id": batch_id,
            "status": "submitted",
            "submitted": len(pending),
            "id_map": id_map,
        }, indent=2))
        print(f"Batch submitted: {batch_id}")

    # ── Poll until complete ───────────────────────────────────────────────────

    print(f"\nPolling every {POLL_INTERVAL}s...")
    while True:
        batch  = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"  [{batch.processing_status}]  processing: {counts.processing}  "
              f"succeeded: {counts.succeeded}  errored: {counts.errored}")

        if batch.processing_status == "ended":
            break

        time.sleep(POLL_INTERVAL)

    # ── Retrieve and save results ─────────────────────────────────────────────

    print(f"\nBatch ended. Retrieving results...")
    written = 0
    n_ok = n_truncated = n_api_error = n_no_tool_use = 0
    dropped_degree = dropped_evidence = 0

    # Merge id_map from current build + saved state (for resume case)
    merged_id_map = {**saved_id_map, **id_map}
    # Evidence spans are validated against the source text, so the description
    # has to be on hand at parse time.
    descriptions = {r["id"]: r["description"] for r in records}

    with open(out_path, "a", encoding="utf-8") as f:
        for result in client.messages.batches.results(batch_id):
            rec_id = merged_id_map.get(result.custom_id, result.custom_id)
            skills: list[dict] = []
            meta: dict = {}

            if result.result.type != "succeeded":
                status = "api_error"
                n_api_error += 1
            else:
                try:
                    skills, meta = parse_skill_response(
                        result.result.message, descriptions.get(rec_id, "")
                    )
                    if meta["truncated"]:
                        status = "truncated"
                        n_truncated += 1
                    else:
                        status = "ok"
                        n_ok += 1
                    dropped_degree   += meta["n_dropped_degree"]
                    dropped_evidence += meta["n_dropped_evidence"]
                except ValueError:
                    status = "no_tool_use"
                    n_no_tool_use += 1

            out = {
                "id": rec_id,
                "status": status,
                "prompt_version": PROMPT_VERSION,
                "model": MODEL,
                "skills": skills,
            }
            f.write(json.dumps(out) + "\n")
            f.flush()
            written += 1

    errors = n_api_error + n_no_tool_use + n_truncated
    print(f"Written: {written}  |  ok: {n_ok}  truncated: {n_truncated}  "
          f"api_error: {n_api_error}  no_tool_use: {n_no_tool_use}")
    print(f"Dropped — degrees: {dropped_degree}  |  unmatched evidence: {dropped_evidence}")
    state_path.write_text(json.dumps({
        "batch_id": batch_id,
        "status":   "results_saved",
        "written":  written,
        "errors":   errors,
    }, indent=2))
    print(f"\nDone. Results → {out_path}")


if __name__ == "__main__":
    main()
