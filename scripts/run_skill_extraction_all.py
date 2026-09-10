"""
Submit all pending skill extraction batches simultaneously, poll until all
complete, then save results.

Scope is the one locked on 2026-08-26: LinkedIn-only, 2021 = the three roles
that existed in volume that year, 2026 = the Core 6. 385 + 2,841 = 3,226
records.

Resumable at two levels: an in-flight batch is picked up from its state file,
and any record whose last result was not status == "ok" is resubmitted on the
next run.

Run from project root:
  python3 scripts/run_skill_extraction_all.py
"""

import anthropic
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from skill_extraction_prompt import (  # noqa: E402
    PROMPT_VERSION,
    build_request_params,
    parse_skill_response,
)

# ── Config ────────────────────────────────────────────────────────────────────

COMBOS = [
    ("2021", "data_analyst"),
    ("2021", "data_engineer"),
    ("2021", "data_scientist"),
    ("2026", "data_analyst"),
    ("2026", "data_engineer"),
    ("2026", "data_scientist"),
    ("2026", "machine_learning_engineer"),
    ("2026", "ai_engineer"),
    ("2026", "analytics_engineer"),
]

MODEL         = "claude-haiku-4-5-20251001"
EXTRACTED_DIR = Path("data/extracted")
SKILLS_DIR    = Path("data/skills")
POLL_INTERVAL = 60   # seconds between status checks across all batches

# ── Load .env ─────────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Helpers ───────────────────────────────────────────────────────────────────

def with_retry(fn, *, what: str, attempts: int = 6, base: float = 5.0):
    """Call fn(), retrying transient network failures with exponential backoff.

    A batch of this size polls for a long time, and a single DNS blip or dropped
    connection would otherwise abort the whole run — as happened on the first
    launch, at the poll on line ~184. The batches themselves are unaffected by a
    client-side crash (they live server-side for 29 days), but crashing means
    nobody is there to retrieve the results.
    """
    for i in range(attempts):
        try:
            return fn()
        except (anthropic.APIConnectionError, anthropic.APITimeoutError,
                anthropic.InternalServerError, anthropic.RateLimitError) as e:
            if i == attempts - 1:
                raise
            delay = base * (2 ** i)
            print(f"    [retry {i+1}/{attempts-1}] {what}: {type(e).__name__} — "
                  f"waiting {delay:.0f}s")
            time.sleep(delay)


def sanitize_id(raw_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id)
    return safe[:64]


def load_done_ids(out_path: Path) -> set[str]:
    """IDs already extracted SUCCESSFULLY — status == "ok" only.

    Failures stay pending so a re-run retries them rather than leaving a
    zero-skill row in the dataset.
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


def load_records(year: str, occ: str) -> list[dict]:
    path = EXTRACTED_DIR / year / f"{occ}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = anthropic.Anthropic()

    # ── Determine which combos need running ───────────────────────────────────
    pending = []
    for year, occ in COMBOS:
        out_path   = SKILLS_DIR / year / f"{occ}.jsonl"
        state_path = SKILLS_DIR / year / f"{occ}_batch_state.json"
        records    = load_records(year, occ)
        n_expected = len(records)
        done_ids   = load_done_ids(out_path)

        # Resume in-progress batch if state file exists and not yet saved
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("status") != "results_saved":
                print(f"  RESUME {year}/{occ}  (batch {state['batch_id']})")
                pending.append({
                    "year": year, "occ": occ,
                    "out_path": out_path, "state_path": state_path,
                    "batch_id": state["batch_id"],
                    "id_map": state.get("id_map", {}),
                    "records": records,
                    "n_expected": n_expected,
                })
                continue

        to_run = [r for r in records if r["id"] not in done_ids]

        if not to_run:
            print(f"  SKIP  {year}/{occ}  (all {len(done_ids)} already done)")
            continue

        # Build id map
        id_map = {}
        orig_to_safe = {}
        for rec in to_run:
            safe = sanitize_id(rec["id"])
            if safe in id_map and id_map[safe] != rec["id"]:
                for i in range(1, 10000):
                    candidate = sanitize_id(rec["id"])[:59] + f"_{i:04d}"
                    if candidate not in id_map:
                        safe = candidate
                        break
            id_map[safe]            = rec["id"]
            orig_to_safe[rec["id"]] = safe

        print(f"  SUBMIT {year}/{occ}  ({len(to_run)} of {n_expected} records)")
        batch_requests = [
            {
                "custom_id": orig_to_safe[rec["id"]],
                "params":    build_request_params(rec["description"], MODEL),
            }
            for rec in to_run
        ]
        batch    = with_retry(
            lambda: client.messages.batches.create(requests=batch_requests),
            what=f"submit {year}/{occ}")
        batch_id = batch.id

        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({
            "batch_id":       batch_id,
            "status":         "submitted",
            "submitted":      len(to_run),
            "prompt_version": PROMPT_VERSION,
            "id_map":         id_map,
        }, indent=2))

        pending.append({
            "year": year, "occ": occ,
            "out_path": out_path, "state_path": state_path,
            "batch_id": batch_id, "id_map": id_map,
            "records": records,
            "n_expected": n_expected,
        })

    if not pending:
        print("\nAll combos already complete.")
        return

    print(f"\n{len(pending)} batch(es) submitted. Polling every {POLL_INTERVAL}s ...\n")

    # ── Poll all batches until all ended ──────────────────────────────────────
    while True:
        all_ended = True
        for job in pending:
            if job.get("ended"):
                continue
            batch  = with_retry(
                lambda j=job: client.messages.batches.retrieve(j["batch_id"]),
                what=f"poll {job['year']}/{job['occ']}")
            counts = batch.request_counts
            status = batch.processing_status
            print(f"  {job['year']}/{job['occ']:<32} [{status}]  "
                  f"processing={counts.processing}  succeeded={counts.succeeded}  errored={counts.errored}")
            if status == "ended":
                job["ended"] = True
            else:
                all_ended = False

        if all_ended:
            print("\nAll batches ended. Retrieving results ...\n")
            break

        time.sleep(POLL_INTERVAL)

    # ── Retrieve and save results ─────────────────────────────────────────────
    for job in pending:
        year, occ     = job["year"], job["occ"]
        out_path      = job["out_path"]
        state_path    = job["state_path"]
        merged_id_map = job["id_map"]

        # Reload state id_map in case of resume
        if state_path.exists():
            saved_state   = json.loads(state_path.read_text())
            merged_id_map = {**saved_state.get("id_map", {}), **merged_id_map}

        # Evidence spans are validated against the source text.
        descriptions = {r["id"]: r["description"] for r in job["records"]}

        written = skipped = 0
        n_ok = n_truncated = n_api_error = n_no_tool_use = 0
        dropped_degree = dropped_evidence = 0

        # Already-written ok rows. A crash mid-retrieval leaves the state file
        # saying "submitted", so the next run resumes the same batch and
        # re-fetches every result in it — these would otherwise be duplicated.
        already = load_done_ids(out_path)

        # Fetch the whole result set as a unit so a mid-stream network failure
        # retries cleanly instead of leaving a half-written file. Results stay
        # retrievable server-side for 29 days, so re-fetching is always safe.
        results = with_retry(
            lambda j=job: list(client.messages.batches.results(j["batch_id"])),
            what=f"fetch results {year}/{occ}")

        with open(out_path, "a", encoding="utf-8") as f:
            for result in results:
                rec_id = merged_id_map.get(result.custom_id, result.custom_id)
                if rec_id in already:
                    skipped += 1
                    continue
                skills: list[dict] = []

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

                f.write(json.dumps({
                    "id":             rec_id,
                    "status":         status,
                    "prompt_version": PROMPT_VERSION,
                    "model":          MODEL,
                    "skills":         skills,
                }) + "\n")
                f.flush()
                written += 1

        errors = n_api_error + n_no_tool_use + n_truncated
        state_path.write_text(json.dumps({
            "batch_id":       job["batch_id"],
            "status":         "results_saved",
            "prompt_version": PROMPT_VERSION,
            # Keep the id_map after saving. Batch results stay retrievable for
            # 29 days, so re-parsing them offline is free — but only if the
            # custom_id -> record_id mapping survives. Dropping it here meant
            # rebuilding it by hand to diagnose the evidence-drop rate.
            "id_map":         merged_id_map,
            "written":        written,
            "ok":             n_ok,
            "truncated":      n_truncated,
            "api_error":      n_api_error,
            "no_tool_use":    n_no_tool_use,
            "dropped_degree":   dropped_degree,
            "dropped_evidence": dropped_evidence,
            "skipped_already_written": skipped,
        }, indent=2))
        print(f"  {year}/{occ}  → {out_path}  ({written} written, {n_ok} ok, {errors} to retry, "
              f"{dropped_evidence} evidence-dropped"
              + (f", {skipped} already present)" if skipped else ")"))

    print("\nDone. Re-run this script to retry any non-ok rows.")


if __name__ == "__main__":
    main()
