"""Item 4 — model first pass over what the deterministic rules cannot settle.

CLAUDE.md requires every Anthropic API step to name its model, purpose, and
review status. All three live in config/resolver_policy.toml under
[adjudication] and are read from there, never hardcoded here.

CLAUDE.md also says not to call the API for work that can be done in the Claude
Code conversation. That rule was written when a human was always present. This
pipeline must run unattended when new job ads arrive, and 1,918 items is not a
conversation. The rule is superseded for this step, deliberately and in writing.

WHAT REACHES THE MODEL. Two deterministic passes run first and between them
remove 1,414 obvious edges and 354 below-limit acronym pairs. The model sees
only genuinely ambiguous items — 1,918 on a first run, ~96 on a later one.

WHY BATCHES OF 25. Measured: one item per request costs 848 input tokens to ask
a ~67-token question, because the shared system prompt and tool schema are 781
tokens. The prefix is billed once per REQUEST, so 25 items per request divides
that overhead by 25 — $2.20 to $0.76 on a full run. Prompt caching cannot help:
Sonnet 5's minimum cacheable prefix is 1,024 tokens and ours is 781, so caching
silently never fires. See policy `adjudication.items_per_request`.

WHY `uncertain` IS FIRST-CLASS. A wrong merge is irreversible and corrupts a
published figure with no signal that anything is wrong. An `uncertain` costs a
person thirty seconds. The prompt says so explicitly, and uncertain items are
escalated to Opus once, then queued for a human — never guessed.

Run:
  python3 pipeline/adjudicate.py --calibrate   # 24 known-answer pairs, blind
  python3 pipeline/adjudicate.py --edges       # the 1,894-edge workload
  python3 pipeline/adjudicate.py --dry-run     # build batches, send nothing
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from resolver_policy import load as load_policy

PROMPT_VERSION = "adjudicate-v1.0.0"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "resolver"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read .env into the environment without printing it.

    The key lives in a gitignored, untracked .env. Loading it in-process keeps
    it out of shell history and out of any log — a credential echoed once is a
    credential to rotate.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ── The prompt ───────────────────────────────────────────────────────────────
# Deliberately short. The measured prefix is 781 tokens and Sonnet 5's cache
# minimum is 1,024 — padding this to reach the cache threshold would cost more
# than it saves at 25 items per request. Kept lean instead.
SYSTEM = """You adjudicate skill-catalogue relationships for a labour-market dataset.

Two kinds of question:

PAIR — does this acronym, as used in job advertisements, mean this spelled-out
phrase? Answer `merge` only if they are the same skill. Initials that coincide
are not a match: "ai" does not mean "apache iceberg".

EDGE — is the child a specific case of the parent? Answer `rollup` only if
everyone who has the child skill necessarily has the parent skill. Shared
vocabulary is not a relationship: "agent-to-agent communication" is a protocol
between software agents, not a human communication skill.

Answer `uncertain` whenever you would be guessing. An uncertain answer costs a
person thirty seconds. A wrong merge is irreversible and corrupts a published
figure with no signal that anything is wrong. Prefer uncertain to a guess.

Answer every item you are given, once, using its exact id."""

TOOL = {
    "name": "record_verdicts",
    "description": "Record one verdict per item, in the order given.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["merge", "keep_separate", "rollup",
                                     "related", "reject", "uncertain"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "verdict", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    },
}


def pair_item(i: int, c: dict) -> dict:
    return {
        "id": f"pair-{i}",
        "kind": "pair",
        "key": (c["acronym"], c["spelled_out"]),
        "text": (f"[pair-{i}] PAIR  acronym {c['acronym']!r} "
                 f"({c['acronym_mentions']} mentions)  vs  spelled-out "
                 f"{c['spelled_out']!r} ({c['gain_mentions']} mentions)"),
    }


def edge_item(i: int, e: dict) -> dict:
    return {
        "id": f"edge-{i}",
        "kind": "edge",
        "key": (e["parent"], e["child"]),
        "text": (f"[edge-{i}] EDGE  parent {e['parent']!r} "
                 f"({e['parent_mentions']} mentions)  ->  child "
                 f"{e['child']!r} ({e['child_mentions']} mentions)"),
    }


def chunk(items: list, n: int) -> list[list]:
    return [items[i:i + n] for i in range(0, len(items), n)]


def build_requests(items: list[dict], model: str, size: int) -> list[dict]:
    """One Batch API request per chunk of `size` items."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    reqs = []
    for b, group in enumerate(chunk(items, size)):
        body = "\n".join(it["text"] for it in group)
        reqs.append(Request(
            custom_id=f"batch-{b}",
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=16000,
                system=SYSTEM,
                tools=[TOOL],
                tool_choice={"type": "tool", "name": "record_verdicts"},
                messages=[{"role": "user", "content":
                           f"Adjudicate each item.\n\n{body}"}],
            ),
        ))
    return reqs


def run_standard(client, requests: list) -> dict[str, dict]:
    """Send the same requests synchronously, one call each.

    Used when the workload is below `adjudication.batch_api_min_items`. The
    Batch API is the not-latency-sensitive tier — that is why it is half price
    — and there is no SLA under 24 hours. On a 24-item calibration run a person
    is waiting on, that trades an unbounded wait for about one cent.
    """
    verdicts: dict[str, dict] = {}
    for i, req in enumerate(requests, 1):
        print(f"  request {i}/{len(requests)}")
        msg = client.messages.create(**req["params"])
        for block in msg.content:
            if block.type == "tool_use":
                for v in block.input.get("verdicts", []):
                    verdicts[v["id"]] = v
    return verdicts


def run_batch(client, requests: list, poll: int = 30) -> dict[str, dict]:
    """Submit, poll to completion, return {item_id: verdict}.

    Results arrive in ANY order and are keyed by custom_id, never by position —
    a positional assumption here would silently mis-assign verdicts to items.
    """
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch {batch.id} submitted ({len(requests)} requests)")

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  {batch.processing_status}: "
              f"{batch.request_counts.processing} processing, "
              f"{batch.request_counts.succeeded} done")
        time.sleep(poll)

    print(f"  succeeded {batch.request_counts.succeeded}, "
          f"errored {batch.request_counts.errored}")

    verdicts: dict[str, dict] = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            print(f"  ! {result.custom_id}: {result.result.type}")
            continue
        for block in result.result.message.content:
            if block.type == "tool_use":
                for v in block.input.get("verdicts", []):
                    verdicts[v["id"]] = v
    return verdicts


def main(argv: list[str]) -> int:
    load_dotenv()
    policy = load_policy()
    model = policy.get("adjudication.model")
    escalation = policy.get("adjudication.escalation_model")
    size = policy.get("adjudication.items_per_request")
    batch_min = policy.get("adjudication.batch_api_min_items")

    calibrate = "--calibrate" in argv
    do_edges = "--edges" in argv
    dry = "--dry-run" in argv

    if not (calibrate or do_edges):
        print(__doc__.split("Run:")[1])
        return 1

    items: list[dict] = []
    if calibrate:
        cand = json.loads((OUT / "acronym_gap_candidates.json").read_text())
        pool = cand["to_adjudicate"] or cand["already_decided"]
        items += [pair_item(i, c) for i, c in enumerate(pool)]
    if do_edges:
        pre = json.loads((OUT / "edge_prepass.json").read_text())
        items += [edge_item(i, e) for i, e in enumerate(pre["remainder"])]

    reqs = build_requests(items, model, size)
    # Batch or standard is chosen by workload size, not as a blanket setting.
    use_batch = (policy.get("adjudication.use_batch_api")
                 and len(items) >= batch_min)
    send = run_batch if use_batch else run_standard
    print(f"model {model} (escalation {escalation} on uncertain)")
    print(f"{len(items):,} items -> {len(reqs):,} requests of {size}")
    print(f"transport: {'batch (50% off, no latency SLA)' if use_batch else 'standard'}"
          f"   [batch_api_min_items={batch_min}]")

    if dry:
        print("\n--dry-run: nothing sent. First request body:\n")
        print(reqs[0]["params"]["messages"][0]["content"][:600])
        return 0

    client = anthropic.Anthropic()
    print("\nSONNET PASS")
    verdicts = send(client, reqs)

    # Escalate uncertain ONCE to Opus. `uncertain` is the only route — a
    # confident-but-wrong Sonnet verdict is not re-asked, because a second
    # opinion on a confident answer is a double run, which the policy forbids.
    unsure = [it for it in items if verdicts.get(it["id"], {}).get("verdict") == "uncertain"]
    if unsure:
        print(f"\nOPUS ESCALATION — {len(unsure):,} uncertain")
        esc = build_requests(unsure, escalation, size)
        for item_id, v in send(client, esc).items():
            v["escalated"] = True
            verdicts[item_id] = v

    by_item = {it["id"]: it for it in items}
    out = [{
        "id": it["id"], "kind": it["kind"], "key": list(it["key"]),
        "verdict": verdicts.get(it["id"], {}).get("verdict", "MISSING"),
        "reason": verdicts.get(it["id"], {}).get("reason", ""),
        "escalated": verdicts.get(it["id"], {}).get("escalated", False),
        "model": escalation if verdicts.get(it["id"], {}).get("escalated") else model,
        "prompt_version": PROMPT_VERSION,
    } for it in by_item.values()]

    path = OUT / ("calibration_model_verdicts.json" if calibrate
                  else "edge_model_verdicts.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    missing = sum(1 for r in out if r["verdict"] == "MISSING")
    print(f"\n{len(out):,} verdicts -> {path.name}"
          + (f"   ({missing} MISSING — item never answered)" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
