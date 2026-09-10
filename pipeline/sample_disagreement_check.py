"""How often would the general model (Sonnet) and the narrow homonym gate
(Haiku) disagree, on edges NEITHER has ever been scored against?

The 332-edge calibration measured accuracy against known human answers. This
measures something different: on the 1,726 edges nobody has reviewed yet, how
big would the "send to a human" queue be if we only escalated disagreements
between the two models, instead of sending everything?

A seeded random sample of 199 (kept under batch_api_min_items=200 so both runs
stay on the fast standard API — the same reason earlier interactive runs were
forced off the batch queue). Reproducible: same seed, same sample, every time
this is re-run.

Ground truth does not exist for this sample — it CANNOT be scored for
accuracy. It only sizes the disagreement queue. Run both model scripts on this
sample first, or the scoring script will find nothing to compare.

MODELS: claude-sonnet-5 (general adjudication prompt, adjudicate.py's SYSTEM),
claude-haiku-4-5-20251001 (narrow homonym prompt, homonym_gate.py's SYSTEM).
Purpose: measure disagreement-queue size for a proposed second-opinion routing
policy. Not yet human-reviewed as a production step.

Run: python3 pipeline/sample_disagreement_check.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adjudicate import (SYSTEM as SONNET_SYSTEM, TOOL as SONNET_TOOL,
                        edge_item, load_dotenv, run_standard)
from homonym_gate import SYSTEM as HAIKU_SYSTEM, TOOL as HAIKU_TOOL
from resolver_decisions import edge_key, load as load_decisions
from resolver_level1 import LEVEL1_VERSION

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "resolver"
SAMPLE_N = 199
SEED = 20260828


def build_requests(items, model, system, tool, size=25):
    reqs = []
    for b in range(0, len(items), size):
        group = items[b:b + size]
        body = "\n".join(it["text"] for it in group)
        reqs.append(Request(
            custom_id=f"batch-{b}",
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=16000, system=system, tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
                messages=[{"role": "user", "content": f"Adjudicate each item.\n\n{body}"}],
            ),
        ))
    return reqs


def run(client, items, model, system, tool, label):
    reqs = build_requests(items, model, system, tool)
    print(f"\n{label}: {len(items)} items -> {len(reqs)} requests ({model})")
    verdicts = {}
    for i, req in enumerate(reqs, 1):
        print(f"  request {i}/{len(reqs)}")
        msg = client.messages.create(**req["params"])
        for block in msg.content:
            if block.type == "tool_use":
                for v in block.input.get("verdicts", []):
                    verdicts[v["id"]] = v
    return verdicts


def main():
    load_dotenv()
    decisions = load_decisions(level1_version=LEVEL1_VERSION)
    decided_keys = set(decisions.edges.keys())
    pre = json.loads((OUT / "edge_prepass.json").read_text())
    unreviewed = [e for e in pre["remainder"]
                 if edge_key(e["parent"], e["child"]) not in decided_keys]
    print(f"{len(unreviewed)} edges have no human verdict at all")

    rng = random.Random(SEED)
    sample = rng.sample(unreviewed, SAMPLE_N)
    items = [edge_item(i, e) for i, e in enumerate(sample)]
    print(f"sampled {len(items)} (seed {SEED}, reproducible)")

    client = anthropic.Anthropic()
    sonnet = run(client, items, "claude-sonnet-5", SONNET_SYSTEM, SONNET_TOOL, "SONNET (general)")
    haiku = run(client, items, "claude-haiku-4-5-20251001", HAIKU_SYSTEM, HAIKU_TOOL, "HAIKU (narrow)")

    out = [{"id": it["id"], "key": list(it["key"]),
           "sonnet": sonnet.get(it["id"], {}).get("verdict", "MISSING"),
           "sonnet_reason": sonnet.get(it["id"], {}).get("reason", ""),
           "haiku": haiku.get(it["id"], {}).get("verdict", "MISSING"),
           "haiku_reason": haiku.get(it["id"], {}).get("reason", "")}
          for it in items]
    path = OUT / "sample_disagreement.json"
    path.write_text(json.dumps({"seed": SEED, "sample_n": SAMPLE_N,
                                "pool_n": len(unreviewed), "rows": out}, indent=2) + "\n")
    print(f"\n{len(out)} rows -> {path.name}")


if __name__ == "__main__":
    main()
