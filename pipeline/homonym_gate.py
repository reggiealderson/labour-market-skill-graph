"""A narrower question than full adjudication: is this child a NAMED PRODUCT
whose real meaning has nothing to do with the parent word's general sense?

WHY THIS EXISTS. The full adjudication prompt (adjudicate.py) scored 96.4% on
edges overall but missed 9 of 13 known-bad edges — all misses were a product
name hiding inside a common English parent word. A cheap string rule was also
tried (analysis_brand_word_check.py) and rejected: 2.2% precision, because most
brand-word edges (`azure sql`, `aws lambda`) ARE genuinely correct. Neither a
general prompt nor a keyword match separates the two classes.

This asks the narrow question directly, with one worked example of each class
in the prompt, rather than the general "is this a valid rollup" question the
production prompt asks. It is a hypothesis, not yet policy — this script only
measures whether asking a smaller, sharper question does better.

MODEL: claude-haiku-4-5-20251001, per instruction to prioritise Haiku for an
experimental check like this. Purpose: test whether a cheaper, narrower model
can catch the proper-noun-collision fault the general model missed. Not yet
human-reviewed as a production step — this run's OUTPUT is what gets reviewed.

Run: python3 pipeline/homonym_gate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from adjudicate import build_requests, edge_item, load_dotenv, run_standard
from resolver_decisions import edge_key, load as load_decisions
from resolver_level1 import LEVEL1_VERSION

import anthropic

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "resolver"
MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You check one narrow failure in a skills catalogue: a CHILD skill
string that names a specific PRODUCT or company tool, where that product's real
meaning has nothing to do with the general sense of the PARENT word — even
though the parent word appears inside the child's name.

Example of the fault:
  parent "access", child "microsoft access"
  Microsoft Access is a database application (a named product). It is not a
  case of general computer "access" (permissions, entry). -> reject

Example that is NOT the fault — read this one carefully, it is the trap:
  parent "sql", child "azure sql"
  Azure SQL is Microsoft's product name too, but the product itself IS a form
  of SQL. Someone with Azure SQL skill genuinely has SQL skill. -> rollup

The test: does knowing the child skill necessarily mean you know the parent
skill, in its ordinary sense? A vendor-specific tool that is honestly a form of
the parent skill is `rollup`. A product whose name happens to contain the
parent word, but which does something unrelated to the parent's ordinary
meaning, is `reject`.

Answer `uncertain` if you are not confident which case this is.
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
                        "verdict": {"type": "string",
                                   "enum": ["rollup", "reject", "uncertain"]},
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


def build_requests_custom(items, model, size):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    reqs = []
    for b in range(0, len(items), size):
        group = items[b:b + size]
        body = "\n".join(it["text"] for it in group)
        reqs.append(Request(
            custom_id=f"batch-{b}",
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=16000, system=SYSTEM, tools=[TOOL],
                tool_choice={"type": "tool", "name": "record_verdicts"},
                messages=[{"role": "user", "content": f"Check each item.\n\n{body}"}],
            ),
        ))
    return reqs


def main():
    load_dotenv()
    decisions = load_decisions(level1_version=LEVEL1_VERSION)
    human_edges = {k: v for k, v in decisions.edges.items() if v["by"] == "human"}

    prepass = json.loads((OUT / "edge_prepass.json").read_text())
    by_key = {(e["parent"], e["child"]): e
             for e in prepass["obvious"] + prepass["remainder"]}

    items, skipped = [], []
    for i, key in enumerate(human_edges):
        parent, child = key.split(" || ")
        e = by_key.get((parent, child))
        if e is None:
            skipped.append(key)
            continue
        items.append(edge_item(i, e))
    if skipped:
        print(f"{len(skipped)} human decisions have no mention record, skipped")
    print(f"{len(items)} items -> {MODEL}, blind, narrow prompt\n")

    reqs = build_requests_custom(items, MODEL, 25)
    client = anthropic.Anthropic()

    verdicts = {}
    for i, req in enumerate(reqs, 1):
        print(f"  request {i}/{len(reqs)}")
        msg = client.messages.create(**req["params"])
        for block in msg.content:
            if block.type == "tool_use":
                for v in block.input.get("verdicts", []):
                    verdicts[v["id"]] = v

    out = [{
        "id": it["id"], "kind": it["kind"], "key": list(it["key"]),
        "verdict": verdicts.get(it["id"], {}).get("verdict", "MISSING"),
        "reason": verdicts.get(it["id"], {}).get("reason", ""),
        "model": MODEL,
    } for it in items]

    path = OUT / "homonym_gate_verdicts.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    missing = sum(1 for r in out if r["verdict"] == "MISSING")
    print(f"\n{len(out)} verdicts -> {path.name}"
          + (f"  ({missing} MISSING)" if missing else ""))


if __name__ == "__main__":
    main()
