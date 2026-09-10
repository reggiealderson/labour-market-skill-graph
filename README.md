# Labour-market skill knowledge graph

Turn a folder of job advertisements into a **skill knowledge graph** per role:
which skills are demanded, how common each is, and which skills are demanded
together. This is a minimal, runnable extract of the pipeline behind the
interactive graph at
**[reggiealderson.com/skill-graph](https://reggiealderson.com/skill-graph)**.

You bring the job ads (in a documented schema) and an Anthropic API key; the
pipeline produces the graph JSON that the viewer renders. It ships a **worked
example** for six data/AI roles — swap the taxonomy and occupation lists for
your own domain.

> The raw job-ad dataset is **not** included and never leaves the pipeline;
> outputs are concept-level (skill labels + counts), never ad text.

## What the graph shows

- **Nodes** — skills (*concepts*) demanded in a role, carrying penetration
  (share of the role's ads), type (language, library, tool, method, …), and AI
  facets. A skill that is common but does not co-occur tightly is kept as an
  isolated node.
- **Edges** — skill pairs demanded together more than chance predicts
  (association-rule co-occurrence: support, confidence, lift).

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
# put your ads at data/extracted/<year>/<occupation>.jsonl  (see schema/)
# then run the stages in docs/RUN.md, ending at:
python3 pipeline/analytics/export_graph.py
# -> data/analytics/graph/<occupation>_2026.json
```

Full command sequence: **[`docs/RUN.md`](docs/RUN.md)**.

## The pipeline

Job ads → skill extraction → normalise → block (local embeddings) → semantic
merge into stable concepts → typing → category assignment → AI facets →
per-role penetration and co-occurrence → graph JSON.

- Overview: [`docs/pipeline_overview.md`](docs/pipeline_overview.md)
- Co-occurrence method (support/confidence/lift, the hub clause):
  [`docs/adjacency_method.md`](docs/adjacency_method.md)
- Input schema: [`schema/input_job_ads.md`](schema/input_job_ads.md)
- Output schema: [`schema/graph_data_contract.md`](schema/graph_data_contract.md)

Four LLM stages (extraction, merge, typing, assignment, facets) call the
Anthropic API; every model output is recorded as model-made and is meant to be
human-reviewed. Blocking embeds locally (`intfloat/e5-small-v2`) — no embedding
API.

## Layout

| path | what |
|------|------|
| `pipeline/` | Stage modules + `pipeline/analytics/` (penetration, adjacency, graph export). |
| `scripts/` | Stage runners (`run_*.py`, `classify_facet.py`). |
| `config/` | All rule content — normalisation, blocking, resolver, typing — versioned YAML/TOML, never in code. |
| `taxonomy/` | The worked-example categories + facets, and their validators' targets. |
| `tools/` | Taxonomy validators (no API needed). |
| `schema/` | Input and output data contracts. |
| `sample/` | A synthetic input file and one real example graph output. |
| `docs/` | How to run, pipeline overview, method. |

## Design principles (from the source project)

1. All rule content lives in versioned config, never in Python.
2. No rule depends on the current corpus.
3. Model output is never mistaken for human output — provenance columns
   (`model`, `by`) travel with every model-made row.
4. Nodes are a population, not a by-product of edges.

## Scope and honesty

This is the same code that produced the published graph, minus site-specific
ingestion (a cloud data pull) and the author's website-publishing step. It has
not been re-run end to end inside this public repo — you supply the data and
key. The taxonomy validators and local stages run standalone. See the scope note
in [`docs/RUN.md`](docs/RUN.md).

## Interpretation and limits

Built from cross-sectional ad samples. It shows which skills are *demanded
together*, not that one causes demand for another. Penetration is ad-level
incidence within a role-year sample — a snapshot, not a census.
