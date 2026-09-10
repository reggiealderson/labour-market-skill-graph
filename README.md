# Labour-market skill knowledge graph

Turn a folder of job advertisements into a **skill knowledge graph** per role:
which skills are demanded, how common each is, and which skills are demanded
together. This is a minimal, runnable extract of the pipeline behind the
interactive graph at
**[reggiealderson.com/skill-graph](https://reggiealderson.com/skill-graph)**.

You bring the job ads (in a documented schema) and an Anthropic API key; the
pipeline produces the graph JSON that the viewer renders.

**This is a data-science / AI domain pipeline, by design.** The method was built
for data-science, analytics, and AI hiring, and that shows in its hand-authored
parts: the six occupations, the nine concept **types**, the 27-category
**taxonomy** (`taxonomy/categories.yaml`), and the four **AI facets**
(`taxonomy/facets.yaml`) all encode judgement calls about *data/AI* skills — for
example what separates a *library* from a *method*, or what counts as "core AI"
versus "AI-adjacent". They ship as a **worked example**. The machinery around
them is general: to use another field, you swap this domain content (see
[`docs/RUN.md`](docs/RUN.md), "Adapting to your own data").

> The raw job-ad dataset is **not** included and never leaves the pipeline;
> outputs are concept-level (skill labels + counts), never ad text.

## What the graph shows

- **Nodes** — skills (*concepts*) demanded in a role, carrying penetration
  (share of the role's ads), type (language, library, tool, method, …), and AI
  facets. A skill that is common but does not co-occur tightly is kept as an
  isolated node.
- **Edges** — skill pairs that are demanded together more than chance predicts,
  weighted by a per-cell co-occurrence statistic.

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

Each stage is deterministic and versioned. LLM stages run at temperature 0 and
their output is recorded as model-made and meant for human review; blocking
embeds locally (`intfloat/e5-small-v2`), no embedding API.

1. **Skill extraction** *(LLM)* — read the skills mentioned in each ad's
   description.
2. **Normalisation** — fold casing, punctuation, and spelling variants of each
   skill surface form. Rules live in `config/`, never in code.
3. **Blocking** *(local embeddings)* — group plausibly-duplicate forms with
   nearest-neighbour search, so the merge step compares only likely pairs.
4. **Semantic merge** *(LLM)* — decide which forms name the same skill, producing
   stable *concepts* (hash-derived ids; new ads reuse existing ids).
5. **Typing** *(LLM)* — assign each concept one of nine data/AI types (language,
   library, tool, method, knowledge_area, …).
6. **Category assignment** *(LLM)* — place each concept in the domain taxonomy
   (`taxonomy/categories.yaml`); prevalence signals withheld so counts can't bias
   it.
7. **AI facets** *(LLM)* — apply the four AI facets (`taxonomy/facets.yaml`):
   AI-relatedness, wave, workflow layer, vendor ecosystem.
8. **Penetration** — per role, the share of ads mentioning each concept (the
   node's prevalence).
9. **Adjacency** — a per-cell co-occurrence statistic for each skill pair (the
   edge weight).
10. **Graph export** — assemble one JSON per role: nodes (the concept
    population) + edges (pairs that clear the keep thresholds).

Full detail — including how the method is grounded in the data-science domain —
is in **[`docs/pipeline_overview.md`](docs/pipeline_overview.md)**.

- Input schema: [`schema/input_job_ads.md`](schema/input_job_ads.md)
- Output schema: [`schema/graph_data_contract.md`](schema/graph_data_contract.md)
- How to run: [`docs/RUN.md`](docs/RUN.md)

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
