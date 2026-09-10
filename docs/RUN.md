# Running the pipeline

End to end: a folder of job ads → per-role skill knowledge graphs. Run every
command **from the repository root**.

## 0. Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...        # required by the LLM stages
```

- The LLM stages call the Anthropic API (semantic merge, typing, category
  assignment, facet classification, and skill extraction). They cost tokens.
- The blocking stage embeds locally with `intfloat/e5-small-v2`
  (`config/embedding.yml`); `sentence-transformers` downloads it on first run.

## 1. Provide your input

Put your ads at `data/extracted/<year>/<occupation>.jsonl`, one file per cell,
in the schema at [`../schema/input_job_ads.md`](../schema/input_job_ads.md). A
synthetic example is at `sample/extracted/2026/data_scientist.jsonl`.

## 2. Stages, in order

| # | Stage | Command | Needs |
|---|-------|---------|-------|
| 1 | Skill extraction | `python3 scripts/run_skill_extraction_all.py` | API |
| 2 | Normalisation | `python3 scripts/run_normalise_all.py` | local |
| 3 | Blocking (pick threshold) | `python3 scripts/run_blocking.py --diagnostics` | local |
| 3 | Blocking (build groups) | `python3 scripts/run_blocking.py` | local |
| 4 | Semantic merge → concepts | `python3 scripts/run_merge_batch.py` | API |
| 5 | Typing | `python3 scripts/run_typing.py` | API |
| 6 | Category assignment | `python3 scripts/run_stage_f.py` | API |
| 7 | Facets (one per facet) | `python3 scripts/classify_facet.py <facet_id> <concept_csv> --facets-yaml taxonomy/facets.yaml --out taxonomy/concept_facet.csv` | API |
| 8 | Analytics backbone | `python3 pipeline/analytics/build_posting_skill.py` | local |
| 8 | Penetration | `python3 pipeline/analytics/compute_penetration.py` | local |
| 8 | Adjacency (edges + nodes) | `python3 pipeline/analytics/compute_adjacency.py` | local |
| 9 | Export graph JSON | `python3 pipeline/analytics/export_graph.py` | local |

Output: `data/analytics/graph/<occupation>_2026.json` + `index.json`, in the
schema at [`../schema/graph_data_contract.md`](../schema/graph_data_contract.md).

Several LLM stages use the Anthropic **Message Batches** API and are
resume-safe: a re-run picks up in-flight batches and re-submits only records
that did not succeed.

## 3. Validate the taxonomy (no API needed)

The taxonomy is rule content. Validate it any time you edit it:

```bash
python3 tools/validate_categories.py
python3 tools/validate_facets.py
```

## Adapting to your own data

This repo ships the author's **worked example** — six data/AI occupations, a
27-category taxonomy (`taxonomy/categories.yaml`), and four AI facets
(`taxonomy/facets.yaml`). To use it for a different domain:

1. **Occupations.** The occupation lists are hard-coded in the stage runners
   (e.g. `scripts/run_normalise_all.py`). Edit them to your cells. Docstrings in
   the runners reference the author's corpus counts — those are illustrative.
2. **Taxonomy.** Replace `taxonomy/categories.yaml` and `taxonomy/facets.yaml`
   with your own, then re-run the validators. Category assignment and facets are
   only as meaningful as the taxonomy you give them.
3. **Config.** Normalisation, blocking threshold, resolver policy, and typing
   rules live in `config/*.yml|.toml|.yaml` — all versioned, none in code.

## Honest scope note

This is a faithful extract of the private pipeline, minus site-specific
ingestion (a cloud data pull) and the author's website-publishing step. The code
is the same code that produced the graph at reggiealderson.com/skill-graph. It
has **not** been re-run end to end inside this public repo — you supply the data
and API key. The taxonomy validators and the local (non-API) stages run
standalone; the API stages are wired exactly as in the source project.
