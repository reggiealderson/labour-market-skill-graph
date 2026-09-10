# Input data contract — job advertisements

The pipeline begins **after** raw ingestion. You supply one JSONL file per
occupation-year cell; the pipeline does the rest. The original project pulls
these from a cloud store, but that step is site-specific and is **not** part of
this repository — bring your own ads in the format below.

## Location and naming

```
data/extracted/<year>/<occupation>.jsonl
```

- `<year>` — a snapshot label, e.g. `2026`.
- `<occupation>` — a slug, e.g. `data_scientist`. One file per occupation.
- One JSON object per line (JSONL / NDJSON).

Occupation slugs are yours to choose; they become the cells the whole pipeline
reports on. The worked example uses six: `data_analyst`, `data_engineer`,
`data_scientist`, `machine_learning_engineer`, `ai_engineer`,
`analytics_engineer`.

## Record schema

Each line is one job ad:

| field | type | required | notes |
|-------|------|----------|-------|
| `id` | string | yes | Unique within the file. Used for dedup and as the posting key. |
| `job_title` | string | yes | The advertised title. |
| `company` | string | no | Employer name. Blank allowed. |
| `description` | string | yes | Full ad body. This is what skills are extracted from. |
| `date` | string | no | ISO date the ad was posted/seen. Blank allowed. |
| `source_url` | string | no | Where the ad came from. |
| `location` | string | no | Free-text location. |

Minimal valid line:

```json
{"id": "abc123", "job_title": "Data Scientist", "company": "Acme", "description": "We are hiring a data scientist with Python, SQL and machine learning experience...", "date": "2026-03-01", "source_url": "", "location": "Remote, US"}
```

## What the pipeline expects of the sample

- **Descriptions carry the signal.** Skills are read from `description` by the
  extraction stage, so thin or templated descriptions yield thin graphs. The
  reference project drops ads with descriptions under ~200 characters and
  near-duplicate bodies before sampling.
- **One cell = one comparable sample.** Every figure (penetration, adjacency)
  is computed per `<year>/<occupation>` cell and never pooled across cells, so
  keep sample sizes per cell roughly comparable if you intend to compare cells.
- **No raw ad text leaves the pipeline.** Downstream outputs are concept-level
  (skill labels + counts), never ad bodies — see
  [`graph_data_contract.md`](graph_data_contract.md).

A synthetic example lives at
[`../sample/extracted/2026/data_scientist.jsonl`](../sample/extracted/2026/data_scientist.jsonl).
It is fabricated, for shape only — not real job ads.
