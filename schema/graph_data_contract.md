# Skill knowledge graph — JSON data contract

This is the seam between the repos. The **labour-market-analytics** repo
produces these files; the **portfolio** repo (Next.js + Sigma.js viewer) and the
public **labour-market-skill-graph** docs repo consume them. As long as this
schema holds, the three repos stay decoupled.

Producer: `pipeline/analytics/export_graph.py`.
Consumer types: `skill-graph-src/components/GraphView.tsx` (`NodeData`,
`EdgeData`, `GraphPayload`, `IndexRow`).

## Files (the 7-file contract)

- `index.json` — one manifest.
- `<occupation>_2026.json` — one per occupation listed in the index (6 today).

All live in `data/analytics/graph/` here, and are copied verbatim to
`skill-graph-src/public/data/` in the portfolio by `scripts/publish_graph.py`.

## index.json

```json
{
  "year": "2026",
  "occupations": [
    { "occupation": "data_scientist", "nodes": 113, "edges": 367 }
  ]
}
```

| field | type | meaning |
|-------|------|---------|
| `year` | string | Snapshot year. Currently `"2026"`. |
| `occupations[]` | array | One row per graph file. |
| `.occupation` | string | Slug; the file is `<occupation>_2026.json`. |
| `.nodes` / `.edges` | int | Counts, for a manifest/overview. |

## `<occupation>_2026.json`

```json
{
  "occupation": "data_scientist",
  "year": "2026",
  "nodes": [ /* NodeData */ ],
  "edges": [ /* EdgeData */ ]
}
```

### Node

| field | type | source | notes |
|-------|------|--------|-------|
| `id` | string | concept_id | Stable key; edges reference it. |
| `label` | string | concept_label | Display name (e.g. `"Python"`). |
| `category` | string | `concept_category.csv` primary_category | May be `""`. |
| `type` | string | Stage D `concept_type.csv` | e.g. tool, library, method. May be `""`. |
| `relatedness` | enum | facet_ai_relatedness | `core_ai` \| `ai_adjacent` \| `ai_independent`. Defaults to `ai_independent` if unfaceted. |
| `wave` | string[] | facet_ai_wave | May be empty. |
| `workflow` | string[] | facet_ai_workflow | May be empty. |
| `vendor` | string[] | facet_vendor_ecosystem | May be empty. |
| `penetration` | float | penetration.csv (2026, concept grain) | Share of that occupation's ads, 0–1. |
| `hits` | int | penetration.csv | Ads containing the concept. |
| `n_ads` | int | penetration.csv | Total ads in the cell (500 for most roles). |
| `degree` | int | computed | Number of incident edges. **May be 0** — see isolated nodes. |

### Edge

| field | type | source | notes |
|-------|------|--------|-------|
| `source` / `target` | string | concept_id | Reference node `id`s. Undirected pair. |
| `weight` | float | adjacency `geo_mean_conf` | Geometric mean of the two confidences. Edge strength. |
| `lift` | float | adjacency `lift` | Co-occurrence vs chance. `> 1` = attract. |

## Node population vs edges (important)

Nodes are a **population in their own right**, not derived from edges. A concept
is a node for a role if it appears in `>= max(3, 3% of ads)` in that 2026 cell.
An edge is added only if the pair passes the graph floor:

```
count_ab >= max(3, 3% of ads)  AND
  ( lift > 1.3
    OR  (a hub skill — support >= 60% — with lift > 1.05) )
```

Consequences the viewer must handle:

1. **Isolated nodes exist** (`degree == 0`). A prevalent skill with no
   qualifying edge is still a legitimate node. Do not drop degree-0 nodes.
2. **The hub clause.** `lift` has a ceiling of `1 / support`, so a near-universal
   skill (Python at 84% of data-scientist ads) cannot reach `lift 1.3` however
   tightly it pairs. The hub clause relaxes only the lift bar, and only for
   skills in `>= 60%` of ads, so the most defining skill of a role is not
   structurally excluded. See `pipeline/analytics/compute_adjacency.py`.

## Producing and publishing

```
python3 pipeline/analytics/compute_adjacency.py   # -> adjacency.csv, graph_nodes.csv
python3 pipeline/analytics/export_graph.py         # -> data/analytics/graph/*.json
python3 scripts/publish_graph.py                   # copy to portfolio + rebuild viewer
```

`publish_graph.py` does not commit or deploy — it stages the portfolio tree for
review.

## Changing the schema

Any change to node/edge fields is a breaking change to two downstream repos.
When you add or rename a field:

1. Update `export_graph.py` (producer).
2. Update this contract.
3. Update `GraphView.tsx` types + rendering in the portfolio.
4. Update the sample JSON in the public docs repo.
