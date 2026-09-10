# Skill knowledge graph — JSON data contract

This is the contract between the pipeline and any viewer. The pipeline produces
these files; a graph viewer (for example a force-directed web app) consumes
them. As long as this schema holds, the two stay decoupled.

Producer: `pipeline/analytics/export_graph.py`.

## Files (the 7-file contract)

- `index.json` — one manifest.
- `<occupation>_2026.json` — one per occupation listed in the index (6 today).

All live in `data/analytics/graph/`. A viewer reads them verbatim — the JSON is
the whole contract.

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
| `weight` | float | adjacency stat | Symmetric co-occurrence strength, 0–1. Higher = the two skills are demanded together more. |
| `lift` | float | adjacency stat | Ratio of observed to expected co-occurrence. `> 1` = the pair co-occurs more than chance. |

The exact statistics behind `weight` and `lift` are computed in
`pipeline/analytics/compute_adjacency.py`; the viewer only needs their meaning
above.

## Node population vs edges (important)

Nodes are a **population in their own right**, not derived from edges. A concept
is a node for a role if it appears in `>= max(3, 3% of ads)` in that 2026 cell.
An edge is added only if the pair clears the graph keep thresholds in
`compute_adjacency.py`. One consequence the viewer must handle:

- **Isolated nodes exist** (`degree == 0`). A prevalent skill with no qualifying
  edge is still a legitimate node. Do not drop degree-0 nodes.

## Producing and publishing

```
python3 pipeline/analytics/compute_adjacency.py   # -> adjacency.csv, graph_nodes.csv
python3 pipeline/analytics/export_graph.py         # -> data/analytics/graph/*.json
```

The resulting JSON files are what a viewer renders.

## Changing the schema

Any change to node/edge fields is a breaking change for every viewer. When you
add or rename a field:

1. Update `export_graph.py` (producer).
2. Update this contract.
3. Update your viewer's types + rendering.
4. Regenerate the sample JSON under `sample/graph/`.
