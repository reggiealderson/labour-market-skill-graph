# Labour-market skill knowledge graph

A knowledge graph of the skills demanded in data and AI job roles, built from job
advertisements. For each role, the graph shows which skills are demanded, how
common each is, and which skills are demanded *together*.

The interactive graph is published at
**[reggiealderson.com/skill-graph](https://reggiealderson.com/skill-graph)**.

This repository is a **public guide to how that graph is built** — the method,
the data contract, and the graph-building code. The full pipeline and the raw job
ads live in a separate private repository; nothing here contains raw ad text.

## What the graph shows

- **Nodes** — skills (called *concepts*) demanded in a role. Node size/among
  attributes carry the skill's *penetration* (share of that role's ads that
  mention it), its type (language, library, tool, method, …), and AI-relatedness
  facets.
- **Edges** — pairs of skills that are demanded together more than chance would
  predict. Edge weight is a symmetric co-occurrence confidence; each edge also
  carries a *lift* value.

Six roles are covered: data analyst, data engineer, data scientist, machine
learning engineer, AI engineer, analytics engineer.

## How it is built (short version)

Job ads → normalised skill mentions → merged into stable concepts → typed and
categorised → AI facets → per-role penetration and co-occurrence → graph JSON.

Full account: [`docs/pipeline_overview.md`](docs/pipeline_overview.md).
The co-occurrence method (support, confidence, lift): [`docs/adjacency_method.md`](docs/adjacency_method.md).

## What is in this repo

| path | what |
|------|------|
| `docs/pipeline_overview.md` | The stages, end to end. |
| `docs/adjacency_method.md` | How two skills are judged "related" (association-rule mining). |
| `schema/graph_data_contract.md` | The exact JSON schema the graph viewer consumes. |
| `graph/compute_adjacency.py` | Builds the co-occurrence edges + node population. |
| `graph/export_graph.py` | Emits one graph JSON per role. |
| `sample/data_scientist_2026.json` | One real graph file, to read against the schema. |
| `sample/index.json` | The manifest that lists every role graph. |

The two scripts in `graph/` are copied unmodified from the private pipeline.
They are here to read, not to run — they expect the private repo's data files.

## A note on the method

Two design choices are worth calling out, both documented in full in the method
doc:

1. **Nodes are a population, not a by-product of edges.** A skill that is common
   in a role but does not co-occur tightly with any other skill is still a node
   (an isolated node). Dropping it would hide real demand.
2. **The hub clause.** The edge metric *lift* has a mathematical ceiling of
   `1 / penetration`. A near-universal skill — Python appears in 84% of
   data-scientist ads — cannot reach the normal lift threshold however tightly it
   pairs, so the most defining skill of a role was being excluded from the graph.
   A targeted clause relaxes the lift bar for such hub skills only.

## Interpretation and limits

The graph is built from cross-sectional samples of job ads. It shows which skills
are *demanded together*, not that one skill causes demand for another. Penetration
is ad-level incidence within a role-year sample. Figures are a snapshot, not a
census of the labour market.

## Licence / use

Documentation and code are shared for transparency about how the published graph
is produced. See the site for the graph itself.
