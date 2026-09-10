# Pipeline overview

How a folder of job advertisements becomes the skill knowledge graph. Each stage
is deterministic and versioned; where a language model is used, its output is
recorded as model-made and human-reviewed.

## Corpus

- ~3,200 job ads across 6 occupations: data analyst, data engineer, data
  scientist, machine learning engineer, AI engineer, analytics engineer.
- Two snapshots: 2021 (first three roles) and 2026 (all six). The published
  graph uses 2026.
- Every figure is computed **per occupation-year cell** — never pooled across
  roles.

## Stages

**A — Normalisation.** Raw skill surface forms from the ads are normalised
(casing, punctuation, spelling variants). ~15,000 surface forms → ~14,800
distinct normalised forms. Rules live in versioned config, not code.

**B — Blocking.** Candidate duplicates are found with k-nearest-neighbours over
sentence embeddings (cosine threshold 0.88), so the merge step only compares
plausible pairs rather than all pairs.

**C — Semantic merge.** A language model judges which blocked forms name the same
skill, producing ~11,300 stable *concepts*. Concept ids are derived from a label
hash and are stable: new ads map into existing ids rather than renumbering.

**D — Typing.** Each concept gets one type from a fixed set of nine (language,
library, tool, method, knowledge_area, practice, transversal_skill, compliance,
industry_context). Concepts without enough evidence are *held* (left untyped) and
excluded from later meaning-bearing steps; their share of mentions is reported as
a coverage figure.

**E — Categories.** A frozen, versioned taxonomy of 27 categories (id, label,
definition, includes/excludes, role). Validated by a schema checker with fixture
tests.

**F — Assignment.** Each concept is assigned a primary category and, where it
spans functions, secondary categories. Runs multiple times at temperature 0; a
label is kept only if it recurs. Mention counts, occupation, and year are
withheld from the request so assignment cannot be biased by prevalence.

**G — Facets.** Four hand-defined facets add an AI lens over concepts:
AI-relatedness (core_ai / ai_adjacent / ai_independent), AI wave, AI workflow
layer, and vendor ecosystem. Classified by a language model at temperature 0 in
batches, every row carrying provenance (`model`, `by=model`), then reviewed.

**Analytics — penetration.** For each cell, penetration = the share of ads that
mention a concept. This is the node's prevalence in the graph.

**Analytics — adjacency (the edges).** For every pair of concepts that co-occur
in an ad, association-rule metrics are computed: support, confidence (both
directions), and lift. See [`adjacency_method.md`](adjacency_method.md). Held and
boilerplate concepts are excluded.

**Export — the graph.** `graph/export_graph.py` assembles one JSON per role:
- **Nodes** = every concept in ≥ max(3, 3% of ads) for that role (a population in
  its own right; isolated nodes are kept).
- **Edges** = co-occurring pairs that clear the graph floor (lift > 1.3, or a hub
  skill in ≥ 60% of ads with lift > 1.05).

The JSON schema is fixed by [`../schema/graph_data_contract.md`](../schema/graph_data_contract.md).

## Where the model is used, and how it is controlled

Three stages call a language model: semantic merge (C), category assignment (F),
and facet classification (G). In every case the model runs at temperature 0,
output is recorded as model-made, and a human reviews it before use. Prevalence
signals (mention counts, occupation, year) are withheld from assignment requests
so the model cannot lean on how common a skill is.

## From data to the published site

1. `graph/compute_adjacency.py` → edges + node population.
2. `graph/export_graph.py` → one graph JSON per role + an index.
3. A publish step copies those JSON files into the portfolio website (a
   Next.js + Sigma.js viewer) and rebuilds it.

The JSON files are the only contract between the pipeline and the viewer, which
keeps the two independent.
