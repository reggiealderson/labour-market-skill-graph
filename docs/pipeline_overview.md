# Pipeline overview

How a folder of job advertisements becomes the skill knowledge graph. Each stage
is deterministic and versioned; where a language model is used, its output is
recorded as model-made and human-reviewed.

## This is a data-science / AI domain pipeline

The method was designed **for and around data-science, analytics, and AI job
ads** — it is not a domain-neutral tool with the data-science parts bolted on.
That grounding shows up in the parts you would change for another field:

- **Occupations.** The worked example is six data/AI roles (data analyst, data
  engineer, data scientist, machine learning engineer, AI engineer, analytics
  engineer). These are hard-coded in the stage runners because the whole project
  compares *these* roles.
- **Types** (Stage D). The nine concept types (language, library, tool, method,
  knowledge_area, practice, transversal_skill, compliance, industry_context)
  were chosen because they are the meaningful distinctions **between data/AI
  skills** — separating, say, a *library* (pandas) from a *method* (regression)
  from a *tool* (Tableau).
- **Categories** (Stage E). The 27-category taxonomy in
  `taxonomy/categories.yaml` is hand-authored to carve up the **data/AI skill
  space** (cloud platforms, ML frameworks, data modelling, and so on). It is a
  worked example, not a universal ontology.
- **Facets** (Stage G). All four facets are explicitly an **AI lens** —
  AI-relatedness, AI wave, AI workflow layer, and AI vendor ecosystem. They only
  make sense for this domain and encode judgement calls about what counts as
  "core AI" versus "AI-adjacent" in data-science hiring.

To apply the pipeline elsewhere, you replace this domain content (occupations,
taxonomy, facets) — the *machinery* around it is general. See
[`RUN.md`](RUN.md), "Adapting to your own data".

## Corpus (the worked example)

- ~3,200 job ads across the six data/AI occupations above.
- Two snapshots: 2021 (the three roles that existed in volume then) and 2026 (all
  six). The published graph uses 2026.
- Every figure is computed **per occupation-year cell** — never pooled across
  roles.

## Stages

**A — Normalisation.** Raw skill surface forms from the ads are normalised
(casing, punctuation, spelling variants). Rules live in versioned config, not
code.

**B — Blocking.** Candidate duplicate forms are found with k-nearest-neighbours
over local sentence embeddings, so the merge step only compares plausible pairs
rather than all pairs.

**C — Semantic merge.** A language model judges which blocked forms name the same
skill, producing stable *concepts*. Concept ids are derived from a label hash and
are stable: new ads map into existing ids rather than renumbering.

**D — Typing.** Each concept gets one type from the fixed data/AI set of nine
(above). Concepts without enough evidence are *held* (left untyped) and excluded
from later meaning-bearing steps; their share of mentions is reported as a
coverage figure.

**E — Categories.** The frozen, versioned data/AI taxonomy (id, label,
definition, includes/excludes, role). Validated by a schema checker with fixture
tests.

**F — Assignment.** Each concept is assigned a primary category and, where it
spans functions, secondary categories. Runs multiple times at temperature 0; a
label is kept only if it recurs. Mention counts, occupation, and year are
withheld from the request so assignment cannot be biased by prevalence.

**G — Facets.** The four hand-defined AI facets are applied over concepts,
classified by a language model at temperature 0 in batches, every row carrying
provenance (`model`, `by=model`), then reviewed.

**Analytics — penetration.** For each cell, penetration = the share of ads that
mention a concept. This is the node's prevalence in the graph.

**Analytics — adjacency (the edges).** For each pair of concepts that co-occur in
an ad within a cell, a co-occurrence statistic is computed and stored with
keep-flags. Held and boilerplate concepts are excluded. The computation lives in
`pipeline/analytics/compute_adjacency.py`.

**Export — the graph.** `pipeline/analytics/export_graph.py` assembles one JSON
per role:
- **Nodes** = every concept in ≥ max(3, 3% of ads) for that role (a population in
  its own right; isolated nodes are kept).
- **Edges** = co-occurring pairs that clear the keep thresholds in
  `compute_adjacency.py`.

The JSON schema is fixed by [`../schema/graph_data_contract.md`](../schema/graph_data_contract.md).

## Where the model is used, and how it is controlled

Several stages call a language model: skill extraction, semantic merge (C),
typing (D), category assignment (F), and facet classification (G). In every case
the model runs at temperature 0, output is recorded as model-made, and a human
reviews it before use. Prevalence signals (mention counts, occupation, year) are
withheld from assignment requests so the model cannot lean on how common a skill
is.

## From data to a viewer

1. `pipeline/analytics/compute_adjacency.py` → edges + node population.
2. `pipeline/analytics/export_graph.py` → one graph JSON per role + an index.
3. A viewer (for example a force-directed web app) renders those JSON files.

The JSON files are the only contract between the pipeline and the viewer, which
keeps the two independent.
