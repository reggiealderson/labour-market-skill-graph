# Skill adjacency (co-occurrence) method

How two skills are judged "related", and how the number behind each graph edge
and each replacement candidate is computed. Used by
`pipeline/analytics/compute_adjacency.py` and `compute_replacement.py`.

## Idea

The method comes from association-rule mining — the family of techniques behind
the classic a-priori / frequent-itemset algorithm. Two skills are related when
ads that mention one tend also to mention the other, in both directions. The
"in both directions" part matters: a rare skill can sit almost always beside a
common one without the reverse being true, and that is not real adjacency.

Everything is computed **per occupation-year cell** — never pooled across
occupations.

## Metrics (for a pair of skills A, B, over the ads in one cell)

- **count(A)** — ads that mention A. **N** — ads in the cell.
- **support(A)** = count(A) / N — the chance a random ad mentions A. (This is
  just A's penetration.)
- **support(A, B)** = count(A and B) / N — the chance an ad mentions both.
- **confidence(A→B)** = support(A,B) / support(A) — of the ads with A, the share
  that also have B. Directional: confidence(A→B) ≠ confidence(B→A).
- **lift(A, B)** = support(A,B) / (support(A) × support(B)) — how much more often
  the pair occurs together than if the two were independent. lift > 1 = they
  attract; = 1 = independent; < 1 = they repel. Lift is what separates a real
  association from "both skills are simply common".
- **edge weight** = geometric mean of confidence(A→B) and confidence(B→A) — one
  symmetric 0–1 number, high only when the pair implies each other *both* ways.

## Filters and floors

A pair is kept only if it clears a floor. Two floors exist because adjacency has
two uses with different needs:

- **Graph floor** (2026 only, for the knowledge graph): joint count ≥
  max(3, 3% of the cell's ads) **and (lift > 1.3, or the pair involves a hub
  skill — one in ≥ 60% of ads — with lift > 1.05)**. The 1.3 lift threshold
  (raised from 1.0) removes edges that exist only because both skills are common,
  which otherwise turn a dense role's graph into a hairball. The hub clause
  corrects a side effect of that: lift has a ceiling of 1 / support, so a
  near-universal skill (e.g. Python in 84% of data-scientist ads has a ceiling of
  1.19) can never reach 1.3, and the most defining skill of a role was being
  excluded. The clause relaxes only the lift bar, and only for hub skills.
- **Analysis floor** (both years, for the replacement analysis): joint count ≥
  max(5, 5% of ads), each skill in ≥ 5 ads, lift > 1. Applied identically to
  2021 and 2026 so the two years are comparable despite very different sample
  sizes.

Held concepts and boilerplate concepts (primary category
`cat_generic_field_reference`) are excluded from adjacency entirely — they carry
no analytical meaning.

Edges are stored once per unordered pair, sorted by concept id, with both
confidences retained so either direction can be read back.

## Note on interpretation

Adjacency is co-occurrence within cross-sectional ad samples. It shows which
skills are demanded together, not that one skill causes another to be demanded.
The replacement analysis builds on it but stays hedged for exactly this reason —
see the replacement section of `docs/ANALYTICS_PLAN.md`.
