"""Stage 4a — skill adjacency (association-rule co-occurrence) per cell.

For every occupation-year cell, for every concept pair that co-occurs in the
same ad, compute:
  count_a, count_b, count_ab, support_a, support_b, support_ab,
  conf_ab, conf_ba, lift, geo_mean_conf (edge weight).

Concepts excluded: held (no type) and boilerplate (primary_category =
cat_generic_field_reference) — no meaning for a graph.

Floors (flags, not filters — both are stored so either use can select):
  passes_graph    : year=2026 AND count_ab >= max(3, 3% of ads) AND
                    ( lift > 1.3
                      OR (a hub skill — support >= 60% — with lift > 1.05) )
  passes_analysis : count_ab >= max(5, 5% of ads) AND count_a>=5 AND count_b>=5
                    AND lift > 1   (both years; used by replacement analysis)

The hub clause exists because lift has a ceiling of 1/support: a near-universal
skill (e.g. Python in 84% of data-scientist ads) cannot reach lift 1.3 no matter
how tightly it pairs. Without the clause the most defining skill of a role is
structurally excluded. The clause relaxes only the lift bar, and only for skills
in >= 60% of ads, so ordinary pairs stay de-cluttered.

Only pairs that pass at least one floor are written.

Nodes are a population in their own right, not derived from edges: a concept is a
graph node for a 2026 cell if it appears in >= max(3, 3% of ads) there. A node
with no passing edge is a legitimate isolated node. Emitted to graph_nodes.csv.

Output: data/analytics/adjacency.csv, data/analytics/graph_nodes.csv
"""
import csv, os, math, collections
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "data/analytics")
BOILERPLATE = "cat_generic_field_reference"
GRAPH_LIFT = 1.3    # graph floor lift (raised from 1.0 to de-clutter the graph)
HUB_SUPPORT = 0.60  # a skill in >= this share of ads is a hub (lift-ceiling relief)
HUB_LIFT = 1.05     # relaxed lift bar a hub edge must still clear


def main():
    n_ads = collections.Counter()
    with open(os.path.join(A, "posting.csv")) as f:
        for r in csv.DictReader(f):
            n_ads[(r["year"], r["occupation"])] += 1

    # per cell: posting_id -> set(concept_id); and concept labels
    cell_ads = collections.defaultdict(lambda: collections.defaultdict(set))
    labels = {}
    with open(os.path.join(A, "posting_skill.csv")) as f:
        for r in csv.DictReader(f):
            if r["held"] == "1" or r["primary_category"] == BOILERPLATE:
                continue
            cell = (r["year"], r["occupation"])
            cell_ads[cell][r["posting_id"]].add(r["concept_id"])
            labels[r["concept_id"]] = r["concept_label"]

    rows = []
    node_rows = []
    report = []
    for cell, ads in cell_ads.items():
        year, occ = cell
        N = n_ads[cell]
        g_min = max(3, math.ceil(0.03 * N))
        a_min = max(5, math.ceil(0.05 * N))

        single = collections.Counter()   # concept -> ad count
        joint = collections.Counter()     # (a,b) sorted -> ad count
        for cset in ads.values():
            for c in cset:
                single[c] += 1
            for a, b in combinations(sorted(cset), 2):
                joint[(a, b)] += 1

        # node population (2026 only): concepts at or above the same 3% floor,
        # independent of whether they form any passing edge.
        if year == "2026":
            for c, cnt in single.items():
                if cnt >= g_min:
                    node_rows.append({
                        "year": year, "occupation": occ,
                        "concept_id": c, "label": labels[c],
                        "count": cnt, "n_ads": N,
                        "support": round(cnt / N, 6),
                    })

        kept = 0
        for (a, b), cab in joint.items():
            ca, cb = single[a], single[b]
            lift = (cab * N) / (ca * cb)
            hub = (max(ca, cb) / N >= HUB_SUPPORT and lift > HUB_LIFT)
            passes_graph = (year == "2026" and cab >= g_min
                            and (lift > GRAPH_LIFT or hub))
            passes_analysis = (cab >= a_min and ca >= 5 and cb >= 5 and lift > 1)
            if not (passes_graph or passes_analysis):
                continue
            conf_ab = cab / ca
            conf_ba = cab / cb
            rows.append({
                "year": year, "occupation": occ,
                "concept_a": a, "label_a": labels[a],
                "concept_b": b, "label_b": labels[b],
                "count_a": ca, "count_b": cb, "count_ab": cab, "n_ads": N,
                "support_a": round(ca / N, 6), "support_b": round(cb / N, 6),
                "support_ab": round(cab / N, 6),
                "conf_ab": round(conf_ab, 6), "conf_ba": round(conf_ba, 6),
                "lift": round(lift, 4),
                "geo_mean_conf": round(math.sqrt(conf_ab * conf_ba), 6),
                "passes_graph": "1" if passes_graph else "0",
                "passes_analysis": "1" if passes_analysis else "0",
            })
            kept += 1
        report.append((f"{year}/{occ}", len(single), len(joint), kept, g_min, a_min))

    with open(os.path.join(A, "adjacency.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(A, "graph_nodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(node_rows[0].keys()))
        w.writeheader(); w.writerows(node_rows)

    print(f"adjacency rows (pass >=1 floor): {len(rows)}")
    print(f"graph node rows (2026, >=3% floor): {len(node_rows)}")
    print(f"{'cell':32s} {'concepts':>8s} {'pairs':>8s} {'kept':>6s} {'gmin':>5s} {'amin':>5s}")
    for cell, nc, npair, kept, gm, am in report:
        print(f"{cell:32s} {nc:8d} {npair:8d} {kept:6d} {gm:5d} {am:5d}")


if __name__ == "__main__":
    main()
