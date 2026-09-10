"""Stage 4a — per-cell skill co-occurrence, for graph edges.

For every occupation-year cell, computes a co-occurrence statistic for each
concept pair that appears together in an ad, and stores it with two pass/keep
flags so downstream steps can select the pairs they need:
  passes_graph    — used to build the 2026 knowledge graph edges.
  passes_analysis — a stricter selection used by other analysis.

Thresholds for both flags are the module constants below. Held concepts (no
type) and boilerplate concepts are excluded — they carry no analytical meaning.

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
GRAPH_LIFT = 1.3    # graph keep threshold (de-clutters common-but-unrelated pairs)
HUB_SUPPORT = 0.60  # a concept in >= this share of ads gets the relaxed threshold
HUB_LIFT = 1.05     # relaxed threshold a hub pair must still clear


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
