"""Stage 4b — replacement CANDIDATES for the 3 shared occupations.

Hedged 3-part rule (all within one occupation, 2021->2026):
  1. X penetration falls, significant   (from change_test, grain=concept)
  2. Y penetration rises, significant
  3. Y takes X's slot at a shared anchor C:
       edge(X,C) existed in 2021 and edge(Y,C) exists in 2026, with
       adj(X,C) falling and adj(Y,C) rising.

This is a candidate generator for HAND REVIEW. It never asserts causation.
Noise floors: X's 2021 rate >= 0.10 and Y's 2026 rate >= 0.10.

Inputs : data/analytics/change_test.csv, data/analytics/adjacency.csv
Output : data/analytics/replacement_candidates.csv
"""
import csv, os, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "data/analytics")
SHARED = ["data_analyst", "data_engineer", "data_scientist"]
RATE_FLOOR = 0.10


def main():
    # change_test concept grain -> per occ: down[], up[], and per-concept stats
    down = collections.defaultdict(list)
    up = collections.defaultdict(list)
    stat = {}  # (occ, concept_id) -> row
    label = {}
    with open(os.path.join(A, "change_test.csv")) as f:
        for r in csv.DictReader(f):
            if r["grain"] != "concept":
                continue
            occ, cid = r["occupation"], r["key"]
            label[cid] = r["label"]
            stat[(occ, cid)] = r
            if r["significant"] == "1":
                if r["direction"] == "down" and float(r["rate_2021"]) >= RATE_FLOOR:
                    down[occ].append(cid)
                elif r["direction"] == "up" and float(r["rate_2026"]) >= RATE_FLOOR:
                    up[occ].append(cid)

    # adjacency (analysis floor) -> edge weight per (occ, year, frozenset(pair))
    #  and neighbour index per (occ, year, concept) -> set(neighbours)
    edge = {}
    nbr = collections.defaultdict(lambda: collections.defaultdict(set))
    with open(os.path.join(A, "adjacency.csv")) as f:
        for r in csv.DictReader(f):
            if r["passes_analysis"] != "1":
                continue
            occ, yr = r["occupation"], r["year"]
            a, b = r["concept_a"], r["concept_b"]
            w = float(r["geo_mean_conf"])
            edge[(occ, yr, frozenset((a, b)))] = w
            nbr[(occ, yr)][a].add(b)
            nbr[(occ, yr)][b].add(a)
            label.setdefault(a, r["label_a"]); label.setdefault(b, r["label_b"])

    def w(occ, yr, x, c):
        return edge.get((occ, yr, frozenset((x, c))), 0.0)

    rows = []
    for occ in SHARED:
        for x in down[occ]:
            anchors_x = nbr[(occ, "2021")].get(x, set())   # C adjacent to X in 2021
            for y in up[occ]:
                if y == x:
                    continue
                anchors_y = nbr[(occ, "2026")].get(y, set())  # C adjacent to Y in 2026
                for c in (anchors_x & anchors_y):
                    if c in (x, y):
                        continue
                    axc21, axc26 = w(occ, "2021", x, c), w(occ, "2026", x, c)
                    ayc21, ayc26 = w(occ, "2021", y, c), w(occ, "2026", y, c)
                    if not (axc26 < axc21 and ayc26 > ayc21):
                        continue
                    xs, ys = stat[(occ, x)], stat[(occ, y)]
                    x_drop = float(xs["rate_2021"]) - float(xs["rate_2026"])
                    y_rise = float(ys["rate_2026"]) - float(ys["rate_2021"])
                    score = x_drop + y_rise + (ayc26 - ayc21) + (axc21 - axc26)
                    rows.append({
                        "occupation": occ,
                        "x_id": x, "x_label": label[x],
                        "y_id": y, "y_label": label[y],
                        "anchor_id": c, "anchor_label": label.get(c, c),
                        "x_rate_2021": xs["rate_2021"], "x_rate_2026": xs["rate_2026"],
                        "x_diff": xs["diff"], "x_p": xs["p_value"],
                        "y_rate_2021": ys["rate_2021"], "y_rate_2026": ys["rate_2026"],
                        "y_diff": ys["diff"], "y_p": ys["p_value"],
                        "adj_xC_2021": round(axc21, 4), "adj_xC_2026": round(axc26, 4),
                        "adj_yC_2021": round(ayc21, 4), "adj_yC_2026": round(ayc26, 4),
                        "score": round(score, 4),
                    })

    # collapse to one row per (occupation, X, Y); anchors become evidence count
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["occupation"], r["x_id"], r["y_id"])].append(r)

    out = []
    for (occ, x, y), items in groups.items():
        items.sort(key=lambda r: r["score"], reverse=True)
        base = items[0]
        top_anchors = "; ".join(dict.fromkeys(i["anchor_label"] for i in items))  # unique, order kept
        top_anchors = "; ".join(top_anchors.split("; ")[:3])
        out.append({
            "occupation": occ,
            "x_label": base["x_label"], "y_label": base["y_label"],
            "x_rate_2021": base["x_rate_2021"], "x_rate_2026": base["x_rate_2026"],
            "x_diff": base["x_diff"], "x_p": base["x_p"],
            "y_rate_2021": base["y_rate_2021"], "y_rate_2026": base["y_rate_2026"],
            "y_diff": base["y_diff"], "y_p": base["y_p"],
            "n_anchors": len(items), "top_anchors": top_anchors,
            "best_score": base["score"],
            "x_id": x, "y_id": y,
        })

    out.sort(key=lambda r: (r["n_anchors"], r["best_score"]), reverse=True)
    with open(os.path.join(A, "replacement_candidates.csv"), "w", newline="") as f:
        w2 = csv.DictWriter(f, fieldnames=list(out[0].keys()) if out else ["occupation"])
        w2.writeheader(); w2.writerows(out)

    print(f"replacement candidates (collapsed): {len(out)}  (from {len(rows)} anchor rows)")
    by = collections.Counter(r["occupation"] for r in out)
    print("by occupation:", dict(by))
    print("\ntop 12 (occupation | X down -> Y up | #anchors | X drop / Y rise):")
    for r in out[:12]:
        xd = f"{float(r['x_rate_2021']):.2f}->{float(r['x_rate_2026']):.2f}"
        yr = f"{float(r['y_rate_2021']):.2f}->{float(r['y_rate_2026']):.2f}"
        print(f"  {r['occupation']:14s} {r['x_label']:20s} -> {r['y_label']:22s} "
              f"n={r['n_anchors']:2d}  X {xd}  Y {yr}")


if __name__ == "__main__":
    main()
