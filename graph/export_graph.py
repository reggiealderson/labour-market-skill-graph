"""Stage 5a — export one knowledge-graph JSON per 2026 occupation.

Nodes = the graph_nodes.csv population for that occupation (every concept in
>= max(3, 3% of ads); a node is a first-class object, not derived from edges).
Edges = graph-floor adjacency pairs (passes_graph=1) among those nodes. A node
with no passing edge is kept as a legitimate isolated node (degree 0).

Node attributes:
  id, label, category (primary_category), type (Stage D type),
  relatedness (facet_ai_relatedness value), wave/workflow/vendor (lists),
  penetration (2026 rate in that occupation), degree.
Edge attributes:
  source, target, weight (geo_mean_conf), lift.

Output: data/analytics/graph/<occupation>_2026.json  (+ index.json)
"""
import csv, os, json, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "data/analytics")
OUT = os.path.join(A, "graph")
os.makedirs(OUT, exist_ok=True)


def load_map(path, key, val):
    m = {}
    with open(os.path.join(ROOT, path)) as f:
        for r in csv.DictReader(f):
            m[r[key]] = (r.get(val) or "").strip()
    return m


def load_facets():
    rel = {}
    multi = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(os.path.join(ROOT, "taxonomy/concept_facet.csv")) as f:
        for r in csv.DictReader(f):
            if r["value"] == "none":
                continue
            if r["facet_id"] == "facet_ai_relatedness":
                rel[r["concept_id"]] = r["value"]
            else:
                short = r["facet_id"].replace("facet_ai_", "").replace("facet_", "")
                multi[r["concept_id"]][short].append(r["value"])
    return rel, multi


def main():
    ctype = load_map("data/concepts/concept_type.csv", "concept_id", "type")
    ccat = load_map("data/concepts/concept_category.csv", "concept_id", "primary_category")
    rel, multi = load_facets()

    # penetration: (occ, concept_id) -> (rate, hits, n_ads)  (2026 concept grain)
    pen = {}
    with open(os.path.join(A, "penetration.csv")) as f:
        for r in csv.DictReader(f):
            if r["year"] == "2026" and r["grain"] == "concept":
                pen[(r["occupation"], r["key"])] = (
                    float(r["rate"]), int(r["hits"]), int(r["n_ads"]))

    # node population per occupation (2026), independent of edges
    node_pop = collections.defaultdict(dict)   # occ -> {concept_id: label}
    with open(os.path.join(A, "graph_nodes.csv")) as f:
        for r in csv.DictReader(f):
            node_pop[r["occupation"]][r["concept_id"]] = r["label"]

    # edges per occupation (2026, graph floor)
    edges = collections.defaultdict(list)
    labels = {}
    with open(os.path.join(A, "adjacency.csv")) as f:
        for r in csv.DictReader(f):
            if r["year"] != "2026" or r["passes_graph"] != "1":
                continue
            edges[r["occupation"]].append(r)
            labels[r["concept_a"]] = r["label_a"]
            labels[r["concept_b"]] = r["label_b"]

    index = []
    for occ in sorted(node_pop):
        erows = edges.get(occ, [])
        node_ids = set(node_pop[occ])
        labels.update(node_pop[occ])
        degree = collections.Counter()
        for r in erows:
            # an edge should never reference a concept outside the node
            # population (both share the same 3% floor); guard anyway.
            node_ids.add(r["concept_a"]); node_ids.add(r["concept_b"])
            degree[r["concept_a"]] += 1; degree[r["concept_b"]] += 1

        nodes = []
        for cid in sorted(node_ids):
            rate, hits, nads = pen.get((occ, cid), (0.0, 0, 0))
            nodes.append({
                "id": cid, "label": labels[cid],
                "category": ccat.get(cid, ""), "type": ctype.get(cid, ""),
                "relatedness": rel.get(cid, "ai_independent"),
                "wave": multi[cid].get("wave", []),
                "workflow": multi[cid].get("workflow", []),
                "vendor": multi[cid].get("vendor_ecosystem", []),
                "penetration": round(rate, 4), "hits": hits, "n_ads": nads,
                "degree": degree[cid],
            })
        out_edges = [{
            "source": r["concept_a"], "target": r["concept_b"],
            "weight": float(r["geo_mean_conf"]), "lift": float(r["lift"]),
        } for r in erows]

        payload = {"occupation": occ, "year": "2026",
                   "nodes": nodes, "edges": out_edges}
        with open(os.path.join(OUT, f"{occ}_2026.json"), "w") as f:
            json.dump(payload, f)
        index.append({"occupation": occ, "nodes": len(nodes), "edges": len(out_edges)})

    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump({"year": "2026", "occupations": index}, f, indent=2)

    print(f"{'occupation':30s} {'nodes':>6s} {'edges':>6s}")
    for row in index:
        print(f"{row['occupation']:30s} {row['nodes']:6d} {row['edges']:6d}")


if __name__ == "__main__":
    main()
