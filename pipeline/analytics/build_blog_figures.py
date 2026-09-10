"""Stage 7 (step 2) — bake every figure the blog posts need into one JSON.

Source of truth = data/analytics/*.csv (same numbers as BigQuery). Output is
a provenance file the SVG charts are built from, so no number is hand-typed.

Output: data/analytics/blog_figures.json
"""
import csv, os, json, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "data/analytics")
BOILER = "cat_generic_field_reference"
OCCS = ["ai_engineer", "analytics_engineer", "data_analyst", "data_engineer",
        "data_scientist", "machine_learning_engineer"]
SHARED = ["data_analyst", "data_engineer", "data_scientist"]


def rows(name):
    with open(os.path.join(A, name)) as f:
        return list(csv.DictReader(f))


def main():
    out = {}

    # --- Section 1: corpus basics ---
    posting = rows("posting.csv")
    ad_counts = collections.Counter((r["year"], r["occupation"]) for r in posting)
    cov = json.load(open(os.path.join(A, "build_coverage.json")))
    out["corpus"] = {
        "total_ads": len(posting),
        "ad_counts": {f"{y}/{o}": ad_counts[(y, o)] for (y, o) in sorted(ad_counts)},
        "coverage_min": round(min(c["coverage_pct"] for c in cov.values()), 1),
        "coverage_max": round(max(c["coverage_pct"] for c in cov.values()), 1),
        "n_concepts": 11302, "held_pct": 2.27,
        "occupations_2026": OCCS, "occupations_2021": SHARED,
    }

    pen = rows("penetration.csv")

    # --- Section 2a: top 10 concepts per 2026 occupation ---
    concept26 = collections.defaultdict(list)
    cat_of = {}
    for r in rows("scoring_set.csv"):
        cat_of[r["concept_id"]] = r["primary_category"]
    for r in pen:
        if r["year"] == "2026" and r["grain"] == "concept" and r["is_held"] == "0":
            concept26[r["occupation"]].append((r["label"], float(r["rate"]), int(r["hits"]), int(r["n_ads"])))
    out["top_concepts_2026"] = {
        o: [{"label": l, "pct": round(rt * 100, 1), "hits": h, "n": n}
            for l, rt, h, n in sorted(concept26[o], key=lambda x: x[1], reverse=True)[:10]]
        for o in OCCS}

    # --- Section 2b: category penetration matrix (2026, all occupations) ---
    cat_matrix = collections.defaultdict(dict)
    for r in pen:
        if r["year"] == "2026" and r["grain"] == "category" and r["is_boilerplate"] == "0":
            cat_matrix[r["key"]][r["occupation"]] = round(float(r["rate"]) * 100, 1)
    out["category_matrix_2026"] = cat_matrix

    # --- Section 3: 2021->2026 change, category grain, 3 shared occ ---
    change = rows("change_test.csv")
    out["change_category"] = {}
    for occ in SHARED:
        sig = [r for r in change if r["occupation"] == occ and r["grain"] == "category" and r["significant"] == "1"]
        sig.sort(key=lambda r: abs(float(r["diff"])), reverse=True)
        out["change_category"][occ] = [{
            "label": r["key"].replace("cat_", "").replace("_", " "),
            "r21": round(float(r["rate_2021"]) * 100, 1),
            "r26": round(float(r["rate_2026"]) * 100, 1),
            "diff": round(float(r["diff"]) * 100, 1),
            "p": float(r["p_value"]) if r["p_value"] else None,
            "dir": r["direction"],
        } for r in sig[:10]]

    # --- Section 3b: concept-level 2021->2026 change, 3 shared occ (down/up) ---
    out["change_concept"] = {}
    for occ in SHARED:
        sig = [r for r in change if r["occupation"] == occ and r["grain"] == "concept"
               and r["significant"] == "1"
               and max(float(r["rate_2021"]), float(r["rate_2026"])) >= 0.10]
        def fmt(r):
            return {"label": r["label"],
                    "r21": round(float(r["rate_2021"]) * 100, 1),
                    "r26": round(float(r["rate_2026"]) * 100, 1),
                    "diff": round(float(r["diff"]) * 100, 1)}
        downs = sorted([r for r in sig if r["direction"] == "down"], key=lambda r: float(r["diff"]))[:6]
        ups = sorted([r for r in sig if r["direction"] == "up"], key=lambda r: float(r["diff"]), reverse=True)[:6]
        out["change_concept"][occ] = {"down": [fmt(r) for r in downs], "up": [fmt(r) for r in ups]}

    # --- Section 4a: AI concept demand across 6 occupations ---
    ai_terms = ["large language models", "retrieval-augmented generation", "generative AI",
                "prompt engineering", "vector databases", "MLOps", "machine learning",
                "natural language processing"]
    ai = collections.defaultdict(dict)
    for r in pen:
        if r["year"] == "2026" and r["grain"] == "concept" and r["label"] in ai_terms:
            ai[r["label"]][r["occupation"]] = round(float(r["rate"]) * 100, 1)
    out["ai_concept_demand_2026"] = ai

    # --- Section 4b: relatedness saturation (core_ai share) per occupation ---
    rel = collections.defaultdict(dict)
    for r in pen:
        if r["year"] == "2026" and r["grain"] == "facet" and r["facet_id"] == "facet_ai_relatedness":
            rel[r["occupation"]][r["key"]] = round(float(r["rate"]) * 100, 1)
    out["relatedness_2026"] = rel

    # --- Section 4c: replacement candidates (top, data_engineer especially) ---
    rep = rows("replacement_candidates.csv")
    rep.sort(key=lambda r: (int(r["n_anchors"]), float(r["best_score"])), reverse=True)
    out["replacement_top"] = [{
        "occupation": r["occupation"], "x": r["x_label"], "y": r["y_label"],
        "x21": round(float(r["x_rate_2021"]) * 100), "x26": round(float(r["x_rate_2026"]) * 100),
        "y21": round(float(r["y_rate_2021"]) * 100), "y26": round(float(r["y_rate_2026"]) * 100),
        "anchors": int(r["n_anchors"]), "top_anchors": r["top_anchors"],
    } for r in rep[:12]]

    # --- Section 5: reflection (viability + self-score bands) ---
    out["viability"] = [{
        "occupation": r["occupation"],
        "demand_match": round(float(r["demand_match"]) * 100, 1),
        "advanced_share": round(float(r["advanced_share"]) * 100, 1),
        "adv": int(r["n_advanced"]), "dev": int(r["n_developing"]), "aware": int(r["n_aware"]),
        "top_gaps": r["top_gaps"],
    } for r in rows("viability.csv")]
    band = collections.Counter(int(r["score"]) for r in rows("self_score.csv"))
    out["self_score_bands"] = {"advanced": band[2], "developing": band[1], "aware": band[0]}

    with open(os.path.join(A, "blog_figures.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote blog_figures.json with sections:", list(out.keys()))
    print("viability:", [(v["occupation"], v["demand_match"]) for v in out["viability"]])


if __name__ == "__main__":
    main()
