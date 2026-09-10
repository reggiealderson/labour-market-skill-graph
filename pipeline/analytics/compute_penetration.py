"""Stage 3 — penetration (+ Wilson 95% interval) at 4 grains, and 2021->2026
change tests (two-proportion z) for the 3 shared occupations.

Grains, per occupation-year cell:
  concept   key = concept_id      (held concepts included, flagged)
  category  key = primary_category (cat_generic_field_reference flagged boilerplate)
  type      key = concept_type
  facet     key = facet value      (value != "none"; facet_id kept)

An ad "has" a key if >= 1 of its concepts maps to that key. Penetration =
distinct ads with the key / all ads in the cell (denominator from posting).

Outputs (data/analytics/): penetration.csv, change_test.csv
"""
import csv, os, math, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "data/analytics")
Z = 1.959963984540054  # 95%
BOILERPLATE = "cat_generic_field_reference"
SHARED = ["data_analyst", "data_engineer", "data_scientist"]


def wilson(hits, n):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = hits / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, max(0.0, centre - half), min(1.0, centre + half)


def two_prop(h1, n1, h2, n2):
    """Return (diff, z, p_value) for 2026 vs 2021. None if undefined."""
    if n1 == 0 or n2 == 0:
        return None, None, None
    p1, p2 = h1 / n1, h2 / n2
    pool = (h1 + h2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p2 - p1, None, None
    z = (p2 - p1) / se
    pval = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return p2 - p1, z, pval


def load_facets():
    m = collections.defaultdict(list)
    with open(os.path.join(ROOT, "taxonomy/concept_facet.csv")) as f:
        for r in csv.DictReader(f):
            if r["value"] != "none":
                m[r["concept_id"]].append((r["facet_id"], r["value"]))
    return m


def main():
    # ad counts per cell
    n_ads = collections.Counter()
    with open(os.path.join(A, "posting.csv")) as f:
        for r in csv.DictReader(f):
            n_ads[(r["year"], r["occupation"])] += 1

    facets = load_facets()
    labels = {}  # concept_id -> label

    # grain key -> set of posting_ids, per cell
    # sets[(year,occ)][grain][ (facet_id,key) ] = set(posting_id)
    sets = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(set)))
    with open(os.path.join(A, "posting_skill.csv")) as f:
        for r in csv.DictReader(f):
            cell = (r["year"], r["occupation"])
            pid = r["posting_id"]
            cid = r["concept_id"]
            labels[cid] = r["concept_label"]
            g = sets[cell]
            g["concept"][("", cid)].add(pid)
            if r["primary_category"]:
                g["category"][("", r["primary_category"])].add(pid)
            if r["concept_type"]:
                g["type"][("", r["concept_type"])].add(pid)
            for fid, val in facets.get(cid, []):
                g["facet"][(fid, val)].add(pid)

    held = set()
    with open(os.path.join(A, "posting_skill.csv")) as f:
        for r in csv.DictReader(f):
            if r["held"] == "1":
                held.add(r["concept_id"])

    # penetration rows
    prows = []
    # index for change test: rate[(occ,grain,fid,key)][year] = (hits, n)
    idx = collections.defaultdict(dict)
    for (year, occ), grains in sets.items():
        n = n_ads[(year, occ)]
        for grain, keys in grains.items():
            for (fid, key), pids in keys.items():
                hits = len(pids)
                p, lo, hi = wilson(hits, n)
                label = labels.get(key, key) if grain == "concept" else key
                prows.append({
                    "year": year, "occupation": occ, "grain": grain,
                    "facet_id": fid, "key": key, "label": label,
                    "n_ads": n, "hits": hits, "rate": round(p, 6),
                    "wilson_low": round(lo, 6), "wilson_high": round(hi, 6),
                    "is_held": "1" if (grain == "concept" and key in held) else "0",
                    "is_boilerplate": "1" if (grain == "category" and key == BOILERPLATE) else "0",
                })
                idx[(occ, grain, fid, key, label)][year] = (hits, n)

    # change tests: 3 shared occupations, keys present in either year
    crows = []
    for (occ, grain, fid, key, label), yr in idx.items():
        if occ not in SHARED:
            continue
        n1 = n_ads[("2021", occ)]
        n2 = n_ads[("2026", occ)]
        h1 = yr.get("2021", (0, n1))[0]
        h2 = yr.get("2026", (0, n2))[0]
        diff, z, pval = two_prop(h1, n1, h2, n2)
        crows.append({
            "occupation": occ, "grain": grain, "facet_id": fid,
            "key": key, "label": label,
            "n_2021": n1, "hits_2021": h1, "rate_2021": round(h1 / n1, 6) if n1 else 0,
            "n_2026": n2, "hits_2026": h2, "rate_2026": round(h2 / n2, 6) if n2 else 0,
            "diff": round(diff, 6) if diff is not None else "",
            "z": round(z, 4) if z is not None else "",
            "p_value": round(pval, 6) if pval is not None else "",
            "significant": "1" if (pval is not None and pval < 0.05) else "0",
            "direction": "up" if (diff or 0) > 0 else ("down" if (diff or 0) < 0 else "flat"),
        })

    with open(os.path.join(A, "penetration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prows[0].keys()))
        w.writeheader(); w.writerows(prows)
    with open(os.path.join(A, "change_test.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crows[0].keys()))
        w.writeheader(); w.writerows(crows)

    print(f"penetration rows : {len(prows)}")
    print(f"change_test rows : {len(crows)}")
    bg = collections.Counter(r["grain"] for r in prows)
    print("penetration by grain:", dict(bg))
    sig = sum(1 for r in crows if r["significant"] == "1")
    print(f"significant changes (p<0.05): {sig} of {len(crows)}")


if __name__ == "__main__":
    main()
