"""Generate HTML heatmap-tables (native labels + precise numbers, theme-safe)
for the blog: a category table (section 1, inside a dropdown) and an AI-skill
table (section 4, shown directly). Cells are coloured by value with black/white
text chosen by luminance.

Output: data/analytics/category_table.html, data/analytics/ai_table.html
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "data/analytics")
FIG = json.load(open(os.path.join(A, "blog_figures.json")))
OCCS = FIG["corpus"]["occupations_2026"]
ACRO = {"ai": "AI", "ml": "ML"}


def pretty(s):
    return " ".join(ACRO.get(w.lower(), w.capitalize()) for w in s.replace("_", " ").split())


def cell(v, vmax):
    t = max(0.0, min(1.0, v/vmax))
    a = (0xe9, 0xf1, 0xfa); b = (0x10, 0x4e, 0x8a)
    r = round(a[0]+(b[0]-a[0])*t); g = round(a[1]+(b[1]-a[1])*t); bl = round(a[2]+(b[2]-a[2])*t)
    lum = 0.299*r + 0.587*g + 0.114*bl
    tc = "#111" if lum > 150 else "#fff"
    return f'<td style="padding:0.4rem 0.3rem;text-align:center;background:#{r:02x}{g:02x}{bl:02x};color:{tc};border:1px solid rgba(128,128,128,0.25);">{v:.0f}</td>'


def table(rows, get, vmax, first_col):
    h = ['<div style="overflow-x:auto;">',
         '<table style="width:100%;border-collapse:collapse;font-size:0.82rem;">',
         '<thead><tr>',
         f'<th style="text-align:left;padding:0.4rem;border:1px solid rgba(128,128,128,0.25);">{first_col}</th>']
    for o in OCCS:
        h.append(f'<th style="padding:0.4rem 0.3rem;border:1px solid rgba(128,128,128,0.25);font-weight:600;">{pretty(o)}</th>')
    h.append('</tr></thead><tbody>')
    for rl in rows:
        h.append('<tr>')
        h.append(f'<td style="padding:0.4rem;border:1px solid rgba(128,128,128,0.25);">{rl}</td>')
        for o in OCCS:
            h.append(cell(get(rl, o), vmax))
        h.append('</tr>')
    h.append('</tbody></table></div>')
    return "".join(h)


def main():
    # category table: top 12 differentiating categories
    mat = FIG["category_matrix_2026"]
    cand = sorted(((max(d.get(o, 0) for o in OCCS)-min(d.get(o, 0) for o in OCCS), c)
                   for c, d in mat.items() if len(d) >= 5), reverse=True)[:12]
    catlabels = [c[1].replace("cat_", "").replace("_", " ") for c in cand]
    catkey = {c[1].replace("cat_", "").replace("_", " "): c[1] for c in cand}
    with open(os.path.join(A, "category_table.html"), "w") as f:
        f.write(table(catlabels, lambda rl, o: mat[catkey[rl]].get(o, 0), 100, "Skill category (% of ads)"))

    # category -> top skills (context for what each category means)
    import csv, collections
    catof = {}
    with open(os.path.join(ROOT, "data/concepts/concept_category.csv")) as f:
        for r in csv.DictReader(f):
            catof[r["concept_id"]] = (r.get("primary_category") or "").strip()
    best = collections.defaultdict(float); label = {}
    with open(os.path.join(A, "penetration.csv")) as f:
        for r in csv.DictReader(f):
            if r["year"] == "2026" and r["grain"] == "concept" and r["is_held"] == "0":
                cid = r["key"]; label[cid] = r["label"]
                best[cid] = max(best[cid], float(r["rate"]))
    by_cat = collections.defaultdict(list)
    for cid, rate in best.items():
        c = catof.get(cid, "")
        if c:
            by_cat[c].append((rate, label[cid]))
    rowsh = ['<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">',
             '<thead><tr><th style="text-align:left;padding:0.4rem;border:1px solid rgba(128,128,128,0.25);">Category</th>'
             '<th style="text-align:left;padding:0.4rem;border:1px solid rgba(128,128,128,0.25);">Most-demanded skills in it</th></tr></thead><tbody>']
    for _, cid in cand:  # the 12 shown in the table, same order
        top = ", ".join(l for _, l in sorted(by_cat.get(cid, []), reverse=True)[:5])
        name = cid.replace("cat_", "").replace("_", " ")
        rowsh.append(f'<tr><td style="padding:0.4rem;border:1px solid rgba(128,128,128,0.25);font-weight:600;">{name}</td>'
                     f'<td style="padding:0.4rem;border:1px solid rgba(128,128,128,0.25);">{top}</td></tr>')
    rowsh.append('</tbody></table>')
    with open(os.path.join(A, "category_skills.html"), "w") as f:
        f.write("".join(rowsh))
    print(f"total categories with data: {len(by_cat)}")

    # ai table
    ai = FIG["ai_concept_demand_2026"]
    terms = list(ai.keys())
    vmax = max(max(ai[t].get(o, 0) for o in OCCS) for t in terms)
    with open(os.path.join(A, "ai_table.html"), "w") as f:
        f.write(table(terms, lambda rl, o: ai[rl].get(o, 0), vmax, "AI skill (% of ads)"))
    print("wrote category_table.html and ai_table.html")


if __name__ == "__main__":
    main()
