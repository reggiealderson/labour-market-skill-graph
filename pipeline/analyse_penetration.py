"""
Compute and display penetration rates for data_scientist, 2021 vs 2026.
Reads from the normed JSONL files.
"""

import json
from collections import Counter
from pathlib import Path

SKILLS_DIR = Path("data/skills")
YEARS      = ["2021", "2026"]
OCCUPATION = "data_scientist"


def load_normed(year: str) -> list[dict]:
    path = SKILLS_DIR / year / f"{OCCUPATION}_normed.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def penetration(records: list[dict]) -> Counter:
    counter: Counter = Counter()
    for r in records:
        seen: set = set()
        for skill in r["skills"]:
            if skill not in seen:
                counter[skill] += 1
                seen.add(skill)
    return counter


def main():
    year_data = {y: load_normed(y) for y in YEARS}
    year_pen  = {y: penetration(year_data[y]) for y in YEARS}
    ns        = {y: len(year_data[y]) for y in YEARS}

    r21, r26 = year_pen["2021"], year_pen["2026"]
    n21, n26 = ns["2021"], ns["2026"]

    all_keys = set(r21) | set(r26)
    rows = [(k, r21[k] / n21, r26[k] / n26) for k in all_keys]

    # ── Overall stats ──────────────────────────────────────────────────────────
    for year in YEARS:
        recs = year_data[year]
        spr  = [len(r["skills"]) for r in recs]
        pen  = year_pen[year]
        print(f"\n{'='*50}")
        print(f"  {year} — {OCCUPATION}")
        print(f"{'='*50}")
        print(f"  Records:            {len(recs)}")
        print(f"  Skills/rec — mean:  {sum(spr)/len(spr):.1f}  median: {sorted(spr)[len(spr)//2]}  max: {max(spr)}")
        print(f"  Unique canonicals:  {len(pen)}")
        print(f"  Zero-skill records: {sum(1 for n in spr if n == 0)}")

    # ── Penetration table: >=4% in either year ─────────────────────────────────
    threshold = 0.04
    notable = [(k, r21, r26) for k, r21, r26 in rows if r21 >= threshold or r26 >= threshold]
    notable.sort(key=lambda x: -x[2])

    print(f"\n{'='*76}")
    print(f"  Penetration rates (≥{threshold*100:.0f}% in either year)  —  {len(notable)} skills")
    print(f"{'='*76}")
    print(f"  {'Skill':<48} {'2021':>7} {'2026':>7} {'delta':>9}")
    print(f"  {'-'*74}")
    for skill, rate21, rate26 in notable:
        delta = rate26 - rate21
        arrow = "UP" if delta > 0.03 else ("DN" if delta < -0.03 else "  ")
        print(f"  {skill:<48} {rate21*100:>6.1f}% {rate26*100:>6.1f}% {arrow} {delta*100:>+6.1f}pp")

    # ── Top risers and fallers ─────────────────────────────────────────────────
    rows_all = sorted(rows, key=lambda x: x[2] - x[1], reverse=True)

    print(f"\n{'='*60}")
    print("  TOP 15 RISING SKILLS  (2021 → 2026)")
    print(f"{'='*60}")
    for skill, rate21, rate26 in rows_all[:15]:
        delta = rate26 - rate21
        print(f"  {skill:<48} {rate21*100:>5.1f}% → {rate26*100:>5.1f}%  (+{delta*100:.1f}pp)")

    print(f"\n{'='*60}")
    print("  TOP 15 FALLING SKILLS  (2021 → 2026)")
    print(f"{'='*60}")
    for skill, rate21, rate26 in rows_all[-15:]:
        delta = rate26 - rate21
        print(f"  {skill:<48} {rate21*100:>5.1f}% → {rate26*100:>5.1f}%  ({delta*100:.1f}pp)")


if __name__ == "__main__":
    main()
