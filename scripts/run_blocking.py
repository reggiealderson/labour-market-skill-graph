"""
Step 2 — blocking. Two modes:

  python3 scripts/run_blocking.py --diagnostics
      Embed every eligible form (populating the on-disk cache) and write
      reports/threshold_samples.csv: 40 random joined pairs at each of
      0.80 / 0.85 / 0.90. STOPS there. A human picks the threshold.

  python3 scripts/run_blocking.py
      Build the final candidate groups at the threshold in config/blocking.yml.
      Writes data/blocking/candidate_group.csv, reports/groups.csv, and
      data/blocking/summary.json.

Nothing is merged. Every form ends in exactly one group.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from blocking import (  # noqa: E402
    config_version, connected_components, knn_edges, load_forms, load_yaml,
    make_encoder, similarity_within_group, split_oversized,
)
from embedding_cache import EmbeddingCache  # noqa: E402

CONFIG_DIR = ROOT / "config"
SURFACE_FORM = ROOT / "data" / "normalise" / "surface_form.csv"
OUT_DIR = ROOT / "data" / "blocking"
CACHE_ROOT = OUT_DIR / "embedding_cache"
REPORTS_DIR = ROOT / "reports"
DIAG_THRESHOLDS = [0.80, 0.85, 0.90]
DIAG_PAIRS_PER_THRESHOLD = 40


def prepare():
    """Shared setup: load config + forms, split short vs eligible, embed
    eligible via cache, build the kNN edge list. Returns everything the two
    modes need."""
    emb_cfg = load_yaml(CONFIG_DIR / "embedding.yml")
    blk_cfg = load_yaml(CONFIG_DIR / "blocking.yml")
    cfg_ver = config_version(emb_cfg, blk_cfg)

    counts = load_forms(SURFACE_FORM)
    all_forms = sorted(counts)  # deterministic order
    min_len = blk_cfg["min_form_length"]

    short = [f for f in all_forms if len(f) < min_len]
    eligible = [f for f in all_forms if len(f) >= min_len]
    print(f"forms: {len(all_forms):,}  eligible: {len(eligible):,}  "
          f"short (< {min_len} chars, pass-through): {len(short):,}")

    cache = EmbeddingCache(
        CACHE_ROOT, emb_cfg["model"]["revision"], emb_cfg["model"]["dimension"]
    )
    n_new = cache.embed_missing(eligible, make_encoder(emb_cfg))
    print(f"embedded {n_new:,} new forms; cache now holds "
          f"{len(cache._forms):,}")

    matrix = cache.matrix_for(eligible)  # aligned to `eligible` order
    edges = knn_edges(eligible, matrix, blk_cfg["knn_k"])
    print(f"kNN edges: {len(edges):,}")

    return emb_cfg, blk_cfg, cfg_ver, counts, short, eligible, matrix, edges


def run_diagnostics():
    _, blk_cfg, _, counts, _short, eligible, _matrix, edges = prepare()
    rng = np.random.default_rng(blk_cfg["random_seed"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in DIAG_THRESHOLDS:
        joined = [(a, b, s) for a, b, s in edges if s >= t]
        if not joined:
            continue
        take = min(DIAG_PAIRS_PER_THRESHOLD, len(joined))
        pick = rng.choice(len(joined), size=take, replace=False)
        for p in sorted(pick):
            a, b, s = joined[p]
            fa, fb = eligible[a], eligible[b]
            rows.append({
                "threshold": f"{t:.2f}",
                "form_a": fa,
                "form_b": fb,
                "similarity": f"{s:.4f}",
                "count_a": counts[fa],
                "count_b": counts[fb],
            })

    out = REPORTS_DIR / "threshold_samples.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "threshold", "form_a", "form_b", "similarity", "count_a", "count_b"
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")
    print("STOP: pick a threshold, set it in config/blocking.yml, then rerun "
          "without --diagnostics.")


def run_final():
    emb_cfg, blk_cfg, cfg_ver, counts, short, eligible, matrix, edges = prepare()
    threshold = blk_cfg["threshold"]
    max_size = blk_cfg["max_group_size"]
    model_rev = emb_cfg["model"]["revision"]

    comps = connected_components(len(eligible), edges, threshold)

    # Each final group: (member_forms, threshold_used). Short forms first as
    # their own single-member groups, recorded with a distinct marker.
    final: list[tuple[list[str], object]] = []
    for f in short:
        final.append(([f], "too_short"))

    for comp in comps:
        pieces = split_oversized(
            comp, edges, threshold, max_size,
            blk_cfg["threshold_step"], blk_cfg["threshold_ceiling"],
        )
        for nodes, used_t in pieces:
            final.append(([eligible[i] for i in nodes], used_t))

    # similarity_within_group per member, computed on the final membership.
    def group_sim(forms: list[str]) -> dict[str, float]:
        if len(forms) == 1:
            return {forms[0]: 1.0}
        idx = [eligible.index(f) for f in forms]
        sims = similarity_within_group(idx, matrix)
        return {eligible[i]: sims[i] for i in idx}

    # Order groups by total mention count desc, then by first form for ties.
    def total(forms):
        return sum(counts[f] for f in forms)

    final.sort(key=lambda g: (-total(g[0]), sorted(g[0])[0]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    cg_rows = []
    group_rows = []
    size_dist: Counter = Counter()
    for gid, (forms, used_t) in enumerate(final):
        forms_sorted = sorted(forms)
        sims = group_sim(forms_sorted)
        size_dist[len(forms_sorted)] += 1
        for form in forms_sorted:
            cg_rows.append({
                "group_id": gid,
                "form": form,
                "similarity_within_group": f"{sims[form]:.4f}",
                "threshold_used": used_t,
                "model_revision": model_rev,
                "config_version": cfg_ver,
            })
        group_rows.append({
            "group_id": gid,
            "size": len(forms_sorted),
            "total_count": total(forms_sorted),
            "threshold_used": used_t,
            "members": " | ".join(forms_sorted),
        })

    with (OUT_DIR / "candidate_group.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "group_id", "form", "similarity_within_group", "threshold_used",
            "model_revision", "config_version",
        ])
        w.writeheader()
        w.writerows(cg_rows)

    with (REPORTS_DIR / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "group_id", "size", "total_count", "threshold_used", "members"
        ])
        w.writeheader()
        w.writerows(group_rows)

    multi = sum(1 for forms, _ in final if len(forms) > 1)
    summary = {
        "config_version": cfg_ver,
        "threshold": threshold,
        "model": f"{emb_cfg['model']['name']}@{model_rev}",
        "n_forms": len(eligible) + len(short),
        "n_groups": len(final),
        "n_multi_member_groups": multi,
        "n_singleton_groups": len(final) - multi,
        "n_excluded_too_short": len(short),
        "size_distribution": {str(k): size_dist[k] for k in sorted(size_dist)},
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"groups: {len(final):,}  multi-member: {multi:,}  "
          f"singletons: {len(final) - multi:,}  too-short: {len(short):,}")
    print(f"size distribution: {dict(sorted(size_dist.items()))}")
    print(f"wrote {OUT_DIR / 'candidate_group.csv'}")
    print(f"wrote {REPORTS_DIR / 'groups.csv'}")
    print(f"wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    if "--diagnostics" in sys.argv[1:]:
        run_diagnostics()
    else:
        run_final()
