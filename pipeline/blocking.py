"""
Blocking engine, step 2 of the skill-taxonomy pipeline.

Groups similar normalised surface forms into small candidate groups for review
in step 3. It does NOT merge, create concepts, or build a hierarchy. Every form
survives, in exactly one group.

Determinism: input is sorted by form before anything runs, embeddings come from
a fixed model revision through an on-disk cache, the kNN graph and its connected
components are deterministic, and every random draw uses one configured seed.
Two runs on the same input give byte-identical output.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml


# ── config ────────────────────────────────────────────────────────────────
def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError(f"{path}: missing top-level 'version'")
    return data


def config_version(embedding_cfg: dict, blocking_cfg: dict) -> str:
    """One version string binding both config files and the engine together."""
    payload = yaml.safe_dump(
        {"embedding": embedding_cfg, "blocking": blocking_cfg}, sort_keys=True
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"blocking-engine-v1.0.0+cfg.{digest}"


# ── input ─────────────────────────────────────────────────────────────────
def load_forms(surface_form_csv: Path) -> dict[str, int]:
    """Distinct normalised forms -> total mention count, from step 1's table.

    Multiple raw forms can share one normalised form, so counts are summed.
    """
    import csv

    counts: dict[str, int] = {}
    with surface_form_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            nf = row["normalised_form"]
            counts[nf] = counts.get(nf, 0) + int(row["count"])
    return counts


# ── embedding ─────────────────────────────────────────────────────────────
def make_encoder(embedding_cfg: dict):
    """Return encode_fn(list[str]) -> (n, D) float32, using the pinned model."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)

    m = embedding_cfg["model"]
    template = embedding_cfg["template"]
    normalize = embedding_cfg["normalize_embeddings"]
    device = embedding_cfg["device"]

    model = SentenceTransformer(
        m["name"], revision=m["revision"], device=device
    )

    def encode_fn(forms: list[str]) -> np.ndarray:
        texts = [template.format(form=f) for f in forms]
        vecs = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    return encode_fn


# ── kNN graph ─────────────────────────────────────────────────────────────
def knn_edges(
    forms: list[str], matrix: np.ndarray, k: int
) -> list[tuple[int, int, float]]:
    """Undirected edges (i, j, cosine) over the k nearest neighbours.

    Vectors are unit-normalised, so cosine is a dot product. Edges are
    de-duplicated with i < j and sorted for determinism. No threshold applied
    here; callers filter by similarity.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(forms)
    k_eff = min(k + 1, n)  # +1 because the nearest neighbour is the point itself
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="brute")
    nn.fit(matrix)
    dist, idx = nn.kneighbors(matrix)

    seen: dict[tuple[int, int], float] = {}
    for i in range(n):
        for d, j in zip(dist[i], idx[i]):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            sim = float(1.0 - d)
            # Same undirected pair can appear from both endpoints; keep one.
            seen[(a, b)] = sim
    return sorted((a, b, s) for (a, b), s in seen.items())


# ── components ────────────────────────────────────────────────────────────
def connected_components(
    n: int, edges: list[tuple[int, int, float]], threshold: float
) -> list[list[int]]:
    """Connected components over edges with similarity >= threshold."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as cc

    kept = [(a, b) for a, b, s in edges if s >= threshold]
    if kept:
        rows = [a for a, _ in kept] + [b for _, b in kept]
        cols = [b for _, b in kept] + [a for a, _ in kept]
        data = np.ones(len(rows), dtype=np.int8)
        g = csr_matrix((data, (rows, cols)), shape=(n, n))
    else:
        g = csr_matrix((n, n), dtype=np.int8)
    n_comp, labels = cc(g, directed=False)
    groups: list[list[int]] = [[] for _ in range(n_comp)]
    for node, lab in enumerate(labels):
        groups[lab].append(node)
    return [sorted(g) for g in groups]


def split_oversized(
    nodes: list[int],
    edges: list[tuple[int, int, float]],
    base_threshold: float,
    max_size: int,
    step: float,
    ceiling: float,
) -> list[tuple[list[int], float]]:
    """Split one component that exceeds max_size by raising its threshold in
    `step` increments until every piece fits. Returns (member_nodes, threshold)
    for each final piece.

    Reuses the global kNN `edges` restricted to this component. Raising the
    threshold only ever DROPS edges, so the induced kNN edge set is exactly
    what a higher threshold would keep — no pairwise recompute needed.
    """
    node_set = set(nodes)
    if len(nodes) <= max_size:
        return [(sorted(nodes), base_threshold)]

    # Induce once: edges with both endpoints in this component, remapped to
    # dense local ids so connected_components works on a small graph.
    local_nodes = sorted(nodes)
    local_id = {g: i for i, g in enumerate(local_nodes)}
    induced = [
        (local_id[a], local_id[b], s)
        for a, b, s in edges
        if a in node_set and b in node_set
    ]

    threshold = base_threshold
    while True:
        threshold = round(threshold + step, 4)
        if threshold > ceiling + 1e-9:
            raise RuntimeError(
                f"component of {len(nodes)} forms still exceeds max_size="
                f"{max_size} at ceiling threshold {ceiling}; global nodes: "
                f"{local_nodes[:10]}..."
            )
        comps = connected_components(len(local_nodes), induced, threshold)
        pieces = [[local_nodes[i] for i in c] for c in comps]
        if len(pieces) > 1 and all(len(p) <= max_size for p in pieces):
            return [(sorted(p), threshold) for p in pieces]
        if len(pieces) > 1:
            # It fragmented, but at least one piece is still too big. Finalise
            # the small pieces here; recurse on the big ones at a higher start.
            out: list[tuple[list[int], float]] = []
            for p in pieces:
                if len(p) <= max_size:
                    out.append((sorted(p), threshold))
                else:
                    out.extend(
                        split_oversized(
                            p, edges, threshold, max_size, step, ceiling
                        )
                    )
            return out
        # Still one blob: raise the threshold and try again.


def similarity_within_group(nodes: list[int], matrix: np.ndarray) -> dict[int, float]:
    """Mean cosine of each member to the other members. Singleton -> 1.0."""
    if len(nodes) == 1:
        return {nodes[0]: 1.0}
    sub = matrix[nodes]
    sims = sub @ sub.T
    out: dict[int, float] = {}
    for a, node in enumerate(nodes):
        others = [sims[a, b] for b in range(len(nodes)) if b != a]
        out[node] = float(np.mean(others))
    return out
