"""
On-disk embedding cache, keyed by (normalised form, model revision).

The revision is the directory name, so vectors made by one model revision can
never be mixed with another's. Within a revision, each form is stored once.
When new advertisements arrive, only forms not already in the cache are
embedded; nothing already cached is ever recomputed.

Layout:  data/blocking/embedding_cache/<revision>/
             forms.txt        one form per line, cache insertion order
             embeddings.npy   float32 matrix (N, D), row i is forms.txt line i
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


class EmbeddingCache:
    def __init__(self, root: Path, model_revision: str, dimension: int):
        self.dir = root / model_revision
        self.dimension = dimension
        self._forms: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, dimension), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        forms_path = self.dir / "forms.txt"
        emb_path = self.dir / "embeddings.npy"
        if forms_path.exists() and emb_path.exists():
            self._forms = forms_path.read_text(encoding="utf-8").splitlines()
            self._matrix = np.load(emb_path)
            if self._matrix.shape != (len(self._forms), self.dimension):
                raise ValueError(
                    f"cache corrupt at {self.dir}: "
                    f"{self._matrix.shape} vs {len(self._forms)} forms"
                )
            self._index = {f: i for i, f in enumerate(self._forms)}

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "forms.txt").write_text(
            "\n".join(self._forms), encoding="utf-8"
        )
        np.save(self.dir / "embeddings.npy", self._matrix)

    def embed_missing(
        self,
        forms: list[str],
        encode_fn: Callable[[list[str]], np.ndarray],
    ) -> int:
        """Embed and append only forms not already cached. Returns how many
        were newly embedded. `encode_fn` maps raw forms -> (n, D) float32."""
        missing = [f for f in forms if f not in self._index]
        # Deterministic order in which new forms are appended.
        missing = sorted(set(missing))
        if not missing:
            return 0
        vecs = encode_fn(missing).astype(np.float32)
        if vecs.shape != (len(missing), self.dimension):
            raise ValueError(
                f"encode_fn returned {vecs.shape}, expected "
                f"{(len(missing), self.dimension)}"
            )
        start = len(self._forms)
        self._forms.extend(missing)
        for j, f in enumerate(missing):
            self._index[f] = start + j
        self._matrix = np.vstack([self._matrix, vecs])
        self._save()
        return len(missing)

    def matrix_for(self, forms: list[str]) -> np.ndarray:
        """Return an (len(forms), D) matrix aligned to the given order."""
        try:
            rows = [self._index[f] for f in forms]
        except KeyError as e:
            raise KeyError(f"form not in cache, embed it first: {e}") from e
        return self._matrix[rows]
