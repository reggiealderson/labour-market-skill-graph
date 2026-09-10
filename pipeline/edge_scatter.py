"""Scatter score — how much a parent's children share BEYOND the parent word.

The review sheet was sorted by closure size. Closure size says what a wrong edge
COSTS. It does not say how LIKELY one is. `tableau` has 2 children on one
subject; `models` has 92 spanning Bayesian models, dbt models, attribution
models and clinical data models. Both are legal under the token rule. Only one
is a meaningful family, and closure size cannot tell them apart.

The score, per parent:

  1. residual(child) = child tokens - parent tokens - level4.content_stopwords
  2. link two children when their residuals share a token
  3. the largest connected group is the family's MAIN SUBJECT
  4. `outside` = children not in that group

`outside` is a count, not a rate, because the rate is meaningless at n=1 (every
single-child parent scores 1.0) and the damage of a scattered family scales with
how many children sit outside it.

WHAT THIS SCORE CANNOT DO. It measures likelihood only, which is the mirror of
the fault it fixes. A parent with one child cannot have a child outside the main
group, so it scores 0 by construction — 468 of the 882 parents with children.
`python` is the clearest case: 2,290 closure postings on a single edge, the most
expensive edge in the catalogue, and scatter rank 395. Neither key is sufficient
alone, so `must_read` is the UNION of both heads. See build_edge_review_sheet.py.

The stopword list is read from the policy, never held here: these are the same
stopwords level 4 used to build the edges, and a second copy would drift.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

# Level 1 folds punctuation, but "/" and "-" survive inside a string
# ("ai/ml models", "end-to-end driving models"). Split on them so the residual
# sees the words, not the joined form.
_SPLIT_CHARS = ("/", "-")


def tokens(s: str) -> list[str]:
    for ch in _SPLIT_CHARS:
        s = s.replace(ch, " ")
    return [t for t in s.split() if t]


def residual(parent: str, child: str, stopwords: set[str]) -> frozenset[str]:
    """What the child says that the parent does not."""
    parent_tokens = set(tokens(parent))
    return frozenset(
        t for t in tokens(child)
        if t not in parent_tokens and t not in stopwords
    )


def groups(parent: str, children: Iterable[str], stopwords: set[str]) -> list[list[str]]:
    """Children grouped by shared residual tokens, largest group first.

    Single-link: two children join when they share ONE residual token, and the
    relation is transitive. That over-groups — `ai` comes out as one group of 91
    joined through common words. Deliberate: over-grouping makes the score
    SMALLER, so the error direction hides scatter rather than inventing it. A
    parent this score calls scattered really is scattered.
    """
    children = sorted(children)          # determinism: the run must be repeatable
    if not children:
        return []

    by_token: dict[str, list[str]] = defaultdict(list)
    for child in children:
        for token in residual(parent, child, stopwords):
            by_token[token].append(child)

    parent_of = {c: c for c in children}

    def find(x: str) -> str:
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x

    for members in by_token.values():
        first = members[0]
        for other in members[1:]:
            a, b = find(first), find(other)
            if a != b:
                parent_of[a] = b

    grouped: dict[str, list[str]] = defaultdict(list)
    for child in children:
        grouped[find(child)].append(child)
    # Size first, then the group's own first member: two groups of equal size
    # must not swap places between runs.
    return sorted(grouped.values(), key=lambda g: (-len(g), g[0]))


def score(parent: str, children: Iterable[str], stopwords: set[str]) -> dict:
    """One parent's scatter figures. `outside` is the ranking key."""
    found = groups(parent, children, stopwords)
    n = sum(len(g) for g in found)
    if n == 0:
        return {"n_children": 0, "n_groups": 0, "largest_group": 0,
                "outside": 0, "scatter_rate": 0.0, "main_subject": []}
    largest = len(found[0])
    return {
        "n_children": n,
        "n_groups": len(found),
        "largest_group": largest,
        "outside": n - largest,
        "scatter_rate": round(1 - largest / n, 4),
        "main_subject": found[0],
    }
