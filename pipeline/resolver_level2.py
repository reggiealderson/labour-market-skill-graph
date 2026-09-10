"""
Resolver, level 2 — automatic merge. TWO rules, with DIFFERENT error rates.

  Rule (a) — identical token sets.  SAFE: it cannot be wrong.
      Merges strings whose token sets are identical after level 1: pure
      reordering or duplicated words. "algorithms and data structures" and
      "data structures and algorithms" are the same skill written two ways.
      A different word order carries no information, so folding it loses none.

  Rule (b) — singular/plural pairs.  SMALL ERROR RATE, NOT ZERO.
      Folds a singular onto its plural when BOTH forms occur in the corpus.
      Plurals are generated from the singular, never stemmed off the plural,
      so no invented word ("data analysi", "databas") can ever be produced.
      It is still not safe: an acronym can look like an English plural. RAPIDS
      is an NVIDIA library, "rapid" is a word. `caps_attested` blocks that class
      (see `merge_plural_pairs`), but the residual rate is above zero.

Do NOT read "level 2 is safe" and trust rule (b)'s output as you would rule
(a)'s. Anything that reports level-2 provenance should distinguish the two.

ORDER: run rule (b) FIRST, then rule (a), then assert no token-set collisions.
Rule (b) creates new collisions — "cloud databases" folds to "cloud database",
which may already exist — and level 4 requires that no two survivors share a
token set. Running (a) first and stopping there hands level 4 bad input.

Everything else that looks mergeable is not:

  - A bare term against a qualified one ("sql" vs "azure sql") is NOT a merge.
    Those are distinct skills and become rollup edges at level 4, which is
    additive and reversible. Merging them would destroy the distinction in every
    downstream count with no way back.
  - Genuinely ambiguous pairs go to level 3 for arbitration.

Level 2 MUST run before level 4. The level-4 subset rule makes a DAG only if no
two strings have equal token sets; if two do, each is a subset of the other and
the edge set contains a cycle. Equal token sets is exactly what level 2 removes.

Canonical choice is deterministic: the most-mentioned member wins, ties broken
by shortest string then alphabetically. Running it twice gives the same answer.
"""

from __future__ import annotations

from collections import Counter, defaultdict

LEVEL2_VERSION = "resolver-level2-v1.1.0"


def token_key(s: str) -> tuple:
    """Order-insensitive identity of a string."""
    return tuple(sorted({t for t in s.replace("/", " ").replace("-", " ").split() if t}))


def merge_identical_token_sets(
    counts: Counter,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (member -> canonical, canonical -> members).

    `counts` maps a level-1 string to its mention count. Only groups with more
    than one member appear in the returned families; singletons map to
    themselves.
    """
    by_key: dict[tuple, list[str]] = defaultdict(list)
    for s in counts:
        by_key[token_key(s)].append(s)

    canonical_of: dict[str, str] = {}
    families: dict[str, list[str]] = {}
    for members in by_key.values():
        # Deterministic: most mentions, then shortest, then alphabetical.
        canonical = sorted(members, key=lambda m: (-counts[m], len(m), m))[0]
        for m in members:
            canonical_of[m] = canonical
        if len(members) > 1:
            families[canonical] = sorted(members)
    return canonical_of, families


def apply_level2(counts: Counter, canonical_of: dict[str, str]) -> Counter:
    """Fold mention counts onto their canonical form."""
    out: Counter = Counter()
    for s, c in counts.items():
        out[canonical_of.get(s, s)] += c
    return out




# ── Rule (b): singular/plural pairs ───────────────────────────────────────────

def plural_forms(singular: str) -> list[str]:
    """Plurals the last word of `singular` could take.

    Generation only. Stemming a plural is ambiguous — "databases" is
    "database"+s or "databas"+es and nothing in the string says which — so the
    resolver never stems. It proposes plurals and folds only on a real hit.
    """
    parts = singular.split()
    if not parts:
        return []
    word, head = parts[-1], parts[:-1]
    cands = [word + "s"]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        cands.append(word + "es")
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        cands.append(word[:-1] + "ies")
    return [" ".join(head + [c]) for c in cands]


def merge_plural_pairs(
    counts: Counter,
    caps_attested: set[str] | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return (member -> canonical, canonical -> members) for plural pairs.

    `caps_attested` holds level-1 strings whose FINAL WORD is written in capitals
    somewhere in the raw text. Those are blocked. It is a test, not a list, so it
    catches the next acronym of this kind and not only the ones known today; it
    found http/https, cm/cms and nir/nirs, which no list held.

    The failure mode it guards is an acronym that reads as an English plural. It
    is not a plural error. The test over-blocks — VLOOKUPS is a real plural
    written in capitals — and that direction is deliberate: a blocked fold is a
    merge not made, and additive; a wrong fold destroys a distinction for good.

    Canonical choice matches rule (a): most mentions, then shortest, then
    alphabetical. Pairs may chain, so members are unioned transitively.
    """
    caps_attested = caps_attested or set()

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for singular in counts:
        for plural in plural_forms(singular):
            if plural in counts and plural not in caps_attested:
                union(singular, plural)

    groups: dict[str, list[str]] = defaultdict(list)
    for s in parent:
        groups[find(s)].append(s)

    canonical_of: dict[str, str] = {}
    families: dict[str, list[str]] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = sorted(members, key=lambda m: (-counts[m], len(m), m))[0]
        for m in members:
            canonical_of[m] = canonical
        families[canonical] = sorted(members)
    return canonical_of, families


def compose(first: dict[str, str], second: dict[str, str]) -> dict[str, str]:
    """Chain two member->canonical maps: apply `first`, then `second`."""
    out = {m: second.get(c, c) for m, c in first.items()}
    for m, c in second.items():
        out.setdefault(m, c)
    return out


__all__ = ["merge_identical_token_sets", "merge_plural_pairs", "plural_forms",
           "compose", "apply_level2", "token_key", "LEVEL2_VERSION"]
