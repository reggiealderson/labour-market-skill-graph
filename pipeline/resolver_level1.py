"""
Resolver, level 1 — deterministic normalisation.

This is production code, not a measurement helper. The merge-sizing script
imports it so that the measurement is made with exactly the code that will run
in production; if the two diverge, the measurement is wrong.

Level 1 is settled and does four things, in order:

  1. Case fold and strip whitespace.
  2. Fold typographic punctuation onto ASCII. Job postings are full of curly
     quotes and en/em dashes, and the same skill is written both ways.
  3. Strip a trailing parenthetical — "Large Language Models (LLMs)" becomes
     "large language models".
  4. Expand acronyms through ONE map shared by every occupation.

Deliberately NOT here:
  - Plural and possessive folding. Not in the settled set; it would merge
    "data model" with "data models" silently, which is almost certainly right
    but has not been decided, and a merge is not reversible downstream.
  - Fuzzy matching of any kind. That is level 2 and above, and it is the level
    whose workload is still being sized.
  - Per-occupation acronym additions. The resolver carries one map for all
    occupations by design — that is what makes each new occupation cheaper than
    the last. The retired per-occupation blocks in pipeline/acronyms.py are not
    applied.

A level-1 collapse is safe: it only ever joins strings that are the same string
once written consistently.
"""

from __future__ import annotations

import hashlib
import re

from acronyms import BASE_ACRONYM_MAP

# LOGIC_VERSION covers the four fixed steps below (case fold, punctuation,
# parenthetical, acronym expansion AS A MECHANISM). Bump it by hand, rarely —
# only when one of those steps itself changes.
_LOGIC_VERSION = "resolver-level1-v1.0"

# One map, all occupations. Seeded from BASE_ACRONYM_MAP; extend HERE, never
# per-occupation.
ACRONYM_MAP: dict[str, str] = dict(BASE_ACRONYM_MAP)

# LEVEL1_VERSION is DERIVED, not hand-written, and that is the point. Level 1's
# logic is settled and changes rarely, but ACRONYM_MAP is explicitly meant to
# grow — the docstring above says "extend HERE" — and every addition changes
# what level1() actually outputs for some string. A hand-maintained version
# string relies on someone remembering to bump it; this doesn't rely on
# anyone remembering anything. data/resolver/decisions.json is keyed on this
# value specifically so that an acronym-map change is caught automatically:
# forget to touch this file and the run stops on the next build, rather than
# silently comparing decisions made under the old map to output from the new
# one. Order-independent: sorted before hashing, so entry order never matters.
_acronym_digest = hashlib.sha256(
    "\n".join(f"{k}\t{v}" for k, v in sorted(ACRONYM_MAP.items())).encode()
).hexdigest()[:12]
LEVEL1_VERSION = f"{_LOGIC_VERSION}+acr.{_acronym_digest}"

_WS = re.compile(r"\s+")
_TRAILING_PAREN = re.compile(r"\s*\([^)]{1,60}\)\s*$")

# Same fold as the extraction parser uses for evidence matching, for the same
# reason: the source text and the model both vary in punctuation style.
_PUNCT_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "´": "'", "`": "'",
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", "​": "",
})


def level1(s: str) -> str:
    """Apply the four settled deterministic steps."""
    s = s.translate(_PUNCT_FOLD).lower().strip()
    s = _WS.sub(" ", s)
    s = _TRAILING_PAREN.sub("", s).strip()
    return ACRONYM_MAP.get(s, s)


__all__ = ["level1", "ACRONYM_MAP", "LEVEL1_VERSION"]
