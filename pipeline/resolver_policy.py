"""Loader for config/resolver_policy.toml.

The policy holds every tunable. Code reads it; no module keeps its own copy of a
number that changes an output. A run stamps `policy_hash` into report.json, so a
published figure always carries the policy that produced it — and two runs under
different hashes are known to be incomparable rather than assumed comparable.

Read-only by design. Nothing writes the policy at run time: a value that a run
can change is not a decision, it is a variable.

Run `python3 pipeline/resolver_policy.py` to print the policy and its hash.
"""

from __future__ import annotations

import hashlib
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

POLICY_PATH = Path(__file__).resolve().parent.parent / "config" / "resolver_policy.toml"

SCHEMA_VERSION = 1

# Every key a run must find. Absent means stop, never fall back to a default:
# a silent default is a decision nobody made and nobody can audit.
REQUIRED: tuple[str, ...] = (
    "counting.closure", "counting.unit", "counting.families_partition",
    "level1.version", "level1.plural_folding",
    "level2.rule_a.enabled", "level2.rule_b.enabled", "level2.rule_b.caps_test",
    "level2.rule_b.caps_test_requires_token_match", "level2.order",
    "candidates.min_mentions", "candidates.fuzzy_cutoff",
    "level3.penetration_threshold",
    "acronym_scan.sort_by", "acronym_scan.min_gain_mentions",
    "acronym_scan.min_dominance_ratio",
    "level4.must_be_standalone", "level4.parent_stoplist",
    "level4.content_stopwords", "level4.stopword_only_edges",
    "level4.obvious_edges.enabled", "level4.obvious_edges.parent_max_tokens",
    "level4.obvious_edges.child_extra_tokens",
    "adjudication.model", "adjudication.escalation_model",
    "adjudication.single_pass", "adjudication.use_batch_api",
    "adjudication.items_per_request", "adjudication.batch_api_min_items",
    "review.must_read", "review.sort_by",
    "review.scatter_top_k", "review.closure_top_k",
    "publication.review_queue", "publication.provisional_when_queue_nonempty",
    "tests.bounds.rollup_edges", "tests.bounds.level3_queue",
    "tests.change_detection.same_corpus_must_be_identical",
    "tests.change_detection.shared_string_tolerance_ratio",
    "tests.change_detection.shared_string_tolerance_floor",
    "tests.change_detection.max_single_parent_drift",
    "tests.change_detection.new_occupation_requires_new_baseline",
    "tests.change_detection.new_occupation_requires_approval",
    "tests.change_detection.known_occupations",
    "tests.change_detection.baseline_dir",
)


class PolicyError(RuntimeError):
    """The policy is missing, malformed, or self-contradictory."""


def _dig(d: dict, dotted: str) -> Any:
    node: Any = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


class Policy:
    """A parsed policy. Access with `p.get("level4.parent_stoplist")`."""

    def __init__(self, raw: dict, digest: str, path: Path):
        self._raw = raw
        self.hash = digest
        self.path = path
        self.version = raw.get("policy_version", "unversioned")

    def get(self, dotted: str, default: Any = ...) -> Any:
        try:
            return _dig(self._raw, dotted)
        except KeyError:
            if default is ...:
                raise PolicyError(
                    f"{self.path.name} has no key {dotted!r}. Add it with a "
                    f"reason and a date; do not let the code default it.")
            return default

    def set_of(self, dotted: str) -> set[str]:
        """A list-valued key as a set. Stoplists are compared, never ordered."""
        return set(self.get(dotted))

    def bounds(self, name: str) -> tuple[float, float]:
        """A two-sided limit. One-sided limits are refused: a lower bound alone
        cannot see an overshoot, which is the fault A3 and the canaries had."""
        v = self.get(f"tests.bounds.{name}")
        if not (isinstance(v, list) and len(v) == 2):
            raise PolicyError(
                f"tests.bounds.{name} must be [low, high]; got {v!r}. "
                f"A one-sided bound cannot detect an overshoot.")
        low, high = v
        if low > high:
            raise PolicyError(f"tests.bounds.{name} is inverted: {v!r}")
        return low, high

    def drift_limit(self, corpus_delta: float) -> float:
        """The allowed edge-drift rate for a given fraction of corpus change.

        ratio x delta, floored so a tiny corpus change is never graded against
        a near-zero limit, and capped at max_single_parent_drift so a large
        corpus change cannot excuse an unlimited amount of drift.
        """
        ratio = self.get("tests.change_detection.shared_string_tolerance_ratio")
        floor = self.get("tests.change_detection.shared_string_tolerance_floor")
        ceiling = self.get("tests.change_detection.max_single_parent_drift")
        return min(max(ratio * corpus_delta, floor), ceiling)

    def as_dict(self) -> dict:
        return self._raw

    def provenance(self) -> dict:
        """The block every run copies into report.json."""
        return {
            "policy_version": self.version,
            "policy_hash": self.hash,
            "policy_path": str(self.path.relative_to(self.path.parent.parent)),
        }


def _validate(raw: dict, path: Path) -> None:
    got = raw.get("schema_version")
    if got != SCHEMA_VERSION:
        raise PolicyError(
            f"{path.name} is schema_version {got!r}; this code reads "
            f"{SCHEMA_VERSION}. Migrate the file, do not coerce it.")

    missing = []
    for key in REQUIRED:
        try:
            _dig(raw, key)
        except KeyError:
            missing.append(key)
    if missing:
        raise PolicyError(
            f"{path.name} is missing required keys: {', '.join(missing)}")

    # Cross-checks: values that are individually legal but jointly wrong.
    if _dig(raw, "level1.plural_folding"):
        raise PolicyError(
            "level1.plural_folding is true. Level 1 sees one string at a time; "
            "a plural rule must see the others. It belongs to level 2 rule (b).")
    if _dig(raw, "counting.unit") not in ("postings", "mentions"):
        raise PolicyError("counting.unit must be 'postings' or 'mentions'")
    if _dig(raw, "acronym_scan.sort_by") != "gain":
        raise PolicyError(
            "acronym_scan.sort_by must be 'gain'. Combined weight is the wrong "
            "key: merging a spelled-out form moves only ITS mentions, not the "
            "acronym's. Same fault as sorting the edge review by the parent's "
            "own weight instead of closure reach.")
    if _dig(raw, "acronym_scan.min_gain_mentions") < 1:
        raise PolicyError("acronym_scan.min_gain_mentions must be >= 1")
    if _dig(raw, "acronym_scan.min_dominance_ratio") < 1.0:
        raise PolicyError(
            "acronym_scan.min_dominance_ratio must be >= 1.0. Below 1.0 the "
            "runner-up outweighs the top expansion and no merge is defensible.")
    if _dig(raw, "review.must_read") != "union":
        raise PolicyError(
            "review.must_read must be 'union'. Closure size is a cost key and "
            "is blind to risk; the scatter score is a risk key and is blind to "
            "cost — a parent with one child scores 0 by construction, which is "
            "468 of 882 parents. Ranking on either key alone cannot see the "
            "fault on its other side, the same reason tests.bounds refuses a "
            "one-sided limit.")
    if _dig(raw, "review.sort_by") not in ("scatter", "closure"):
        raise PolicyError("review.sort_by must be 'scatter' or 'closure'")
    for key in ("review.scatter_top_k", "review.closure_top_k"):
        if _dig(raw, key) < 1:
            raise PolicyError(f"{key} must be >= 1; 0 would drop one key of the union")
    if _dig(raw, "level4.stopword_only_edges") not in ("related", "rollup", "reject"):
        raise PolicyError(
            "level4.stopword_only_edges must be 'related', 'rollup' or 'reject'")
    if list(_dig(raw, "level2.order")) != ["rule_b", "rule_a"]:
        raise PolicyError(
            "level2 order must be rule_b then rule_a: rule (b) creates new "
            "token-set collisions and only rule (a) removes them, and level 4 "
            "requires them gone.")
    ipr = _dig(raw, "adjudication.items_per_request")
    if not 1 <= ipr <= 100:
        raise PolicyError(
            f"adjudication.items_per_request must be in [1, 100]; got {ipr}. "
            f"Above ~50 the token saving is exhausted while one malformed "
            f"response costs that many verdicts.")
    if not _dig(raw, "adjudication.single_pass"):
        raise PolicyError(
            "adjudication.single_pass is false. One Sonnet pass; `uncertain` is "
            "the only route to Opus.")
    cd = raw["tests"]["change_detection"]
    ratio = cd["shared_string_tolerance_ratio"]
    floor = cd["shared_string_tolerance_floor"]
    ceiling = cd["max_single_parent_drift"]
    if not 0 < ratio < 10:
        raise PolicyError(f"shared_string_tolerance_ratio must be in (0, 10); got {ratio}")
    if not 0 <= floor < 1:
        raise PolicyError(f"shared_string_tolerance_floor must be in [0, 1); got {floor}")
    if not 0 < ceiling <= 1:
        raise PolicyError(f"max_single_parent_drift must be in (0, 1]; got {ceiling}")
    if floor > ceiling:
        raise PolicyError(
            f"shared_string_tolerance_floor ({floor}) exceeds "
            f"max_single_parent_drift ({ceiling}) — the floor could never fire "
            f"below the ceiling that is meant to bound it")
    if not cd["new_occupation_requires_new_baseline"]:
        raise PolicyError(
            "new_occupation_requires_new_baseline is false. Measurement showed "
            "the newest occupation (ai_engineer) is the binding case for the "
            "drift ratio; the next new occupation can be worse, and the drift "
            "test would then fail on a correct run. A new occupation must "
            "always get a fresh baseline, never be graded against the old one.")
    if not cd["new_occupation_requires_approval"]:
        raise PolicyError(
            "new_occupation_requires_approval is false. A new occupation's "
            "baseline is not self-certifying; it must be approved.")
    if not isinstance(cd["known_occupations"], list) or not cd["known_occupations"]:
        raise PolicyError("known_occupations must be a non-empty list")
    if not _dig(raw, "tests.change_detection.same_corpus_must_be_identical"):
        raise PolicyError(
            "same_corpus_must_be_identical is false. The resolver is "
            "deterministic or it is not; there is no middle setting.")


@lru_cache(maxsize=None)
def load(path: str | Path = POLICY_PATH) -> Policy:
    """Load, validate and hash the policy. Cached: one policy per process."""
    path = Path(path)
    try:
        blob = path.read_bytes()
    except FileNotFoundError:
        raise PolicyError(
            f"no policy at {path}. The pipeline does not run without one; "
            f"an unwritten decision is not a default.") from None
    raw = tomllib.loads(blob.decode("utf-8"))
    _validate(raw, path)
    return Policy(raw, hashlib.sha256(blob).hexdigest()[:16], path)


if __name__ == "__main__":
    p = load()
    print(f"{p.version}  hash {p.hash}")
    print(f"  counting     closure={p.get('counting.closure')} "
          f"unit={p.get('counting.unit')}")
    print(f"  level2 (b)   caps_test={p.get('level2.rule_b.caps_test')} "
          f"safe={p.get('level2.rule_b.safe')}")
    print(f"  level3       threshold={p.get('level3.penetration_threshold')}")
    print(f"  level4       stopword_only_edges={p.get('level4.stopword_only_edges')!r}")
    print(f"  obvious      parent<={p.get('level4.obvious_edges.parent_max_tokens')} "
          f"child+{p.get('level4.obvious_edges.child_extra_tokens')}")
    print(f"  adjudication {p.get('adjudication.model')} -> "
          f"{p.get('adjudication.escalation_model')} on uncertain, "
          f"batch={p.get('adjudication.use_batch_api')}")
    for name in ("rollup_edges", "level3_queue", "multi_parent_rate"):
        print(f"  bound        {name} = {p.bounds(name)}")
