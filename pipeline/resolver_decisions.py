"""Permanent decisions — read FIRST, before any rule runs.

A rule produces a defensible default. A decision overrides it, in both
directions: an edge the rules never make can be forced in, and one they keep
making can be kept out. `communication -> agent-to-agent communication` is the
reason this file exists. Nothing in that string is wrong, so no re-run removes
it; only a recorded judgement does.

KEYS ARE LEVEL-1 STRINGS. Level 1 is frozen and sees one string at a time, so
its output is stable across runs. Level-2 canonicals are not: they move whenever
rule (b) changes, and keying on them would silently orphan every decision the
next time the plural rule is touched. `level1_version` is checked on load for
exactly that reason.

NOTHING IS DELETED. A decision whose strings have left the corpus is reported
stale, not dropped: new job ads bring strings back, and the judgement still
holds when they do.

Run `python3 pipeline/resolver_decisions.py` to print the file and its shape.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

DECISIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "resolver" / "decisions.json"

SCHEMA_VERSION = 1
SEP = " || "

PAIR_VERDICTS = {"merge", "keep_separate"}
EDGE_VERDICTS = {"rollup", "related", "reject"}
AUTHORS = {"human", "model"}


class DecisionError(RuntimeError):
    """The decisions file is malformed, or contradicts itself."""


def pair_key(a: str, b: str) -> str:
    """Order-independent. A judgement about two strings is not directional."""
    return SEP.join(sorted((a, b)))


def edge_key(parent: str, child: str) -> str:
    """Directional. parent -> child is a different claim from child -> parent."""
    return f"{parent}{SEP}{child}"


def _split(key: str) -> tuple[str, str]:
    parts = key.split(SEP)
    if len(parts) != 2:
        raise DecisionError(
            f"key {key!r} does not hold exactly two strings separated by {SEP!r}")
    return parts[0], parts[1]


class Decisions:
    def __init__(self, raw: dict, path: Path):
        self.path = path
        self._raw = raw
        self.pairs: dict[str, dict] = raw.get("string_pairs", {})
        self.edges: dict[str, dict] = raw.get("edges", {})
        self.level1_version: str = raw["level1_version"]
        # Counters, so a run can prove a decision was actually applied rather
        # than merely present. An unapplied decision is a silent no-op.
        self.applied: dict[str, int] = {"edge_reject": 0, "edge_forced": 0,
                                        "edge_related": 0, "pair": 0}

    # ── reading ──────────────────────────────────────────────────────────────
    def edge_verdict(self, parent: str, child: str) -> str | None:
        entry = self.edges.get(edge_key(parent, child))
        return entry["verdict"] if entry else None

    def pair_verdict(self, a: str, b: str) -> str | None:
        entry = self.pairs.get(pair_key(a, b))
        return entry["verdict"] if entry else None

    # ── applying ─────────────────────────────────────────────────────────────
    def apply_to_edges(self, edges: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (kept_edges, log). Rejected and demoted edges leave the DAG.

        Forced edges (verdict 'rollup' for a pair the rules did not produce) are
        NOT added here: this function only filters what it is given. Forcing is
        done by the caller, which alone knows the mention counts.
        """
        kept, log = [], []
        for e in edges:
            v = self.edge_verdict(e["parent"], e["child"])
            if v is None or v == "rollup":
                kept.append(e)
                continue
            entry = self.edges[edge_key(e["parent"], e["child"])]
            log.append({**e, "verdict": v, "by": entry.get("by"),
                        "note": entry.get("note")})
            self.applied["edge_reject" if v == "reject" else "edge_related"] += 1
        return kept, log

    def forced_edges(self, present: Iterable[tuple[str, str]]) -> list[dict]:
        """Decisions of verdict 'rollup' whose edge the rules did not produce."""
        have = set(present)
        out = []
        for key, entry in self.edges.items():
            if entry["verdict"] != "rollup":
                continue
            p, c = _split(key)
            if (p, c) not in have:
                out.append({"parent": p, "child": c, **entry})
                self.applied["edge_forced"] += 1
        return out

    # ── health ───────────────────────────────────────────────────────────────
    def stale(self, corpus: set[str]) -> list[dict]:
        """Entries naming a string the corpus no longer holds. Reported, never
        removed: the string may return with the next batch of ads."""
        out = []
        for kind, table in (("edge", self.edges), ("pair", self.pairs)):
            for key, entry in table.items():
                a, b = _split(key)
                missing = [s for s in (a, b) if s not in corpus]
                if missing:
                    out.append({"kind": kind, "key": key, "missing": missing,
                                "by": entry.get("by")})
        return out

    def summary(self) -> dict:
        by_author: dict[str, int] = {}
        for table in (self.edges, self.pairs):
            for entry in table.values():
                by_author[entry.get("by", "?")] = by_author.get(entry.get("by", "?"), 0) + 1
        return {
            "edge_decisions": len(self.edges),
            "pair_decisions": len(self.pairs),
            "by_author": by_author,
            "applied": dict(self.applied),
        }

    # ── writing ──────────────────────────────────────────────────────────────
    def record_edge(self, parent: str, child: str, verdict: str, *, by: str,
                    note: str = "", model: str = "", prompt_version: str = "") -> None:
        """Add or update one edge decision. A human entry is never overwritten
        by a model entry: the machine does not get to revise a person."""
        if verdict not in EDGE_VERDICTS:
            raise DecisionError(f"edge verdict {verdict!r} not in {sorted(EDGE_VERDICTS)}")
        if by not in AUTHORS:
            raise DecisionError(f"author {by!r} not in {sorted(AUTHORS)}")
        key = edge_key(parent, child)
        existing = self.edges.get(key)
        if existing and existing.get("by") == "human" and by == "model":
            raise DecisionError(
                f"refusing to overwrite a human decision on {key!r} with a model "
                f"verdict. Record the disagreement for review instead.")
        entry: dict[str, Any] = {"verdict": verdict, "by": by,
                                 "date": date.today().isoformat(), "note": note}
        if by == "model":
            entry["model"] = model
            entry["prompt_version"] = prompt_version
        self.edges[key] = entry

    def record_pair(self, a: str, b: str, verdict: str, *, by: str,
                    canonical: str = "", note: str = "", model: str = "",
                    prompt_version: str = "") -> None:
        """Add or update one string-pair decision.

        This is the rejection memory for the acronym scan. A `keep_separate`
        matters as much as a `merge`: without it, every future scan re-asks a
        question already answered, and the review list never shrinks toward
        zero for a stable occupation. An unmade merge loses nothing, but
        re-asking costs a person's attention every cycle.

        Note what is NOT stored here: a pair below the policy's minimum gain.
        That is a filter outcome, not a judgement — it is recomputed every run
        so the pair returns automatically if new ads raise its gain above the
        limit. Storing it as a decision would freeze a threshold accident into
        the permanent record.
        """
        if verdict not in PAIR_VERDICTS:
            raise DecisionError(f"pair verdict {verdict!r} not in {sorted(PAIR_VERDICTS)}")
        if by not in AUTHORS:
            raise DecisionError(f"author {by!r} not in {sorted(AUTHORS)}")
        if verdict == "merge" and not canonical:
            raise DecisionError(
                f"merging {a!r} and {b!r} without naming a canonical. A merge is "
                f"irreversible; which string survives is part of the decision.")
        if verdict == "merge" and canonical not in (a, b):
            raise DecisionError(
                f"canonical {canonical!r} is neither {a!r} nor {b!r}")
        key = pair_key(a, b)
        existing = self.pairs.get(key)
        if existing and existing.get("by") == "human" and by == "model":
            raise DecisionError(
                f"refusing to overwrite a human decision on {key!r} with a model "
                f"verdict. Record the disagreement for review instead.")
        entry: dict[str, Any] = {"verdict": verdict, "by": by,
                                 "date": date.today().isoformat(), "note": note}
        if canonical:
            entry["canonical"] = canonical
        if by == "model":
            entry["model"] = model
            entry["prompt_version"] = prompt_version
        self.pairs[key] = entry

    def decided_pairs(self) -> set[frozenset]:
        """Every pair already answered, in either direction. The acronym scan
        subtracts this so a person is never asked the same question twice."""
        return {frozenset(_split(k)) for k in self.pairs}

    def save(self) -> None:
        self._raw["string_pairs"] = self.pairs
        self._raw["edges"] = self.edges
        self.path.write_text(json.dumps(self._raw, indent=2) + "\n")


def _validate(raw: dict, path: Path, level1_version: str | None) -> None:
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise DecisionError(
            f"{path.name} is schema_version {raw.get('schema_version')!r}; this "
            f"code reads {SCHEMA_VERSION}. Migrate the file, do not coerce it.")
    if level1_version and raw["level1_version"] != level1_version:
        raise DecisionError(
            f"{path.name} was keyed against level 1 {raw['level1_version']!r} but "
            f"the pipeline runs {level1_version!r}. Level-1 output moved, so every "
            f"key may be orphaned. Re-key the file deliberately; do not run past this.")

    for kind, table, allowed in (("string_pairs", raw.get("string_pairs", {}), PAIR_VERDICTS),
                                 ("edges", raw.get("edges", {}), EDGE_VERDICTS)):
        for key, entry in table.items():
            a, b = _split(key)
            if a == b:
                raise DecisionError(f"{kind} key {key!r} names the same string twice")
            if entry.get("verdict") not in allowed:
                raise DecisionError(
                    f"{kind}[{key!r}] verdict {entry.get('verdict')!r} "
                    f"not in {sorted(allowed)}")
            if entry.get("by") not in AUTHORS:
                raise DecisionError(
                    f"{kind}[{key!r}] by={entry.get('by')!r} not in {sorted(AUTHORS)}")
            if entry.get("by") == "model" and not entry.get("model"):
                raise DecisionError(
                    f"{kind}[{key!r}] is a model verdict with no model id. A model "
                    f"decision must stay distinguishable from a human one.")
            if kind == "string_pairs" and entry["verdict"] == "merge" \
                    and not entry.get("canonical"):
                raise DecisionError(
                    f"string_pairs[{key!r}] merges without naming a canonical")

    # A pair cannot be both merged and held apart, in either key order.
    seen: set[frozenset] = set()
    for key in raw.get("string_pairs", {}):
        fs = frozenset(_split(key))
        if fs in seen:
            raise DecisionError(f"string_pairs holds two entries for {sorted(fs)}")
        seen.add(fs)


def load(path: str | Path = DECISIONS_PATH,
         level1_version: str | None = None) -> Decisions:
    path = Path(path)
    if not path.exists():
        raise DecisionError(
            f"no decisions file at {path}. Create it with empty tables rather "
            f"than letting the pipeline run without one — an absent file and an "
            f"empty file mean different things.")
    raw = json.loads(path.read_text())
    _validate(raw, path, level1_version)
    return Decisions(raw, path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from resolver_level1 import LEVEL1_VERSION

    d = load(level1_version=LEVEL1_VERSION)
    print(f"{d.path.name}  keyed on {d.level1_version}")
    for k, v in d.summary().items():
        print(f"  {k:16} {v}")
    for key, entry in d.edges.items():
        print(f"\n  {key}\n    verdict={entry['verdict']}  by={entry['by']}  "
              f"date={entry['date']}")
        print(f"    {entry['note'][:100]}...")
