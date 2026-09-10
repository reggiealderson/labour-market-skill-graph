"""
Resolver, level 4 — rollup edges (broader/narrower relations).

Level 4 does NOT merge anything. It records that one skill is a narrower form of
another: "azure sql" rolls up into both "azure" and "sql". The entities stay
distinct in the catalogue; the family is a second, coarser level of analysis
computed from them.

This is the whole point. A merge destroys information — you can never recover
the children from a merged parent. A rollup adds information — the parent is
always computable from its children, and a wrong edge is removed without
touching any entity.

Direction is given by the data, not by judgement: if one string's content tokens
are a strict subset of the other's, the subset is the parent.

The result is a DAG, not a tree. "azure sql" has two parents because it really
is both a cloud skill and a SQL skill. A tree would force a false choice.

Two guards on what may act as a parent:

  1. PARENT_STOPLIST — tokens too generic to name a family. A family built on
     "data" or "systems" would contain nearly everything and mean nearly
     nothing.
  2. must_be_standalone — a parent has to occur as a skill string on its own
     somewhere in the corpus. A parent nobody ever wrote by itself is a token,
     not a skill.

CAUTION, and it is the important one: a wrong edge is reversible but NOT
visible. "communication" -> "agent-to-agent communication" is wrong, and while
it is present the communication family number simply looks bigger. Nothing
signals a defect. Reversible means you can fix it once you find it; it does not
help you find it. Review the generated edges, sorted by the parent's weight,
BEFORE publishing any family number.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from resolver_policy import load as load_policy

LEVEL4_VERSION = "resolver-level4-v1.0.0"

# Stoplists live in config/resolver_policy.toml, with their reasons and dates.
# They are read here, not defined here: two homes for one number is worse than
# one bad home, because only one of them gets changed.
_POLICY = load_policy()

# Tokens that may never act as a parent. A family named by one of these would
# swallow most of the catalogue and carry no meaning.
PARENT_STOPLIST = _POLICY.set_of("level4.parent_stoplist")

# Tokens ignored when comparing content, so that "data pipelines" and
# "cloud-native data pipelines" still relate through "pipelines".
CONTENT_STOPWORDS = _POLICY.set_of("level4.content_stopwords")

# What to do with an edge that exists ONLY because a content stopword vanished.
# See build_rollup_edges.
STOPWORD_ONLY_EDGES = _POLICY.get("level4.stopword_only_edges")


def tokens(s: str) -> set[str]:
    return {t for t in s.replace("/", " ").replace("-", " ").split() if t}


def content_tokens(s: str) -> set[str]:
    return tokens(s) - CONTENT_STOPWORDS


def build_rollup_edges(
    counts: Counter,
    *,
    candidate_pairs: list[tuple[str, str]],
    must_be_standalone: bool = True,
) -> tuple[list[dict], dict]:
    """Return (edges, diagnostics).

    Each edge is {"parent": ..., "child": ..., "parent_mentions": ...,
    "child_mentions": ...}. `candidate_pairs` are (a, b) already established as
    similar; direction and admissibility are decided here.

    Pairs the policy demotes rather than admits are returned under
    diagnostics["demoted_to_related"], so the caller can add them to the related
    set. They are not discarded: a demotion is additive and reversible, and the
    dropped word is recorded with each one.
    """
    corpus = set(counts)
    edges: list[dict] = []
    demoted: list[dict] = []
    rejected = Counter()

    for a, b in candidate_pairs:
        ca, cb = content_tokens(a), content_tokens(b)
        if not ca or not cb:
            rejected["empty content tokens"] += 1
            continue
        if ca < cb:
            parent, child = a, b
        elif cb < ca:
            parent, child = b, a
        else:
            # Equal or incomparable — not a rollup. Equal token sets should
            # already have been removed by level 2.
            rejected["not a strict subset"] += 1
            continue

        if tokens(parent) <= PARENT_STOPLIST:
            rejected["parent is all stoplist tokens"] += 1
            continue

        # An edge that survives only because a content stopword vanished.
        # Where the parent's RAW tokens are still inside the child's, dropping
        # the stopword merely widened a relation that already held — that is the
        # case the stopwords exist for. Where they are NOT, the two strings
        # disagree on a word and the subset test never saw the disagreement:
        # "data pipelines" -> "ai pipelines" is a sibling, not a narrowing.
        # Demoting it to `related` is additive; keeping it inflates a family
        # and nothing signals the defect.
        if not tokens(parent) <= tokens(child):
            if STOPWORD_ONLY_EDGES == "related":
                demoted.append({
                    "a": parent, "b": child,
                    "reason": "stopword-only subset",
                    "dropped": sorted(tokens(parent) - tokens(child)),
                })
                rejected["stopword-only subset -> related"] += 1
                continue
            if STOPWORD_ONLY_EDGES == "reject":
                rejected["stopword-only subset -> rejected"] += 1
                continue
            # "rollup" keeps the old behaviour, and the policy records why.
        if must_be_standalone and parent not in corpus:
            rejected["parent never occurs standalone"] += 1
            continue

        edges.append({
            "parent": parent, "child": child,
            "parent_mentions": counts[parent], "child_mentions": counts[child],
        })

    diagnostics = {
        "rejected": dict(rejected),
        "edges": len(edges),
        "stopword_only_policy": STOPWORD_ONLY_EDGES,
        "demoted_to_related": demoted,
    }
    return edges, diagnostics


def assert_acyclic(edges: list[dict]) -> dict:
    """Raise if the edge set contains a cycle. Returns shape diagnostics.

    Do not assume acyclicity — assert it. The subset rule is only antisymmetric
    while no two strings share a token set, which is a property level 2 must
    have established, not one this level can take on trust.
    """
    children: dict[str, set] = defaultdict(set)
    parents: dict[str, set] = defaultdict(set)
    for e in edges:
        children[e["parent"]].add(e["child"])
        parents[e["child"]].add(e["parent"])

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = defaultdict(int)
    stack_path: list[str] = []

    def visit(node: str):
        colour[node] = GREY
        stack_path.append(node)
        for nxt in children.get(node, ()):
            if colour[nxt] == GREY:
                cycle = stack_path[stack_path.index(nxt):] + [nxt]
                raise ValueError(f"rollup edges contain a cycle: {' -> '.join(cycle)}")
            if colour[nxt] == WHITE:
                visit(nxt)
        stack_path.pop()
        colour[node] = BLACK

    for node in list(children) + list(parents):
        if colour[node] == WHITE:
            visit(node)

    multi = {c: p for c, p in parents.items() if len(p) > 1}
    both = set(children) & set(parents)
    return {
        "parents": len(children),
        "children": len(parents),
        "multi_parent_children": len(multi),
        "multi_parent_rate": len(multi) / len(parents) if parents else 0.0,
        "depth_gt_2_nodes": len(both),
        "acyclic": True,
    }


__all__ = ["build_rollup_edges", "assert_acyclic", "PARENT_STOPLIST",
           "content_tokens", "LEVEL4_VERSION"]
