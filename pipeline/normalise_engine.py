"""
Deterministic normalisation engine, step 1 of the concept-taxonomy pipeline.

No model. No embeddings. No corpus-dependent rule targets. Every canonical
target is written by hand in a config/*.yml file; this module only reads
those files and applies them. Adding an occupation requires new YAML lines
only, never a code change.

Five config files, five kinds of rule:
  aliases.yml         exact string -> exact string (acronym / variant)
  vendor_prefixes.yml exact string -> exact string (vendor-prefixed pair)
  orthography.yml     exact string -> exact string (spacing / spelling)
  empty_suffixes.yml  a trailing word, stripped from any form that ends in it
  protected_forms.yml a form no rule may ever change

Engine order per iteration: protected check, then orthography, then aliases,
then vendor_prefixes (first match wins), then empty_suffixes. Repeat until no
rule changes the string, capped at MAX_ITERATIONS. Reaching the cap raises,
rather than silently returning a possibly-unstable result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# v1.1.0: suffix rules now consult config/suffix_guards.yml before firing.
ENGINE_VERSION = "normalise-engine-v1.1.0"
MAX_ITERATIONS = 10

_EXACT_RULE_FILES = ["orthography", "aliases", "vendor_prefixes"]


@dataclass
class Configs:
    versions: dict[str, str]
    config_version: str
    orthography: dict[str, str]
    aliases: dict[str, str]
    vendor_prefixes: dict[str, str]
    suffixes: list[str]
    protected: set[str]
    suffix_pattern_guards: list[str]
    suffix_list_guards: set[str]


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError(f"{path}: missing required top-level 'version' key")
    return data


def _detect_cycles(combined: dict[str, str]) -> None:
    """DFS over the exact-match rule graph. Raises on any cycle.

    Covers orthography + aliases + vendor_prefixes, whose edges are fixed at
    load time. empty_suffixes cannot be represented as a fixed edge (its
    target depends on the input's token structure), so it is not part of
    this graph; the runtime iteration cap in `normalise_one` is the backstop
    for any cycle that only emerges once suffix stripping is combined with
    the exact-match rules.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> None:
        colour[node] = GRAY
        path.append(node)
        nxt = combined.get(node)
        if nxt is not None:
            state = colour.get(nxt, WHITE)
            if state == GRAY:
                cycle = path[path.index(nxt):] + [nxt]
                raise ValueError(
                    "cycle detected in exact-match rules (orthography/"
                    f"aliases/vendor_prefixes): {' -> '.join(cycle)}"
                )
            if state == WHITE:
                visit(nxt, path)
        path.pop()
        colour[node] = BLACK

    for node in combined:
        if colour.get(node, WHITE) == WHITE:
            visit(node, [])


def load_configs(config_dir: Path) -> Configs:
    files = {name: _load_yaml(config_dir / f"{name}.yml")
             for name in [*_EXACT_RULE_FILES, "empty_suffixes",
                          "protected_forms", "suffix_guards"]}

    orthography = dict(files["orthography"].get("rules") or {})
    aliases = dict(files["aliases"].get("rules") or {})
    vendor_prefixes = dict(files["vendor_prefixes"].get("pairs") or {})
    suffixes = list(files["empty_suffixes"].get("suffixes") or [])
    protected = set(files["protected_forms"].get("forms") or [])
    suffix_pattern_guards = list(files["suffix_guards"].get("pattern_guards") or [])
    suffix_list_guards = set(files["suffix_guards"].get("list_guards") or [])

    # A source string must not be defined with different targets in two
    # different exact-match files. Ambiguous otherwise.
    combined: dict[str, str] = {}
    for file_name, rules in [("orthography", orthography), ("aliases", aliases),
                              ("vendor_prefixes", vendor_prefixes)]:
        for src, tgt in rules.items():
            if src in combined and combined[src] != tgt:
                raise ValueError(
                    f"conflicting rule for {src!r}: "
                    f"{combined[src]!r} vs {tgt!r} (from {file_name})"
                )
            combined[src] = tgt
    _detect_cycles(combined)

    versions = {name: data["version"] for name, data in files.items()}
    digest = hashlib.sha256(
        "\n".join(f"{name}:{versions[name]}" for name in sorted(versions))
        .encode("utf-8")
    ).hexdigest()[:16]
    config_version = f"{ENGINE_VERSION}+cfg.{digest}"

    return Configs(
        versions=versions,
        config_version=config_version,
        orthography=orthography,
        aliases=aliases,
        vendor_prefixes=vendor_prefixes,
        suffixes=suffixes,
        protected=protected,
        suffix_pattern_guards=suffix_pattern_guards,
        suffix_list_guards=suffix_list_guards,
    )


def _suffix_result_blocked(result: str, cfg: Configs) -> bool:
    """True if a suffix strip would produce a guarded result."""
    if result in cfg.suffix_list_guards:
        return True
    return any(result.endswith(p) for p in cfg.suffix_pattern_guards)


def _strip_suffix(form: str, cfg: Configs) -> tuple[str, str | None]:
    tokens = form.split()
    if len(tokens) < 2:
        return form, None
    last = tokens[-1]
    if last in cfg.suffixes:
        result = " ".join(tokens[:-1])
        # Guard: do not strip if the head noun would be lost, leaving an
        # adjective / participle that is not a skill on its own.
        if _suffix_result_blocked(result, cfg):
            return form, None
        return result, last
    return form, None


def _apply_first_rule(form: str, cfg: Configs) -> tuple[str, str | None]:
    if form in cfg.orthography:
        return cfg.orthography[form], "orthography"
    if form in cfg.aliases:
        return cfg.aliases[form], "alias"
    if form in cfg.vendor_prefixes:
        return cfg.vendor_prefixes[form], "vendor_prefix"
    stripped, word = _strip_suffix(form, cfg)
    if word is not None:
        return stripped, f"empty_suffix:{word}"
    return form, None


@dataclass
class NormaliseResult:
    input_form: str
    normalised_form: str
    method: str
    trail: list[tuple[str, str, str]] = field(default_factory=list)
    config_version: str = ""


def normalise_one(form: str, cfg: Configs) -> NormaliseResult:
    if form in cfg.protected:
        return NormaliseResult(form, form, "protected", [], cfg.config_version)

    current = form
    trail: list[tuple[str, str, str]] = []
    for _ in range(MAX_ITERATIONS):
        if current in cfg.protected:
            break
        new_form, rule = _apply_first_rule(current, cfg)
        if rule is None:
            break
        trail.append((rule, current, new_form))
        current = new_form
    else:
        raise RuntimeError(
            f"{form!r} did not converge in {MAX_ITERATIONS} iterations: {trail}"
        )

    method = "unchanged" if not trail else "+".join(r for r, _, _ in trail)
    return NormaliseResult(form, current, method, trail, cfg.config_version)
