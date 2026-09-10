"""Validate taxonomy/facets.yaml against the Stage G facet schema.

Checks (all structural; no facet content is generated here):
  1. every facet id is unique and matches the facet id scheme (FACET_ID_SCHEME);
  2. every facet has a non-empty definition and a non-empty applies_to;
  3. every facet has at least one value; every value has a non-empty
     definition and at least one non-empty match rule key — EXCEPT a
     value named as fallback_value, which may omit the match block
     entirely (its concepts come from the fallback mechanism);
  4. every value slug matches the value scheme, and is unique within its facet;
  5. every match rule uses only allowed keys over concept properties
     (concept_label_matches, surface_form_matches, stage_d_type_in), and every
     stage_d_type_in entry is one of the nine Stage D types;
  6. inputs is present and names only allowed concept-property sources;
  7. exclusivity is present and one of ALLOWED_EXCLUSIVITY; unmatched is present
     and one of ALLOWED_UNMATCHED;
  8. no string anywhere inside a facet (any field, value label, match pattern,
     or notes) still contains a scaffold placeholder (TEMPLATE, REPLACE_WITH),
     so the blank template does not pass;
  9. NO facet rule references a Stage F category. A facet is defined over
     concepts, not over category assignments, so it is a hard error for any
     key, source, or pattern anywhere in a facet to name a category id
     (matching the cat_ id scheme), a category source (concept_category and
     the like), or a goes_to edge;
 10. NO facet rule names a specific year. Facet content is corpus-independent
     and must apply to any year in scope, so a four-digit year in any facet
     string is a hard error. (frozen_at at the top level is exempt.)
 11. precedence is present when — and only when — exclusivity is exclusive, and
     when present it lists every value in the facet exactly once and names no
     slug that is not a value in the facet;
 12. fallback_value is present when — and only when — unmatched is declared, and
     when present it names a value in the facet.
 13. assignment, when present, is one of ALLOWED_ASSIGNMENT. When assignment is
     classifier, match blocks are optional on all values (assignment comes from
     an external classifier, not from pattern rules). When assignment is pattern
     (the default when assignment is omitted), match blocks are required on every
     value except the fallback_value.

With --frozen, additionally require top-level frozen_at to be set (not null);
this is for a freeze gate and does not change the default behaviour.

Exit 0 = valid, 1 = one or more failures. Read-only. No API, no corpus.

Usage:  python3 tools/validate_facets.py [facets.yaml] [--frozen]
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FACETS = ROOT / "taxonomy/facets.yaml"

# id schemes: facet_ + lower snake_case; value slug is lower snake_case.
FACET_ID_SCHEME = re.compile(r"^facet_[a-z0-9]+(_[a-z0-9]+)*$")
VALUE_SCHEME = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")

# A Stage F category id. Finding one anywhere in a facet is a hard error
# (check 7): facets are defined over concepts, never over category assignments.
CATEGORY_ID_TOKEN = re.compile(r"\bcat_[a-z0-9]+(?:_[a-z0-9]+)*\b")

# A four-digit year. Facet content must be year-independent (check 8).
YEAR_TOKEN = re.compile(r"\b(?:19|20)\d{2}\b")

# The nine Stage D types a rule may name.
STAGE_D_TYPES = {
    "language", "library", "tool", "method", "knowledge_area",
    "practice", "transversal_skill", "compliance", "industry_context",
}

# Concept-property sources a facet may read (check 6). Category sources are
# absent by design: a facet must not read concept_category.csv.
ALLOWED_INPUTS = {"concept_label", "surface_form", "stage_d_type"}

# The only rule keys a value's match block may use (check 5).
ALLOWED_MATCH_KEYS = {
    "concept_label_matches", "surface_form_matches", "stage_d_type_in",
}

# Allowed values for the required per-facet fields (check 7).
ALLOWED_EXCLUSIVITY = {"exclusive", "multi_valued"}
ALLOWED_UNMATCHED = {"absent", "declared"}
ALLOWED_ASSIGNMENT = {"pattern", "classifier"}

# Scaffold placeholders. Their presence in an authored string means the blank
# template was left in place (check 8), so validation must fail.
PLACEHOLDER_TOKENS = ("TEMPLATE", "REPLACE_WITH")

# Keys that name a category or a category source. Their presence anywhere in a
# facet is a hard error with a clear message (check 7). The category-id token
# scan catches the rest.
CATEGORY_KEYS = {
    "category", "category_id", "categories", "concept_category",
    "concept_category_csv", "primary_category", "secondary_category",
    "goes_to",
}

REQUIRED_FACET_FIELDS = ["id", "label", "definition", "applies_to", "inputs",
                         "values"]
# exclusivity and unmatched are also required, but check_enum_field handles
# their presence check AND allowed-values check in one place — listing them
# here as well would produce duplicate errors when they are missing.
REQUIRED_VALUE_FIELDS = ["value", "label", "definition", "match"]


def fail(errors, msg):
    errors.append(msg)


def _walk(node):
    """Yield (key, value) for every string leaf and every mapping key, deep.

    key is the mapping key a string sits under (or None for list items); value
    is the string leaf. Used by the category-id and year scans so a forbidden
    token is caught wherever an author places it.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            yield ("__key__", str(k))
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)
    elif isinstance(node, str):
        yield ("__str__", node)


def check_no_category_or_year(facet, where, errors):
    """Checks 7 and 8: no category reference, no specific year, anywhere."""
    for kind, text in _walk(facet):
        if kind == "__key__":
            if text in CATEGORY_KEYS:
                fail(errors, f"{where}: key '{text}' references a category — "
                             "a facet is defined over concepts, not over "
                             "category assignments")
        if CATEGORY_ID_TOKEN.search(text):
            hit = CATEGORY_ID_TOKEN.search(text).group(0)
            fail(errors, f"{where}: references category id '{hit}' — a facet "
                         "rule may not reference a category id")
        if YEAR_TOKEN.search(text):
            hit = YEAR_TOKEN.search(text).group(0)
            fail(errors, f"{where}: names year '{hit}' — facet content must be "
                         "year-independent")


def check_inputs(facet, where, errors):
    """Check 6: inputs is present and names only allowed sources."""
    inputs = facet.get("inputs")
    if inputs is None:
        fail(errors, f"{where}: 'inputs' is required")
        return
    if not isinstance(inputs, list) or len(inputs) == 0:
        fail(errors, f"{where}: 'inputs' must be a non-empty list")
        return
    for src in inputs:
        if src not in ALLOWED_INPUTS:
            fail(errors, f"{where}: input '{src}' is not an allowed "
                         f"concept-property source {sorted(ALLOWED_INPUTS)}")


def check_enum_field(facet, field, allowed, where, errors):
    """Check 7: a required per-facet field is present and in its allowed set."""
    if field not in facet:
        fail(errors, f"{where}: '{field}' is required")
        return
    val = facet.get(field)
    if val not in allowed:
        fail(errors, f"{where}: {field} '{val}' is not one of {sorted(allowed)}")


def _value_slugs(facet):
    """The value slugs declared in a facet, order preserved, blanks dropped."""
    values = facet.get("values")
    if not isinstance(values, list):
        return []
    return [v.get("value") for v in values
            if isinstance(v, dict) and v.get("value")]


def check_precedence(facet, where, errors):
    """Check 11: precedence is present iff exclusivity is exclusive, and when
    present it is a permutation of the facet's value slugs."""
    prec = facet.get("precedence")
    if facet.get("exclusivity") != "exclusive":
        if prec is not None:
            fail(errors, f"{where}: precedence must be set only when "
                         "exclusivity is exclusive")
        return
    # exclusivity is exclusive: precedence is required.
    if prec is None:
        fail(errors, f"{where}: precedence is required when exclusivity is "
                     "exclusive")
        return
    if not isinstance(prec, list):
        fail(errors, f"{where}: precedence must be a list of value slugs")
        return

    slugs = _value_slugs(facet)
    known = set(slugs)
    for s in prec:
        if s not in known:
            fail(errors, f"{where}: precedence lists '{s}', which is not a "
                         "value in this facet")
    # every value in the facet appears exactly once in precedence
    for s in slugs:
        n = prec.count(s)
        if n == 0:
            fail(errors, f"{where}: value '{s}' is missing from precedence")
        elif n > 1:
            fail(errors, f"{where}: value '{s}' appears {n} times in "
                         "precedence; expected exactly once")


def check_fallback(facet, where, errors):
    """Check 12: fallback_value is present iff unmatched is declared, and when
    present it names a value in the facet."""
    fb = facet.get("fallback_value")
    if facet.get("unmatched") != "declared":
        if fb is not None:
            fail(errors, f"{where}: fallback_value must be set only when "
                         "unmatched is declared")
        return
    # unmatched is declared: fallback_value is required.
    if fb is None:
        fail(errors, f"{where}: fallback_value is required when unmatched is "
                     "declared")
        return
    if fb not in set(_value_slugs(facet)):
        fail(errors, f"{where}: fallback_value '{fb}' is not a value in this "
                     "facet")


def check_no_placeholder(facet, where, errors):
    """Check 8: no scaffold placeholder survives anywhere in the facet.

    Scans every string leaf in the facet via the shared deep walk, so value
    labels, match patterns, and notes are covered as well as the named
    top-level fields. The blank template must not pass.
    """
    for kind, text in _walk(facet):
        if kind != "__str__":
            continue
        for token in PLACEHOLDER_TOKENS:
            if token in text:
                fail(errors, f"{where}: scaffold placeholder '{token}' survives "
                             f"in {text!r} — the blank template must be replaced "
                             "before validation passes")
                return


def check_match(match, where, errors):
    """Check 5: match uses only allowed keys, has one non-empty rule."""
    if not isinstance(match, dict):
        fail(errors, f"{where}: 'match' must be a mapping")
        return
    if not match:
        fail(errors, f"{where}: 'match' has no rule keys "
                     f"(need one of {sorted(ALLOWED_MATCH_KEYS)})")
        return

    any_nonempty = False
    for key, val in match.items():
        if key not in ALLOWED_MATCH_KEYS:
            fail(errors, f"{where}: match key '{key}' is not allowed "
                         f"(allowed: {sorted(ALLOWED_MATCH_KEYS)})")
            continue
        if not isinstance(val, list):
            fail(errors, f"{where}: match '{key}' must be a list")
            continue
        if val:
            any_nonempty = True
        if key == "stage_d_type_in":
            for t in val:
                if t not in STAGE_D_TYPES:
                    fail(errors, f"{where}: stage_d_type_in '{t}' is not a "
                                 f"Stage D type {sorted(STAGE_D_TYPES)}")
    if not any_nonempty:
        fail(errors, f"{where}: 'match' has no non-empty rule key")


def check_file_level(doc, errors):
    if "version" not in doc:
        fail(errors, "top-level: missing 'version'")
    if "frozen_at" not in doc:
        fail(errors, "top-level: missing 'frozen_at'")

    facets = doc.get("facets")
    if facets is None:
        fail(errors, "top-level: missing 'facets' key")
        return None
    if not isinstance(facets, list):
        fail(errors, "top-level: 'facets' must be a list")
        return None
    if len(facets) == 0:
        fail(errors, "top-level: 'facets' is empty — at least one is required")
        return None
    return facets


def check_facets(doc, errors):
    facets = check_file_level(doc, errors)
    if facets is None:
        return

    for i, f in enumerate(facets):
        where = f"facets[{i}]"
        if not isinstance(f, dict):
            fail(errors, f"{where}: not a mapping")
            continue

        for fld in REQUIRED_FACET_FIELDS:
            if fld not in f:
                fail(errors, f"{where}: missing '{fld}'")

        fid = f.get("id", "")
        where = f"facet '{fid or i}'"

        # 1. facet id scheme + uniqueness
        if fid:
            if not FACET_ID_SCHEME.match(fid):
                fail(errors, f"{where}: id does not match scheme "
                             f"{FACET_ID_SCHEME.pattern}")
            if [x.get("id") for x in facets if isinstance(x, dict)].count(fid) > 1:
                fail(errors, f"{where}: duplicate id")

        # 2. definition + applies_to non-empty
        for fld in ("definition", "applies_to"):
            v = f.get(fld)
            if not (isinstance(v, str) and v.strip()):
                fail(errors, f"{where}: {fld} is empty")

        # 6. inputs present + allowlist
        check_inputs(f, where, errors)

        # 7. exclusivity + unmatched present and in their allowed sets
        check_enum_field(f, "exclusivity", ALLOWED_EXCLUSIVITY, where, errors)
        check_enum_field(f, "unmatched", ALLOWED_UNMATCHED, where, errors)

        # 13. assignment (optional, defaults to pattern)
        assignment = f.get("assignment", "pattern")
        if "assignment" in f and assignment not in ALLOWED_ASSIGNMENT:
            fail(errors, f"{where}: assignment '{assignment}' is not one of "
                         f"{sorted(ALLOWED_ASSIGNMENT)}")

        # 11 + 12. conditional fields: precedence (exclusive) and
        # fallback_value (declared), each a valid reference into this facet
        check_precedence(f, where, errors)
        check_fallback(f, where, errors)

        # 8. no scaffold placeholder anywhere in the facet (deep walk)
        check_no_placeholder(f, where, errors)

        # 9 + 10. no category reference, no specific year, anywhere in facet
        check_no_category_or_year(f, where, errors)

        # 3 + 4 + 5. values
        values = f.get("values")
        if not isinstance(values, list) or len(values) == 0:
            fail(errors, f"{where}: needs at least one value "
                         f"(has {len(values) if isinstance(values, list) else 0})")
            continue

        fb_slug = f.get("fallback_value")  # None when unmatched is absent

        seen = set()
        for j, v in enumerate(values):
            vwhere = f"{where} value[{j}]"
            if not isinstance(v, dict):
                fail(errors, f"{vwhere}: not a mapping")
                continue

            slug = v.get("value", "")
            is_fallback = slug and slug == fb_slug

            # match is optional in two cases:
            #   - the value is the fallback_value (concepts come from the
            #     fallback mechanism, not from pattern matching);
            #   - the facet uses assignment: classifier (concepts come from
            #     an external classifier, not from pattern rules).
            match_optional = is_fallback or assignment == "classifier"
            required_here = [fld for fld in REQUIRED_VALUE_FIELDS
                            if not (fld == "match" and match_optional)]
            for fld in required_here:
                if fld not in v:
                    fail(errors, f"{vwhere}: missing '{fld}'")

            if slug:
                vwhere = f"{where} value '{slug}'"
                if not VALUE_SCHEME.match(slug):
                    fail(errors, f"{vwhere}: value slug does not match scheme "
                                 f"{VALUE_SCHEME.pattern}")
                if slug in seen:
                    fail(errors, f"{vwhere}: duplicate value slug in facet")
                seen.add(slug)

            defn = v.get("definition")
            if not (isinstance(defn, str) and defn.strip()):
                fail(errors, f"{vwhere}: definition is empty")

            if "match" in v:
                check_match(v.get("match"), vwhere, errors)


def validate_file(path, frozen=False):
    """Validate one facets.yaml. Returns a list of error strings.

    `frozen=True` adds one gate: top-level frozen_at must be set (not null).
    It does not relax or alter any other check.
    """
    errors = []
    path = Path(path)
    if not path.exists():
        return [f"{path} not found"]
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    check_facets(doc, errors)
    if frozen and doc.get("frozen_at") is None:
        fail(errors, "top-level: frozen_at is null but --frozen was requested")
    return errors


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    frozen = "--frozen" in argv
    positional = [a for a in argv if not a.startswith("-")]
    target = Path(positional[0]) if positional else FACETS
    errors = validate_file(target, frozen=frozen)

    try:
        n = len(yaml.safe_load(target.read_text()).get("facets") or [])
    except Exception:
        n = 0

    if errors:
        print(f"INVALID — {len(errors)} error(s), {n} facet(s) checked:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"VALID — {n} facet(s) checked, all schema checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
