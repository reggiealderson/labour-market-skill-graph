"""Validate taxonomy/categories.yaml against the Stage E schema.

Checks (all structural; no category content is generated here):
  1. every id is unique and matches the id scheme (ID_SCHEME);
  2. every category has a non-empty definition;
  3. every category has at least two exclusion tests;
  4. every excludes[].goes_to names a category id that exists;
  5. no label contains a Stage D type word (TYPE_WORDS);
  6. esco_hint is non-structural: no other code reads it.

Exit 0 = valid, 1 = one or more failures. Read-only. No API, no corpus.

Usage:  python3 tools/validate_categories.py
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ROOT / "taxonomy/categories.yaml"
SELF = Path(__file__).resolve()

# id scheme: cat_ + lower snake_case
ID_SCHEME = re.compile(r"^cat_[a-z0-9]+(_[a-z0-9]+)*$")

# Stage D type words forbidden in a label (per spec).
TYPE_WORDS = ["tool", "library", "language", "method", "practice", "knowledge"]

# Directories whose .py files must NOT reference esco_hint (check 6).
CODE_DIRS = ["pipeline", "scripts", "tools", "taxonomy"]

REQUIRED_FIELDS = ["id", "label", "definition", "includes", "excludes", "role"]
VALID_ROLES = {"load_bearing", "coverage"}


def fail(errors, msg):
    errors.append(msg)


def check_file_level(doc, errors):
    """File-level checks that run independently of the category loop."""
    if "version" not in doc:
        fail(errors, "top-level: missing 'version'")
    if "frozen_at" not in doc:
        fail(errors, "top-level: missing 'frozen_at'")

    cats = doc.get("categories")
    if cats is None:
        fail(errors, "top-level: missing 'categories' key")
        return None
    if not isinstance(cats, list):
        fail(errors, "top-level: 'categories' must be a list")
        return None
    if len(cats) == 0:
        fail(errors, "top-level: 'categories' is empty — at least one is required")
        return None
    return cats


def check_categories(doc, errors):
    cats = check_file_level(doc, errors)
    if cats is None:
        return

    ids = set()
    # first pass: collect ids for goes_to resolution
    for i, c in enumerate(cats):
        cid = c.get("id") if isinstance(c, dict) else None
        if cid:
            ids.add(cid)

    for i, c in enumerate(cats):
        where = f"categories[{i}]"
        if not isinstance(c, dict):
            fail(errors, f"{where}: not a mapping")
            continue

        # required fields present
        for fld in REQUIRED_FIELDS:
            if fld not in c:
                fail(errors, f"{where}: missing '{fld}'")

        cid = c.get("id", "")
        where = f"category '{cid or i}'"

        # 1. id scheme + uniqueness
        if cid:
            if not ID_SCHEME.match(cid):
                fail(errors, f"{where}: id does not match scheme "
                             f"{ID_SCHEME.pattern}")
            if list(x.get("id") for x in cats).count(cid) > 1:
                fail(errors, f"{where}: duplicate id")

        # 2. definition non-empty
        defn = c.get("definition")
        if not (isinstance(defn, str) and defn.strip()):
            fail(errors, f"{where}: definition is empty")

        # 5. label must not contain a Stage D type word
        label = c.get("label", "")
        if isinstance(label, str):
            low = label.lower()
            for w in TYPE_WORDS:
                if re.search(rf"\b{re.escape(w)}\b", low):
                    fail(errors, f"{where}: label contains Stage D type "
                                 f"word '{w}' ({label!r})")

        # role enum
        role = c.get("role")
        if role is not None and role not in VALID_ROLES:
            fail(errors, f"{where}: role '{role}' not in {sorted(VALID_ROLES)}")

        # 3 + 4. exclusion tests
        exc = c.get("excludes")
        if not isinstance(exc, list) or len(exc) < 2:
            fail(errors, f"{where}: needs at least 2 exclusion tests "
                         f"(has {len(exc) if isinstance(exc, list) else 0})")
            exc = exc if isinstance(exc, list) else []
        for j, e in enumerate(exc):
            if not isinstance(e, dict):
                fail(errors, f"{where}: excludes[{j}] not a mapping")
                continue
            if not e.get("concept_shape"):
                fail(errors, f"{where}: excludes[{j}] missing 'concept_shape'")
            goes_to = e.get("goes_to")
            if not goes_to:
                fail(errors, f"{where}: excludes[{j}] missing 'goes_to'")
            elif goes_to not in ids:
                fail(errors, f"{where}: excludes[{j}] goes_to '{goes_to}' "
                             f"is not a defined category id")


def check_esco_hint_unread(errors):
    """Check 6: no .py file except this validator references esco_hint."""
    token = "esco" + "_hint"  # split so this validator's own scan is not a hit
    for d in CODE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            if py.resolve() == SELF:
                continue
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if token in text:
                fail(errors, f"esco_hint is read by code: {py.relative_to(ROOT)} "
                             "(esco_hint must be non-structural)")


def validate_file(path, check_esco=True):
    """Validate one categories.yaml. Returns a list of error strings.

    `check_esco=True` also runs the repo-wide esco_hint source scan (check 6),
    which is independent of the file being validated. Tests pass False.
    """
    errors = []
    path = Path(path)
    if not path.exists():
        return [f"{path} not found"]
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        return [f"YAML parse error: {e}"]

    check_categories(doc, errors)
    if check_esco:
        check_esco_hint_unread(errors)
    return errors


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    target = Path(argv[0]) if argv else CATEGORIES
    errors = validate_file(target)

    try:
        n = len(yaml.safe_load(target.read_text()).get("categories") or [])
    except Exception:
        n = 0

    if errors:
        print(f"INVALID — {len(errors)} error(s), {n} categor(y/ies) checked:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"VALID — {n} categor(y/ies) checked, all schema checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
