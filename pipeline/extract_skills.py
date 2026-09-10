"""
LLM skill extraction using Claude Sonnet 4.6.

Processes each job record individually, extracting verbatim skill strings.
Checkpoint-safe: already-processed records are skipped on restart.

Input:  data/extracted/{year}/{occupation}.jsonl
Output: data/skills/{year}/{occupation}.jsonl
        One JSON line per record: {"id": "...", "skills": [...]}
        Records with parse errors: {"id": "...", "skills": [], "extraction_error": true}
"""

import anthropic
import json
import os
import re
import time
from pathlib import Path

# Load .env from project root
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

MODEL         = "claude-haiku-4-5-20251001"
EXTRACTED_DIR = Path("data/extracted")
SKILLS_DIR    = Path("data/skills")
DELAY         = 0.05   # seconds between API calls

PROMPT_TEMPLATE = """\
Extract all skills from the job posting below.

A SKILL is a specific, learnable capability that a candidate is expected to have or develop — \
whether listed as required or as a nice-to-have.
Extract skills verbatim — copy the exact words from the posting. Do not paraphrase, \
summarise, or infer skills that are not explicitly named. Do not synthesise skill names \
by combining or paraphrasing words from the description — every extracted item must \
appear as a recognisable phrase in the source text.

INCLUDE:
- Technical tools and software (e.g. "Python", "Tableau", "Salesforce", "AWS", "Docker")
- Technical disciplines and methods (e.g. "machine learning", "financial modeling", "RESTful API design")
- Frameworks and methodologies (e.g. "Agile", "Scrum", "Six Sigma", "GAAP")
- Domain knowledge stated as a requirement (e.g. "HIPAA compliance", "options trading", "IFRS accounting")
- Soft skills stated explicitly as requirements (e.g. "communication skills", "leadership", "stakeholder management")
- Professional certifications (e.g. "CPA", "PMP", "AWS Certified Solutions Architect", "RN license")

DO NOT INCLUDE:
- Job responsibilities or duties ("manage a team of engineers", "prepare quarterly reports")
- Years of experience ("5+ years experience", "3-5 years in marketing")
- Academic degrees ("Bachelor's degree", "MBA", "Master's in Nursing") — professional \
certifications like CPA or PMP ARE skills; academic degrees are not
- Company or industry descriptions ("Fortune 500 company", "fast-growing startup", "healthcare industry")
- Compensation or benefits ("competitive salary", "health insurance", "401k")
- Work arrangement or location ("remote work", "travel up to 20%", "based in New York")
- Vague character traits not framed as a job requirement ("passionate about technology", "entrepreneurial mindset")

HARD NEGATIVES — these look like skills but are not, or violate the verbatim rule:
- "Experience with Python" → extract "Python", not "experience with Python"
- "Knowledge of GAAP" → extract "GAAP", not "knowledge of GAAP"
- "Ability to manage multiple projects" → responsibility, not a skill — exclude
- "Strong attention to detail" → vague character trait — exclude
- "Bachelor's degree in Computer Science" → exclude (degree, not a skill)
- "Work in a fast-paced environment" → work condition — exclude
- "MBA" → academic degree — exclude even when listed as a preferred qualification
- "communicate effectively" → vague character trait — exclude; "communication skills" \
stated as a named requirement is acceptable
- "Adobe Creative Suite" → extract "Adobe Creative Suite" — do NOT infer or list "Photoshop", \
"Illustrator", "InDesign" unless those names appear explicitly elsewhere in the posting
- "common algorithms and data structures" → extract as one phrase, do not split into \
"algorithms" and "data structures" separately

Return a JSON array of skill strings, verbatim from the text. No duplicates. No explanation. \
If no skills are found, return [].

Job posting:
{description}\
"""


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def parse_skills(text: str) -> list[str]:
    """Parse a JSON array from model output, tolerating markdown fences."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    parsed = json.loads(text.strip())
    if isinstance(parsed, list):
        return [s for s in parsed if isinstance(s, str)]
    return []


def extract_skills(client: anthropic.Anthropic, description: str) -> tuple[list[str], bool]:
    """Return (skills, had_error)."""
    prompt = PROMPT_TEMPLATE.format(description=description)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text
    try:
        return parse_skills(raw), False
    except Exception:
        return [], True


def main():
    client = anthropic.Anthropic()
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    for year in ("2021", "2026"):
        year_in  = EXTRACTED_DIR / year
        year_out = SKILLS_DIR / year
        year_out.mkdir(parents=True, exist_ok=True)

        for jsonl_path in sorted(year_in.glob("*.jsonl")):
            occ      = jsonl_path.stem
            out_path = year_out / f"{occ}.jsonl"

            records  = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            done_ids = load_done_ids(out_path)
            pending  = [r for r in records if r["id"] not in done_ids]

            print(f"\n{year}/{occ}: {len(done_ids)} already done, {len(pending)} to process")
            if not pending:
                continue

            errors = 0
            with open(out_path, "a", encoding="utf-8") as f:
                for i, rec in enumerate(pending, 1):
                    skills, had_error = extract_skills(client, rec["description"])
                    if had_error:
                        errors += 1
                    out = {"id": rec["id"], "skills": skills}
                    if had_error:
                        out["extraction_error"] = True
                    f.write(json.dumps(out) + "\n")
                    f.flush()

                    if i % 100 == 0 or i == len(pending):
                        print(f"  {i}/{len(pending)} done  ({errors} errors so far)")

                    time.sleep(DELAY)

            print(f"  {year}/{occ} complete. Errors: {errors}/{len(pending)}")

    print("\nAll extractions complete.")


if __name__ == "__main__":
    main()
