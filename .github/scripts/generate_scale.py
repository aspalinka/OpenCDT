"""Generate a clinical scale JSON from a medical paper URL.

Usage:
    # Validate URL only
    uv run python .github/scripts/generate_scale.py --mode validate \
        --url "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456/" \
        --allowed-domains .github/config/allowed_domains.json

    # Full generation pipeline
    ANTHROPIC_API_KEY=sk-ant-... uv run python .github/scripts/generate_scale.py \
        --mode generate --url "https://..." --issue-number 1 \
        --examples scales/cha2ds2_vasc.json scales/wells_dvt.json \
        --output-dir scales
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

SUSPICIOUS_PATTERNS = re.compile(r"[@]|\.\.\/|%00")

_USER_AGENT = "Mozilla/5.0 (compatible; OpenCDT-bot/1.0)"


def extract_url(text: str) -> str | None:
    """Return the first HTTPS URL found in *text*, or None."""
    match = re.search(r"https://[^\s<>\"')\]]+", text)
    return match.group(0) if match else None


def validate_url(url: str, allowed_domains: list[str]) -> tuple[bool, str]:
    """Check that *url* is HTTPS, on an allowed domain, and not suspicious."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return False, "Only HTTPS URLs are allowed"
    if SUSPICIOUS_PATTERNS.search(url):
        return False, "URL contains suspicious characters"
    domain = parsed.hostname or ""
    if not any(domain == d or domain.endswith(f".{d}") for d in allowed_domains):
        return False, f"Domain '{domain}' is not in the allowlist: {allowed_domains}"
    return True, "OK"


# ---------------------------------------------------------------------------
# Content fetching
# ---------------------------------------------------------------------------


def fetch_pmc(url: str) -> str:
    """Fetch full text from a PMC article via E-utilities."""
    import requests
    from bs4 import BeautifulSoup

    match = re.search(r"PMC(\d+)", url)
    if not match:
        raise ValueError(f"Could not extract PMC ID from URL: {url}")
    pmc_id = match.group(1)
    api_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pmc&id={pmc_id}&rettype=xml"
    )
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml-xml")
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)


def fetch_pdf(url: str) -> bytes:
    """Download a PDF and return raw bytes."""
    import requests

    resp = requests.get(
        url, timeout=60, headers={"User-Agent": _USER_AGENT}
    )
    resp.raise_for_status()
    if b"%PDF" not in resp.content[:10]:
        raise ValueError("Response does not appear to be a valid PDF")
    return resp.content


def fetch_html(url: str) -> str:
    """Download an HTML page and extract readable text."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(
        url, timeout=30, headers={"User-Agent": _USER_AGENT}
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _is_pmc_url(url: str) -> bool:
    """Check if URL points to a PMC article (any format, including PDF)."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return (
        "pmc" in hostname and "nih.gov" in hostname
    ) or "/pmc/articles/PMC" in url


def _is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF based on extension or path pattern."""
    lower = url.lower()
    if lower.endswith(".pdf"):
        return True
    # Common publisher pattern: /pdf/ in the path (e.g. nejm.org/doi/pdf/...)
    path = urlparse(lower).path
    return "/pdf/" in path or path.endswith("/pdf")


def fetch_content(url: str) -> str | bytes:
    """Dispatcher: choose the best fetching strategy based on URL."""
    if _is_pmc_url(url):
        return fetch_pmc(url)
    if _is_pdf_url(url):
        return fetch_pdf(url)
    return fetch_html(url)


SYSTEM_PROMPT = """\
You are a clinical informatics expert. Your task is to extract a clinical \
scoring scale from the provided medical paper and produce a JSON object that \
conforms exactly to the JSON Schema provided in the user message. 
This is a high-precision information extraction task. 
Guessing, normalization, or completion of missing information is strictly forbidden.

**Core extraction principles:**
1. Extract ONLY information explicitly stated in the paper.
2. If information is not explicitly stated, ambiguous, or unclear, return null for that field.
3. Do NOT assume typical scoring structures (e.g., Yes=1, No=0).
4. Do NOT infer missing coefficients or thresholds.
5. Do NOT reconstruct logic unless explicitly described.
6. Do NOT search externally.
7. Do NOT "fix", normalize, or improve the scale.
8. Prioritize the main body of the article (e.g., Methods, Results, figures, and clinical recommendations) \
over information presented only in the abstract.

**Required vs nullable fields:**
- ALWAYS required (never null): `name`, `items` (these define the scale — if you cannot extract them, the extraction has failed)
- Nullable if not explicitly stated: `formula`, `min_score`, `max_score`, `interpretation`, \
`description`, `purpose`, `when_to_use`, `when_not_to_use`, `category`, `notes`, `full_name`
- Default to empty list if not applicable: `tags`, `constraints`, `references`

**Item rules:**
- All item labels MUST be snake_case matching regex: ^[a-z][a-z0-9_]*$
- Labels must reflect item meaning but must not invent new variables.
- Do NOT create items not explicitly defined in the paper.
- If the paper describes categories (e.g., age 65–74, ≥75), create separate items only if they are explicitly separate scoring components.
- If scoring structure is unclear, return null value.

**Point assignment rules:**
- Extract numeric point values exactly as written.
- If the paper provides a scoring table, use it exactly.
- Do NOT assume binary scoring unless explicitly stated.
- If point values are not explicitly provided, return null value.
- If coefficients are provided instead of points, extract them exactly.
- Do NOT round or transform values.

**Formula rules:**
- Construct the formula ONLY if explicitly derivable from stated point assignments.
- The formula must use only:
    - item labels
    - numeric literals
    - operators (+, -, *)
- No function calls.
- No Python builtins.
- No conditional expressions.
- If the paper describes the score as "sum of components", use additive form.
- If derivation is unclear, return null value for formula.

Every label used in formula MUST exist in items.
Every item SHOULD appear in formula unless explicitly stated otherwise.

**Min/max and constraints rules:**
- Compute min_score and max_score ONLY if strictly computable from explicitly stated scoring rules.
- If mutual exclusivity or constraints prevent exact computation, return null value.
- Do NOT assume theoretical maximum unless explicitly stated.
- If the paper explicitly states that certain items are mutually exclusive, list them in `constraints`.
- Do NOT infer mutual exclusivity unless clearly described.

**Interpretation rules:**
- Extract interpretation ranges EXACTLY as written.
- Preserve bin structure exactly (e.g., "1–2", "3", "4–5").
- Do NOT split multi-score bins into single-score entries.
- Do NOT expand bins to cover missing values.
- Do NOT enforce full coverage if the paper does not provide it.
- Do NOT merge or simplify categories.
- If interpretation text is vague, reproduce it exactly.
- If no interpretation is provided, return null value.
- If interpretation structure cannot be safely represented numerically,
return null value for interpretation.

**Critical interpretation rules:**
- Interpretation must prioritize CLINICAL IMPLICATIONS / MANAGEMENT RECOMMENDATIONS over risk estimates.
- If the paper provides both (a) risk/probability by score and (b) clinical actions (e.g., outpatient vs inpatient vs ICU), then:
  - Use the clinical action wording as the primary interpretation label.
  - Include risk estimates only as secondary information (e.g., in a description/notes field, or appended after the action).
- If only risk estimates are provided and no clinical implications are stated, then use risk-based interpretation.
- Do NOT convert risk estimates into clinical actions unless explicitly stated.

**References:**
- Extract citation exactly as written in the paper.
- Include PMID only if explicitly present.
- Do NOT search externally.
- If unavailable, return null value.

Return ONLY the JSON object. No markdown fences, no commentary."""


def get_json_schema() -> str:
    """Generate JSON Schema from the ClinicalScale Pydantic model."""
    from opencdt.models import ClinicalScale

    schema = ClinicalScale.model_json_schema()
    return json.dumps(schema, indent=2)


def build_user_prompt(
    paper_content: str,
    example_scales: list[str],
    json_schema: str,
) -> str:
    """Build the user message with schema, examples, and paper content."""
    parts = [
        "## JSON Schema\n\n"
        "The output MUST conform to this JSON Schema:\n\n"
        f"```json\n{json_schema}\n```\n",
        "\n## Example scales\n\n"
        "Here are two valid scale definitions for reference:\n",
    ]
    for i, example in enumerate(example_scales, 1):
        parts.append(f"### Example {i}\n```json\n{example}\n```\n")
    parts.append(
        "\n## Paper content\n\n"
        f"{paper_content}\n\n"
        "Extract the clinical scoring scale from this paper and return ONLY "
        "the JSON object."
    )
    return "\n".join(parts)


def generate_scale(
    content: str | bytes,
    example_scales: list[str],
    url: str,
) -> dict:
    """Call Claude API to generate a scale JSON from paper content."""
    import anthropic

    client = anthropic.Anthropic()
    json_schema = get_json_schema()

    if isinstance(content, bytes):
        # PDF: use Claude's native document support
        user_content: list[dict] = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(content).decode(),
                },
            },
            {
                "type": "text",
                "text": build_user_prompt(
                    "[PDF document provided above]", example_scales, json_schema
                ),
            },
        ]
    else:
        user_content = [
            {
                "type": "text",
                "text": build_user_prompt(content, example_scales, json_schema),
            }
        ]

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)

    return json.loads(raw)


def validate_scale(scale_dict: dict) -> None:
    """Validate generated JSON by loading through ClinicalScale."""
    # Import here so the script can also be used for URL validation without
    # needing all project dependencies installed.
    from opencdt.models import ClinicalScale

    ClinicalScale(**scale_dict)


def commit_and_push(
    scale_dict: dict,
    output_dir: str,
    issue_number: int,
) -> str:
    """Write scale JSON, commit, and push. Returns branch name."""
    name = scale_dict.get("name", "unknown")
    sanitized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    filename = f"{sanitized}.json"
    filepath = Path(output_dir) / filename

    run_id = os.environ.get("GITHUB_RUN_ID", "")
    suffix = f"-{run_id}" if run_id else ""
    branch = f"add-scale-{sanitized}{suffix}"

    filepath.write_text(json.dumps(scale_dict, indent=2, ensure_ascii=False) + "\n")

    def run(cmd: list[str]) -> str:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Command {cmd} failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result.stdout.strip()

    run(["git", "config", "user.name", "OpenCDT"])
    run(["git", "config", "user.email", "OpenCDT@users.noreply.github.com"])
    run(["git", "checkout", "-b", branch])
    run(["git", "add", str(filepath)])
    run(
        [
            "git",
            "commit",
            "-m",
            f"Add {name} scale\n\nGenerated from issue #{issue_number}",
        ]
    )
    run(["git", "push", "-u", "origin", branch])

    return branch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clinical scale from paper")
    parser.add_argument(
        "--mode",
        choices=["validate", "generate"],
        required=True,
        help="'validate' checks URL only; 'generate' runs the full pipeline",
    )
    parser.add_argument("--url", required=True, help="Paper URL")
    parser.add_argument("--issue-number", type=int, help="GitHub issue number")
    parser.add_argument(
        "--examples", nargs="*", default=[], help="Paths to example scale JSON files"
    )
    parser.add_argument(
        "--output-dir", default="scales", help="Directory for generated scale"
    )
    parser.add_argument(
        "--allowed-domains",
        default=".github/config/allowed_domains.json",
        help="Path to allowed domains JSON",
    )
    args = parser.parse_args()

    allowed = json.loads(Path(args.allowed_domains).read_text())

    ok, msg = validate_url(args.url, allowed)
    if not ok:
        print(f"::error::URL validation failed: {msg}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "validate":
        print(f"URL is valid: {args.url}")
        return

    if not args.issue_number:
        print("::error::--issue-number is required for generate mode", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching content from {args.url}...")
    content = fetch_content(args.url)

    example_scales = [Path(p).read_text() for p in args.examples]

    print("Generating scale via Claude API...")
    scale_dict = generate_scale(content, example_scales, args.url)

    print("Validating generated scale...")
    validate_scale(scale_dict)
    print("Validation passed!")

    print("Committing and pushing...")
    branch = commit_and_push(scale_dict, args.output_dir, args.issue_number)
    print(f"Pushed branch: {branch}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"branch={branch}\n")
            f.write(f"scale_name={scale_dict.get('name', 'unknown')}\n")
            f.write(f"scale_description={scale_dict.get('description', '')}\n")


if __name__ == "__main__":
    main()
