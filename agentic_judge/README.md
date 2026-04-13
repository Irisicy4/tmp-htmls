# Agentic Judge

Second-pass verification layer for EvolveBench. Re-navigates source URLs using a Playwright + GPT-4o vision browser agent to verify tasks the static LLM judge (`test.py`) marked as passed. The gap between static pass rate and agentic verification rate is a key finding in the paper.

## How it works

1. `run.py` reads result JSONs from `results/modal-updated/` (or `results/synthetic/`)
2. Filters to tasks where `eval.passed == true`
3. Stratified-samples by domain (from `task.yaml` comment line)
4. For each task: extracts URLs from the task instruction (slots 1–3, ground truth), then from `task_result` (fills remaining capacity)
5. Navigates each URL with headless Chromium via Playwright, takes screenshots
6. Selects domain-specific criteria from a single `prompts/verify_prompt.txt` (6 archetypes: shopping, research_data, travel_planning, software_tech, content_media, general)
7. Passes screenshots + page titles to GPT-4o — returns `{verified, finding, confidence}` in `<Answer>` tags
8. Writes one `*_agentic_verification.json` per task to `results/original/` or `results/synthetic/`

## Requirements

- `OPENAI_API_KEY` environment variable
- Playwright installed: `playwright install chromium`
- Python dependencies already in `pyproject.toml` (openai, playwright, pyyaml)

## Running verification

> **Note:** Run all commands from the repo root directory.

```bash
# Verify 30 tasks sampled across both datasets (default)
python -m agentic_judge.run --dataset both --sample-size 30

# Verify only the original dataset
python -m agentic_judge.run --dataset original --sample-size 15

# Verify specific tasks by name
python -m agentic_judge.run --task-names task-01-im-looking-for-backpack-under task-05-go-to-nbacom-and-check

# Preview which tasks would be verified without running
python -m agentic_judge.run --dataset original --sample-size 10 --dry-run
```

## Analyzing results

```bash
# Compute agreement, catch rate, and unverifiable breakdown
python -m agentic_judge.analysis.agreement

# Print paper-ready table and headline numbers
python -m agentic_judge.analysis.report
```

## Output files

Each verified task produces one file in `results/original/` or `results/synthetic/`:

```json
{
  "task_name": "task-01-im-looking-for-backpack-under",
  "category": "Shopping",
  "dataset": "original",
  "static_judge_passed": true,
  "static_judge_score": 3.45,
  "static_judge_dimensions": {"constraint_satisfaction": 5, "result_specificity": 3},
  "agentic_verified": true,
  "agentic_finding": "Amazon product page loaded and lists backpacks in the correct price range.",
  "null_reason": null,
  "verification_method": "browser_navigation",
  "deliverable_url": "https://www.amazon.com/dp/B09YRC9Y3G",
  "time_seconds": 14.2,
  "timestamp": "2026-04-04T10:22:00+00:00"
}
```

### `verification_method` values

| Value | Meaning |
|---|---|
| `browser_navigation` | Playwright loaded ≥1 page successfully AND took a screenshot |
| `url_check` | Playwright attempted but all URLs errored (timeout/SSL/404); no screenshot |
| `unverifiable` | No URLs found in instruction or task_result; Playwright never launched |

### `null_reason` values (when `agentic_verified` is null)

| Value | Meaning |
|---|---|
| `no_urls` | No URLs found in instruction or task_result |
| `navigation_error` | All Playwright navigations failed |
| `gpt_uncertain` | GPT-4o could not determine correctness from page content |
| `null` | Not applicable — `agentic_verified` is true or false |

`agreement_report.json` is written to `results/agreement_report.json` and includes `unverifiable_by_reason` counts for all three reasons — include this in the paper to pre-empt reviewer questions.

## Domain-to-archetype prompt mapping

Tasks are judged using prompts tailored to their domain:

| Archetype | Domains | Prompt file |
|---|---|---|
| Shopping | Shopping | `prompts/verify_prompt.txt` (shopping guidance) |
| Research & Data | Data & ML Engineering, Finance & Economics, Legal, Insurance & Actuarial, Medical & Clinical & Bio, Marketing & Analytics | `prompts/verify_prompt.txt` (research_data guidance) |
| Travel & Planning | Travel & Planning, Real Estate, Logistics & Supply Chain | `prompts/verify_prompt.txt` (travel_planning guidance) |
| Software Engineering | Software Engineering | `prompts/verify_prompt.txt` (software_tech guidance) |
| Content & Media | (Self) Media, Design, HR & Recruiting | `prompts/verify_prompt.txt` (content_media guidance) |
| General (fallback) | Daily Activities, unknown domains | `prompts/verify_prompt.txt` (general guidance) |

## Integration with static judge

Static judge pipeline: `modal_runner.py` → `test.py` per task → result JSON with `eval.passed`.

Agentic judge pipeline: `run.py` reads those result JSONs → navigates → writes `*_agentic_verification.json` → `agreement.py` compares the two.

Neither pipeline modifies the other's files.
