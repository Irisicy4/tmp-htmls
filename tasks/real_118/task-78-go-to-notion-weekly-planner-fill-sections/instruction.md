Go to https://www.notion.so/templates/weekly-planner and open the Weekly Planner template. Fill in all the empty sections of the planner with appropriate placeholder data — including goals, daily tasks, notes, and any other incomplete fields — to make the page fully complete.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `template_url` (string) — Should be the Notion weekly planner template URL.
- `sections_filled` (list[object]) — Each section you filled in.
- `sections_filled[].name` (string) — Section label as shown on the planner.
- `sections_filled[].content_preview` (string) — First ~120 chars of the content you filled in.
- `sections_filled[].placeholder_or_real` (string) — 'placeholder' (fake but realistic) or 'real'.
- `completion_evidence` (string) — URL to your filled copy OR path to a screenshot.

**Optional but graded if present:**

- `sections_total` (integer) — Sections found on the template, filled or not.
- `sections_skipped` (list[string]) — Names of sections you couldn't access/fill.
- `editing_method` (string) — How you edited (duplicate template + edit, fork, etc.).
- `notion_account_used` (boolean) — Did you sign into a Notion account?

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "template_url": "https://www.notion.so/templates/weekly-planner",
  "sections_filled": [
    {
      "name": "Goals",
      "content_preview": "1. Ship Q3 OKR ...",
      "placeholder_or_real": "placeholder"
    },
    {
      "name": "Monday tasks",
      "content_preview": "Standup @ 9, ...",
      "placeholder_or_real": "placeholder"
    }
  ],
  "completion_evidence": "/output/notion_filled.png"
}
```
