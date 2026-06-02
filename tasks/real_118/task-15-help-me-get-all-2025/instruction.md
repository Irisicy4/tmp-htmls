Help me get all 2025 publication info from https://m.douban.com/subject_collection/ECNA7Y7GA and save as an Excel file.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `source_url` (string) — Exact Douban subject_collection URL used.
- `output_file.path` (string) — Absolute path to the .xlsx file you created.
- `output_file.format` (string) — 'xlsx' or 'xls'.
- `output_file.sheet_name` (string) — Sheet name used.
- `output_file.row_count` (integer) — Number of data rows (excluding header).
- `output_file.column_headers` (list[string]) — Column headers in order.
- `publications` (list[object]) — All 2025 publications, structured.
- `publications[].title` (string) — Title.
- `publications[].year` (integer) — Publication year — must be 2025.

**Optional but graded if present:**

- `publications[].author` (string) — Author.
- `publications[].rating` (number 0-10) — Douban rating.
- `publications[].genre` (string) — 
- `publications[].url` (string) — Per-item URL on Douban.
- `filter_methodology` (string) — How you confirmed only 2025 items were kept.
- `total_items_seen` (integer) — Items seen on the page before year-filtering.

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "source_url": "https://m.douban.com/subject_collection/ECNA7Y7GA",
  "output_file": {
    "path": "/output/douban_2025.xlsx",
    "format": "xlsx",
    "sheet_name": "publications",
    "row_count": 14,
    "column_headers": [
      "title",
      "year",
      "author",
      "rating"
    ]
  },
  "publications": [
    {
      "title": "...",
      "year": 2025,
      "author": "..."
    }
  ]
}
```
