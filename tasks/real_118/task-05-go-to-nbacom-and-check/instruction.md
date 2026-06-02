go to nba.com and check for Jayson Tatum's current 3-point status
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `source_url` (string) — Canonical nba.com URL where the stats were read.
- `player_name` (string) — Player name as shown on the page (should be 'Jayson Tatum').
- `season` (string) — Season label (e.g. '2024-25').
- `current_3pt_stats.attempts` (number) — 3PT attempts (season total or per-game).
- `current_3pt_stats.makes` (number) — 3PT makes (matching the attempts basis).
- `current_3pt_stats.pct` (number 0-1 or 0-100) — 3PT percentage.
- `current_3pt_stats.basis` (string) — 'season-total' OR 'per-game'.
- `data_freshness` (string) — When you fetched it (ISO date or 'as of YYYY-MM-DD').

**Optional but graded if present:**

- `games_played` (integer) — Season games played.
- `last_game_3pt` (object) — {date, makes, attempts} for last game played.
- `career_context` (string) — Where this season ranks vs his career average.
- `milestone_progress` (string) — All-time 3PT record progress, if relevant.
- `alternate_sources_checked` (list[string]) — Other source URLs you double-checked (e.g. basketball-reference).

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "source_url": "https://www.nba.com/player/1628369/jayson-tatum",
  "player_name": "Jayson Tatum",
  "season": "2024-25",
  "current_3pt_stats": {
    "attempts": 8.6,
    "makes": 3.0,
    "pct": 0.349,
    "basis": "per-game"
  },
  "data_freshness": "as of 2026-06-01"
}
```
