I bought a crate of Clemenules clementines and they all had seeds—shouldn't this variety be seedless? Can someone confirm whether Clemenules are normally seedless?
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `normally_seedless` (boolean) — True if Clemenules clementines are normally seedless under standard agronomic conditions.
- `cross_pollination_explanation` (string) — Brief explanation of how seeds end up in normally-seedless varieties (cross-pollination from compatible cultivars).
- `source_url` (string) — Authoritative source URL (university extension, FAO, USDA, peer-reviewed paper, etc.).
- `evidence_quote` (string) — Verbatim ≤30-word quote from the source supporting the claim.

**Optional but graded if present:**

- `alternative_sources` (list[string]) — Additional source URLs you cross-checked.
- `conditions_for_seeds` (list[string]) — Specific conditions that cause seed development (e.g. proximity to Murcott, Page, etc.).

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "normally_seedless": true,
  "cross_pollination_explanation": "Clemenules are parthenocarpic; seeds form when bees move pollen from a compatible nearby mandarin.",
  "source_url": "https://www.mdpi.com/2073-4395/11/10/2023",
  "evidence_quote": "Clemenules is normally seedless under isolation; seed numbers rise sharply when planted near compatible varieties."
}
```
