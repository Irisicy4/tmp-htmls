Please compose an email to David Sullivan (dgs@bu.edu). Tell him that the one-day basic Java class he requested last time is available on either November 29th or November 30th. Since the weather has gotten colder, write politely and include a reminder to be careful not to catch a cold. Stop just before pressing the send button — do not actually send it.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `recipient` (string) — Recipient email address (should be dgs@bu.edu).
- `subject` (string) — Email subject line.
- `body` (string) — Full email body text.
- `dates_mentioned` (list[string]) — Dates offered in the body (Nov 29 / Nov 30).
- `includes_cold_warning` (boolean) — True if you included the cold-weather caution per the brief.
- `sent` (boolean) — Did you click send? (Should be false — task says stop before sending.)

**Optional but graded if present:**

- `salutation` (string) — Opening line.
- `signature` (string) — Sign-off line.
- `draft_path` (string) — Path where the draft was saved, if anywhere.

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "recipient": "dgs@bu.edu",
  "subject": "Java one-day class \u2014 Nov 29 or 30 availability",
  "body": "Dear Mr. Sullivan, ...",
  "dates_mentioned": [
    "November 29",
    "November 30"
  ],
  "includes_cold_warning": true,
  "sent": false
}
```
