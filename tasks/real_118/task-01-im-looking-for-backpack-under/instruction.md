im looking for a backpack under $75 that has all the features of this one: https://www.amazon.com/dp/B09YRC9Y3G please do some research and find 3-5 optionsand summarize their key features and prices, comparing them to the original.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `reference_product.url` (string) — Canonical Amazon URL of the original backpack.
- `reference_product.name` (string) — Product name as shown on the page.
- `reference_product.features` (list[string]) — Key features pulled from the original listing.
- `options` (list[object]) — 3-5 alternative backpacks.
- `options[].name` (string) — Product name.
- `options[].url` (string) — Canonical product URL (Amazon, manufacturer, etc.).
- `options[].price` (number (USD)) — Selling price in USD.
- `options[].features` (list[string]) — 3+ feature bullets.
- `options[].comparison_notes` (string) — How it compares to the original (pros and cons).

**Optional but graded if present:**

- `options[].asin` (string) — Amazon ASIN if applicable.
- `options[].rating` (number 1-5) — Review rating.
- `options[].review_count` (integer) — Number of reviews.
- `options[].verified_in_stock` (boolean) — True if you saw stock indicator.
- `reference_product.price_estimate` (number (USD)) — Your estimate of the original's MSRP.
- `recommendation` (string) — Your top pick and why.

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "reference_product": {
    "url": "https://www.amazon.com/dp/B09YRC9Y3G",
    "name": "...",
    "features": [
      "...",
      "..."
    ]
  },
  "options": [
    {
      "name": "MATEIN ...",
      "url": "https://www.amazon.com/dp/B07...",
      "price": 64.99,
      "features": [
        "40L expandable",
        "USB port"
      ],
      "comparison_notes": "..."
    }
  ]
}
```
