Help me choose the perfect birthday gift for my 28-year-old male friend. He likes gaming, computers and gadgets. The budget should be 10,000-15,000 yen, and it should be something you can buy on Amazon. Pick 3 different options and add them to your shopping cart.
<!-- agentic_judge.grounded addendum -->

---

## Required output format

After completing the task, output your result as a JSON block delimited by `=== JSON RESULT ===` on its own line, the JSON, and `=== END JSON ===` on its own line. Include EVERY required field below; optional fields are rewarded when present.

**Required fields:**

- `platform` (string) — Should be 'amazon.co.jp'.
- `recipient_profile` (object) — Snapshot of the recipient used.
- `recipient_profile.age` (integer) — 
- `recipient_profile.interests` (list[string]) — 
- `options` (list[object]) — Exactly 3 chosen items.
- `options[].name` (string) — Product name.
- `options[].url` (string) — Amazon.co.jp product URL.
- `options[].asin` (string) — Amazon ASIN.
- `options[].price_jpy` (integer) — Price in JPY (yen, no symbol).
- `options[].category` (string) — Category fit ('gaming'|'computers'|'gadgets').
- `options[].fit_notes` (string) — Why it fits the profile.
- `options[].cart_action` (string) — 'added'|'attempted'|'not_added' — be honest.
- `cart_url_or_screenshot` (string) — Cart URL or path to cart screenshot.

**Optional but graded if present:**

- `options[].rating` (number 1-5) — Review rating.
- `options[].review_count` (integer) — 
- `options[].prime_eligible` (boolean) — 
- `options[].delivery_estimate` (string) — 
- `budget_used_jpy` (integer) — Sum of the 3 prices.

**Anti-fabrication note:** the grader fetches the URLs you cite and verifies the figures and quotes against the live page. Be precise; do not paraphrase numeric facts.

**Example shape (values illustrative, not literal):**

```json
{
  "platform": "amazon.co.jp",
  "recipient_profile": {
    "age": 28,
    "interests": [
      "gaming",
      "computers",
      "gadgets"
    ]
  },
  "options": [
    {
      "name": "...",
      "url": "https://www.amazon.co.jp/dp/B0...",
      "asin": "B0XXXX",
      "price_jpy": 12800,
      "category": "gaming",
      "fit_notes": "...",
      "cart_action": "added"
    }
  ],
  "cart_url_or_screenshot": "https://www.amazon.co.jp/gp/cart/view.html"
}
```
