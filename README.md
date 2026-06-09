# tmp-htmls

Static HTML hosting for the Agentic Judge presentation decks.

Pages are deployed by the `.github/workflows/pages.yml` workflow on every push to `main`.

After the first push, enable Pages once: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Files

- [`index.html`](./index.html) — landing page linking to v1 and v2
- [`v1.html`](./v1.html) — original 3-judge design (single pass, substring-match faithfulness)
- [`v2.html`](./v2.html) — Effectiveness as main judge with LLM-agentic Faithfulness sub-agent and pending-question loop
