"""agentic_judge.grounded — pre-run constraint-grounded judging.

Sibling to the existing post-run URL verifier under agentic_judge/.  This
module judges a task by composing three layers:

  1. Hard-constraint predicates (Python-checkable) — typed assertions on the
     agent's over-reported summary JSON.
  2. Faithfulness judge — fetches the URLs the agent cites and verifies the
     claims in prose match what's actually on those pages.
  3. Effectiveness judge — LLM-as-judge on the soft rubric (the existing
     gpt-4o-style anchors) for quality dimensions.

Anti-cheat principle: each task's `SUMMARY_SCHEMA` asks the agent to emit
MORE fields than `HARD_CONSTRAINTS` actually validates.  The agent doesn't
know which subset will be enforced, so the cheapest cheating strategy
(fabricating the few checked numbers) is dominated by genuinely doing the
task.

See `agentic_judge/grounded/README.md` for the full contract.
"""
__version__ = "0.1.0"
