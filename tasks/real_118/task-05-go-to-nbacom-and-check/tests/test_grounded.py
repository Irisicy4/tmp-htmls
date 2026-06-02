"""Grounded (deterministic) judging spec for task-05-go-to-nbacom-and-check.

Lives alongside the existing LLM-as-judge `test_task.py` — same `tests/`
folder, two independent grading layers:

  * test_task.py     — LLM-as-judge (Harbor's authoritative verdict).
                       Source of truth for the soft rubric.
  * test_grounded.py — typed hard-constraint predicates + URL faithfulness
                       verification (this file).  Adds machine-checkable
                       gating; does NOT redefine the soft rubric.

The soft-rubric data (DIMENSIONS / DIMENSION_WEIGHTS / TASK_RUBRIC) is
**inherited from the sibling test_task.py** at import time — if you tune
the LLM judge's anchors, this file picks up the change automatically.

Only the constraint-specific data — SUMMARY_SCHEMA, HARD_CONSTRAINTS,
FAITHFULNESS_CHECKS — is defined per task in this file.

Call via:

  spec = importlib.util.spec_from_file_location(
      "test_grounded", "tasks/real_118/task-05-go-to-nbacom-and-check/tests/test_grounded.py"
  )
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  m.grade(result)        # hard + faithfulness, no LLM
  m.grade_with_llm(result)  # full mix incl. effectiveness LLM call
"""
import importlib.util as _ilu
import pathlib as _pth
import sys as _sys


def _load_sibling_test_task():
    """Load tests/test_task.py from this directory; tests/ is not a package
    so we use importlib.util by file path."""
    here = _pth.Path(__file__).resolve().parent
    target = here / "test_task.py"
    spec = _ilu.spec_from_file_location(f"{here.name}_test_task", target)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llm = _load_sibling_test_task()

# --- Soft-rubric data: inherited from the LLM-as-judge sibling --------
DIMENSION_WEIGHTS = dict(_llm.DIMENSION_WEIGHTS)
DIMENSIONS = list(getattr(_llm, "DIMENSIONS", DIMENSION_WEIGHTS.keys()))
TASK_RUBRIC = _llm.USER_PROMPT_TEMPLATE   # includes the per-axis 1-5 anchors

from agentic_judge.grounded.framework.constraints import (
    HardConstraint, EqualsValue, FieldPresent, AllURLsMatch,
    JSONSchemaConforms, ContainsAllSubstrings,
)
from agentic_judge.grounded.framework.summary_schema import SummarySchema, RequiredField, OptionalField

SUMMARY_SCHEMA = SummarySchema(
    required=[
        RequiredField("source_url",                 "string",   "Canonical nba.com URL where the stats were read."),
        RequiredField("player_name",                "string",   "Player name as shown on the page (should be 'Jayson Tatum')."),
        RequiredField("season",                     "string",   "Season label (e.g. '2024-25')."),
        RequiredField("current_3pt_stats.attempts", "number",   "3PT attempts (season total or per-game)."),
        RequiredField("current_3pt_stats.makes",    "number",   "3PT makes (matching the attempts basis)."),
        RequiredField("current_3pt_stats.pct",      "number 0-1 or 0-100","3PT percentage."),
        RequiredField("current_3pt_stats.basis",    "string",   "'season-total' OR 'per-game'."),
        RequiredField("data_freshness",             "string",   "When you fetched it (ISO date or 'as of YYYY-MM-DD')."),
    ],
    optional=[
        OptionalField("games_played",               "integer",  "Season games played."),
        OptionalField("last_game_3pt",              "object",   "{date, makes, attempts} for last game played."),
        OptionalField("career_context",             "string",   "Where this season ranks vs his career average."),
        OptionalField("milestone_progress",         "string",   "All-time 3PT record progress, if relevant."),
        OptionalField("alternate_sources_checked",  "list[string]", "Other source URLs you double-checked (e.g. basketball-reference)."),
    ],
    examples={
        "source_url": "https://www.nba.com/player/1628369/jayson-tatum",
        "player_name": "Jayson Tatum",
        "season": "2024-25",
        "current_3pt_stats": {"attempts": 8.6, "makes": 3.0, "pct": 0.349, "basis": "per-game"},
        "data_freshness": "as of 2026-06-01",
    },
)

HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    AllURLsMatch("source_url" if False else "current_3pt_stats", "basis", r"^(season-total|per-game)$",
                  name="basis_value_valid"),  # placeholder; replaced below
]
# Simpler — direct value check
HARD_CONSTRAINTS = [
    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths()),
    EqualsValue("player_name", "Jayson Tatum", case_insensitive=True, name="player_matches"),
    ContainsAllSubstrings("source_url", ["nba.com"], name="source_is_nba_dot_com"),
    FieldPresent("data_freshness", name="freshness_reported"),
    FieldPresent("season", name="season_reported"),
]

def FAITHFULNESS_CHECKS(summary: dict) -> list[dict]:
    src = summary.get("source_url")
    if not src:
        return []
    pct = (summary.get("current_3pt_stats") or {}).get("pct")
    claim_pct = ""
    if pct is not None:
        try:
            # The page may show "34.9%" or ".349" — verify either
            p = float(pct)
            pct_str = f"{p*100:.1f}%" if p < 1.0 else f"{p:.1f}%"
            claim_pct = f"re:{pct_str.replace('.', r'\\\\.').replace('%', r'\\\\%')}"
        except Exception:
            pass
    out = [{"url": src, "claim": "Tatum"}]
    if claim_pct:
        out.append({"url": src, "claim": claim_pct})
    for alt in (summary.get("alternate_sources_checked") or [])[:2]:
        out.append({"url": alt, "claim": "Tatum"})
    return out


# --- generic entry points (don't customise per task) -----------------

def grade(result: dict) -> dict:
    """Hard constraints + faithfulness only; no LLM call."""
    from agentic_judge.grounded.framework.extractor import extract_summary_json
    from agentic_judge.grounded.framework.faithfulness import (
        verify_url_claims, aggregate as _agg_faith,
    )
    agent_text = result.get("task_result") or ""
    if not isinstance(agent_text, str) or not agent_text.strip():
        for m in reversed(result.get("conversation") or []):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content") or ""
                if isinstance(c, str) and len(c) > 20:
                    agent_text = c
                    break
    summary, src = extract_summary_json(agent_text)
    if summary is None:
        return {"summary_parsed": False, "summary_source": src,
                "hard_report": [{"name": "summary_json_parseable",
                                  "passed": False,
                                  "detail": f"could not extract JSON (src={src})"}],
                "faithfulness_report": {}}
    hard = [c.check(summary) for c in HARD_CONSTRAINTS]
    hard_report = [r.to_dict() for r in hard]
    faith = {"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0, "details": []}
    try:
        findings = verify_url_claims(FAITHFULNESS_CHECKS(summary))
        faith = _agg_faith(findings)
    except Exception as e:
        faith["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return {"summary_parsed": True, "summary_source": src,
            "summary_json": summary, "hard_report": hard_report,
            "hard_pass_rate": sum(1 for r in hard if r.passed) / max(len(hard), 1),
            "faithfulness_report": faith}


def grade_with_llm(result: dict) -> dict:
    """Full mix: hard + faithfulness + LLM effectiveness judge."""
    from agentic_judge.grounded.framework import grounded_judge_test
    return grounded_judge_test(result, _sys.modules[__name__])


def main() -> int:
    import json, sys
    if len(sys.argv) != 2:
        print("usage: test_grounded.py <result.json>", file=sys.stderr)
        return 2
    print(json.dumps(grade(json.load(open(sys.argv[1]))), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
