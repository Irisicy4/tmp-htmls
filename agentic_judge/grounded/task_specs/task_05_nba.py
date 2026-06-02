"""Spec for task-05-go-to-nbacom-and-check (Sports / Data)."""
from ..framework.constraints import (
    HardConstraint, EqualsValue, FieldPresent, AllURLsMatch,
    JSONSchemaConforms, ContainsAllSubstrings,
)
from ..framework.summary_schema import SummarySchema, RequiredField, OptionalField


TASK_INSTRUCTION = "Go to nba.com and check Jayson Tatum's current 3-point status."

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


DIMENSIONS = ["platform_execution", "stat_accuracy_recency", "interpretive_completeness", "response_clarity"]
DIMENSION_WEIGHTS = {
    "platform_execution":        0.30,
    "stat_accuracy_recency":     0.35,
    "interpretive_completeness": 0.20,
    "response_clarity":          0.15,
}

TASK_RUBRIC = """A. Platform Execution (0.30)
  5 — Pulled stats directly from nba.com player page; source_url is canonical.
  4 — nba.com but indirect (e.g. team page rather than player).
  3 — nba.com but a generic stats page (no player context).
  2 — Other site (ESPN, Basketball-Reference) instead of nba.com.
  1 — No live data fetched.

B. Stat Accuracy & Recency (0.35)
  5 — Current-season makes/attempts/percentage; data_freshness within last 7 days.
  4 — Current season but one metric ambiguous.
  3 — Career data only, or one season old.
  2 — Stale data (>1 season old).
  1 — Fabricated or absent.

C. Interpretive Completeness (0.20)
  5 — Season stats + games played + career context OR milestone progress.
  4 — Season stats + one of context/milestone.
  3 — Season stats only.
  2 — Partial stats.
  1 — No stat coverage.

D. Response Clarity (0.15)
  5 — Structured, easy to scan, numbers labelled with basis (per-game vs total).
  4 — Clear with minor formatting issue.
  3 — Readable but mixed units.
  2 — Difficult to follow.
  1 — Unintelligible.
"""
