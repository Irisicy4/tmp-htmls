#!/usr/bin/env python3
"""Regenerate SUMMARY_SCHEMA + HARD_CONSTRAINTS + FAITHFULNESS_URL_KEYS
for tasks in tasks/real_118_revise_84/, one at a time, from the
curator-edited modified_review_full_prompt.txt.

Design rules baked into the LLM prompt:
  * Every URL field is at the same nesting depth as the fields the
    grounder will extract from that URL page. Siblings-of-URL = the
    web-grounding-check target set.
  * FAITHFULNESS is a list of JSON path patterns (like
    'options[].url', 'reference_product.url') — no per-task Python
    function.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "harness"))

for line in (REPO / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("LLM_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]

TASKS_DIR = REPO / "tasks" / "real_118_revise_84"

REGEN_PROMPT = """You are converting a task brief into a machine-checkable specification for a web-agent grading framework. Emit ONLY a JSON object.

## Design rules

1. The summary_schema is what the agent will fill in. It MUST be shaped so that:
   * Every URL field lives at the SAME nesting depth as the fields extracted from THAT URL's page.
   * If the agent must gather N options from N different pages, use `options[]` (list of objects). Each entry MUST contain `url` PLUS every field the grader will read off that URL's page.
   * A `.url` sibling and its co-siblings form the web-grounding target set for that URL.
   * Do NOT put URL fields in a top-level `sources` bag disconnected from the fields they source.

REQUIRED vs OPTIONAL — apply this rule strictly:
   * REQUIRED = anything the brief explicitly asks the agent to report ("for each X, report … name, URL, price, features…" ⇒ every one of those is required, at the per-item depth).
   * REQUIRED = anything the brief specifies a count/range/bound for ("3-5 options" ⇒ `options[]` is required).
   * OPTIONAL = bonus fields like retrieval timestamps, screenshots, alternative-source URLs, recommendations, cross-references — never the primary deliverable.
   * When in doubt, mark REQUIRED.

Type hints: use `"string"`, `"number"`, `"integer"`, `"boolean"`, `"list[string]"`, `"list[object]"` — not `"array"`.

2. HARD_CONSTRAINTS are typed predicates. Encode every countable/bounded rule from the brief. Do NOT emit JSONSchemaConforms — it is added automatically. Available types (use EXACT argument names shown):
     - ListLengthInRange   : {"type":"ListLengthInRange","path":"options","low":3,"high":5,"name":"quantity_3_to_5"}
     - AllValuesLE         : {"type":"AllValuesLE","list_path":"options","sub_path":"price","ceiling":75.0,"name":"all_prices_under_75"}
     - AllValuesGE         : {"type":"AllValuesGE","list_path":"options","sub_path":"rating","floor":4.0,"name":"..."}
     - AllValuesBetween    : {"type":"AllValuesBetween","list_path":"options","sub_path":"price","low":100,"high":500,"name":"..."}
     - EqualsValue         : {"type":"EqualsValue","path":"sent","expected":false,"name":"..."}
     - FieldPresent        : {"type":"FieldPresent","path":"reference_product.url","name":"reference_url_present"}
     - AllURLsMatch        : {"type":"AllURLsMatch","list_path":"options","url_field":"url","url_pattern":"^https?://","name":"every_option_has_url"}
     - ContainsAllSubstrings: {"type":"ContainsAllSubstrings","path":"summary_text","substrs":["safety","cleanliness"],"name":"..."}

CRITICAL: for AllValuesLE / AllValuesGE / AllValuesBetween / AllURLsMatch, `list_path` is the LIST field alone (no brackets, no dots into children), and `sub_path` is the key inside each list item. Example: for "every option's price ≤ 75", use list_path="options", sub_path="price", NOT list_path="options[].price".

3. FAITHFULNESS_URL_KEYS is a flat list of JSON paths (use `[]` for list-of-objects).

## Output schema

Return ONLY a JSON object:

{
  "summary_schema": {
    "required": [{"path": "...", "type_hint": "...", "description": "..."}, ...],
    "optional": [{"path": "...", "type_hint": "...", "description": "..."}, ...],
    "examples": { ... shallow example ... }
  },
  "hard_constraints": [
    {"type": "ListLengthInRange", "path": "options", "low": 3, "high": 5, "name": "quantity_3_to_5"},
    ...
  ],
  "faithfulness_url_keys": ["reference_product.url", "options[].url"]
}

No preamble, no code fence.

## TASK BRIEF

__BRIEF__
"""


def _chat(prompt, model="gpt-4o-mini"):
    from openai import OpenAI
    kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    r = OpenAI(**kwargs).chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def _strip_fences(text):
    import re
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _render_spec(task_name, plan):
    ss = plan["summary_schema"]
    hcs = plan["hard_constraints"]
    keys = plan["faithfulness_url_keys"]

    # Drop any JSONSchemaConforms the LLM emitted; we always prepend it.
    hcs = [h for h in hcs if h.get("type") != "JSONSchemaConforms"]
    types_used = sorted({hc["type"] for hc in hcs} | {"JSONSchemaConforms"})
    ctor_import = ", ".join(types_used)

    def _f(f, cls):
        desc = (f.get("description") or "").replace('"', '\\"')
        return f'        {cls}("{f["path"]}", "{f.get("type_hint","any")}", "{desc}"),'

    req = "\n".join(_f(f, "RequiredField") for f in ss.get("required", []))
    opt = "\n".join(_f(f, "OptionalField") for f in ss.get("optional", []))

    hc_lines = ['    JSONSchemaConforms(required_paths=SUMMARY_SCHEMA.required_paths(), name="summary_schema_conforms"),']
    for hc in hcs:
        t = hc["type"]
        args = []
        for k, v in hc.items():
            if k == "type": continue
            if isinstance(v, str):
                args.append(f'{k}="{v}"')
            else:
                args.append(f'{k}={v!r}')
        hc_lines.append(f'    {t}({", ".join(args)}),')

    keys_lines = ",\n".join(f'    "{k}"' for k in keys)
    # Use pprint for the examples so Python booleans are True/False, not true/false.
    import pprint as _pp
    examples_py = _pp.pformat(ss.get("examples", {}), width=88, sort_dicts=False)

    return f'''"""Grounded (deterministic) judging spec for {task_name}.

Regenerated from modified_review_full_prompt.txt. Design:
  * URL fields sit at the same depth as fields extracted from that URL.
  * FAITHFULNESS_URL_KEYS lists the JSON paths whose URLs the framework
    fetches; sibling schema fields become the extraction targets.
"""
import importlib.util as _ilu
import pathlib as _pth
import sys as _sys


def _load_sibling_test_task():
    here = _pth.Path(__file__).resolve().parent
    target = here / "test_task.py"
    spec = _ilu.spec_from_file_location(f"{{here.name}}_test_task", target)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_llm = _load_sibling_test_task()
# Some legacy test_task.py files don't declare DIMENSION_WEIGHTS.
# Fall back to the framework's four canonical dimensions with equal weight.
_DEFAULT_DIMS = ["accuracy", "completeness", "sources", "quality"]
DIMENSION_WEIGHTS = dict(getattr(_llm, "DIMENSION_WEIGHTS",
                                  {{d: 0.25 for d in _DEFAULT_DIMS}}))
DIMENSIONS = list(getattr(_llm, "DIMENSIONS", DIMENSION_WEIGHTS.keys()))
TASK_RUBRIC = getattr(_llm, "USER_PROMPT_TEMPLATE", "")

from agentic_judge.grounded.framework.constraints import (
    HardConstraint, {ctor_import},
)
from agentic_judge.grounded.framework.summary_schema import (
    SummarySchema, RequiredField, OptionalField,
)

SUMMARY_SCHEMA = SummarySchema(
    required=[
{req}
    ],
    optional=[
{opt}
    ],
    examples={examples_py},
)

HARD_CONSTRAINTS: list[HardConstraint] = [
{chr(10).join(hc_lines)}
]

FAITHFULNESS_URL_KEYS = [
{keys_lines}
]


def grade(result: dict) -> dict:
    from agentic_judge.grounded.framework.extractor import extract_summary_json
    from agentic_judge.grounded.framework.faithfulness import (
        verify_url_keys, aggregate as _agg_faith,
    )
    agent_text = result.get("task_result") or ""
    if not isinstance(agent_text, str) or not agent_text.strip():
        for m in reversed(result.get("conversation") or []):
            if isinstance(m, dict) and m.get("role") == "assistant":
                c = m.get("content") or ""
                if isinstance(c, str) and c.strip():
                    agent_text = c
                    break
    summary, src = extract_summary_json(agent_text)
    if summary is None:
        return {{
            "summary_parsed": False,
            "summary_source": src,
            "hard_report": [{{"name": "summary_json_parseable",
                              "passed": False,
                              "detail": f"could not extract JSON (src={{src}})"}}],
            "faithfulness_report": {{}},
        }}
    hard_report = [c.check(summary).to_dict() for c in HARD_CONSTRAINTS]
    faith = {{"score_5": 0.0, "fetched": 0, "matched": 0, "total": 0, "details": []}}
    try:
        findings = verify_url_keys(summary, FAITHFULNESS_URL_KEYS, SUMMARY_SCHEMA)
        faith = _agg_faith(findings)
    except Exception as e:
        faith["details"].append({{"error": f"{{type(e).__name__}}: {{e}}"}})
    return {{
        "summary_parsed": True,
        "summary_source": src,
        "hard_report": hard_report,
        "faithfulness_report": faith,
        "summary_json": summary,
    }}


def grade_with_llm(result: dict) -> dict:
    from agentic_judge.grounded.framework.runner import grounded_judge_test
    return grounded_judge_test(result, spec=_sys.modules[__name__])
'''


def process_one(task_name, dry_run=False):
    task_dir = TASKS_DIR / task_name
    if not task_dir.is_dir():
        raise SystemExit(f"missing {task_dir}")
    brief_path = task_dir / "modified_review_full_prompt.txt"
    if not brief_path.is_file():
        brief_path = task_dir / "review_full_prompt.txt"
    brief = brief_path.read_text().strip()

    prompt = REGEN_PROMPT.replace("__BRIEF__", brief)
    raw = _chat(prompt)
    plan_text = _strip_fences(raw)
    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[{task_name}] invalid JSON: {e}\n---raw---\n{raw[:800]}")

    for key in ("summary_schema", "hard_constraints", "faithfulness_url_keys"):
        if key not in plan:
            raise SystemExit(f"[{task_name}] plan missing key: {key}")

    spec_src = _render_spec(task_name, plan)

    if dry_run:
        print(f"=== [{task_name}] plan ===")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print()
        print(f"=== [{task_name}] rendered test_grounded.py ===")
        print(spec_src)
        return plan

    tests_dir = task_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_grounded.py").write_text(spec_src)
    (task_dir / "instruction.md").write_text(brief + "\n")

    print(f"✓ {task_name}")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.all:
        tasks = sorted([p.name for p in TASKS_DIR.iterdir() if p.is_dir()])
    elif args.task:
        tasks = [args.task]
    else:
        ap.error("provide a task name or --all")

    for t in tasks:
        try:
            process_one(t, dry_run=args.dry_run)
        except SystemExit as e:
            print(f"✗ {t}: {e}", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"✗ {t}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    main()
