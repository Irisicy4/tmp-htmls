#!/usr/bin/env python3
"""Regenerate LLM-judge rubrics for a given list of tasks in updated-deprivacy-100.

Reads the current TASK_INSTRUCTION from each task's test_task.py, calls an LLM
to produce a task-specific rubric (system prompt, constraints, evidence questions,
dimensions with weights and 1-5 anchors), and rewrites the top of test_task.py
while preserving the helper functions and `test(result)` entry point.
"""
import os, re, sys, json, ast, pathlib, argparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "tasks" / "updated-deprivacy-100"

# Load .env
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("OPENAI_API_KEY", os.environ.get("UNIAPI_KEY", ""))
base = os.environ.get("UNIAPI_BASE", "")
if base and "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = f"{base}/v1"

META_SYSTEM = """You design evaluation rubrics for an AI-agent benchmark.
Given a task instruction, produce a strict JSON rubric the benchmark's LLM judge will use to score agent responses on a 1-5 scale across 3-5 dimensions.

Output ONLY valid JSON (no markdown fences, no prose) with this exact schema:
{
  "category": "short category label, e.g. 'Web Research', 'Data & ML Engineering', 'Content Creation'",
  "system_prompt": "one paragraph (3-6 sentences) describing what the judge is evaluating, referencing the concrete subject of THIS task",
  "constraints": ["bullet", "points", "of task-specific constraints derived from the instruction"],
  "evidence_questions": ["bullet", "points", "the judge should answer about the agent trace before scoring"],
  "dimensions": [
    {
      "name": "snake_case_name",
      "weight": 0.30,
      "description": "one sentence",
      "anchors": {"5": "...", "4": "...", "3": "...", "2": "...", "1": "..."}
    }
  ]
}

Requirements:
- 3 to 5 dimensions; weights must sum to exactly 1.0.
- Dimensions must be specific to the task's actual goals, not generic ("did agent try hard" is bad).
- Each anchor is one concrete sentence referencing observable evidence.
- Keep the system_prompt and anchors grounded in the specific entities/URLs/actions in the instruction."""


def call_llm(task_instruction: str, model: str) -> dict:
    import openai
    client = openai.OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": META_SYSTEM},
            {"role": "user", "content": f"Task instruction:\n{task_instruction}"},
        ],
        temperature=0.2,
        max_tokens=2500,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL)
    data = json.loads(raw)
    # normalize weights
    total = sum(d["weight"] for d in data["dimensions"])
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"weights sum to {total}, not 1.0")
    # force exact sum
    data["dimensions"][-1]["weight"] = round(
        data["dimensions"][-1]["weight"] + (1.0 - total), 4
    )
    return data


def render_rubric(task_slug: str, instruction: str, rubric: dict) -> str:
    """Render the top of test_task.py (everything before helper functions)."""
    dims = rubric["dimensions"]
    constraints_block = "\n".join(f"- {c}" for c in rubric["constraints"])
    evidence_block = "\n".join(f"- {q}" for q in rubric["evidence_questions"])

    # Dimension scoring sections
    letters = "ABCDE"
    dim_sections = []
    for i, d in enumerate(dims):
        section = (
            f"#### {letters[i]}. {d['name'].replace('_', ' ').title()}\n"
            f"{d['description']}\n\n"
            + "\n".join(f"{k} — {v}" for k, v in sorted(d['anchors'].items(), reverse=True))
        )
        dim_sections.append(section)
    dim_block = "\n\n".join(dim_sections)

    json_fields = ",\n  ".join(f'"{d["name"]}": <1-5>' for d in dims)
    reasoning_fields = ",\n    ".join(
        f'"{d["name"]}": "<one sentence citing specific evidence>"' for d in dims
    )

    weights_lines = "\n".join(
        f'    "{d["name"]}":{" " * max(1, 20 - len(d["name"]))}{d["weight"]},' for d in dims
    )

    # Escape any triple quotes in user-supplied strings
    def esc(s): return s.replace('"""', '\\"\\"\\"')

    return f'''"""
LLM-as-judge evaluator for EvolveBench {task_slug}.

Category: {rubric["category"]}
Task: {esc(instruction[:200])}
"""

import os, json, re

TASK_INSTRUCTION = """{esc(instruction)}"""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """{esc(rubric["system_prompt"])}

You will be given:
1. The task instruction
2. The agent's final response
3. A summary of the agent's tool-call trace (what it actually did, if available)

Follow the evaluation steps exactly in order."""

USER_PROMPT_TEMPLATE = """## Task Instruction
{{task_instruction}}

## Task-Specific Constraints
{esc(constraints_block)}

## Agent Final Response
{{agent_response}}

## Agent Tool-Call Trace (what the agent actually did)
{{execution_summary}}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
{esc(evidence_block)}

### Step 2: Dimension Scoring

{esc(dim_block)}

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  {json_fields},
  "dimension_reasoning": {{{{
    {reasoning_fields}
  }}}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}}}
</Answer>
"""

DIMENSION_WEIGHTS = {{
{weights_lines}
}}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())
'''


def rewrite_test_file(path: pathlib.Path, new_header: str):
    src = path.read_text()
    # find the line after `DIMENSIONS = list(DIMENSION_WEIGHTS.keys())`
    m = re.search(r"^DIMENSIONS\s*=\s*list\(DIMENSION_WEIGHTS\.keys\(\)\).*?$", src, re.MULTILINE)
    if not m:
        raise RuntimeError(f"{path}: could not find DIMENSIONS line")
    tail = src[m.end():]
    # strip any leading blank lines from tail
    tail = "\n" + tail.lstrip("\n")
    new_src = new_header + tail
    # validate
    ast.parse(new_src)
    path.write_text(new_src)


def read_task_instruction(path: pathlib.Path) -> str:
    src = path.read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
           and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "TASK_INSTRUCTION":
            return ast.literal_eval(node.value)
    raise RuntimeError(f"No TASK_INSTRUCTION in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for slug in args.tasks:
        path = BASE / slug / "tests" / "test_task.py"
        inst = read_task_instruction(path)
        print(f"\n=== {slug} ===\n  instruction: {inst[:120]}...")
        rubric = call_llm(inst, args.model)
        print(f"  category: {rubric['category']}")
        print(f"  dimensions: {[(d['name'], d['weight']) for d in rubric['dimensions']]}")
        header = render_rubric(slug, inst, rubric)
        if args.dry_run:
            # validate parses when combined with tail
            src = path.read_text()
            m = re.search(r"^DIMENSIONS\s*=\s*list.*$", src, re.MULTILINE)
            tail = "\n" + src[m.end():].lstrip("\n")
            ast.parse(header + tail)
            print(f"  [dry-run] would write {len(header)} chars (parsed OK)")
        else:
            rewrite_test_file(path, header)
            print(f"  PATCHED")


if __name__ == "__main__":
    main()
