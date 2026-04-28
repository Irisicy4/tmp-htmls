#!/usr/bin/env python3
"""Convert updated-deprivacy-100 task test_task.py files to use the shared
run_judge() in harness/evaluator.py.

Each task currently has ~200 lines of inlined judge logic + prompt scaffolding.
After conversion, each file has only the task-specific rubric pieces and
delegates execution to the shared engine.
"""
import re, ast, pathlib, sys

BASE = pathlib.Path(__file__).resolve().parent.parent / "tasks" / "updated-deprivacy-100"

TEMPLATE = '''"""LLM-as-judge evaluator for {slug}.

Category: {category}
Task: {task_summary}
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = {task_instruction!r}

SYSTEM_PROMPT_EXTRA = {system_prompt_extra!r}

CONSTRAINTS = """{constraints}"""

EVIDENCE_QUESTIONS = """{evidence_questions}"""

DIMENSION_RUBRICS = """{dimension_rubrics}"""

DIMENSION_WEIGHTS = {{
{weights_lines}
}}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
'''


def extract_str_assign(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
           and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    return None


def extract_section(text, start_pattern, end_pattern):
    """Return text between start_pattern (regex) and end_pattern (regex), stripped."""
    sm = re.search(start_pattern, text)
    if not sm:
        return None
    em = re.search(end_pattern, text[sm.end():])
    if not em:
        return None
    return text[sm.end():sm.end() + em.start()].strip("\n").strip()


def extract_system_prompt_extra(system_prompt):
    """Strip canonical header + footer; return the middle (task-specific extra)."""
    # Canonical header starts with this; we want what's after the first sentence.
    HEADER_PREFIX = "You are an expert evaluator for AI agent benchmarks."
    FOOTER_MARKER = "\n\nYou will be given:"
    body = system_prompt
    if FOOTER_MARKER in body:
        body = body.split(FOOTER_MARKER, 1)[0]
    body = body.strip()
    if body.startswith(HEADER_PREFIX):
        body = body[len(HEADER_PREFIX):].lstrip()
    # body is the extra context. May be empty string.
    return body


def extract_category(src):
    # docstring at top has "Category: X"
    m = re.search(r"Category:\s*(.+?)\n", src)
    return m.group(1).strip() if m else "General"


def convert(slug, dry_run=False):
    path = BASE / slug / "tests" / "test_task.py"
    src = path.read_text()
    tree = ast.parse(src)

    task_instruction = extract_str_assign(tree, "TASK_INSTRUCTION")
    system_prompt = extract_str_assign(tree, "SYSTEM_PROMPT")
    user_prompt_template = extract_str_assign(tree, "USER_PROMPT_TEMPLATE")
    dimension_weights = extract_str_assign(tree, "DIMENSION_WEIGHTS")

    if not all([task_instruction, system_prompt, user_prompt_template, dimension_weights]):
        return f"{slug}: SKIP (missing required assignments)"

    constraints = extract_section(
        user_prompt_template,
        r"##\s+Task-Specific Constraints\s*\n",
        r"\n\s*##\s+Agent Final Response",
    )
    evidence = extract_section(
        user_prompt_template,
        r"###\s+Step 1:[^\n]*\n",
        r"\n\s*###\s+Step 2:",
    )
    dimensions = extract_section(
        user_prompt_template,
        r"###\s+Step 2:[^\n]*\n",
        r"\n\s*###\s+Step 3:",
    )

    if constraints is None or evidence is None or dimensions is None:
        return f"{slug}: SKIP (could not parse rubric sections)"

    extra = extract_system_prompt_extra(system_prompt)
    category = extract_category(src)
    task_summary = task_instruction.split("\n", 1)[0][:200]

    weights_lines = "\n".join(
        f"    {k!r}: {v}," for k, v in dimension_weights.items()
    )

    new_src = TEMPLATE.format(
        slug=slug,
        category=category,
        task_summary=task_summary,
        task_instruction=task_instruction,
        system_prompt_extra=extra,
        constraints=constraints.replace('"""', '\\"\\"\\"'),
        evidence_questions=evidence.replace('"""', '\\"\\"\\"'),
        dimension_rubrics=dimensions.replace('"""', '\\"\\"\\"'),
        weights_lines=weights_lines,
    )

    # Validate
    ast.parse(new_src)

    if dry_run:
        return f"{slug}: OK (would write {len(new_src)} chars; was {len(src)})"
    path.write_text(new_src)
    return f"{slug}: PATCHED ({len(src)} -> {len(new_src)} chars)"


def main():
    apply = "--apply" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    if only:
        slugs = only
    else:
        slugs = sorted(p.name for p in BASE.glob("task-*"))
    for slug in slugs:
        try:
            print(convert(slug, dry_run=not apply))
        except Exception as e:
            print(f"{slug}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
