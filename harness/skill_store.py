"""
Skill persistence and retrieval.

Stores skills as markdown files with YAML frontmatter in a directory.
Supports two kinds of skills:
  - Per-task skills (legacy): one file per task
  - Cross-task workflows (AWM-style): multiple abstract sub-routines
    from batch induction across tasks

Retrieval uses LLM-based search to pick relevant workflows for new tasks.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Split a markdown file into YAML frontmatter dict and body text."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    fm_text, body = match.group(1), match.group(2)
    if _YAML_AVAILABLE:
        fm = yaml.safe_load(fm_text) or {}
    else:
        fm = {}
        for line in fm_text.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip()
    return fm, body


def _build_frontmatter(metadata: Dict[str, Any]) -> str:
    """Serialize metadata dict to YAML frontmatter block."""
    if _YAML_AVAILABLE:
        fm_body = yaml.dump(metadata, default_flow_style=False, sort_keys=False).strip()
    else:
        lines = []
        for k, v in metadata.items():
            if isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{k}: {v}")
        fm_body = "\n".join(lines)
    return f"---\n{fm_body}\n---\n"


# ---------------------------------------------------------------------------
# Injection framing
# ---------------------------------------------------------------------------

# Reflection injection (Reflexion-style): per-task feedback from Phase 1
REFLECTION_INJECTION_PREFIX = """\
IMPORTANT — You are re-attempting a task you tried before. \
Below is the evaluator's feedback on your previous attempt. \
Study the weak dimensions carefully and change your approach to address them. \
The evaluator will check the SAME dimensions again.

"""

REFLECTION_INJECTION_SUFFIX = """

---

"""

# Workflow injection (AWM-style): cross-task reusable sub-routines
WORKFLOW_INJECTION_PREFIX = """\
Below are reusable workflow sub-routines learned from similar past tasks. \
These encode what evaluators specifically check for and common failure modes. \
You MUST pay attention to the Evaluator Criteria and Failure Patterns sections — \
they describe exactly how your output will be scored.

Use these workflows as hard constraints on quality criteria, not just soft suggestions. \
Adapt the placeholder variables to your specific task.

"""

WORKFLOW_INJECTION_SUFFIX = """
---

Now, here is your actual task:

"""


class SkillStore:
    """Manages a directory of skill markdown files."""

    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def save_skill(
        self,
        skill_md: str,
        task_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write a skill markdown file with frontmatter.

        Returns the path to the saved file.
        """
        meta = {
            "task_name": task_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            meta.update(metadata)

        content = _build_frontmatter(meta) + "\n" + skill_md.strip() + "\n"

        safe_name = re.sub(r"[^\w\-]", "_", task_name)
        file_path = self.skills_dir / f"{safe_name}.md"

        # Avoid overwriting: append counter if file exists
        counter = 1
        while file_path.exists():
            file_path = self.skills_dir / f"{safe_name}_{counter}.md"
            counter += 1

        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def save_workflows(
        self,
        workflows: List[str],
        source_tasks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Path]:
        """Save multiple cross-task workflows from batch induction.

        Each workflow is saved as a separate file with shared metadata
        indicating it came from cross-task induction.

        Returns list of paths to saved files.
        """
        paths = []
        for i, workflow_md in enumerate(workflows):
            # Extract title from workflow markdown for the filename
            title_match = re.search(r"#\s*Workflow:\s*(.+)", workflow_md)
            if title_match:
                title = title_match.group(1).strip()
                safe_title = re.sub(r"[^\w\-]", "_", title)[:60]
            else:
                safe_title = f"workflow_{i}"

            meta = {
                "workflow_type": "cross-task",
                "source_tasks": source_tasks,
                "workflow_index": i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if metadata:
                meta.update(metadata)

            content = _build_frontmatter(meta) + "\n" + workflow_md.strip() + "\n"

            file_path = self.skills_dir / f"{safe_title}.md"
            counter = 1
            while file_path.exists():
                file_path = self.skills_dir / f"{safe_title}_{counter}.md"
                counter += 1

            with open(file_path, "w") as f:
                f.write(content)
            paths.append(file_path)

        return paths

    def list_skills(self) -> List[Dict[str, Any]]:
        """Read all skill .md files and return list of {path, metadata, summary}.

        Summary is the first ~500 chars of the body for quick LLM scanning.
        """
        skills = []
        for md_file in sorted(self.skills_dir.glob("*.md")):
            text = md_file.read_text()
            fm, body = _parse_frontmatter(text)
            skills.append({
                "path": str(md_file),
                "metadata": fm,
                "summary": body[:500].strip(),
                "full_body": body,
            })
        return skills

    def load_skill(self, skill_path: str) -> str:
        """Read a single skill file and return its full content."""
        return Path(skill_path).read_text()

    def search_skills(
        self,
        task_description: str,
        top_k: int = 3,
    ) -> List[str]:
        """Use an LLM to pick the most relevant skills for a task.

        Returns the full body content of the selected skills.
        """
        all_skills = self.list_skills()
        if not all_skills:
            return []

        skill_summaries = ""
        for i, s in enumerate(all_skills):
            meta = s["metadata"]
            name = meta.get("task_name", meta.get("workflow_type", f"skill_{i}"))
            tags = meta.get("tags", [])
            score = meta.get("score", "?")
            wtype = meta.get("workflow_type", "per-task")
            skill_summaries += (
                f"\n[{i}] {name} (type={wtype}, score={score}, tags={tags})\n"
                f"{s['summary']}\n"
            )

        prompt = (
            "You are selecting relevant workflow sub-routines for a new task. "
            "Given the task description and a list of available workflows, "
            f"pick up to {top_k} workflows that are most relevant.\n\n"
            "Prefer cross-task workflows (they are more general and abstract) "
            "over per-task skills. Pick workflows whose Evaluator Criteria and "
            "Failure Patterns sections would help the agent avoid common mistakes.\n\n"
            "Return ONLY a JSON array of workflow indices, e.g. [0, 2]. "
            "If no workflows are relevant, return [].\n\n"
            f"Task:\n{task_description[:2000]}\n\n"
            f"Available workflows:\n{skill_summaries}"
        )

        try:
            import openai
            base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=base_url)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            content = response.choices[0].message.content.strip()
            # Parse the JSON array from the response
            match = re.search(r"\[[\d\s,]*\]", content)
            if match:
                indices = json.loads(match.group())
                selected = []
                for idx in indices:
                    if 0 <= idx < len(all_skills):
                        selected.append(all_skills[idx]["full_body"])
                return selected[:top_k]
        except Exception as e:
            print(f"[SkillStore] search_skills error: {e}")

        return []

    # ------------------------------------------------------------------
    # Reflections (Reflexion-style per-task feedback)
    # ------------------------------------------------------------------

    def save_reflection(self, task_name: str, reflection: Dict[str, Any]) -> Path:
        """Save a per-task reflection JSON file.

        Reflections are stored as .reflection.json files (not .md) to
        distinguish them from skills/workflows.
        """
        safe_name = re.sub(r"[^\w\-]", "_", task_name)
        file_path = self.skills_dir / f"{safe_name}.reflection.json"
        with open(file_path, "w") as f:
            json.dump(reflection, f, indent=2)
        return file_path

    def load_reflection(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Load a per-task reflection for a specific task, if it exists."""
        safe_name = re.sub(r"[^\w\-]", "_", task_name)
        file_path = self.skills_dir / f"{safe_name}.reflection.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path) as f:
                return json.load(f)
        except Exception:
            return None

    def format_reflection_injection(self, reflection: Dict[str, Any]) -> str:
        """Format a per-task reflection for injection into Phase 2.

        This is the primary per-task improvement signal: the evaluator's
        exact feedback on the previous attempt.
        """
        text = reflection.get("reflection_text", "")
        if not text:
            return ""
        return (
            REFLECTION_INJECTION_PREFIX
            + text
            + REFLECTION_INJECTION_SUFFIX
        )

    # ------------------------------------------------------------------
    # Combined injection: reflection (per-task) + workflows (cross-task)
    # ------------------------------------------------------------------

    def format_full_injection(
        self,
        reflection: Optional[Dict[str, Any]] = None,
        workflow_bodies: Optional[List[str]] = None,
    ) -> str:
        """Format the complete injection: reflection first, then workflows.

        Reflection is task-specific and is the primary signal.
        Workflows are supplementary cross-task patterns.
        """
        parts = []

        # Reflection comes first — it's the most relevant signal
        if reflection:
            ref_text = self.format_reflection_injection(reflection)
            if ref_text:
                parts.append(ref_text)

        # Workflows are supplementary
        if workflow_bodies:
            combined = "\n\n---\n\n".join(workflow_bodies)
            parts.append(
                WORKFLOW_INJECTION_PREFIX
                + combined
                + "\n"
            )

        if not parts:
            return ""

        return "".join(parts) + WORKFLOW_INJECTION_SUFFIX

    def format_injection(self, skill_bodies: List[str]) -> str:
        """Format selected workflows for injection into the task instruction.

        Uses stronger framing than the old 'here are some strategies' approach:
        explicitly tells the agent these encode evaluator criteria and failure modes.
        """
        if not skill_bodies:
            return ""

        combined = "\n\n---\n\n".join(skill_bodies)
        return (
            WORKFLOW_INJECTION_PREFIX
            + combined
            + WORKFLOW_INJECTION_SUFFIX
        )
