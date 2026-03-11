"""
Skill persistence and retrieval.

Stores skills as markdown files with YAML frontmatter in a directory.
Retrieves relevant skills using LLM-based search.
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
                lines.append(f"{k}: [{', '.join(str(x) for x in v)}]")
            else:
                lines.append(f"{k}: {v}")
        fm_body = "\n".join(lines)
    return f"---\n{fm_body}\n---\n"


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
            name = meta.get("task_name", f"skill_{i}")
            tags = meta.get("tags", [])
            score = meta.get("score", "?")
            skill_summaries += (
                f"\n[{i}] {name} (score={score}, tags={tags})\n"
                f"{s['summary']}\n"
            )

        prompt = (
            "You are selecting relevant skills for a new task. "
            "Given the task description and a list of available skills, "
            f"pick up to {top_k} skills that are most relevant. "
            "Return ONLY a JSON array of skill indices, e.g. [0, 2]. "
            "If no skills are relevant, return [].\n\n"
            f"Task:\n{task_description}\n\n"
            f"Available skills:\n{skill_summaries}"
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
