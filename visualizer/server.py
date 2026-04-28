#!/usr/bin/env python3
"""HTTP server for the agent visualization UI.

Expects a Harbor-style run directory produced by harbor_modal_runner:

    <run-tag>/
        result.json              # job-level aggregate (Harbor canonical)
        config.json
        job.log
        results.csv              # optional, written by post-hoc rejudges
        task-01-.../             # trial dirs sit DIRECTLY under run dir
            result.json          # judge verdict (score, passed, feedback, _instruction)
            agent_result.json    # cocoa-style: has visualization_data.iterations
            agent/
                trajectory.json  # terminus ATIF: steps[]
                command-N/       # openhands: per-command stdout/stderr
                artifact_*       # collected /output/ files

Also tolerates the legacy `<run>/trials/<task>/` layout. --data-dir can point
at either the run folder or its trials/ subfolder; the server auto-detects.
"""

import argparse
import csv
import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _looks_like_trials_root(d: Path) -> bool:
    """A directory is a 'trials root' if it directly contains trial dirs
    (a child folder with result.json or an agent/ subfolder)."""
    if not d.is_dir():
        return False
    for child in d.iterdir():
        if not child.is_dir():
            continue
        if (child / "result.json").exists() or (child / "agent").is_dir() or (child / "agent_result.json").exists():
            return True
    return False


def _resolve_trials_dir(data_dir: Path) -> Path | None:
    """Find the directory that contains trial subdirs.

    Supports the canonical Harbor layout (trials directly under <run>/) and
    the legacy <run>/trials/ layout. Also tolerates a doubled-up nesting
    (<run>/<run>/...) from older `modal volume get` collects.
    """
    if not data_dir.is_dir():
        return None
    if data_dir.name == "trials":
        return data_dir
    legacy = data_dir / "trials"
    if legacy.is_dir():
        return legacy
    # Doubled-nest from older collect runs: results/<tag>/<tag>/...
    nested = data_dir / data_dir.name
    if nested.is_dir():
        if (nested / "trials").is_dir():
            return nested / "trials"
        if _looks_like_trials_root(nested):
            return nested
    if _looks_like_trials_root(data_dir):
        return data_dir
    for child in data_dir.iterdir():
        if (child / "trials").is_dir():
            return child / "trials"
    return None


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _split_instruction(raw: str) -> tuple[str, str]:
    """Return (instruction, skills) from the raw task prompt.

    Two conventions share the same ``\\n---\\n`` delimiter:

    * Skill injection (legacy): ``<skill>\\n---\\nNow, here is your actual task:\\n\\n<task>``
      → instruction is the LAST chunk, skills are everything before.
    * Harbor runner (current): ``<task>\\n---\\n<output-dir boilerplate>``
      → instruction is the FIRST chunk, skills are empty.

    Disambiguate on the ``Now, here is your actual task`` marker.
    """
    if "\n---\n" not in raw:
        return raw.strip(), ""
    parts = raw.split("\n---\n")
    last = parts[-1].strip()
    for prefix in ("Now, here is your actual task:\n\n", "Now, here is your actual task:\n"):
        if last.startswith(prefix):
            return last[len(prefix):], "\n---\n".join(parts[:-1]).strip()
    # No skill marker — first chunk is the task, trailing chunks are boilerplate.
    return parts[0].strip(), ""


def _load_csv_fallback(trials_dir: Path) -> dict[str, dict]:
    """Some runs had trials re-judged locally after the fact and only results.csv
    was updated, not the per-trial result.json. Read the CSV (from the run root)
    once so the server can fall back to it.
    """
    run_root = trials_dir.parent
    csv_path = run_root / "results.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                try:
                    score = float(r["score"]) if r.get("score") not in (None, "") else None
                except (ValueError, TypeError):
                    score = None
                out[r["task"]] = {
                    "score": score,
                    "passed": r.get("passed") == "yes",
                    "feedback": r.get("failure_reason") or "",
                    "category": r.get("category") or "",
                }
    except Exception:
        return {}
    return out


def _task_meta(trial_dir: Path, csv_fallback: dict[str, dict]) -> dict:
    """Pull score/passed out of a trial's result.json, falling back to results.csv."""
    try:
        r = _load_json(trial_dir / "result.json")
        return {"passed": r.get("passed"), "score": r.get("score")}
    except Exception:
        row = csv_fallback.get(trial_dir.name)
        if row:
            return {"passed": row["passed"], "score": row["score"]}
        return {"passed": None, "score": None}


# ---------- per-agent trajectory → visualization_data adapters ----------


def _terminus_to_iterations(traj: dict) -> list[dict]:
    """Convert a terminus ATIF trajectory into {iteration, think, actions[]} blocks.

    Each agent step becomes one iteration. tool_calls become actions whose
    observation is the batched terminal output attached to that step.
    """
    iterations = []
    for step in traj.get("steps", []):
        if step.get("source") != "agent":
            continue
        think = step.get("message") or ""
        tool_calls = step.get("tool_calls") or []
        # Flatten the observation.results[].content into a single terminal string.
        obs = step.get("observation") or {}
        terminal = ""
        if isinstance(obs, dict):
            for r in obs.get("results") or []:
                if isinstance(r, dict) and r.get("content"):
                    terminal += r["content"]
                elif isinstance(r, str):
                    terminal += r

        actions = []
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                fn = tc.get("function_name") or tc.get("name") or "tool_call"
                args = tc.get("arguments") or {}
                action = {"action_type": fn, **{k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))}}
                # Attach the full terminal observation to the LAST action so the
                # UI's right panel shows it once instead of repeating N times.
                actions.append({
                    "action": action,
                    "observation": terminal if i == len(tool_calls) - 1 else "",
                    "screenshot": "",
                })
        else:
            # No tool calls — still surface the observation (e.g. final answer).
            actions.append({
                "action": {"action_type": "message"},
                "observation": terminal,
                "screenshot": "",
            })

        iterations.append({
            "iteration": len(iterations) + 1,
            "think": think,
            "actions": actions,
        })
    return iterations


def _openhands_to_iterations(agent_dir: Path) -> list[dict]:
    """Synthesize iterations from openhands' per-command dirs. Each command-N/
    folder has command.txt + stdout.txt + return-code.txt."""
    if not agent_dir.is_dir():
        return []
    cmd_dirs = sorted(
        (p for p in agent_dir.iterdir() if p.is_dir() and p.name.startswith("command-")),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    iterations = []
    for i, d in enumerate(cmd_dirs):
        cmd = (d / "command.txt").read_text(errors="replace") if (d / "command.txt").exists() else ""
        out = (d / "stdout.txt").read_text(errors="replace") if (d / "stdout.txt").exists() else ""
        err = (d / "stderr.txt").read_text(errors="replace") if (d / "stderr.txt").exists() else ""
        rc = (d / "return-code.txt").read_text(errors="replace").strip() if (d / "return-code.txt").exists() else ""
        observation = out
        if err.strip():
            observation += ("\n[stderr]\n" + err)
        if rc and rc != "0":
            observation += f"\n[exit {rc}]"
        iterations.append({
            "iteration": i + 1,
            "think": "",
            "actions": [{
                "action": {"action_type": "bash_command", "keystrokes": cmd},
                "observation": observation,
                "screenshot": "",
            }],
        })
    return iterations


def _build_visualization_data(trial_dir: Path, csv_fallback: dict[str, dict]) -> dict:
    """Return {task_description, task_instruction, task_skills, iterations, eval}."""
    result: dict = {}
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            result = _load_json(result_path)
        except Exception:
            result = {}
    if not result:
        row = csv_fallback.get(trial_dir.name, {})
        result = {
            "passed": row.get("passed"),
            "score": row.get("score"),
            "feedback": row.get("feedback", ""),
            "_instruction": "",
        }

    raw_instruction = result.get("_instruction") or ""
    instruction, skills = _split_instruction(raw_instruction)

    iterations: list[dict] = []
    task_description = raw_instruction

    # 1. Cocoa-style: agent_result.json already carries visualization_data.
    agent_result_path = trial_dir / "agent_result.json"
    if agent_result_path.exists():
        try:
            ar = _load_json(agent_result_path)
            vd = ar.get("visualization_data") or {}
            if vd.get("iterations"):
                iterations = vd["iterations"]
                task_description = vd.get("task_description") or raw_instruction
        except Exception:
            pass

    # 2. Terminus-style: agent/trajectory.json with ATIF steps.
    if not iterations:
        traj_path = trial_dir / "agent" / "trajectory.json"
        if traj_path.exists():
            try:
                iterations = _terminus_to_iterations(_load_json(traj_path))
            except Exception:
                pass

    # 3. Openhands-style: agent/command-N/ dirs.
    if not iterations:
        iterations = _openhands_to_iterations(trial_dir / "agent")

    return {
        "task_description": task_description,
        "task_instruction": instruction,
        "task_skills": skills,
        "iterations": iterations,
        "eval": {
            "passed": bool(result.get("passed")),
            "feedback": result.get("feedback", "") or "",
            "score": result.get("score"),
            "dimension_scores": result.get("dimension_scores"),
        },
        "task_result": result.get("_task_result", "") or "",
    }


# ---------- HTTP handler ----------


class VisualizationHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, trials_dir=None, compare_dir=None,
                 artifacts_dir=None, compare_artifacts_dir=None,
                 csv_fallback=None, compare_csv_fallback=None, **kwargs):
        self.trials_dir = trials_dir
        self.compare_dir = compare_dir
        self.artifacts_dir = artifacts_dir
        self.compare_artifacts_dir = compare_artifacts_dir
        self.csv_fallback = csv_fallback or {}
        self.compare_csv_fallback = compare_csv_fallback or {}
        super().__init__(*args, **kwargs)

    def _resolve_task_artifacts_dir(self, task: str, artifacts_dir, trials_dir):
        """Return the on-disk dir for a task's artifacts and the filename
        prefix (if any) that marks artifact files. Supports:
          - Harbor canonical <run>/<task>/artifacts/<file>
          - Legacy <run>/artifacts/<task>/<file>
          - Legacy <run>/<task>/agent/artifact_<file>
        """
        if trials_dir is not None:
            d = trials_dir / task / "artifacts"
            if d.is_dir() and any(d.iterdir()):
                return d, ""
        if artifacts_dir is not None:
            d = artifacts_dir / task
            if d.is_dir():
                return d, ""
        if trials_dir is not None:
            d = trials_dir / task / "agent"
            if d.is_dir():
                return d, "artifact_"
        return None, ""

    def do_GET(self):
        parsed = urlparse(self.path)
        path, qs = parsed.path, parse_qs(parsed.query)
        if path == "/api/list":
            return self._handle_list(self.trials_dir, self.csv_fallback)
        if path == "/api/list2":
            return self._handle_list(self.compare_dir, self.compare_csv_fallback)
        if path == "/api/data":
            return self._handle_data(qs, self.trials_dir, self.csv_fallback)
        if path == "/api/data2":
            return self._handle_data(qs, self.compare_dir, self.compare_csv_fallback)
        if path == "/api/compare":
            return self._handle_compare()
        if path == "/api/artifacts":
            return self._handle_artifact_list(qs, self.artifacts_dir, self.trials_dir)
        if path == "/api/artifacts2":
            return self._handle_artifact_list(qs, self.compare_artifacts_dir, self.compare_dir)
        if path.startswith("/artifacts/"):
            return self._serve_artifact(path[len("/artifacts/"):], self.artifacts_dir, self.trials_dir)
        if path.startswith("/artifacts2/"):
            return self._serve_artifact(path[len("/artifacts2/"):], self.compare_artifacts_dir, self.compare_dir)
        if path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def _handle_artifact_list(self, qs: dict, artifacts_dir, trials_dir):
        task = qs.get("task", [None])[0]
        if not task:
            return self._send_json({"files": []})
        d, prefix = self._resolve_task_artifacts_dir(task, artifacts_dir, trials_dir)
        if d is None:
            return self._send_json({"files": []})
        files = []
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            if prefix and not p.name.startswith(prefix):
                continue
            if p.name == "manifest.json":
                continue
            display_name = p.name[len(prefix):] if prefix else p.name
            files.append({"name": display_name, "size": p.stat().st_size})
        self._send_json({"files": files})

    def _serve_artifact(self, rel: str, artifacts_dir, trials_dir):
        import mimetypes
        if ".." in rel:
            return self.send_error(404)
        # rel is "<task>/<filename>"
        parts = rel.split("/", 1)
        if len(parts) != 2:
            return self.send_error(404)
        task, fname = parts
        d, prefix = self._resolve_task_artifacts_dir(task, artifacts_dir, trials_dir)
        if d is None:
            return self.send_error(404)
        target = (d / f"{prefix}{fname}").resolve()
        try:
            target.relative_to(d.resolve())
        except ValueError:
            return self.send_error(403)
        if not target.is_file():
            return self.send_error(404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _handle_list(self, trials_dir: Path | None, csv_fallback: dict):
        if not trials_dir or not trials_dir.is_dir():
            return self.send_error(404, "Trials directory not found")
        try:
            files = []
            seen: set[str] = set()
            for d in sorted(trials_dir.iterdir(), key=self._trial_sort_key):
                if not d.is_dir():
                    continue
                has_result = (d / "result.json").exists()
                if not has_result and d.name not in csv_fallback:
                    continue
                files.append({"name": d.name, **_task_meta(d, csv_fallback)})
                seen.add(d.name)
            # Surface CSV-only tasks whose trial dir never materialized on the volume.
            for task_name in csv_fallback:
                if task_name in seen:
                    continue
                files.append({"name": task_name, "passed": csv_fallback[task_name]["passed"],
                              "score": csv_fallback[task_name]["score"]})
            files.sort(key=lambda f: self._trial_sort_key(Path(f["name"])))
            self._send_json({"files": files, "has_compare": self.compare_dir is not None})
        except Exception as exc:
            self.send_error(500, f"Error listing trials: {exc}")

    def _handle_data(self, qs: dict, trials_dir: Path | None, csv_fallback: dict):
        name = qs.get("file", [None])[0]
        if not name or not trials_dir:
            return self.send_error(400, "Missing file parameter or directory not configured")
        # Strip a legacy .json suffix if the frontend passes one through.
        if name.endswith(".json"):
            name = name[:-5]
        trial_dir = trials_dir / name
        if not trial_dir.is_dir() and name not in csv_fallback:
            return self.send_error(404, f"Trial not found: {name}")
        try:
            self._send_json(_build_visualization_data(trial_dir, csv_fallback))
        except Exception as exc:
            self.send_error(500, f"Error building trial view: {exc}")

    def _handle_compare(self):
        if not self.trials_dir or not self.compare_dir:
            return self._send_json({"rows": []})
        try:
            rows = []
            task_names = set()
            for d in self.trials_dir.iterdir():
                if d.is_dir():
                    task_names.add(d.name)
            task_names |= set(self.csv_fallback.keys())
            for name in sorted(task_names, key=lambda n: self._trial_sort_key(Path(n))):
                m1 = _task_meta(self.trials_dir / name, self.csv_fallback)
                m2 = _task_meta(self.compare_dir / name, self.compare_csv_fallback)
                rows.append({
                    "task": name,
                    "p1_passed": m1["passed"],
                    "p1_score": m1["score"],
                    "p2_passed": m2["passed"],
                    "p2_score": m2["score"],
                })
            self._send_json({"rows": rows})
        except Exception as exc:
            self.send_error(500, f"Error building comparison: {exc}")

    @staticmethod
    def _trial_sort_key(path: Path):
        name = path.name
        if name.startswith("task-"):
            try:
                return (int(name.split("-")[1]), name)
            except (ValueError, IndexError):
                pass
        return (10**9, name)

    def log_message(self, format, *args):
        pass


def _make_handler(trials_dir: Path, compare_dir: Path | None,
                  csv_fallback: dict, compare_csv_fallback: dict,
                  artifacts_dir: Path | None = None,
                  compare_artifacts_dir: Path | None = None):
    class Handler(VisualizationHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                trials_dir=trials_dir,
                compare_dir=compare_dir,
                artifacts_dir=artifacts_dir,
                compare_artifacts_dir=compare_artifacts_dir,
                csv_fallback=csv_fallback,
                compare_csv_fallback=compare_csv_fallback,
                **kwargs,
            )
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Harbor run visualization server")
    parser.add_argument("--data-dir", required=True, help="Path to a run folder (or its trials/ subfolder)")
    parser.add_argument("--compare-dir", default=None, help="Optional second run for side-by-side comparison")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    trials_dir = _resolve_trials_dir(data_dir)
    if not trials_dir:
        print(f"Error: could not find a trials/ folder under {data_dir}")
        return

    compare_trials = None
    if args.compare_dir:
        compare_root = Path(args.compare_dir).resolve()
        compare_trials = _resolve_trials_dir(compare_root)
        if not compare_trials:
            print(f"Error: could not find a trials/ folder under {compare_root}")
            return

    csv_fallback = _load_csv_fallback(trials_dir)
    compare_csv_fallback = _load_csv_fallback(compare_trials) if compare_trials else {}

    # Artifacts dir is sibling to trials/ (e.g. .../<run-tag>/artifacts/<task>/<file>).
    artifacts_dir = trials_dir.parent / "artifacts"
    if not artifacts_dir.is_dir():
        artifacts_dir = None
    compare_artifacts_dir = None
    if compare_trials:
        compare_artifacts_dir = compare_trials.parent / "artifacts"
        if not compare_artifacts_dir.is_dir():
            compare_artifacts_dir = None

    os.chdir(Path(__file__).parent)
    handler_cls = _make_handler(trials_dir, compare_trials, csv_fallback, compare_csv_fallback,
                                artifacts_dir=artifacts_dir, compare_artifacts_dir=compare_artifacts_dir)
    server = HTTPServer((args.host, args.port), handler_cls)

    print(f"Visualization server: http://{args.host}:{args.port}")
    print(f"Trials dir:    {trials_dir}")
    if compare_trials:
        print(f"Compare trials: {compare_trials}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
