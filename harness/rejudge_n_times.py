"""
Re-run the LLM judge N times per task to characterize judge variance.

For each (task, judge_model, call_index) we save a row containing the full raw
judge response, parsed dimension scores, weighted overall score, pass/fail,
plus the input hash so the colleague can verify all calls of a task got the
same prompt.

Two rubric flavors in tasks/updated-deprivacy-100/:
  A (94 tasks): SYSTEM_PROMPT + USER_PROMPT_TEMPLATE + DIMENSION_WEIGHTS,
                judge replies with JSON inside <Answer>...</Answer>.
  B (6 tasks):  Single RUBRIC string + DIMENSIONS list, judge replies with bare
                JSON; weights embedded in RUBRIC text (parsed by regex).

Resumable: skips (task, judge_model, call_index) tuples already in the JSONL.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import openai


# ── Env ───────────────────────────────────────────────────────────────────────

def load_env(env_file: Path) -> None:
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


# ── Task module loading ───────────────────────────────────────────────────────

def load_test_module(task_name: str, tasks_root: Path):
    test_path = tasks_root / task_name / "tests" / "test_task.py"
    if not test_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"tt_{task_name}", test_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def detect_flavor(mod) -> str:
    """A: SYSTEM_PROMPT/USER_PROMPT_TEMPLATE/DIMENSION_WEIGHTS.  B: RUBRIC."""
    if all(hasattr(mod, k) for k in ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "DIMENSION_WEIGHTS")):
        return "A"
    if hasattr(mod, "RUBRIC") and hasattr(mod, "DIMENSIONS"):
        return "B"
    return "?"


def extract_weights_from_rubric(rubric_text: str, dims: list[str]) -> dict[str, float]:
    """Pull `weight 0.36` out of RUBRIC text per dimension."""
    weights: dict[str, float] = {}
    for d in dims:
        m = re.search(rf"{re.escape(d)}.*?weight\s*([\d.]+)", rubric_text, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                weights[d] = float(m.group(1))
            except ValueError:
                pass
    return weights


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(mod, trace: dict) -> dict:
    flavor = detect_flavor(mod)
    agent_response = mod._extract_response(trace)
    execution_summary = trace.get("execution_summary", "") or ""

    if flavor == "A":
        fmt_kwargs = dict(
            task_instruction=getattr(mod, "TASK_INSTRUCTION", ""),
            agent_response=agent_response,
            execution_summary=execution_summary or "Not available.",
        )
        try:
            user_content = mod.USER_PROMPT_TEMPLATE.format(**fmt_kwargs)
        except (KeyError, IndexError, ValueError):
            tmpl = mod.USER_PROMPT_TEMPLATE
            sentinels = {
                "\x00TI\x00": fmt_kwargs["task_instruction"],
                "\x00AR\x00": fmt_kwargs["agent_response"],
                "\x00ES\x00": fmt_kwargs["execution_summary"],
            }
            tmpl = tmpl.replace("{task_instruction}", "\x00TI\x00")
            tmpl = tmpl.replace("{agent_response}", "\x00AR\x00")
            tmpl = tmpl.replace("{execution_summary}", "\x00ES\x00")
            tmpl = tmpl.replace("{{", "{").replace("}}", "}")
            for k, v in sentinels.items():
                tmpl = tmpl.replace(k, v)
            user_content = tmpl
        system = mod.SYSTEM_PROMPT
        dim_weights = dict(mod.DIMENSION_WEIGHTS)
        dimensions = list(mod.DIMENSION_WEIGHTS)
    elif flavor == "B":
        system = ""
        user_content = f"{mod.RUBRIC}\n\nAgent response to evaluate:\n\n{agent_response}"
        if execution_summary:
            user_content += (
                "\n\nVerified agent tool-call trace (ground truth of what the agent actually did):\n"
                + execution_summary
            )
        dimensions = list(mod.DIMENSIONS)
        dim_weights = extract_weights_from_rubric(mod.RUBRIC, dimensions)
    else:
        raise ValueError(f"Unknown rubric flavor for module {mod.__name__}")

    h = hashlib.sha256(((system or "") + "||" + user_content).encode("utf-8")).hexdigest()[:16]
    return {
        "flavor": flavor,
        "system": system,
        "user_content": user_content,
        "input_hash": h,
        "agent_response_len": len(agent_response),
        "execution_summary_len": len(execution_summary),
        "dimensions": dimensions,
        "dimension_weights": dim_weights,
        "pass_threshold": float(mod.PASS_THRESHOLD),
    }


# ── Judge call & parsing ──────────────────────────────────────────────────────

def call_judge(client, model: str, system: str, user: str, temperature: float,
               max_tokens: int = 1024, timeout: int = 120) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    completion = client.chat.completions.create(
        model=model,
        messages=msgs,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return completion.choices[0].message.content


_ANSWER_RE = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_json(text: str):
    if not text:
        return None
    m = _ANSWER_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return None


def score_from_parsed(parsed, prompt_meta: dict) -> dict:
    dims = prompt_meta["dimensions"]
    weights = prompt_meta["dimension_weights"]
    threshold = prompt_meta["pass_threshold"]

    if not isinstance(parsed, dict):
        return {"parse_error": "judge response is not a JSON object"}

    dim_scores: dict = {}
    missing = []
    for d in dims:
        v = parsed.get(d)
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            dim_scores[d] = int(v)
        else:
            dim_scores[d] = None
            missing.append(d)

    if weights and all(dim_scores.get(d) is not None for d in dims):
        overall = round(sum(dim_scores[d] * weights.get(d, 0.0) for d in dims), 4)
        overall_source = "recomputed"
    else:
        try:
            overall = round(float(parsed.get("overall_score")), 4)
            overall_source = "judge_reported"
        except (TypeError, ValueError):
            overall = None
            overall_source = "missing"

    passed = (overall is not None) and (overall >= threshold)

    return {
        "dimension_scores": dim_scores,
        "overall_score": overall,
        "overall_source": overall_source,
        "passed": bool(passed) if overall is not None else None,
        "evidence_summary": parsed.get("evidence_summary"),
        "dimension_reasoning": parsed.get("dimension_reasoning") or parsed.get("reasoning"),
        "parse_error": ("missing dims: " + ",".join(missing)) if missing else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="cocoa-deprivacy-100")
    p.add_argument("--results-root", default="results")
    p.add_argument("--tasks-root", default="tasks/updated-deprivacy-100")
    p.add_argument("--out-dir", default="results/judge_variance")
    p.add_argument("--n-calls", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-models", default="gpt-4o,claude-sonnet-4-20250514")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of tasks (smoke test). 0 = all.")
    p.add_argument("--tasks", default="",
                   help="Comma-separated task names (default: all in source).")
    args = p.parse_args()

    load_env(Path(".env"))
    if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("LLM_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not api_key or not base_url:
        sys.exit("Missing OPENAI_API_KEY or LLM_BASE_URL/OPENAI_BASE_URL")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    print(f"base_url = {base_url}")

    source = args.source
    results_root = Path(args.results_root)
    tasks_root = Path(args.tasks_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_path = out_dir / f"{source}_runs.jsonl"
    inputs_path = out_dir / f"{source}_inputs.jsonl"
    summary_path = out_dir / f"{source}_summary.json"

    trials_root = results_root / source / "trials"

    def find_trace(task_dir: Path) -> Path | None:
        """Cocoa traces live in one of three places, in priority order."""
        for rel in ("agent/result.json", "agent_result.json", "result.json"):
            p = task_dir / rel
            if p.exists():
                return p
        return None

    if args.tasks:
        tasks_avail = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks_avail = sorted(
            x.name for x in trials_root.iterdir()
            if x.is_dir() and find_trace(x) is not None
        )
    if args.limit:
        tasks_avail = tasks_avail[: args.limit]
    print(f"Tasks: {len(tasks_avail)} under {trials_root}")

    judge_models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    print(f"Judges: {judge_models}  calls/task/model: {args.n_calls}  "
          f"temp: {args.temperature}  workers: {args.workers}")

    existing_keys = set()
    if runs_path.exists():
        with open(runs_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing_keys.add((r["task"], r["judge_model"], r["call_index"]))
                except Exception:
                    pass
        print(f"Resume: {len(existing_keys)} calls already on disk")

    inputs_seen = set()
    if inputs_path.exists():
        with open(inputs_path) as f:
            for line in f:
                try:
                    inputs_seen.add(json.loads(line)["task"])
                except Exception:
                    pass

    task_meta: dict[str, dict] = {}
    for task_name in tasks_avail:
        mod = load_test_module(task_name, tasks_root)
        if mod is None:
            print(f"  [skip] {task_name}: no test_task.py")
            continue
        trace_path = find_trace(trials_root / task_name)
        if trace_path is None:
            print(f"  [skip] {task_name}: no trace file found")
            continue
        try:
            with open(trace_path) as f:
                trace = json.load(f)
            # The eval-output file uses _task_result instead of task_result
            if "task_result" not in trace and "_task_result" in trace:
                trace["task_result"] = trace.get("_task_result")
            meta = build_prompt(mod, trace)
        except Exception as e:
            print(f"  [skip] {task_name}: prompt build failed: {e}")
            continue
        task_meta[task_name] = meta

        if task_name not in inputs_seen:
            with open(inputs_path, "a") as f:
                f.write(json.dumps({
                    "source": source,
                    "task": task_name,
                    "rubric_flavor": meta["flavor"],
                    "input_hash": meta["input_hash"],
                    "agent_response_len": meta["agent_response_len"],
                    "execution_summary_len": meta["execution_summary_len"],
                    "dimensions": meta["dimensions"],
                    "dimension_weights": meta["dimension_weights"],
                    "pass_threshold": meta["pass_threshold"],
                    "system_prompt": meta["system"],
                    "user_content": meta["user_content"],
                    "task_instruction": getattr(mod, "TASK_INSTRUCTION", ""),
                }) + "\n")

    print(f"Prompts ready for {len(task_meta)} tasks")

    todo = [
        (t, jm, i)
        for t in task_meta
        for jm in judge_models
        for i in range(1, args.n_calls + 1)
        if (t, jm, i) not in existing_keys
    ]
    print(f"Calls to make: {len(todo)} (skipped {len(existing_keys)} already done)")
    if not todo:
        print("Nothing to do.")
        return

    write_lock = threading.Lock()
    progress = {"done": 0, "ok": 0, "err": 0}

    def do_one(task_name: str, jm: str, idx: int) -> dict:
        meta = task_meta[task_name]
        started_dt = datetime.now(timezone.utc)
        api_err = None
        raw = None
        try:
            raw = call_judge(client, jm, meta["system"], meta["user_content"], args.temperature)
        except Exception as e:
            api_err = f"{type(e).__name__}: {e}"
        finished_dt = datetime.now(timezone.utc)

        scored: dict = {"parse_error": None}
        if raw is not None:
            parsed = extract_json(raw)
            if parsed is None:
                scored = {"parse_error": "could not extract JSON"}
            else:
                scored = score_from_parsed(parsed, meta)

        return {
            "source": source,
            "task": task_name,
            "judge_model": jm,
            "call_index": idx,
            "temperature": args.temperature,
            "rubric_flavor": meta["flavor"],
            "input_hash": meta["input_hash"],
            "raw_response": raw,
            "dimension_scores": scored.get("dimension_scores"),
            "overall_score": scored.get("overall_score"),
            "overall_source": scored.get("overall_source"),
            "passed": scored.get("passed"),
            "pass_threshold": meta["pass_threshold"],
            "dimension_weights": meta["dimension_weights"],
            "evidence_summary": scored.get("evidence_summary"),
            "dimension_reasoning": scored.get("dimension_reasoning"),
            "parse_error": scored.get("parse_error"),
            "api_error": api_err,
            "started_at": started_dt.isoformat(),
            "finished_at": finished_dt.isoformat(),
            "duration_sec": round((finished_dt - started_dt).total_seconds(), 3),
        }

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(do_one, t, jm, i): (t, jm, i) for (t, jm, i) in todo}
        for fut in as_completed(futs):
            t, jm, i = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {
                    "source": source, "task": t, "judge_model": jm, "call_index": i,
                    "temperature": args.temperature,
                    "raw_response": None, "dimension_scores": None,
                    "overall_score": None, "passed": None,
                    "parse_error": None, "api_error": f"worker: {e}",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
            with write_lock:
                with open(runs_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            progress["done"] += 1
            if rec.get("api_error") or rec.get("parse_error"):
                progress["err"] += 1
            else:
                progress["ok"] += 1
            if progress["done"] % 10 == 0 or progress["done"] == len(todo):
                el = time.time() - t0
                rate = progress["done"] / el if el else 0
                eta = (len(todo) - progress["done"]) / rate if rate else 0
                print(f"  [{progress['done']}/{len(todo)}]  ok={progress['ok']}  err={progress['err']}  "
                      f"elapsed={el:.0f}s  eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    summary = {
        "source": source,
        "tasks_count": len(task_meta),
        "judge_models": judge_models,
        "n_calls_per_task_per_model": args.n_calls,
        "temperature": args.temperature,
        "workers": args.workers,
        "calls_attempted_this_run": len(todo),
        "calls_ok_this_run": progress["ok"],
        "calls_err_this_run": progress["err"],
        "elapsed_sec": round(elapsed, 1),
        "runs_path": str(runs_path),
        "inputs_path": str(inputs_path),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. {progress['ok']} ok, {progress['err']} errors in {elapsed:.1f}s")
    print(f"Runs   : {runs_path}")
    print(f"Inputs : {inputs_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
