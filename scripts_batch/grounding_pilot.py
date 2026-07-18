#!/usr/bin/env python3
"""Pilot three alternative Layer-② grounding checks against the same
agent outputs. Layers ① and ③ unchanged; only how the sub-agent verifies
web claims differs.

Options:
  A — Schema-guided forward re-extraction + cross-check
  B — NLI-style entailment (LLM-approximated: entails / neutral / contradicts)
  C — RAGAS-style atomic-claim faithfulness

Reads agent outputs already stored under run_results_two_phase/by_task/,
so we do not re-dispatch the agent — the pilot only changes the judge.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "harness"))

# Load .env for API keys
for line in (REPO_ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
if not os.environ.get("OPENAI_BASE_URL") and os.environ.get("LLM_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]

from agentic_judge.grounded.framework.faithfulness import _fetch_text, _is_safe_url
from agentic_judge.grounded.framework.extractor import extract_summary_json


RESULTS_ROOT = Path("/Users/icy/Desktop/act/2020604-evolvebench/_local_only/run_results_two_phase/by_task")
OUT_PATH = Path("/Users/icy/Desktop/act/2020604-evolvebench/_local_only/grounding_options_pilot.md")

PILOT_TASKS = [
    ("real_118", "task-01-im-looking-for-backpack-under"),
    ("long_horizon/web_research", "task-ai-regulation-tracker"),
    ("long_horizon/web_research", "task-w13-open-dataset-finder"),
    ("long_horizon/stem_research", "task-ibuprofen-pharmacokinetics"),
    ("long_horizon/web_research", "task-w15-clinical-trial-finder"),
]

MAX_URLS_PER_TASK = 6  # sample the first N URLs to keep the pilot cheap


def _openai_chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    kwargs = {"api_key": key}
    if base:
        kwargs["base_url"] = base
    client = OpenAI(**kwargs)
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def _load_summary(task_dir: Path) -> tuple[dict, str]:
    m = json.load((task_dir / "modal_verdict.json").open())
    text = m.get("task_result", "") or ""
    summary, _src = extract_summary_json(text)
    return summary or {}, text


def _walk_urls(summary) -> list[tuple[str, str]]:
    """Return (schema-path, url) pairs found anywhere in the summary."""
    out: list[tuple[str, str]] = []

    def rec(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                rec(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                rec(v, f"{prefix}[]")
        elif isinstance(obj, str):
            for m in re.finditer(r"https?://[^\s\"'<>()\\\]]+", obj):
                out.append((prefix, m.group(0)))

    rec(summary)
    seen = set()
    dedup = []
    for p, u in out:
        if u in seen:
            continue
        seen.add(u)
        dedup.append((p, u))
    return dedup


# -------------- Option A: forward re-extraction --------------------------

def option_a(task_dir: Path, summary: dict, page_cache: dict) -> dict:
    """Ask the judge to independently extract expected values from each URL
    without seeing the agent's answer, then cross-check."""
    findings = []
    started = time.time()
    urls = _walk_urls(summary)[:MAX_URLS_PER_TASK]
    total_tokens_est = 0

    for schema_path, url in urls:
        page = page_cache.get(url)
        if not page or page.get("error"):
            findings.append({"url": url, "path": schema_path, "match": False,
                             "kind": "unfetched", "detail": page.get("error", "?") if page else "?"})
            continue
        text = page["text"][:6000]
        # Ask the LLM to describe what the page is *about* — a schema-agnostic extraction
        prompt = (
            "You are the grading system. You have a webpage the agent cited. "
            "Extract in one short sentence what the page is primarily about "
            "(topic, entity, main product/service). Do NOT read the agent's answer.\n\n"
            f"URL: {url}\nPAGE TEXT (truncated):\n{text}\n\n"
            "Return only the one-sentence description, no preamble."
        )
        try:
            judge_desc = _openai_chat(prompt).strip()
            total_tokens_est += len(prompt.split()) + len(judge_desc.split())
        except Exception as e:
            findings.append({"url": url, "path": schema_path, "match": False,
                             "kind": "llm_error", "detail": str(e)[:120]})
            continue

        # Now check whether the agent's context around this URL is consistent
        agent_context = _agent_context_for_url(summary, url)
        cross = _openai_chat(
            "You have (1) an independent one-sentence description of a webpage, "
            "and (2) the surrounding context the agent wrote when citing it. "
            "Rate agreement: EXACT, PARAPHRASE, CONTRADICTION, or UNKNOWN. "
            "Reply with a single word.\n\n"
            f"WEBPAGE DESCRIPTION: {judge_desc}\n"
            f"AGENT CONTEXT: {agent_context[:400]}\n"
        ).strip().upper().split()[0]
        findings.append({
            "url": url, "path": schema_path,
            "judge_extracted": judge_desc[:200],
            "agent_context": agent_context[:200],
            "kind": cross,
            "match": cross in ("EXACT", "PARAPHRASE"),
        })
    dt = time.time() - started
    agreed = sum(1 for f in findings if f["match"])
    return {"option": "A",
            "urls_checked": len(findings),
            "agreement_rate": (agreed / len(findings)) if findings else None,
            "score_5": (5.0 * agreed / len(findings)) if findings else 0.0,
            "wall_s": round(dt, 1),
            "tokens_est": total_tokens_est,
            "findings": findings}


def _agent_context_for_url(summary: dict, url: str) -> str:
    """Return the JSON sub-object nearest the URL, as a compact string."""
    def rec(obj):
        if isinstance(obj, dict):
            if any(v == url for v in obj.values() if isinstance(v, str)):
                return json.dumps({k: (v[:120] if isinstance(v, str) else v) for k, v in obj.items()}, ensure_ascii=False)[:400]
            for v in obj.values():
                r = rec(v)
                if r: return r
        elif isinstance(obj, list):
            for v in obj:
                r = rec(v)
                if r: return r
        return None
    return rec(summary) or json.dumps(summary)[:400]


# -------------- Option B: NLI-style entailment via LLM ------------------

def option_b(task_dir: Path, summary: dict, page_cache: dict) -> dict:
    findings = []
    started = time.time()
    urls = _walk_urls(summary)[:MAX_URLS_PER_TASK]
    total_tokens_est = 0

    for schema_path, url in urls:
        page = page_cache.get(url)
        if not page or page.get("error"):
            findings.append({"url": url, "path": schema_path, "verdict": "unfetched",
                             "detail": page.get("error", "?") if page else "?"})
            continue
        # Build a claim from the agent's context around this URL
        claim = _agent_context_for_url(summary, url)[:300]
        premise = page["text"][:6000]
        prompt = (
            "You are an NLI classifier. Given the PREMISE (webpage text) and "
            "the HYPOTHESIS (agent claim), decide: ENTAILS, NEUTRAL, or CONTRADICTS. "
            "Reply with a single word.\n\n"
            f"PREMISE:\n{premise}\n\nHYPOTHESIS:\n{claim}\n"
        )
        try:
            v = _openai_chat(prompt).strip().upper().split()[0]
            total_tokens_est += len(prompt.split()) + 1
        except Exception as e:
            v = "ERROR"
        findings.append({"url": url, "path": schema_path, "verdict": v})
    dt = time.time() - started
    entails = sum(1 for f in findings if f.get("verdict") == "ENTAILS")
    total = sum(1 for f in findings if f.get("verdict") in ("ENTAILS", "NEUTRAL", "CONTRADICTS"))
    return {"option": "B",
            "urls_checked": len(findings),
            "entails": entails, "neutral": sum(1 for f in findings if f.get("verdict")=="NEUTRAL"),
            "contradicts": sum(1 for f in findings if f.get("verdict")=="CONTRADICTS"),
            "citation_precision": (entails / total) if total else None,
            "score_5": (5.0 * entails / len(findings)) if findings else 0.0,
            "wall_s": round(dt, 1),
            "tokens_est": total_tokens_est,
            "findings": findings}


# -------------- Option C: RAGAS-style atomic claim faithfulness ---------

def option_c(task_dir: Path, summary: dict, page_cache: dict, agent_prose: str = "") -> dict:
    findings = []
    started = time.time()
    prose = (agent_prose or "")[:8000]
    urls = _walk_urls(summary)[:MAX_URLS_PER_TASK]
    total_tokens_est = 0

    # 1) Decompose prose into atomic factual claims
    decompose_prompt = (
        "Decompose the following text into a list of atomic factual claims "
        "(one per line, no numbering, no bullets, each a self-contained "
        "declarative sentence). Aim for 5-10 claims covering the most "
        "important facts. Do not add claims not present in the text.\n\n"
        f"TEXT:\n{prose}\n"
    )
    try:
        raw = _openai_chat(decompose_prompt)
        total_tokens_est += len(decompose_prompt.split()) + len(raw.split())
        atomic = [l.strip("-• \t") for l in raw.splitlines() if l.strip()][:10]
    except Exception as e:
        return {"option": "C", "error": f"decompose failed: {e}", "score_5": 0.0}

    # 2) Build a corpus from the union of fetched pages
    corpus = ""
    for _p, u in urls:
        pg = page_cache.get(u)
        if pg and not pg.get("error"):
            corpus += f"\n=== {u} ===\n" + pg["text"][:2000]
    corpus = corpus[:12000]

    # 3) For each atomic claim, verify against the corpus
    supported = 0
    for claim in atomic:
        prompt = (
            "Given the CORPUS (concatenated webpages cited by the agent), "
            "decide if the CLAIM is SUPPORTED by any part of it. "
            "Reply with a single word: SUPPORTED, UNSUPPORTED, or CONTRADICTED.\n\n"
            f"CORPUS:\n{corpus}\n\nCLAIM: {claim}\n"
        )
        try:
            v = _openai_chat(prompt).strip().upper().split()[0]
            total_tokens_est += len(prompt.split()) + 1
        except Exception:
            v = "ERROR"
        if v == "SUPPORTED": supported += 1
        findings.append({"claim": claim[:180], "verdict": v})

    dt = time.time() - started
    faith = supported / len(atomic) if atomic else 0
    return {"option": "C",
            "atomic_claims": len(atomic),
            "supported": supported,
            "faithfulness": faith,
            "score_5": 5.0 * faith,
            "wall_s": round(dt, 1),
            "tokens_est": total_tokens_est,
            "findings": findings}


# --------------------- Baseline (current) ---------------------------------

def option_baseline(task_dir: Path, summary: dict, page_cache: dict) -> dict:
    """Re-invoke the current LLM-verifier on the same URLs for comparison."""
    findings = []
    started = time.time()
    urls = _walk_urls(summary)[:MAX_URLS_PER_TASK]
    total_tokens_est = 0
    for schema_path, url in urls:
        page = page_cache.get(url)
        if not page or page.get("error"):
            findings.append({"url": url, "supported": False,
                             "detail": page.get("error", "?") if page else "?"})
            continue
        claim = _agent_context_for_url(summary, url)[:300]
        prompt = (
            "You are a source-checker. PAGE is the webpage text; CLAIM is "
            "the agent's context around citing that page. Does the PAGE "
            "support the CLAIM? Reply YES or NO with one short reason.\n\n"
            f"PAGE:\n{page['text'][:6000]}\n\nCLAIM: {claim}\n"
        )
        try:
            r = _openai_chat(prompt).strip()
            supported = r.upper().startswith("YES")
            total_tokens_est += len(prompt.split()) + len(r.split())
        except Exception as e:
            supported = False; r = f"error: {e}"
        findings.append({"url": url, "supported": supported, "verdict": r[:120]})
    dt = time.time() - started
    yes = sum(1 for f in findings if f.get("supported"))
    return {"option": "baseline",
            "urls_checked": len(findings),
            "supported": yes,
            "citation_precision": (yes / len(findings)) if findings else None,
            "score_5": 5.0 * yes / len(findings) if findings else 0.0,
            "wall_s": round(dt, 1),
            "tokens_est": total_tokens_est,
            "findings": findings}


# --------------------- Runner ---------------------------------------------

def run_one_task(task_name: str, subdir: str) -> dict:
    task_dir = REPO_ROOT / "tasks" / subdir / task_name
    result_dir = RESULTS_ROOT / task_name
    if not result_dir.exists():
        return {"task": task_name, "error": "no verdict dir"}

    summary, agent_text = _load_summary(result_dir)
    if not summary:
        return {"task": task_name, "error": "no summary JSON in agent output"}

    # Fetch pages once — shared across all four modes
    urls = _walk_urls(summary)[:MAX_URLS_PER_TASK]
    print(f"[{task_name}] fetching {len(urls)} URLs...")
    page_cache = {}
    for _p, u in urls:
        safe, reason = _is_safe_url(u)
        if not safe:
            page_cache[u] = {"error": reason}
            continue
        ok, status, text, err = _fetch_text(u)
        page_cache[u] = {"text": text, "error": None if ok else err}

    print(f"[{task_name}] running baseline / A / B / C ...")
    bl = option_baseline(task_dir, summary, page_cache)
    a = option_a(task_dir, summary, page_cache)
    b = option_b(task_dir, summary, page_cache)
    c = option_c(task_dir, summary, page_cache, agent_prose=agent_text)
    return {"task": task_name, "n_urls": len(urls),
            "baseline": bl, "A": a, "B": b, "C": c}


def main():
    results = []
    for subdir, task in PILOT_TASKS:
        try:
            r = run_one_task(task, subdir)
            results.append(r)
        except Exception as e:
            results.append({"task": task, "error": f"{type(e).__name__}: {e}"})

    # Report
    lines = [
        "# Grounding-check options — 5-task pilot",
        "",
        f"Same agent outputs (two-phase codex verdicts) re-graded with four "
        f"different Layer-② implementations. Each option scores web-grounding on "
        f"the same URLs. Layers ① (hard constraints) and ③ (rubric) unchanged.",
        "",
        "## Per-task scores (score_5 = grounding component only)",
        "",
        "| task | n_urls | baseline | A (re-extract) | B (NLI) | C (RAGAS) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['task']} | — | — | — | — | — (error: {r.get('error')}) |")
            continue
        def _s(o): return f"{o['score_5']:.2f}" if isinstance(o.get('score_5'),(int,float)) else "?"
        lines.append(
            f"| {r['task']} | {r['n_urls']} | {_s(r['baseline'])} | {_s(r['A'])} | {_s(r['B'])} | {_s(r['C'])} |"
        )

    # Aggregate
    def _agg(key):
        xs = [r[key]['score_5'] for r in results if key in r and isinstance(r[key].get('score_5'),(int,float))]
        return round(sum(xs)/len(xs), 2) if xs else None
    def _wc(key):
        xs = [r[key].get('wall_s',0) for r in results if key in r]
        return round(sum(xs), 1)
    def _tk(key):
        xs = [r[key].get('tokens_est',0) for r in results if key in r]
        return sum(xs)
    lines += [
        "",
        "## Aggregate",
        "",
        "|             | baseline | A (re-extract) | B (NLI) | C (RAGAS) |",
        "|-------------|---:|---:|---:|---:|",
        f"| mean score_5 | {_agg('baseline')} | {_agg('A')} | {_agg('B')} | {_agg('C')} |",
        f"| wall-clock s  | {_wc('baseline')} | {_wc('A')} | {_wc('B')} | {_wc('C')} |",
        f"| ~tokens (LLM in+out approx) | {_tk('baseline')} | {_tk('A')} | {_tk('B')} | {_tk('C')} |",
        "",
        "## Per-task detail",
        "",
    ]
    for r in results:
        if "error" in r:
            continue
        lines.append(f"### {r['task']}")
        for key in ("baseline", "A", "B", "C"):
            o = r[key]
            lines.append(f"- **{key}**: {json.dumps({k:v for k,v in o.items() if k!='findings'}, ensure_ascii=False)}")
        lines.append("")
    OUT_PATH.write_text("\n".join(lines))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
