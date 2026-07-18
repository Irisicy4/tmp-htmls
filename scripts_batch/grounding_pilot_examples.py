#!/usr/bin/env python3
"""One task, ALL invocations of baseline / A / B / C, logged verbatim
into markdown. Page content is factored out to placeholders so the
prompts stay readable."""
from __future__ import annotations

import json, os, re, sys
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

from agentic_judge.grounded.framework.faithfulness import _fetch_text
from agentic_judge.grounded.framework.extractor import extract_summary_json


def _chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
    if os.environ.get("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
    r = OpenAI(**kwargs).chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
    t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _agent_ctx(summary, target_url):
    def rec(obj):
        if isinstance(obj, dict):
            if any(v == target_url for v in obj.values() if isinstance(v, str)):
                return json.dumps({k:(v[:120] if isinstance(v,str) else v) for k,v in obj.items()}, ensure_ascii=False)[:400]
            for v in obj.values():
                r = rec(v)
                if r: return r
        elif isinstance(obj, list):
            for v in obj:
                r = rec(v)
                if r: return r
        return None
    return rec(summary) or json.dumps(summary)[:400]


def _get_path(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


TASK = "task-05-go-to-nbacom-and-check"
TASK_SUBDIR = "real_118"
MAX_URLS = 6

verdict_path = Path(f"/Users/icy/Desktop/act/2020604-evolvebench/_local_only/run_results_two_phase/by_task/{TASK}/modal_verdict.json")
d = json.load(verdict_path.open())
summary, _src = extract_summary_json(d.get("task_result",""))
agent_prose = d.get("task_result","") or ""

# Enumerate URLs from the summary in walk order
def walk_urls(obj):
    out = []
    seen = set()
    def rec(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(o, list):
            for v in o:
                rec(v, f"{prefix}[]")
        elif isinstance(o, str):
            for m in re.finditer(r"https?://[^\s\"'<>()\\\]]+", o):
                u = m.group(0)
                if u not in seen:
                    seen.add(u); out.append((prefix, u))
    rec(obj)
    return out

url_pairs = walk_urls(summary)[:MAX_URLS]
urls = [u for _p, u in url_pairs]

# Fetch each URL once
pages = {}
for u in urls:
    ok, status, text, err = _fetch_text(u)
    pages[u] = {"ok": ok, "text": text[:3000] if ok else f"[FETCH ERROR: {err}]"}

# Load schema
import importlib.util as _ilu
spec_path = REPO / "tasks" / TASK_SUBDIR / TASK / "tests" / "test_grounded.py"
spec = _ilu.spec_from_file_location("_ex_spec", spec_path)
task_mod = _ilu.module_from_spec(spec); sys.modules["_ex_spec"] = task_mod
task_mod.__name__ = "_ex_spec"; spec.loader.exec_module(task_mod)
schema = task_mod.SUMMARY_SCHEMA


def url_placeholder(u: str) -> str:
    """Compact label like URL_1_PAGE for placeholder use in prompts."""
    return f"{{{{URL_{urls.index(u)+1}_PAGE}}}}"


log = []
log.append(f"# Raw conversation examples — every invocation, all four methods")
log.append(f"")
log.append(f"**Task:** `{TASK}`  ")
log.append(f"**Number of URLs graded:** {len(urls)}  ")
log.append(f"**Model:** `gpt-4o-mini`, temperature 0  ")
log.append(f"")
log.append(f"All prompts below reference the fetched page content as `{{{{URL_i_PAGE}}}}` placeholders. "
           f"The actual page content is inlined once, at the top of each URL section, so the reader can "
           f"see what each placeholder resolves to. LLM responses are verbatim.")
log.append("")

# =========== 1. Agent JSON =============================================
log.append("---\n\n## Full agent output JSON")
log.append("")
log.append("```json")
log.append(json.dumps(summary, indent=2, ensure_ascii=False))
log.append("```")
log.append("")

# =========== 2. All fetched pages ======================================
log.append("---\n\n## Fetched pages (referenced by placeholder from every prompt below)")
log.append("")
for i, u in enumerate(urls, 1):
    log.append(f"### `URL_{i}_PAGE` — {u}")
    log.append("")
    log.append("```")
    log.append(pages[u]["text"])
    log.append("```")
    log.append("")

# =========== 3. Baseline for every URL =================================
log.append("---\n\n## Baseline (current framework) — one call per URL")
log.append("")
log.append("For each URL, we assemble a *claim* (the agent's JSON sub-object nearest that URL) and ask "
           "an LLM whether the fetched page supports the claim. Score is `5 × (# YES / # URLs)`.")
log.append("")

baseline_verdicts = []
for i, u in enumerate(urls, 1):
    ctx = _agent_ctx(summary, u)
    ph = url_placeholder(u)
    prompt = (
        "You are a source-checker. PAGE is the webpage text; CLAIM is "
        "the agent's context around citing that page. Does the PAGE "
        "support the CLAIM? Reply YES or NO with one short reason.\n\n"
        f"PAGE:\n{ph}\n\nCLAIM: {ctx[:300]}\n"
    )
    real_prompt = prompt.replace(ph, pages[u]["text"])
    resp = _chat(real_prompt)
    resp_clean = _strip_fences(resp)
    verdict = "YES" if resp_clean.upper().startswith("YES") else "NO"
    baseline_verdicts.append(verdict)
    log.append(f"### Call {i} of {len(urls)} — {u}")
    log.append("")
    log.append("**Prompt:**")
    log.append("```")
    log.append(prompt.rstrip())
    log.append("```")
    log.append("")
    log.append("**LLM response:**")
    log.append("```")
    log.append(resp_clean)
    log.append("```")
    log.append("")

yes_count = baseline_verdicts.count("YES")
log.append(f"**Baseline aggregate:** {yes_count}/{len(urls)} YES → score_5 = **{5*yes_count/len(urls):.2f}**")
log.append("")

# =========== 4. Option A: schema-guided per URL =========================
log.append("---\n\n## Option A — schema-guided forward re-extraction (two turns per URL)")
log.append("")
log.append("For each URL: **Turn 1** asks the judge to independently fill the schema fields the agent "
           "claimed to source from that URL, using only the page. **Turn 2** compares the judge's "
           "extraction against the agent's per field.")
log.append("")

def schema_fields_for_url(u):
    """Fields whose enclosing object contains this URL."""
    out = []
    for f in schema.required:
        parts = f.path.split(".")
        v = _get_path(summary, f.path)
        if v is None: continue
        enclosing = summary if len(parts)==1 else _get_path(summary, ".".join(parts[:-1]))
        if isinstance(enclosing, dict):
            if any(x == u for x in enclosing.values() if isinstance(x, str)):
                out.append((f.path, f.type_hint, f.description))
    return out

a_all_verdicts = []
for i, u in enumerate(urls, 1):
    fields = schema_fields_for_url(u)
    if not fields:
        log.append(f"### Call {i} of {len(urls)} — {u}")
        log.append("")
        log.append("*(no required fields in the agent's JSON are sourced from this URL — skipping)*")
        log.append("")
        continue
    ph = url_placeholder(u)
    schema_block = "\n".join(f"- {p} ({t}) — {desc}" for p, t, desc in fields)
    a1p = (
        "You are the grading system. Extract the following typed fields "
        "*from the webpage below* into a JSON object matching the field paths "
        "exactly. Use `null` for any field you cannot verify from the page. "
        "Do NOT infer, do NOT guess — only report what the page states.\n\n"
        f"URL: {u}\n\nEXPECTED FIELDS:\n{schema_block}\n\n"
        f"PAGE TEXT:\n{ph}\n\n"
        "Return only a JSON object, no preamble."
    )
    a1_real = a1p.replace(ph, pages[u]["text"])
    a1r = _chat(a1_real)
    a1r_clean = _strip_fences(a1r)

    agent_reported = {p: _get_path(summary, p) for p, _, _ in fields}
    a2p = (
        "You are given two JSON objects filling the same typed schema slots. "
        "For each field, decide whether the AGENT's value and the JUDGE's independent "
        "extraction from the same page agree. Output a JSON object mapping each field "
        "path to one of: EXACT, EQUIVALENT (numerically or semantically equal), "
        "MISMATCH (differ meaningfully), or JUDGE_NULL (judge could not extract).\n\n"
        f"AGENT REPORTED:\n{json.dumps(agent_reported, indent=2)}\n\n"
        f"JUDGE EXTRACTED (from the same page):\n{a1r_clean}\n\n"
        "Return only the JSON verdict object."
    )
    a2r = _chat(a2p)
    a2r_clean = _strip_fences(a2r)
    try:
        verdict_map = json.loads(a2r_clean)
        matches = sum(1 for v in verdict_map.values() if v in ("EXACT","EQUIVALENT"))
        total = len(verdict_map)
        a_all_verdicts.append((matches, total))
    except Exception:
        a_all_verdicts.append((0, 1))

    log.append(f"### Call {i} of {len(urls)} — {u}")
    log.append("")
    log.append("**Turn 1 — schema-guided extraction prompt:**")
    log.append("```")
    log.append(a1p.rstrip())
    log.append("```")
    log.append("")
    log.append("**Turn 1 response (judge's independent extraction):**")
    log.append("```json")
    log.append(a1r_clean)
    log.append("```")
    log.append("")
    log.append("**Turn 2 — per-field cross-check prompt:**")
    log.append("```")
    log.append(a2p.rstrip())
    log.append("```")
    log.append("")
    log.append("**Turn 2 response (per-field verdict):**")
    log.append("```json")
    log.append(a2r_clean)
    log.append("```")
    log.append("")

a_total = sum(t for _, t in a_all_verdicts)
a_match = sum(m for m, _ in a_all_verdicts)
log.append(f"**Option A aggregate:** {a_match}/{a_total} fields EXACT-or-EQUIVALENT → score_5 = **{5*a_match/max(a_total,1):.2f}**")
log.append("")

# =========== 5. Option B: NLI per URL ==================================
log.append("---\n\n## Option B — NLI-style entailment (one call per URL)")
log.append("")
log.append("For each URL, the fetched page is the PREMISE and the agent's JSON sub-object citing that "
           "URL is the HYPOTHESIS. LLM returns ENTAILS / NEUTRAL / CONTRADICTS. Score is "
           "`5 × (# ENTAILS / # URLs)`.")
log.append("")

b_verdicts = []
for i, u in enumerate(urls, 1):
    ctx = _agent_ctx(summary, u)
    ph = url_placeholder(u)
    bp = (
        "You are an NLI classifier. Given the PREMISE (webpage text) and "
        "the HYPOTHESIS (agent claim), decide: ENTAILS, NEUTRAL, or CONTRADICTS. "
        "Reply with a single word.\n\n"
        f"PREMISE:\n{ph}\n\nHYPOTHESIS:\n{ctx[:300]}\n"
    )
    real_bp = bp.replace(ph, pages[u]["text"])
    bresp = _chat(real_bp)
    bresp_clean = _strip_fences(bresp).upper().split()[0] if bresp else "ERROR"
    b_verdicts.append(bresp_clean)
    log.append(f"### Call {i} of {len(urls)} — {u}")
    log.append("")
    log.append("**Prompt:**")
    log.append("```")
    log.append(bp.rstrip())
    log.append("```")
    log.append("")
    log.append("**LLM response:**")
    log.append("```")
    log.append(bresp_clean)
    log.append("```")
    log.append("")

b_yes = sum(1 for v in b_verdicts if v == "ENTAILS")
log.append(f"**Option B aggregate:** {b_yes}/{len(urls)} ENTAILS → score_5 = **{5*b_yes/len(urls):.2f}**  "
           f"(NEUTRAL: {b_verdicts.count('NEUTRAL')}, CONTRADICTS: {b_verdicts.count('CONTRADICTS')})")
log.append("")

# =========== 6. Option C: decompose + verify per claim ==================
log.append("---\n\n## Option C — RAGAS-style atomic-claim faithfulness")
log.append("")
log.append("**Stage 1** decomposes the agent's prose into atomic claims (one LLM call). "
           "**Stage 2** verifies each claim against the concatenation of all fetched pages "
           "(one LLM call per claim). Score is `5 × (# SUPPORTED / # atomic claims)`.")
log.append("")

c1p = (
    "Decompose the following text into a list of atomic factual claims "
    "(one per line, no numbering, no bullets, each a self-contained "
    "declarative sentence). Aim for 5-10 claims covering the most "
    "important facts. Do not add claims not present in the text.\n\n"
    f"TEXT:\n{agent_prose[:4000]}\n"
)
c1r = _chat(c1p)
c1r_clean = _strip_fences(c1r)
atomic = [l.strip("-• \t") for l in c1r_clean.splitlines() if l.strip()][:10]

log.append("### Stage 1 — decompose prose into atomic claims (1 call)")
log.append("")
log.append("**Prompt:**")
log.append("```")
log.append(c1p.rstrip())
log.append("```")
log.append("")
log.append("**LLM response (atomic claims):**")
log.append("```")
log.append(c1r_clean)
log.append("```")
log.append("")

corpus_placeholder = "{{ALL_URLS_CONCATENATED}}"
corpus_real = "\n\n".join(f"=== {u} ===\n{pages[u]['text']}" for u in urls)

log.append(f"### Stage 2 — verify each claim against `{corpus_placeholder}` (one call per claim)")
log.append("")

c_verdicts = []
for j, claim in enumerate(atomic, 1):
    c2p = (
        "Given the CORPUS (concatenated webpages cited by the agent), "
        "decide if the CLAIM is SUPPORTED by any part of it. "
        "Reply with a single word: SUPPORTED, UNSUPPORTED, or CONTRADICTED.\n\n"
        f"CORPUS:\n{corpus_placeholder}\n\nCLAIM: {claim}\n"
    )
    real_c2p = c2p.replace(corpus_placeholder, corpus_real[:15000])
    c2r = _chat(real_c2p)
    c2r_clean = _strip_fences(c2r).upper().split()[0] if c2r else "ERROR"
    c_verdicts.append(c2r_clean)
    log.append(f"#### Claim {j} of {len(atomic)}")
    log.append("")
    log.append(f"> {claim}")
    log.append("")
    log.append("**Prompt:**")
    log.append("```")
    log.append(c2p.rstrip())
    log.append("```")
    log.append("")
    log.append("**LLM response:**")
    log.append("```")
    log.append(c2r_clean)
    log.append("```")
    log.append("")

c_yes = sum(1 for v in c_verdicts if v == "SUPPORTED")
log.append(f"**Option C aggregate:** {c_yes}/{len(atomic)} atomic claims SUPPORTED → score_5 = **{5*c_yes/max(len(atomic),1):.2f}**  "
           f"(UNSUPPORTED: {c_verdicts.count('UNSUPPORTED')}, CONTRADICTED: {c_verdicts.count('CONTRADICTED')})")
log.append("")

# =========== Summary ====================================================
log.append("---\n\n## Method summary for this task")
log.append("")
log.append("| method | invocations | verdict | score_5 |")
log.append("|---|---|---|---|")
log.append(f"| baseline | {len(urls)} | {yes_count} YES / {len(urls)} | {5*yes_count/len(urls):.2f} |")
log.append(f"| A (re-extract) | {sum(1 for _u in urls if schema_fields_for_url(_u))} URLs × 2 turns | {a_match} EXACT-or-EQUIVALENT / {a_total} fields | {5*a_match/max(a_total,1):.2f} |")
log.append(f"| B (NLI) | {len(urls)} | {b_yes} ENTAILS / {len(urls)} | {5*b_yes/len(urls):.2f} |")
log.append(f"| C (RAGAS) | 1 decompose + {len(atomic)} verify | {c_yes} SUPPORTED / {len(atomic)} | {5*c_yes/max(len(atomic),1):.2f} |")

out = Path("/Users/icy/Desktop/act/2020604-evolvebench/_local_only/grounding_options_examples.md")
out.write_text("\n".join(log))
print(f"wrote {out}")
