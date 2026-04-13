"""
LLM-as-judge evaluator for EvolveBench task-39.
Category: Marketing & Analytics
"""
import os, json, re

TASK_INSTRUCTION = 'Help me search Xiaohongshu for highly-liked posts about AI + psychology and AI companionship products, and organize the main points of each post and the corresponding bloggers into a document.'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks assessing a Xiaohongshu content research and document compilation task.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Platform: Xiaohongshu (小红书) specifically\n- Topics: AI + psychology AND AI companionship products (both topic areas required)\n- Filter: highly-liked posts (sort by likes or engagement)\n- Output: structured document with per-post main points AND blogger info\n- Document must be saved (not just shown in response)\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent navigate Xiaohongshu? Cite evidence.\n- Were both topic areas (AI+psychology AND AI companionship) searched?\n- How many posts were found and summarised?\n- Are blogger names/handles included?\n- Was a document saved?\n\n### Step 2: Dimension Scoring\n\n#### A. Platform Execution\nDid the agent navigate Xiaohongshu as instructed?\n\n5 — Agent navigated Xiaohongshu, searched for both topic areas, and retrieved posts sorted by likes.\n4 — Agent accessed Xiaohongshu for one topic area; the other was supplemented from elsewhere.\n3 — Agent referenced Xiaohongshu content without clearly navigating the platform.\n2 — Agent used a different platform (e.g. WeChat, Weibo) without explanation.\n1 — No platform navigation.\n\n#### B. Topic Relevance\nDo the posts cover both required topic areas?\n\n5 — Posts from both "AI + psychology" and "AI companionship products" included; topics clearly distinguished.\n4 — Both topics present but one area has fewer posts or less depth.\n3 — Only one topic area represented; the other is missing or barely mentioned.\n2 — Posts are broadly AI-related but not specifically psychology or companionship.\n1 — Topics ignored or irrelevant posts included.\n\n#### C. Content Extraction\nAre the main points of each post extracted and blogger info included?\n\n5 — 5+ posts summarised with: main point(s) per post AND blogger name/handle; summaries are substantive (not just post titles).\n4 — 3–4 posts summarised with main points; blogger info present for most.\n3 — 2–3 posts summarised; blogger info missing for some; summaries are thin.\n2 — Posts listed by title only without main point extraction.\n1 — No post content extracted.\n\n#### D. Document Saved\nWas the compiled content saved to a document?\n\n5 — Document saved (file, Google Docs, etc.); path, URL, or trace confirmation; document is well-structured.\n4 — Document creation attempted; trace confirms write; content present.\n3 — Content structured in response that could be saved; no file created.\n2 — Content present but unstructured; would require significant reformatting.\n1 — No document output.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "platform_execution": <1-5>,\n  "topic_relevance": <1-5>,\n  "content_extraction": <1-5>,\n  "document_saved": <1-5>,\n  "dimension_reasoning": {{"platform_execution": "<one sentence>", "topic_relevance": "<one sentence>", "content_extraction": "<one sentence>", "document_saved": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'platform_execution': 0.25, 'topic_relevance': 0.25, 'content_extraction': 0.25, 'document_saved': 0.25}
DIMENSIONS = list(DIMENSION_WEIGHTS.keys())

def _extract_response(result):
    t = result.get("task_result") or ""
    if isinstance(t, str) and t.strip(): return t
    for m in reversed(result.get("conversation") or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content") or ""
            if isinstance(c, str) and len(c) > 20: return c
    return ""

def _parse(text):
    m = re.search(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        try: return json.loads(m.group(1).strip())
        except Exception: pass
    try: return json.loads(text.strip())
    except Exception: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def _call(agent_response, execution_summary):
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": USER_PROMPT_TEMPLATE.format(task_instruction=TASK_INSTRUCTION, agent_response=agent_response, execution_summary=execution_summary or "Not available.")}],
            max_tokens=1024)
        return _parse(completion.choices[0].message.content)
    except Exception as e: return {"error": str(e)}

def _vote(votes):
    valid = [v for v in votes if v and "error" not in v and all(d in v for d in DIMENSIONS)]
    if not valid: return votes[0] if votes else {"error": "All calls failed"}
    agg = {d: sorted([v[d] for v in valid])[len(valid)//2] for d in DIMENSIONS}
    overall = sum(agg[d] * DIMENSION_WEIGHTS[d] for d in DIMENSIONS)
    agg["overall_score"] = round(overall, 2); agg["passed"] = overall >= PASS_THRESHOLD
    median = sorted(valid, key=lambda v: abs(v.get("overall_score",0)-overall))[0]
    agg["evidence_summary"] = median.get("evidence_summary",""); agg["dimension_reasoning"] = median.get("dimension_reasoning",{}); agg["_votes_used"] = len(valid)
    return agg

def test(result):
    agent_response = _extract_response(result)
    execution_summary = result.get("execution_summary", "")
    if not agent_response.strip():
        return {"passed": False, "feedback": "No response found from agent.", "details": {"task_completed": result.get("status") == "success"}}
    first = _call(agent_response, execution_summary)
    if first and "error" not in first:
        overall = first.get("overall_score", 0)
        scores = _vote([first, _call(agent_response, execution_summary), _call(agent_response, execution_summary)]) if abs(float(overall) - PASS_THRESHOLD) <= 0.5 else (first.__setitem__("_votes_used", 1) or first)
    else:
        scores = first or {"error": "Judge call failed", "overall_score": 0}
    overall = scores.get("overall_score", 0); passed = scores.get("passed", float(overall) >= PASS_THRESHOLD)
    lines = [f"Overall score: {overall}/5  (threshold: {PASS_THRESHOLD})"] + [f"  {d}: {scores[d]}/5" for d in DIMENSIONS if d in scores]
    if scores.get("evidence_summary"): lines.append(f"\nEvidence summary: {scores['evidence_summary']}")
    if scores.get("dimension_reasoning"):
        lines.append("\nDimension reasoning:")
        for d, r in scores["dimension_reasoning"].items(): lines.append(f"  {d}: {r}")
    if scores.get("_votes_used", 1) > 1: lines.append(f"\n(Borderline: {scores['_votes_used']} calls, majority vote)")
    return {"passed": bool(passed), "feedback": "\n".join(lines), "details": {"task_completed": result.get("status") == "success", "overall_score": overall, "dimension_scores": {d: scores.get(d) for d in DIMENSIONS}, "evidence_summary": scores.get("evidence_summary"), "dimension_reasoning": scores.get("dimension_reasoning"), "pass_threshold": PASS_THRESHOLD, "votes_used": scores.get("_votes_used", 1)}}