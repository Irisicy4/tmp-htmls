"""
LLM-as-judge evaluator for EvolveBench task-42.
Category: Software Engineering
"""
import os, json, re

TASK_INSTRUCTION = 'Create a complete modern website for a French art museum with 3D. Here is the style of site I want: https://www.ciaokombucha.com'
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = 'You are an expert evaluator for AI agent benchmarks. This is a creative web development task requiring 3D elements in the style of a specific reference site (ciaokombucha.com — known for bold typography, scroll animations, and immersive 3D/WebGL elements). The agent should have fetched the reference site to understand the style.'

USER_PROMPT_TEMPLATE = '## Task Instruction\n{task_instruction}\n\n## Task-Specific Constraints\n- Reference style: https://www.ciaokombucha.com — bold typography, scroll-triggered animations, immersive 3D/WebGL, dark aesthetic, cinematic feel\n- Theme: French art museum — must include relevant museum content (collections, exhibitions, visitor info)\n- 3D requirement: must use Three.js, WebGL, or equivalent for actual 3D elements (not just CSS transforms)\n- Completeness: "complete" website with multiple sections (hero, collections, exhibitions, about, visit)\n\n## Agent Final Response\n{agent_response}\n\n## Agent Tool-Call Trace\n{execution_summary}\n\n---\n\n## Evaluation Instructions\n\n### Step 1: Evidence Analysis\n- Did the agent visit the reference site? Cite evidence.\n- What 3D technology was used (Three.js, WebGL, CSS 3D)?\n- How many website sections are present?\n- Does the visual style match the reference (bold type, dark, immersive)?\n- Was a file created and saved?\n\n### Step 2: Dimension Scoring\n\n#### A. Reference Style Adoption\nDoes the website reflect the style of ciaokombucha.com?\n\n5 — Clear influence: bold/large typography, dark or rich colour palette, scroll animations, cinematic/immersive feel — all present.\n4 — 2–3 style elements present; overall feel is close but one key element missing.\n3 — Modern design but generic; doesn\'t capture the specific reference aesthetic.\n2 — Standard bootstrap/template design with no reference to the style guide.\n1 — No website or completely wrong visual direction.\n\n#### B. 3D Implementation\nAre actual 3D elements implemented using an appropriate library?\n\n5 — Three.js or WebGL used for real-time 3D elements (e.g. rotating sculpture, 3D gallery, particle system, 3D hero animation).\n4 — 3D elements present using CSS 3D transforms or a simpler library; not full WebGL but visually 3D.\n3 — Animated elements present but no true 3D depth (e.g. parallax only).\n2 — Static layout with no 3D or animation of any kind.\n1 — No 3D; agent described what 3D would look like without implementing it.\n\n#### C. Museum Content\nDoes the website have appropriate French art museum content?\n\n5 — Multiple sections with museum-appropriate content: collection highlights, current/upcoming exhibitions, visit information (hours, address), about the museum.\n4 — 3–4 sections with relevant content; one area thin or placeholder-only.\n3 — 2–3 sections present; content is very generic (not museum-specific).\n2 — Single-page skeleton with minimal content.\n1 — No museum content; purely generic template.\n\n#### D. Code Completeness\nIs the website code complete and runnable?\n\n5 — Complete HTML/CSS/JS in one or few files; external libraries correctly imported; no obvious syntax errors; all sections linked together.\n4 — Mostly complete; one section or feature is stubbed or broken.\n3 — Significant code present but incomplete; would need fixes to run.\n2 — Skeleton code only; most features not implemented.\n1 — No code; only a description.\n\n### Step 3: Output\n<Answer>\n{{\n  "evidence_summary": "<2-3 sentences>",\n  "reference_style_adoption": <1-5>,\n  "3d_implementation": <1-5>,\n  "museum_content": <1-5>,\n  "code_completeness": <1-5>,\n  "dimension_reasoning": {{"reference_style_adoption": "<one sentence>", "3d_implementation": "<one sentence>", "museum_content": "<one sentence>", "code_completeness": "<one sentence>"}},\n  "overall_score": <weighted average, one decimal>,\n  "passed": <true or false>\n}}\n</Answer>'

DIMENSION_WEIGHTS = {'reference_style_adoption': 0.25, '3d_implementation': 0.3, 'museum_content': 0.2, 'code_completeness': 0.25}
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