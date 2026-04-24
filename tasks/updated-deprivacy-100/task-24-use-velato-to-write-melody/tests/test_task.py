"""
LLM-as-judge evaluator for EvolveBench task-24.

Category: Software Engineering
Task: Use Velato to write a Christmas melody that outputs "Merry Christmas" when compiled.
      Save the project. Include compilation instructions.

Velato is an esoteric programming language where code is represented as music/MIDI.
Notes encode instructions; the melody must be both musically Christmas-like AND
produce "Merry Christmas" as program output when compiled.
"""

import os, json, re

TASK_INSTRUCTION = """Use Velato to write a melody and save the project. The melody should have a Christmas feel, and when the program represented by the melody is compiled, it should output a greeting like 'Merry Christmas'. Ensure the note arrangement encodes the letters logically, and the generated code must include compilation instructions."""
PASS_THRESHOLD = 3.0

SYSTEM_PROMPT = """You are an expert evaluator for AI agent benchmarks. You are assessing a task involving Velato — an esoteric programming language where programs are encoded as MIDI music.

Key context:
- Velato programs are MIDI files where note intervals encode instructions
- The melody must work as both: (a) a Christmas-sounding piece of music, and (b) a valid Velato program that outputs "Merry Christmas"
- This is a highly technical and niche task; partial credit should be given generously for correct understanding of Velato even if execution is imperfect"""

USER_PROMPT_TEMPLATE = """## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Language: Velato specifically (not abc notation, LilyPond, or generic MIDI)
- Output: the compiled program must output "Merry Christmas" (or equivalent greeting)
- Christmas feel: melody should use Christmas-associated notes/patterns (pentatonic, carol-like)
- File saved: a project file (.mid or Velato-compatible format) must be saved
- Compilation instructions: must be included (how to run the Velato interpreter on the file)

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis
- Did the agent demonstrate understanding of how Velato encodes programs in music?
- Did the agent produce a MIDI or Velato project file?
- Does the encoded program logically map to "Merry Christmas" output?
- Were compilation/run instructions provided?
- Is there a Christmas feel to the melody?

### Step 2: Dimension Scoring

#### A. Velato Understanding
Does the agent correctly understand and apply Velato's encoding rules?

5 — Agent correctly explains Velato's note-interval encoding and maps specific notes to produce "Merry Christmas"; encoding is logically correct.
4 — Agent shows correct understanding of Velato but one aspect of the encoding is imprecise.
3 — Agent demonstrates general awareness of Velato but encoding details are vague or partially wrong.
2 — Agent confused Velato with another music-coding tool or made significant encoding errors.
1 — Agent has no understanding of Velato or fabricated a response.

#### B. Program Correctness
Does the note arrangement logically encode a program that would output "Merry Christmas"?

5 — Note sequence demonstrably encodes "Merry Christmas" via Velato's rules; agent verifies the encoding.
4 — Encoding is mostly correct with minor errors that would likely still produce the right output.
3 — Encoding approach is correct but has errors that might produce wrong/partial output.
2 — Encoding is attempted but fundamentally flawed; output would not be "Merry Christmas".
1 — No valid encoding; response is generic or fabricated.

#### C. File & Project Saved
Was a project file created and saved?

5 — MIDI or Velato project file created and saved; file path or confirmation provided.
4 — File creation attempted; trace confirms file write but no explicit path in response.
3 — Agent provided note sequence/code that could be saved but did not create the file.
2 — Agent described what to do without producing any file output.
1 — No file output of any kind.

#### D. Compilation Instructions & Christmas Feel
Were compilation instructions included, and does the melody have a Christmas character?

5 — Complete compilation instructions provided (Velato interpreter, how to run) AND melody is described as having Christmas character (carol-like intervals, familiar patterns).
4 — Compilation instructions present; Christmas feel mentioned but not well-integrated.
3 — Either compilation instructions OR Christmas feel addressed, not both.
2 — Neither adequately addressed.
1 — No compilation instructions and no Christmas character.

### Step 3: Output
<Answer>
{{
  "evidence_summary": "<2-3 sentences>",
  "velato_understanding": <1-5>,
  "program_correctness": <1-5>,
  "file_saved": <1-5>,
  "compilation_christmas": <1-5>,
  "dimension_reasoning": {{"velato_understanding": "<one sentence>", "program_correctness": "<one sentence>", "file_saved": "<one sentence>", "compilation_christmas": "<one sentence>"}},
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false>
}}
</Answer>
"""

DIMENSION_WEIGHTS = {"velato_understanding": 0.25, "program_correctness": 0.35, "file_saved": 0.25, "compilation_christmas": 0.15}
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