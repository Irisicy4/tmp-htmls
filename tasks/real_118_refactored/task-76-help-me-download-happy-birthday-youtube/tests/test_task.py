import sys
sys.path.insert(0, "/harness")
from evaluator import evaluate

TASK_INSTRUCTION = ("""Help me download a song named Happy Birthday from YouTube.""")

USER_PROMPT_TEMPLATE = ("""## Task Instruction
{task_instruction}

## Task-Specific Constraints
- Source: YouTube specifically
- File: audio download (MP3 or equivalent)
- Must use a legitimate download method available in the browser environment
- Should confirm the file was downloaded successfully

## Agent Final Response
{agent_response}

## Agent Tool-Call Trace (what the agent actually did)
{execution_summary}

---

## Evaluation Instructions

### Step 1: Evidence Analysis (do this before scoring)
- Did the agent search YouTube for Happy Birthday?
- What tool or method was used to download audio?
- Was the download successful?
- What file format and location was the file saved to?

### Step 2: Dimension Scoring

#### A. Youtube Access (0.2)
Did the agent find a Happy Birthday video on YouTube?

5 — Agent navigated to YouTube and found a suitable Happy Birthday audio/video.
4 — Agent found the content but with some difficulty.
3 — Agent found Happy Birthday content from a non-YouTube source.
2 — Agent described what to search without searching.
1 — No YouTube search.

#### B. Download Method (0.35)
Did the agent use an appropriate download method?

5 — Used a working method (yt-dlp, online converter, or browser tool) and executed it.
4 — Used a valid method but with some issues.
3 — Identified a valid method without executing it.
2 — Suggested an invalid or unavailable method.
1 — No download method.

#### C. Download Success (0.35)
Was the download completed successfully?

5 — Download confirmed with file name, format, and save location.
4 — Download appeared successful but without full confirmation.
3 — Download initiated but completion unclear.
2 — Download failed.
1 — No download attempted.

#### D. File Quality (0.1)
Is the downloaded file appropriate?

5 — Audio-only MP3 or equivalent, correct song, reasonable quality.
4 — Correct song but video format instead of audio.
3 — Correct content but unexpected format.
2 — Wrong content downloaded.
1 — No file.

### Step 3: Output
Respond ONLY with valid JSON inside <Answer></Answer> tags:

<Answer>
{{
  "evidence_summary": "<2-3 sentences summarising Step 1 findings>",
  "dimension_reasoning": {{
    "youtube_access": "<one sentence citing specific evidence>",
    "download_method": "<one sentence citing specific evidence>",
    "download_success": "<one sentence citing specific evidence>",
    "file_quality": "<one sentence citing specific evidence>",
  }},
  "youtube_access": <1-5>,
  "download_method": <1-5>,
  "download_success": <1-5>,
  "file_quality": <1-5>,
  "overall_score": <weighted average, one decimal>,
  "passed": <true or false based on overall_score >= 3.0>
}}
</Answer>
""")

DIMENSION_WEIGHTS = {
    "youtube_access": 0.2,
    "download_method": 0.35,
    "download_success": 0.35,
    "file_quality": 0.1,
}


def test(result):
    return evaluate(result, TASK_INSTRUCTION, USER_PROMPT_TEMPLATE, DIMENSION_WEIGHTS)
