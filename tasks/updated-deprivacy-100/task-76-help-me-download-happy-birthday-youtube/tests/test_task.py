"""LLM-as-judge evaluator for task-76-help-me-download-happy-birthday-youtube.

Category: Daily Activities
Task: Help me download a song named Happy Birthday from YouTube.
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = 'Help me download a song named Happy Birthday from YouTube.'

SYSTEM_PROMPT_EXTRA = 'Your job is to assess whether an AI agent successfully completed a browser-based task.\n\nThe task involves finding and downloading an audio file of Happy Birthday from YouTube.'

CONSTRAINTS = """- Source: YouTube specifically
- File: audio download (MP3 or equivalent)
- Must use a legitimate download method available in the browser environment
- Should confirm the file was downloaded successfully"""

EVIDENCE_QUESTIONS = """- Did the agent search YouTube for Happy Birthday?
- What tool or method was used to download audio?
- Was the download successful?
- What file format and location was the file saved to?"""

DIMENSION_RUBRICS = """#### A. Youtube Access (0.2)
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
1 — No file."""

DIMENSION_WEIGHTS = {
    'youtube_access': 0.2,
    'download_method': 0.35,
    'download_success': 0.35,
    'file_quality': 0.1,
}


def test(result):
    return run_judge(
        result,
        task_instruction=TASK_INSTRUCTION,
        system_prompt_extra=SYSTEM_PROMPT_EXTRA,
        constraints=CONSTRAINTS,
        evidence_questions=EVIDENCE_QUESTIONS,
        dimension_rubrics=DIMENSION_RUBRICS,
        dimension_weights=DIMENSION_WEIGHTS,
    )
