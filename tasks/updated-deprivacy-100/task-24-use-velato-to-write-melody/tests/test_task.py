"""LLM-as-judge evaluator for task-24-use-velato-to-write-melody.

Category: Software Engineering
Task: Use Velato to write a melody and save the project. The melody should have a Christmas feel, and when the program represented by the melody is compiled, it should output a greeting like 'Merry Christma
"""

import sys
sys.path.insert(0, "/harness")
from evaluator import run_judge

TASK_INSTRUCTION = "Use Velato to write a melody and save the project. The melody should have a Christmas feel, and when the program represented by the melody is compiled, it should output a greeting like 'Merry Christmas'. Ensure the note arrangement encodes the letters logically, and the generated code must include compilation instructions."

SYSTEM_PROMPT_EXTRA = 'You are assessing a task involving Velato — an esoteric programming language where programs are encoded as MIDI music.\n\nKey context:\n- Velato programs are MIDI files where note intervals encode instructions\n- The melody must work as both: (a) a Christmas-sounding piece of music, and (b) a valid Velato program that outputs "Merry Christmas"\n- This is a highly technical and niche task; partial credit should be given generously for correct understanding of Velato even if execution is imperfect'

CONSTRAINTS = """- Language: Velato specifically (not abc notation, LilyPond, or generic MIDI)
- Output: the compiled program must output "Merry Christmas" (or equivalent greeting)
- Christmas feel: melody should use Christmas-associated notes/patterns (pentatonic, carol-like)
- File saved: a project file (.mid or Velato-compatible format) must be saved
- Compilation instructions: must be included (how to run the Velato interpreter on the file)"""

EVIDENCE_QUESTIONS = """- Did the agent demonstrate understanding of how Velato encodes programs in music?
- Did the agent produce a MIDI or Velato project file?
- Does the encoded program logically map to "Merry Christmas" output?
- Were compilation/run instructions provided?
- Is there a Christmas feel to the melody?"""

DIMENSION_RUBRICS = """#### A. Velato Understanding
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
1 — No compilation instructions and no Christmas character."""

DIMENSION_WEIGHTS = {
    'velato_understanding': 0.25,
    'program_correctness': 0.35,
    'file_saved': 0.25,
    'compilation_christmas': 0.15,
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
