"""
DEMO 2 - multi-agent: orchestrator + specialist workers.

                        ┌──────────────┐
            task ─────▶ │ orchestrator │  plans, delegates, decides when done
                        └──────┬───────┘
          ask_coder     ask_tester  ask_reviewer      (each is just a tool…)
               ▼             ▼            ▼
          ┌───────┐     ┌────────┐   ┌──────────┐
          │ coder │     │ tester │   │ reviewer │   (…that runs a whole agent)
          └───────┘     └────────┘   └──────────┘
            own prompt, own tools, own fresh context

Patterns you'll see in the trace:
  1. Delegation      - orchestrator hands work to a specialist
  2. Parallel fan-out - tester and reviewer run at the same time
  3. Feedback loop   - failures go back to the coder until tests pass
  4. Verification gate - plain code (not an LLM) runs the tests after the
                         orchestrator reports; a failure sends it back to work

    python demo_multi.py          # offline: scripted models, real tools
    python demo_multi.py --live   # real Claude drives every agent
    python demo_multi.py --gemini # real Gemini drives every agent
"""

import shutil
import subprocess
import sys

from agentic.agent import Agent, as_tool
from agentic.llm import pick_llm, text, tool_call
from agentic.tools import LIST_FILES, READ_FILE, RUN_PYTHON, WORKSPACE, WRITE_FILE

TASK = ("Build palindrome.py with a function is_palindrome(s) that ignores case, spaces and "
        "punctuation. Make sure it is tested and reviewed before you report back.")

# ---- scripted decisions for offline mode (tool outputs are real) -----------

V1 = '''def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]
'''
V2 = '''def is_palindrome(s: str) -> bool:
    """True if s reads the same backwards, ignoring case, spaces and punctuation."""
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
'''
TESTS = '''from palindrome import is_palindrome

cases = {
    "racecar": True,
    "Never odd or even": True,
    "A man, a plan, a canal: Panama": True,
    "hello": False,
    "": True,
}
failed = 0
for s, want in cases.items():
    got = is_palindrome(s)
    ok = got == want
    failed += not ok
    print(("PASS" if ok else "FAIL"), repr(s), "->", got)
print(f"{len(cases) - failed}/{len(cases)} passed")
raise SystemExit(failed)
'''

MOCK_ORCHESTRATOR = [
    [text("Plan: coder writes it, then tester and reviewer check it in parallel."),
     tool_call("ask_coder", task="Create palindrome.py with is_palindrome(s). It must ignore case, spaces and punctuation.")],
    [text("Code is in. Sending to tester and reviewer at the same time."),
     tool_call("ask_tester", task="Write and run test_palindrome.py for palindrome.is_palindrome, incl. punctuation cases."),
     tool_call("ask_reviewer", task="Review palindrome.py against the spec: ignore case, spaces, punctuation.")],
    [text("Both found the same bug: punctuation isn't stripped. Sending it back to the coder."),
     tool_call("ask_coder", task="Fix palindrome.py: it fails on 'A man, a plan, a canal: Panama' because punctuation "
                                 "is not removed. Then run test_palindrome.py and confirm all tests pass.")],
    [text("Done. palindrome.py handles case, spaces and punctuation; 5/5 tests pass; "
          "reviewer's issue was fixed in one iteration.")],
]
MOCK_CODER = [  # the coder agent is called twice; its script covers both runs
    [tool_call("write_file", path="palindrome.py", content=V1)],
    [text("Wrote palindrome.py (lowercases and strips spaces).")],
    [tool_call("write_file", path="palindrome.py", content=V2)],
    [tool_call("run_python", path="test_palindrome.py")],
    [text("Fixed: now keeps only alphanumeric chars. test_palindrome.py: 5/5 passed.")],
]
MOCK_TESTER = [
    [tool_call("write_file", path="test_palindrome.py", content=TESTS)],
    [tool_call("run_python", path="test_palindrome.py")],
    [text("4/5 passed. FAIL: 'A man, a plan, a canal: Panama' returned False - punctuation is not ignored.")],
]
MOCK_REVIEWER = [
    [tool_call("read_file", path="palindrome.py")],
    [text("Issue: only spaces are removed, so commas/colons break the check. "
          "Suggest filtering with str.isalnum(). Also add type hints and a docstring.")],
]

# ---- wiring ----------------------------------------------------------------

def build(argv: list) -> Agent:
    def llm(script):  # each agent gets its own backend (and its own mock script)
        return pick_llm(argv, script)

    coder = Agent("coder", "You are a Python developer. Write clean code to the workspace. "
                  "When asked to fix something, fix it and re-run any tests.",
                  [WRITE_FILE, READ_FILE, RUN_PYTHON, LIST_FILES], llm(MOCK_CODER))
    tester = Agent("tester", "You are a QA engineer. Write a runnable test script (plain asserts/prints, "
                   "no pytest), run it, and report exactly what passed and failed. Never create or edit the "
                   "code under test; if it is missing or broken, report that instead.",
                   [WRITE_FILE, READ_FILE, RUN_PYTHON, LIST_FILES], llm(MOCK_TESTER))
    reviewer = Agent("reviewer", "You are a strict code reviewer. Read the code and list concrete issues "
                     "against the spec. You cannot edit files.",
                     [READ_FILE, LIST_FILES], llm(MOCK_REVIEWER))

    return Agent(
        "orchestrator",
        "You are a tech lead. You do not write code yourself; delegate to your team via tools. "
        "Run only independent work in parallel: code must exist before it is tested or reviewed. Loop until tests pass and review issues are fixed, "
        "then give a short final report.",
        [as_tool(coder, "Delegate implementation or fixes to the coder."),
         as_tool(tester, "Have the tester write and run tests."),
         as_tool(reviewer, "Have the reviewer review code (read-only).")],
        llm(MOCK_ORCHESTRATOR),
    )


# ---- verification gate -----------------------------------------------------
#
# Agents report what they *believe*. This step checks what is *true*, in plain
# code that the model can't skip or argue with. Two checks:
#   1. Our own acceptance cases, taken from the spec. They live here, outside
#      workspace/, so no agent can edit them to make them pass.
#   2. Every test_*.py the agents wrote (those tests can be wrong too).

ACCEPTANCE = {
    "racecar": True,
    "A man, a plan, a canal: Panama": True,
    "No 'x' in Nixon": True,
    "Was it a car or a cat I saw?": True,
    "hello": False,
    "palindrome": False,
    "": True,
}
MAX_ROUNDS = 3


def _run(args: list) -> subprocess.CompletedProcess:
    # A fresh interpreter each time, so we test the file as it is on disk right now.
    return subprocess.run([sys.executable, *args], cwd=WORKSPACE, capture_output=True, text=True, timeout=60)


def verify() -> tuple[bool, str]:
    lines, ok = [], True

    check = ("from palindrome import is_palindrome\n"
             f"for s, want in {ACCEPTANCE!r}.items():\n"
             "    got = is_palindrome(s)\n"
             "    print(('PASS' if got == want else 'FAIL'), repr(s), 'expected', want, 'got', got)\n")
    proc = _run(["-c", check])
    failed = [l for l in proc.stdout.splitlines() if l.startswith("FAIL")]
    if proc.returncode != 0 or failed:
        ok = False
        lines.append("Acceptance cases (from the spec):")
        lines += failed or [proc.stderr.strip()[-800:]]
    else:
        lines.append(f"Acceptance cases: {len(ACCEPTANCE)}/{len(ACCEPTANCE)} passed")

    for test in sorted(WORKSPACE.glob("test_*.py")):
        proc = _run([test.name])
        if proc.returncode == 0:
            lines.append(f"{test.name}: passed")
        else:
            ok = False
            lines.append(f"{test.name}: FAILED (exit code {proc.returncode})\n"
                         + (proc.stdout + proc.stderr).strip()[-1500:])
    return ok, "\n".join(lines)


def show(title: str, body: str, color: str):
    print(f"\n\033[1;{color}m[verifier] {title}\033[0m")
    for line in body.splitlines():
        print(f"           {line}")


if __name__ == "__main__":
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir()
    orchestrator = build(sys.argv)

    task = TASK
    for round_no in range(1, MAX_ROUNDS + 1):
        report = orchestrator.run(task)
        ok, details = verify()
        if ok:
            show(f"round {round_no}: all checks passed", details, "32")
            break
        show(f"round {round_no}: checks FAILED, sending back to the orchestrator", details, "31")
        # Agent.run starts with an empty context, so the retry task must carry everything it needs.
        task = (f"{TASK}\n\nA previous attempt reported:\n{report}\n\n"
                f"But automatic verification failed:\n{details}\n\n"
                "Get this fixed. The spec is the source of truth: if a test contradicts the spec, "
                "fix the test; otherwise fix the code.")
    else:
        report += f"\n\n(UNVERIFIED: checks still failing after {MAX_ROUNDS} rounds)"

    print("\n" + "=" * 70 + "\nFINAL REPORT\n" + "=" * 70 + "\n" + report)
