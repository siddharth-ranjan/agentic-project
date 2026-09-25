"""
DEMO 1 - a single agent.

One LLM, a handful of tools, and the loop in agentic/agent.py.
Watch it: think -> call tool -> read result -> think again -> ... -> answer.

    python demo_single.py          # offline: scripted model, real tools
    python demo_single.py --live   # real Claude makes every decision
    python demo_single.py --gemini # real Gemini makes every decision
"""

import sys

from agentic.agent import Agent
from agentic.llm import pick_llm, text, tool_call
from agentic.tools import CALCULATOR, RUN_PYTHON, WRITE_FILE

TASK = ("If I invest $1000 at 5% annual compound interest, what do I have after 10 years? "
        "Then write a script interest.py that prints the balance for each year, and run it "
        "to double-check your number.")

SCRIPT = """\
balance = 1000.0
for year in range(1, 11):
    balance *= 1.05
    print(f"year {year:2d}: ${balance:,.2f}")
"""

# What a model would plausibly decide, turn by turn. Tool outputs are NOT scripted.
MOCK = [
    [text("I'll compute the closed form first: 1000 * 1.05^10."),
     tool_call("calculator", expression="1000 * 1.05 ** 10")],
    [text("≈ $1628.89. Now I'll write a year-by-year script to verify."),
     tool_call("write_file", path="interest.py", content=SCRIPT)],
    [tool_call("run_python", path="interest.py")],
    [text("After 10 years you'd have **$1,628.89**. The year-by-year script agrees "
          "(year 10: $1,628.89), so the calculation checks out.")],
]

if __name__ == "__main__":
    llm = pick_llm(sys.argv, MOCK)
    agent = Agent(
        name="assistant",
        system="You are a careful assistant. Use tools to compute and verify rather than guessing.",
        tools=[CALCULATOR, WRITE_FILE, RUN_PYTHON],
        llm=llm,
    )
    print("\n" + agent.run(TASK))
