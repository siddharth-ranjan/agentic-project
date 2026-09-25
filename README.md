# Agentic workflows: a hands-on demo

A small agent framework, about 200 lines, with no framework dependencies, so every part is visible.

```
agentic/
  agent.py   ← THE LOOP (read this first) + as_tool() for multi-agent
  tools.py   ← tools = Python function + JSON schema
  llm.py     ← ClaudeLLM (real) / MockLLM (scripted, offline)
  trace.py   ← coloured, indented trace output
demo_single.py   ← one agent
demo_multi.py    ← orchestrator + coder / tester / reviewer
workspace/       ← the only folder the agents can read or write (git-ignored)
docs/agent-loop.html ← step-through diagrams of both demos
```

Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`

## Run it

```bash
.venv/bin/python demo_single.py        # offline
.venv/bin/python demo_multi.py         # offline

export GEMINI_API_KEY=...              # Google AI Studio key
.venv/bin/python demo_single.py --gemini   # real Gemini decides everything
.venv/bin/python demo_multi.py  --gemini

export ANTHROPIC_API_KEY=sk-ant-...    # or: ant auth login
.venv/bin/python demo_single.py --live     # real Claude decides everything
.venv/bin/python demo_multi.py  --live
```

In **offline mode** the model's *decisions* are scripted, but the tools really
run. The files, test output, and failures you see are all real. In **live mode**
Claude (`claude-opus-5`) or Gemini (`gemini-3.5-flash-lite`, override with
`GEMINI_MODEL`) makes every decision, so each run can take a different path.
Swapping the model only means swapping the class in `llm.py`; the loop in
`agent.py` stays the same.

## Concept 1: an agent is an LLM + tools + a loop

```
messages = [task]
while True:
    reply = llm(system, messages, tools)      # THINK
    if reply has no tool calls: return reply  # DONE
    results = run(reply.tool_calls)           # ACT   (your code runs it, not the LLM)
    messages += reply + results               # OBSERVE, then loop again
```

The model can't execute anything. It emits a request like
`{"type":"tool_use","name":"run_python","input":{...}}`. Your code runs it and
sends back a `tool_result`. Because the model sees results, including errors,
it can adapt its next step. That feedback is what makes it "agentic."

## Concept 2: multi-agent means agents used as tools

`as_tool(agent)` wraps an entire agent as a tool. When the orchestrator calls
`ask_coder(task=...)`, a separate agent starts with:

- **its own system prompt** (a specialised role)
- **its own tool set** (the reviewer can't write files, which is least privilege)
- **a fresh, empty context**, so it sees only the task it was given, not the whole history

Only the worker's *final answer* returns to the orchestrator. That isolation
keeps every context small and focused, which is the main reason to split work
across agents.

Patterns shown in `demo_multi.py`:

| Pattern | Where in the trace |
|---|---|
| **Orchestrator–worker** | the orchestrator never writes code; it only delegates |
| **Parallel fan-out** | tester and reviewer run at the same time (two tool calls in one turn) |
| **Evaluator–optimizer loop** | failed tests and review notes go back to the coder, who fixes and re-runs |
| **Verification gate** | after the orchestrator reports, plain Python runs fixed acceptance cases plus every `test_*.py`; a failure sends the task back (up to 3 rounds) |

## When *not* to use agents

If the steps are known ahead of time, a plain **workflow** is cheaper and more
predictable: fixed code that calls the LLM at set points (for example
draft → review → fix as three hard-coded calls). Use agents when the *path*
depends on what gets discovered along the way.

## Try next

1. Run `--live` and compare Claude's path with the scripted one.
2. Add a tool in `tools.py` (e.g. `http_get`) and give it to one agent.
3. Edit `ACCEPTANCE` in `demo_multi.py` so it expects something the agents weren't told, and watch the gate send the work back.
4. Break something on purpose (bad path, syntax error) and watch the agent recover from the `is_error` result.
5. Further reading: Anthropic's "Building effective agents" guide and the
   `anthropics/claude-cookbooks` repo (`patterns/agents/`).
