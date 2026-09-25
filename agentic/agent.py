"""
THE AGENT LOOP - the single most important idea in agentic systems.

    agent = LLM + tools + a loop

    messages = [user task]
    loop:
        reply = LLM(system, messages, tools)
        messages += reply
        if reply has no tool calls:  -> done, return the text
        run each requested tool
        messages += tool results      -> go around again

The model decides *what* to do next; our code does it and reports back.
That feedback loop is what makes it "agentic" rather than one-shot.
"""

from concurrent.futures import ThreadPoolExecutor

from .tools import Tool
from . import trace


class Agent:
    def __init__(self, name: str, system: str, tools: list[Tool], llm, max_turns: int = 15):
        self.name = name
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.max_turns = max_turns

    def run(self, task: str, depth: int = 0) -> str:
        log = trace.Logger(self.name, depth)
        log.task(task)

        # Fresh memory per run. Each agent only ever sees ITS OWN conversation -
        # this isolation is what makes multi-agent systems scale.
        messages = [{"role": "user", "content": task}]
        schemas = [t.schema() for t in self.tools.values()]

        for turn in range(1, self.max_turns + 1):
            # 1. THINK - ask the model what to do next
            reply = self.llm.respond(self.system, messages, schemas)
            messages.append({"role": "assistant", "content": reply["content"]})

            for block in reply["content"]:
                if block["type"] == "text" and block["text"].strip():
                    log.thought(turn, block["text"])

            if reply["stop_reason"] == "refusal":
                return "(the model declined this request)"

            calls = [b for b in reply["content"] if b["type"] == "tool_use"]
            if not calls:
                # 3. DONE - no tool requested means the model has its final answer
                final = "\n".join(b["text"] for b in reply["content"] if b["type"] == "text")
                log.done(turn, final)
                return final

            # 2. ACT - run the requested tools. Independent calls run in parallel.
            for c in calls:
                log.call(turn, c["name"], c["input"])
            with ThreadPoolExecutor() as pool:
                results = list(pool.map(self._execute, calls))
            for c, (out, is_err) in zip(calls, results):
                log.result(c["name"], out, is_err)

            # 3. OBSERVE - all results go back in ONE user message
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"], "content": out, "is_error": is_err}
                for c, (out, is_err) in zip(calls, results)
            ]})

        return "(stopped: hit max_turns)"

    def _execute(self, call: dict) -> tuple[str, bool]:
        tool = self.tools.get(call["name"])
        if tool is None:
            return f"unknown tool {call['name']}", True
        try:
            return tool.fn(**call["input"]), False
        except Exception as e:  # errors go back to the model so it can recover
            return f"{type(e).__name__}: {e}", True


def as_tool(agent: Agent, description: str) -> Tool:
    """
    THE MULTI-AGENT TRICK: wrap a whole agent as a tool.

    To the orchestrator, "ask the coder" looks exactly like calling a calculator.
    Under the hood it spins up a sub-agent with its own prompt, tools and a
    clean context, runs its full loop, and returns only the final answer.
    """
    def delegate(task: str) -> str:
        return agent.run(task, depth=1)  # depth=1 just indents its trace

    return Tool(
        name=f"ask_{agent.name}",
        description=description,
        input_schema={"type": "object",
                      "properties": {"task": {"type": "string",
                                              "description": "Complete, self-contained instructions."}},
                      "required": ["task"]},
        fn=delegate,
    )
