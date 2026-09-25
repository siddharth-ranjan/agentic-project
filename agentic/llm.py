"""
The "brain" of an agent: something that takes a conversation + tool list and
returns the next assistant message.

Two interchangeable backends:
  * ClaudeLLM - the real Claude API (needs ANTHROPIC_API_KEY or `ant auth login`)
  * GeminiLLM - Google Gemini (needs GEMINI_API_KEY or GOOGLE_API_KEY)
  * MockLLM   - replays a scripted list of responses, so you can watch the
                agent loop run offline. The *tools still execute for real*;
                only the model's decisions are pre-written.

Both return the same shape - a dict:
    {"stop_reason": "tool_use" | "end_turn" | ...,
     "content": [ {"type": "text", "text": ...},
                  {"type": "tool_use", "id": ..., "name": ..., "input": {...}} ]}
"""

import itertools

MODEL = "claude-opus-5"
GEMINI_MODEL = "gemini-3.5-flash-lite"  # override with the GEMINI_MODEL env var


class ClaudeLLM:
    def __init__(self, model: str = MODEL):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def respond(self, system: str, messages: list, tools: list) -> dict:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            # If a safety classifier declines, the server retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        return {
            "stop_reason": response.stop_reason,
            # Plain dicts, so they can be appended straight back into `messages`.
            "content": [b.model_dump(exclude_none=True) for b in response.content],
        }


class GeminiLLM:
    """Google Gemini backend. Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the env.

    The agent loop keeps messages in one neutral shape (the dicts above); this
    class translates them to Gemini's Content/Part format and back. That
    translation is all it takes to swap the "brain" - the loop doesn't change.
    """

    def __init__(self, model: str | None = None):
        import os
        from google import genai

        self.client = genai.Client()
        self.model = model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)

    def respond(self, system: str, messages: list, tools: list) -> dict:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"])
                for t in tools])] if tools else None,
            # We run the loop ourselves, so turn off the SDK's automatic tool calling.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._generate(self._to_contents(messages), config)

        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.finish_reason in (
                types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT):
            return {"stop_reason": "refusal", "content": []}

        content = []
        for part in candidate.content.parts or []:
            if part.function_call:
                fc = part.function_call
                content.append({"type": "tool_use", "id": fc.id or f"call_{next(_ids)}",
                                "name": fc.name, "input": dict(fc.args or {}),
                                # Gemini requires its original part (with thought_signature) sent back.
                                "_gemini_part": part})
            elif part.text and not part.thought:
                content.append({"type": "text", "text": part.text, "_gemini_part": part})
        has_tool = any(b["type"] == "tool_use" for b in content)
        return {"stop_reason": "tool_use" if has_tool else "end_turn", "content": content}

    def _generate(self, contents, config, attempts: int = 6):
        """Call Gemini, waiting out 429 rate limits (the free tier allows ~15 requests/min)."""
        import re
        import time
        from google.genai import errors

        for attempt in range(attempts):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except errors.ClientError as e:
                if e.code != 429 or attempt == attempts - 1:
                    raise
                m = re.search(r"retry in ([\d.]+)s", str(e))
                wait = float(m.group(1)) + 1 if m else 30
                print(f"      ⏳ rate limited by Gemini, waiting {wait:.0f}s…")
                time.sleep(wait)

    @staticmethod
    def _to_contents(messages: list) -> list:
        from google.genai import types

        names = {}  # tool_use_id -> tool name (Gemini function responses are matched by name)
        contents = []
        for m in messages:
            if isinstance(m["content"], str):
                contents.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
                continue
            parts = []
            for b in m["content"]:
                if "_gemini_part" in b:
                    parts.append(b["_gemini_part"])
                    if b["type"] == "tool_use":
                        names[b["id"]] = b["name"]
                elif b["type"] == "tool_result":
                    key = "error" if b.get("is_error") else "result"
                    part = types.Part.from_function_response(
                        name=names[b["tool_use_id"]], response={key: b["content"]})
                    part.function_response.id = b["tool_use_id"]
                    parts.append(part)
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=parts))
        return contents


def _load_dotenv():
    """Read KEY=value lines from ./.env into the environment (existing vars win)."""
    import os
    from pathlib import Path

    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.removeprefix("export ").strip(), value.strip().strip("\"'"))


def pick_llm(argv: list, script: list):
    """--live -> Claude, --gemini -> Gemini, otherwise the offline scripted mock."""
    _load_dotenv()
    if "--gemini" in argv:
        return GeminiLLM()
    if "--live" in argv:
        return ClaudeLLM()
    return MockLLM(script)


_ids = itertools.count(1)


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def tool_call(name: str, **inputs) -> dict:
    return {"type": "tool_use", "id": f"toolu_mock_{next(_ids)}", "name": name, "input": inputs}


class MockLLM:
    """Replays one scripted assistant turn per call.

    Each script entry is a list of content blocks. If it contains a tool_use
    block, stop_reason is "tool_use"; otherwise "end_turn" - exactly like the API.
    """

    def __init__(self, script: list[list[dict]]):
        self.script = list(script)

    def respond(self, system: str, messages: list, tools: list) -> dict:
        if not self.script:
            return {"stop_reason": "end_turn", "content": [text("(mock script exhausted)")]}
        content = self.script.pop(0)
        has_tool = any(b["type"] == "tool_use" for b in content)
        return {"stop_reason": "tool_use" if has_tool else "end_turn", "content": content}
