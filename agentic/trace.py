"""Pretty, indented terminal trace so you can see who is doing what."""

import json
import threading

_lock = threading.Lock()
COLORS = {"orchestrator": "35", "coder": "33", "tester": "36", "reviewer": "32", "assistant": "34"}


def _short(s: str, n: int = 300) -> str:
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + f" … (+{len(s) - n} chars)"


class Logger:
    def __init__(self, name: str, depth: int):
        self.name, self.pad = name, "    " * depth
        self.color = COLORS.get(name, "37")

    def _p(self, msg: str):
        with _lock:
            for i, line in enumerate(msg.splitlines() or [""]):
                prefix = f"\033[{self.color}m[{self.name}]\033[0m " if i == 0 else " " * (len(self.name) + 3)
                print(f"{self.pad}{prefix}{line}")

    def task(self, t):            self._p(f"\033[1m▶ TASK:\033[0m {_short(t)}")
    def thought(self, turn, t):   self._p(f"💭 (turn {turn}) {_short(t)}")
    def call(self, turn, name, a): self._p(f"🔧 (turn {turn}) {name}({_short(json.dumps(a), 160)})")
    def result(self, name, out, err): self._p(f"{'❌' if err else '📥'} {name} → {_short(out, 200)}")
    def done(self, turn, t):      self._p(f"\033[1m✅ DONE after {turn} turn(s):\033[0m {_short(t)}")
