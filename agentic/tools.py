"""
Tools = plain Python functions + a JSON schema describing them to the model.

The model never runs anything itself. It *asks* for a tool call (name + JSON
input); our code runs the function and sends the result back as text.

All file tools are confined to ./workspace so the agents can't touch anything else.
"""

import ast
import operator
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., str]

    def schema(self) -> dict:
        """The part the model sees."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def _safe_path(name: str) -> Path:
    p = (WORKSPACE / name).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError(f"path escapes workspace: {name}")
    return p


# ---- implementations -------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}


def calculator(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")
    return str(ev(ast.parse(expression, mode="eval").body))


def write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to workspace/{path}"


def read_file(path: str) -> str:
    return _safe_path(path).read_text()


def list_files() -> str:
    files = sorted(str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*") if p.is_file())
    return "\n".join(files) or "(workspace is empty)"


def run_python(path: str) -> str:
    """Run a script from the workspace. (A real system would use a proper sandbox.)"""
    proc = subprocess.run([sys.executable, str(_safe_path(path))], cwd=WORKSPACE,
                          capture_output=True, text=True, timeout=30)
    return f"exit code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


# ---- schemas ---------------------------------------------------------------

CALCULATOR = Tool("calculator", "Evaluate an arithmetic expression, e.g. '(3+4)*2'.",
                  {"type": "object", "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}, calculator)

WRITE_FILE = Tool("write_file", "Create or overwrite a file in the workspace.",
                  {"type": "object", "properties": {"path": {"type": "string"},
                                                    "content": {"type": "string"}},
                   "required": ["path", "content"]}, write_file)

READ_FILE = Tool("read_file", "Read a file from the workspace.",
                 {"type": "object", "properties": {"path": {"type": "string"}},
                  "required": ["path"]}, read_file)

LIST_FILES = Tool("list_files", "List all files in the workspace.",
                  {"type": "object", "properties": {}}, list_files)

RUN_PYTHON = Tool("run_python", "Run a Python file from the workspace and return its output.",
                  {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}, run_python)
