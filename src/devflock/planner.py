"""The manager: turns a project (existing code, or just an idea) into a
validated Plan of modules + a task DAG that the packer and scheduler consume.

The model's output is never trusted: it is parsed, schema-checked, and
cycle-checked here, and on any problem the exact error list is fed back to the
same session for a retry. Small local models get JSON wrong often enough that
this loop is part of the design, not an afterthought.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import PurePosixPath
from typing import Any, Optional, Protocol

from .packer import ModuleSpec, Plan, TaskSpec

logger = logging.getLogger("devflock.planner")

PLAN_SCHEMA_DOC = """\
Reply with ONE JSON object and nothing else (no prose before or after):
{
  "summary": "2-3 sentences: what is being built/changed and how it is split",
  "modules": [
    {"name": "auth", "path": "src/auth", "weight": 3, "depends_on": []}
  ],
  "tasks": [
    {"id": "t1", "module": "auth",
     "description": "specific, self-contained instructions for one worker",
     "depends_on": [],
     "acceptance_cmd": "a shell command that exits 0 only when the task is truly done"}
  ]
}
Rules:
- "path" is a folder relative to the repo root; it must not be absolute or contain "..".
- Modules must be loosely coupled: each owns its folder and talks to others only through
  small, explicitly named interfaces. No two modules may share a file.
- "weight" is a rough relative size (1 = small, 5 = large).
- Each task touches ONE module, roughly 1-8 files, and must be completable without asking
  anyone. If task B needs something task A creates, put A in B's depends_on. Put cross-module
  interface definitions in early tasks so later tasks can run in parallel.
- "acceptance_cmd" runs with the module's folder as the working directory. It must be
  deterministic and fast (tests, a type check, an import check). Never use "true".
- Ids unique. Only reference modules/tasks that exist. No dependency cycles.
"""

SYSTEM_CONTEXT = """\
You are the DevFlock manager. A team of independent coding workers will carry out your plan
in parallel; each worker owns only the module folders assigned to it and cannot see your
reasoning. Your job is to design a modular decomposition that maximises safe parallelism.
"""


def build_prompt(mode: str, idea: Optional[str], requested_n: Optional[int]) -> str:
    if mode == "idea":
        task = (f"Design a new project from this idea, from scratch:\n\n{idea}\n\n"
                "Choose a sensible stack and layout. Make the FIRST tasks create the shared "
                "interface/contract files and scaffolding, so the remaining tasks can run in parallel.")
    else:
        task = ("Study the existing project in the current directory (use Glob, Grep and Read; "
                "you cannot modify anything) and propose the modules that already exist or that "
                "the requested work naturally splits into.")
        if idea:
            task += f"\n\nRequested work:\n{idea}"
        else:
            task += ("\n\nNo specific feature was requested: produce a plan of tasks that "
                     "document each module's public interface and add missing tests.")
    size = ""
    if requested_n:
        size = (f"\nThe person has {requested_n} workers. Design at least {requested_n} independent "
                f"modules where that is natural (they can be packed onto fewer workers later), "
                f"but do not invent artificial modules.")
    return f"{SYSTEM_CONTEXT}\n{task}\n{size}\n\n{PLAN_SCHEMA_DOC}"


# ---------------------------------------------------------------- parsing

def extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    last_err: Exception | None = None
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"no valid JSON object found ({last_err})" if last_err else "no JSON object found in reply")


def _safe_path(p: str) -> bool:
    pp = PurePosixPath(p)
    return bool(p) and not pp.is_absolute() and ".." not in pp.parts


def _find_cycle(deps: dict[str, list[str]]) -> Optional[list[str]]:
    WHITE, GREY, BLACK = 0, 1, 2
    color = {k: WHITE for k in deps}
    stack: list[str] = []

    def visit(n: str) -> Optional[list[str]]:
        color[n] = GREY
        stack.append(n)
        for d in deps.get(n, []):
            if d not in color:
                continue
            if color[d] == GREY:
                return stack[stack.index(d):] + [d]
            if color[d] == WHITE:
                cyc = visit(d)
                if cyc:
                    return cyc
        stack.pop()
        color[n] = BLACK
        return None

    for n in list(deps):
        if color[n] == WHITE:
            cyc = visit(n)
            if cyc:
                return cyc
    return None


def validate_plan(data: Any) -> tuple[Optional[Plan], list[str]]:
    errs: list[str] = []
    if not isinstance(data, dict):
        return None, ["top level must be a JSON object"]
    mods_raw, tasks_raw = data.get("modules"), data.get("tasks")
    if not isinstance(mods_raw, list) or not mods_raw:
        errs.append('"modules" must be a non-empty list')
    if not isinstance(tasks_raw, list) or not tasks_raw:
        errs.append('"tasks" must be a non-empty list')
    if errs:
        return None, errs

    modules: list[ModuleSpec] = []
    names: set[str] = set()
    paths: set[str] = set()
    for i, m in enumerate(mods_raw):
        if not isinstance(m, dict) or not isinstance(m.get("name"), str) or not m["name"].strip():
            errs.append(f"modules[{i}] needs a string 'name'")
            continue
        name = m["name"].strip()
        raw_path = str(m.get("path") or name).strip()
        path = raw_path
        if name in names:
            errs.append(f"duplicate module name {name!r}")
        if not _safe_path(raw_path):
            errs.append(f"module {name!r}: path {raw_path!r} must be relative with no '..'")
        else:
            path = str(PurePosixPath(raw_path))  # normalises "./src/api/" -> "src/api"
        if not _safe_path(raw_path):
            pass
        elif path in paths:
            errs.append(f"module {name!r}: path {path!r} already used by another module")
        try:
            weight = float(m.get("weight", 1))
        except (TypeError, ValueError):
            weight = 1.0
            errs.append(f"module {name!r}: weight must be a number")
        names.add(name)
        paths.add(path)
        modules.append(ModuleSpec(name=name, path=path, weight=max(weight, 0.1),
                                  depends_on=[str(d) for d in (m.get("depends_on") or [])]))
    for m in modules:
        for d in m.depends_on:
            if d not in names:
                errs.append(f"module {m.name!r} depends on unknown module {d!r}")

    tasks: list[TaskSpec] = []
    ids: set[str] = set()
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict) or not isinstance(t.get("id"), str) or not t["id"].strip():
            errs.append(f"tasks[{i}] needs a string 'id'")
            continue
        tid = t["id"].strip()
        if tid in ids:
            errs.append(f"duplicate task id {tid!r}")
        ids.add(tid)
        if t.get("module") not in names:
            errs.append(f"task {tid!r}: unknown module {t.get('module')!r}")
        desc = t.get("description")
        if not isinstance(desc, str) or len(desc.strip()) < 10:
            errs.append(f"task {tid!r}: 'description' must be a specific instruction")
        acc = t.get("acceptance_cmd")
        if not isinstance(acc, str) or acc.strip() in ("", "true", ":"):
            errs.append(f"task {tid!r}: 'acceptance_cmd' must be a real verification command")
        tasks.append(TaskSpec(id=tid, module=str(t.get("module")), description=str(desc or ""),
                              depends_on=[str(d) for d in (t.get("depends_on") or [])],
                              acceptance_cmd=acc if isinstance(acc, str) else None))
    for t in tasks:
        for d in t.depends_on:
            if d not in ids:
                errs.append(f"task {t.id!r} depends on unknown task {d!r}")
    if not errs:
        cyc = _find_cycle({t.id: t.depends_on for t in tasks})
        if cyc:
            errs.append("task dependency cycle: " + " -> ".join(cyc))
        mcyc = _find_cycle({m.name: m.depends_on for m in modules})
        if mcyc:
            errs.append("module dependency cycle: " + " -> ".join(mcyc))
    if errs:
        return None, errs
    return Plan(modules=modules, tasks=tasks), []


# ---------------------------------------------------------------- session

class PlannerSession(Protocol):
    async def run_task(self, prompt: str, *, max_turns: int = ..., allowed_tools: Any = ...,
                       resume: bool = ..., timeout_s: float = ...) -> Any: ...


class PlanningFailed(RuntimeError):
    pass


async def make_plan(session: PlannerSession, mode: str, idea: Optional[str],
                    requested_n: Optional[int] = None, max_attempts: int = 3,
                    timeout_s: float = 600.0) -> tuple[Plan, str]:
    """Returns (plan, summary). `session` is a devflock.worker.Worker whose cwd is
    the project (existing-project modes) -- read-only tools only, so planning
    can never modify anything."""
    tools = [] if mode == "idea" else ["Read", "Glob", "Grep"]
    prompt = build_prompt(mode, idea, requested_n)
    last_errs: list[str] = []
    for attempt in range(1, max_attempts + 1):
        res = await session.run_task(prompt, max_turns=30, allowed_tools=tools,
                                     resume=attempt > 1, timeout_s=timeout_s)
        if not res.ok:
            last_errs = [f"session error: {res.error}"]
            logger.warning("planner attempt %d failed: %s", attempt, res.error)
            prompt = "Your previous reply could not be received. " + PLAN_SCHEMA_DOC
            continue
        try:
            plan, errs = validate_plan(extract_json(res.summary))
        except ValueError as e:
            plan, errs = None, [str(e)]
        if plan:
            summary = ""
            try:
                summary = str(extract_json(res.summary).get("summary", ""))
            except Exception:
                pass
            return plan, summary
        last_errs = errs
        logger.info("planner attempt %d invalid: %s", attempt, errs)
        prompt = ("Your plan was rejected. Fix ALL of these problems and reply with the complete "
                  "corrected JSON object only:\n- " + "\n- ".join(errs))
    raise PlanningFailed("manager could not produce a valid plan after "
                         f"{max_attempts} attempts. Last problems: {last_errs}")
