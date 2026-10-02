"""Turns the manager's plan (modules + a task dependency DAG) into N regions
that N workers can own, and recommends a good N. Pure logic, no LLM calls,
so it's cheap to re-run whenever the person adjusts the worker count in the
wizard.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict, deque


@dataclasses.dataclass
class ModuleSpec:
    name: str
    depends_on: list[str] = dataclasses.field(default_factory=list)
    weight: float = 1.0  # relative size, e.g. rough file/LOC estimate
    path: str = ""       # folder relative to the repo root; defaults to the module name


@dataclasses.dataclass
class TaskSpec:
    id: str
    module: str
    description: str
    depends_on: list[str] = dataclasses.field(default_factory=list)
    acceptance_cmd: str | None = None


@dataclasses.dataclass
class Plan:
    modules: list[ModuleSpec]
    tasks: list[TaskSpec]


def max_parallelism(plan: Plan) -> int:
    """Widest set of tasks that could run at once, i.e. the largest 'layer'
    of the task DAG under Kahn's algorithm. This is the natural ceiling for
    how many workers actually help."""
    indeg: dict[str, int] = {t.id: 0 for t in plan.tasks}
    children: dict[str, list[str]] = defaultdict(list)
    for t in plan.tasks:
        for d in t.depends_on:
            indeg[t.id] += 1
            children[d].append(t.id)

    frontier = deque([tid for tid, d in indeg.items() if d == 0])
    widest = len(frontier)
    while frontier:
        nxt = []
        for tid in frontier:
            for c in children[tid]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    nxt.append(c)
        widest = max(widest, len(nxt))
        frontier = deque(nxt)
    return max(widest, 1)


def recommend_n(plan: Plan, hard_cap: int = 12) -> int:
    n_modules = max(len(plan.modules), 1)
    return max(1, min(max_parallelism(plan), n_modules, hard_cap))


@dataclasses.dataclass
class RegionAssignment:
    region: str
    modules: list[str]


def pack(plan: Plan, n: int) -> list[RegionAssignment]:
    """Greedy longest-processing-time-first bin packing of modules into n
    regions by weight, keeping modules whole (a module is never split across
    regions -- that would break the 'a region == a folder' invariant)."""
    n = max(1, min(n, max(len(plan.modules), 1)))
    mods = sorted(plan.modules, key=lambda m: m.weight, reverse=True)
    bins: list[list[str]] = [[] for _ in range(n)]
    bin_weight = [0.0] * n
    for m in mods:
        i = min(range(n), key=lambda i: bin_weight[i])
        bins[i].append(m.name)
        bin_weight[i] += m.weight
    names = [f"region-{i+1}" if len(bins[i]) != 1 else bins[i][0] for i in range(n)]
    return [RegionAssignment(region=names[i], modules=bins[i]) for i in range(n) if bins[i]]


def module_to_region(assignments: list[RegionAssignment]) -> dict[str, str]:
    return {m: a.region for a in assignments for m in a.modules}
