"""Turns a validated Plan + N endpoints into running workers, and runs them."""
from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Callable, Optional

from . import gitops
from .config import Endpoint, RunConfig
from .packer import Plan, RegionAssignment, module_to_region
from .scheduler import RegionWorker, Scheduler
from .state import State

logger = logging.getLogger("devflock.runner")


def assign_endpoints(assignments: list[RegionAssignment], workers: list[Endpoint]) -> dict[str, Endpoint]:
    if len(workers) < len(assignments):
        raise ValueError(f"{len(assignments)} regions need {len(assignments)} workers, got {len(workers)}")
    return {a.region: workers[i] for i, a in enumerate(assignments)}


async def run_flock(cfg: RunConfig, plan: Plan, assignments: list[RegionAssignment],
                    on_started: Optional[Callable[[State, dict[str, RegionWorker]], None]] = None,
                    scheduler_hook: Optional[Callable] = None) -> dict:
    project = Path(cfg.project_dir).resolve()
    gitops.ensure_repo(project)
    state = State(project / ".devflock" / "state.db")
    mods = {m.name: m for m in plan.modules}
    m2r = module_to_region(assignments)
    eps = assign_endpoints(assignments, cfg.workers)

    region_workers: dict[str, RegionWorker] = {}
    module_dir: dict[str, Path] = {}
    try:
        for a in assignments:
            wt = gitops.create_region_worktree(project, a.region)
            dirs = [gitops.region_subdir(wt, mods[m].path or m) for m in a.modules]
            for m, d in zip(a.modules, dirs):
                module_dir[m] = d
            primary, extras = dirs[0], dirs[1:]  # packer sorts heaviest module first
            rw = RegionWorker(f"worker-{a.region}", eps[a.region], a.region, primary, extra_write_roots=extras)
            await rw.start()
            region_workers[a.region] = rw
            state.upsert_region(a.region, str(primary))
            state.upsert_worker(rw.worker_id, base_url=eps[a.region].base_url,
                                gateway_port=rw.gateway.port, region=a.region, status="idle")

        for t in plan.tasks:
            d = module_dir[t.module]
            desc = f"Module `{t.module}` (folder: {d}).\n\n{t.description}"
            acc = f"cd {shlex.quote(str(d))} && {t.acceptance_cmd}" if t.acceptance_cmd else None
            state.add_task(t.id, m2r[t.module], desc, depends_on=t.depends_on, acceptance_cmd=acc)

        sched = Scheduler(cfg, project, state, region_workers)
        if on_started:
            on_started(state, region_workers)
        counts = await sched.run()

        order = [a.region for a in assignments]
        done_regions = [r for r in order if any(
            row["region"] == r and row["status"] == "done"
            for row in state.conn.execute("SELECT region, status FROM tasks"))]
        conflicts = gitops.integrate(project, done_regions) if done_regions else []
        failed = [dict(r) for r in state.conn.execute(
            "SELECT id, region, status FROM tasks WHERE status IN ('failed','blocked')")]
        tok = state.conn.execute(
            "SELECT COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(output_tokens),0) o FROM ledger").fetchone()
        return {"counts": counts, "conflicts": conflicts, "failed": failed,
                "tokens_in": tok["i"], "tokens_out": tok["o"], "integrated": done_regions,
                "branch": "devflock/integration",
                "base_branch": gitops.current_branch(project)}
    finally:
        for rw in region_workers.values():
            await rw.stop()
        state.close()
