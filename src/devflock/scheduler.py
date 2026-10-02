"""The scheduler owns the task queue and the worker pool. Each worker is
pinned to one region (its git worktree); tasks for that region only ever run
on that worker, so `resume` keeps working and the `.devflock` file stays a
faithful memory of "what this worker has done here."

Loop, once per pass:
  1. ask State for ready tasks (deps satisfied);
  2. hand each to its region's (idle) worker;
  3. run concurrently, verify on completion, repair up to N times on
     verify failure by feeding the error back to the *same* session;
  4. commit the region's worktree if the task succeeded;
  5. repeat until no tasks are pending/running.

This is intentionally simple (no work-stealing across regions yet -- see
README "Next steps"). It is the piece most worth hardening once you've run
it against your real Qwen boxes and seen real failure modes.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from . import gitops, verifier
from .tools import build_region_tools
from .config import Endpoint, RunConfig
from .gateway import Gateway, Upstream
from .region import RegionMemory, public_only
from .state import State
from .worker import Worker, build_region_seed_prompt

logger = logging.getLogger("devflock.scheduler")


class RegionWorker:
    """Pairs a Gateway (network path to one submachine) with a Worker
    (the Claude Code session pinned to one region's worktree)."""

    def __init__(self, worker_id: str, endpoint: Endpoint, region: str, region_dir: Path,
                 extra_write_roots: Optional[list[Path]] = None):
        self.worker_id = worker_id
        self.extra_write_roots = extra_write_roots or []
        self.endpoint = endpoint
        self.region = region
        self.region_dir = region_dir
        self.gateway: Optional[Gateway] = None
        self.worker: Optional[Worker] = None
        self.busy = False

    async def start(self):
        self.gateway = Gateway(self.worker_id,
                                Upstream(self.endpoint.base_url, self.endpoint.api_key,
                                         self.endpoint.auth_style))
        port = await self.gateway.start()
        self.worker = Worker(self.worker_id, port, self.endpoint.api_key,
                              self.region_dir, model_name=self.endpoint.model_name,
                              extra_write_roots=self.extra_write_roots)

    async def stop(self):
        if self.gateway:
            await self.gateway.stop()

    async def swap_endpoint(self, endpoint: Endpoint):
        """Point this worker's gateway at a fresh tunnel (new Kaggle
        session) without losing worktree state; the Claude Code session
        itself is still lost, so also call self.worker.recycle_session()."""
        self.endpoint = endpoint
        assert self.gateway is not None
        self.gateway.upstream = Upstream(endpoint.base_url, endpoint.api_key, endpoint.auth_style)
        if self.worker:
            self.worker.recycle_session()


class Scheduler:
    def __init__(self, cfg: RunConfig, project_dir: Path, state: State,
                 region_workers: dict[str, RegionWorker], max_concurrency: Optional[int] = None):
        self.cfg = cfg
        self.project_dir = project_dir
        self.state = state
        self.workers = region_workers
        self.max_concurrency = max_concurrency or len(region_workers)
        self._sem = asyncio.Semaphore(self.max_concurrency)

    def _neighbor_summaries(self, exclude_region: str) -> dict[str, str]:
        out = {}
        for r in self.state.all_regions():
            if r["name"] == exclude_region:
                continue
            out[r["name"]] = public_only(Path(r["path"]), r["name"])
        return out

    async def _run_one_task(self, task_row) -> None:
        task_id = task_row["id"]
        region = task_row["region"]
        rw = self.workers.get(region)
        if rw is None:
            self.state.set_task_status(task_id, "blocked", assigned_worker=None)
            self.state.log_event("no_worker_for_region", task=task_id, detail=region)
            return

        async with self._sem:
            rw.busy = True
            self.state.set_task_status(task_id, "running", assigned_worker=rw.worker_id)
            self.state.log_event("task_start", worker=rw.worker_id, task=task_id)
            try:
                await self._attempt_task(task_row, task_id, region, rw)
            except Exception as e:
                # A bug here must never leave a task stuck at "running" forever --
                # that would silently wedge the whole scheduler loop (see README
                # "Known sharp edges"). Fail loudly and move on.
                logger.exception("worker[%s] task %s: unhandled exception", rw.worker_id, task_id)
                self.state.set_task_status(task_id, "failed")
                self.state.log_event("task_crashed", worker=rw.worker_id, task=task_id, detail=repr(e))
            finally:
                rw.busy = False

    async def _attempt_task(self, task_row, task_id: str, region: str, rw: "RegionWorker") -> None:
        neighbors = self._neighbor_summaries(region)
        prompt = build_region_seed_prompt(
            rw.region_dir, region, task_row["description"], task_row["acceptance_cmd"],
            neighbors, write_roots=rw.worker.write_roots)
        mcp_servers = None
        if self.cfg.enable_cross_region_tools and len(self.workers) > 1:
            region_dirs = {r: w.region_dir for r, w in self.workers.items()}
            mcp_servers = {"devflock": build_region_tools(self.state, region_dirs, region, rw.worker_id)}

        attempts = 0
        ok = False
        last_error = ""
        while attempts < self.cfg.max_repair_attempts and not ok:
            attempts += 1
            result = await rw.worker.run_task(prompt if attempts == 1 else (
                f"The acceptance check failed:\n{last_error}\n"
                f"Please fix it in your region and try again."
            ), resume=(attempts > 1), max_turns=self.cfg.max_turns_per_task,
                timeout_s=self.cfg.task_timeout_s)

            if not result.ok:
                last_error = result.error or "worker returned an error"
                self.state.log_event("task_worker_error", worker=rw.worker_id, task=task_id,
                                      detail=last_error)
                continue

            self.state.log_cost(rw.worker_id, task_id, result.input_tokens,
                                 result.output_tokens, result.cost_usd)

            v = verifier.verify(rw.region_dir, task_row["acceptance_cmd"])
            if v.ok:
                ok = True
            else:
                last_error = f"stdout:\n{v.stdout}\nstderr:\n{v.stderr}"
                self.state.set_task_status(task_id, "verify_failed")
                self.state.log_event("verify_failed", worker=rw.worker_id, task=task_id,
                                      detail=last_error[:500])

        if ok:
            gitops.commit_region(self.project_dir, region, f"devflock: {task_id} - {task_row['description'][:60]}")
            self.state.set_task_status(task_id, "done")
            self.state.log_event("task_done", worker=rw.worker_id, task=task_id)
            self.state.upsert_region(region, str(rw.region_dir), owner_worker=rw.worker_id)
        else:
            self.state.set_task_status(task_id, "failed")
            self.state.log_event("task_failed", worker=rw.worker_id, task=task_id, detail=last_error[:1000])

    async def run(self, poll_interval: float = 1.0):
        pending_futures: set[asyncio.Task] = set()
        while True:
            self.state.block_orphaned_tasks()
            ready = self.state.ready_tasks()
            assignable = []
            for t in ready:
                rw = self.workers.get(t["region"])
                if rw is None or not rw.busy:  # missing worker -> _run_one_task marks it blocked
                    assignable.append(t)
            for t in assignable:
                self.state.set_task_status(t["id"], "running")  # claim immediately to avoid double-dispatch
                pending_futures.add(asyncio.create_task(self._run_one_task(t)))

            counts = self.state.counts_by_status()
            outstanding = counts.get("pending", 0) + counts.get("running", 0)
            if outstanding == 0 and not pending_futures:
                break
            if pending_futures:
                done, pending_futures = await asyncio.wait(pending_futures, timeout=poll_interval)
                for d in done:
                    exc = d.exception()
                    if exc is not None:
                        logger.error("task coroutine raised: %r", exc, exc_info=exc)
                        self.state.log_event("scheduler_exception", detail=repr(exc))
            else:
                await asyncio.sleep(poll_interval)

        return self.state.counts_by_status()
