"""Failure path: a task that can never pass verification must fail cleanly,
its dependents must become 'blocked', and the scheduler must EXIT (this used
to spin forever). Uses a mock that never emits tool calls, so nothing gets
written and the acceptance check always fails."""
import asyncio, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from aiohttp import web
from devflock import gitops
from devflock.config import Endpoint, RunConfig
from devflock.scheduler import RegionWorker, Scheduler
from devflock.state import State
from mock_llama_server import make_app


async def main():
    project = Path(tempfile.mkdtemp(prefix="devflock-fail-"))
    gitops.ensure_repo(project)
    wt = gitops.create_region_worktree(project, "backend")
    wt2 = gitops.create_region_worktree(project, "frontend")
    bdir = gitops.region_subdir(wt, "backend")
    fdir = gitops.region_subdir(wt2, "frontend")

    runner = web.AppRunner(make_app("sk-x", flaky=False, tool_capable=False))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 18590).start()
    ep = Endpoint(name="w", base_url="http://127.0.0.1:18590", api_key="sk-x")
    rwB = RegionWorker("wB", ep, "backend", bdir); await rwB.start()
    rwF = RegionWorker("wF", ep, "frontend", fdir); await rwF.start()

    st = State(project / ".devflock" / "state.db")
    st.upsert_region("backend", str(bdir)); st.upsert_region("frontend", str(fdir))
    st.add_task("A", "backend", "Say hello.", acceptance_cmd="test -f never_created.txt")
    st.add_task("B", "frontend", "Depends on A.", depends_on=["A"])
    st.add_task("C", "frontend", "Depends on B.", depends_on=["B"])

    cfg = RunConfig(project_dir=str(project), manager=ep, workers=[ep], mode="fixed",
                    max_repair_attempts=2, max_turns_per_task=2, task_timeout_s=20)
    t0 = time.time()
    try:
        counts = await asyncio.wait_for(Scheduler(cfg, project, st, {"backend": rwB, "frontend": rwF}).run(), 60)
    except asyncio.TimeoutError:
        print("[FAIL] scheduler did not exit (would spin forever)"); return 1
    ok = counts == {"failed": 1, "blocked": 2}
    print(f"[{'PASS' if ok else 'FAIL'}] exited in {time.time()-t0:.1f}s with counts={counts} "
          f"(expected 1 failed, 2 blocked: dependents cascade)")
    await rwB.stop(); await rwF.stop(); await runner.cleanup()
    return 0 if ok else 1

sys.exit(asyncio.run(main()))
