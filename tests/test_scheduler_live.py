import asyncio
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web

from devflock import gitops
from devflock.config import Endpoint, RunConfig
from devflock.scheduler import RegionWorker, Scheduler
from devflock.state import State
from mock_llama_server import make_app


async def run_mock(port, key):
    app = make_app(key, flaky=False, tool_capable=True)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


async def main():
    ok = True
    project = Path(tempfile.mkdtemp(prefix="devflock-sched-"))
    print("PROJECT_DIR:", project, flush=True)
    gitops.ensure_repo(project)

    backend_wt = gitops.create_region_worktree(project, "backend")
    frontend_wt = gitops.create_region_worktree(project, "frontend")
    backend_dir = gitops.region_subdir(backend_wt, "backend")
    frontend_dir = gitops.region_subdir(frontend_wt, "frontend")

    mockA = await run_mock(18390, "sk-A")
    mockB = await run_mock(18391, "sk-B")

    epA = Endpoint(name="workerA", base_url="http://127.0.0.1:18390", api_key="sk-A")
    epB = Endpoint(name="workerB", base_url="http://127.0.0.1:18391", api_key="sk-B")

    rwA = RegionWorker("workerA", epA, "backend", backend_dir)
    rwB = RegionWorker("workerB", epB, "frontend", frontend_dir)
    await rwA.start()
    await rwB.start()

    state = State(project / ".devflock" / "state.db")
    state.upsert_region("backend", str(backend_dir))
    state.upsert_region("frontend", str(frontend_dir))
    state.add_task("t-backend-1", "backend",
                    "Use the Write tool to create hello.py with a print statement.",
                    acceptance_cmd="test -f hello.py")
    state.add_task("t-frontend-1", "frontend",
                    "Use the Write tool to create hello.py (as a placeholder page).",
                    depends_on=["t-backend-1"],
                    acceptance_cmd="test -f hello.py")

    cfg = RunConfig(project_dir=str(project), manager=epA, workers=[epA, epB], mode="fixed",
                     task_timeout_s=20.0)
    sched = Scheduler(cfg, project, state, {"backend": rwA, "frontend": rwB})

    print("--- running scheduler ---")
    final_counts = await sched.run()
    print("final task counts:", final_counts)
    ok &= final_counts.get("done") == 2

    backend_file = (backend_dir / "hello.py").exists()
    frontend_file = (frontend_dir / "hello.py").exists()
    print(f"[{'PASS' if backend_file else 'FAIL'}] backend region file written")
    print(f"[{'PASS' if frontend_file else 'FAIL'}] frontend region file written")
    ok &= backend_file and frontend_file

    # dependency ordering: frontend task must have started after backend was done
    events = list(reversed(state.recent_events(200)))
    t_backend_done = next(e["t"] for e in events if e["kind"] == "task_done" and e["task"] == "t-backend-1")
    t_frontend_start = next(e["t"] for e in events if e["kind"] == "task_start" and e["task"] == "t-frontend-1")
    dep_respected = t_frontend_start >= t_backend_done
    print(f"[{'PASS' if dep_respected else 'FAIL'}] dependency order respected "
          f"(backend done @ {t_backend_done:.2f}, frontend start @ {t_frontend_start:.2f})")
    ok &= dep_respected

    print(f"total cost: ${state.total_cost():.6f}")

    conflicted = gitops.integrate(project, ["backend", "frontend"])
    print(f"[{'PASS' if not conflicted else 'FAIL'}] integration merge, conflicts={conflicted}")
    ok &= not conflicted

    import subprocess
    def on_branch(path):
        return subprocess.run(["git", "cat-file", "-e", f"devflock/integration:{path}"], cwd=project).returncode == 0
    merged_ok = on_branch("backend/hello.py") and on_branch("frontend/hello.py")
    print(f"[{'PASS' if merged_ok else 'FAIL'}] integration branch has backend/hello.py and frontend/hello.py "
          f"(disjoint paths, no conflict)")
    ok &= merged_ok

    await rwA.stop()
    await rwB.stop()
    await mockA.cleanup()
    await mockB.cleanup()
    state.close()

    print("\nALL SCHEDULER TESTS " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
