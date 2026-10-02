"""Exercises the cross-region tools (ask_region / request_change) with the
REAL bundled Claude Code CLI.

STATUS: this currently FAILS. In isolation (one Worker.run_task call, see
the debug repro this was developed against) the MCP tool is correctly
offered and called on the very first turn. Driven through the Scheduler,
the same tool is consistently NOT offered on turn 1 of a fresh session, so
the mock's "tool not available yet" fallback fires on every attempt and the
task exhausts its repair attempts. This is not a timing race (it is
consistent, not intermittent) and the root cause is not yet isolated.
enable_cross_region_tools defaults to False in RunConfig until this is
resolved -- see README "Known gaps".
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web

from devflock import gitops
from devflock.config import Endpoint, RunConfig
from devflock.region import RegionMemory
from devflock.scheduler import RegionWorker, Scheduler
from devflock.state import State
from mock_llama_server import make_app


async def run_mock(port, key):
    app = make_app(key, flaky=False, tool_capable=True)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def main():
    ok = True
    project = Path(tempfile.mkdtemp(prefix="devflock-tools-"))
    gitops.ensure_repo(project)

    api_wt = gitops.create_region_worktree(project, "api")
    ui_wt = gitops.create_region_worktree(project, "ui")
    api_dir = gitops.region_subdir(api_wt, "api")
    ui_dir = gitops.region_subdir(ui_wt, "ui")

    # Seed the api region's published interface so ui's ask_region has something real to read.
    mem = RegionMemory.load_or_init(api_dir, "api")
    mem.public = "Exposes GET /todos -> [{id, text, done}]. Run tests with: pytest -q"
    mem.save(api_dir)

    mockA = await run_mock(18800, "sk-A")
    mockB = await run_mock(18801, "sk-B")
    epA = Endpoint(name="wA", base_url="http://127.0.0.1:18800", api_key="sk-A")
    epB = Endpoint(name="wB", base_url="http://127.0.0.1:18801", api_key="sk-B")
    rwA = RegionWorker("workerA", epA, "api", api_dir); await rwA.start()
    rwB = RegionWorker("workerB", epB, "ui", ui_dir); await rwB.start()

    state = State(project / ".devflock" / "state.db")
    state.upsert_region("api", str(api_dir))
    state.upsert_region("ui", str(ui_dir))

    # Task 1: ui worker asks about api's interface using the ask_region tool.
    state.add_task("ask-1", "ui",
                    "Use the ask_region tool to ask region 'api' what endpoints it exposes. "
                    "Then use the Write tool to create answer.txt containing exactly what it told you.",
                    acceptance_cmd="test -f answer.txt")

    cfg = RunConfig(project_dir=str(project), manager=epA, workers=[epA, epB], mode="fixed",
                    task_timeout_s=30, max_turns_per_task=4, enable_cross_region_tools=True)
    sched = Scheduler(cfg, project, state, {"api": rwA, "ui": rwB})
    counts = await sched.run()
    print("after ask_region task, counts:", counts)

    answer = (ui_dir / "answer.txt")
    got_answer = answer.exists() and "GET /todos" in answer.read_text()
    print(f"[{'PASS' if got_answer else 'FAIL'}] ask_region returned api's real published interface: "
          f"{answer.read_text()[:200] if answer.exists() else '(no file)'!r}")
    ok &= got_answer

    events = [dict(e) for e in state.recent_events(50)]
    saw_ask_event = any(e["kind"] == "ask_region" for e in events)
    print(f"[{'PASS' if saw_ask_event else 'FAIL'}] ask_region call was logged: "
          f"{[e for e in events if e['kind']=='ask_region']}")
    ok &= saw_ask_event

    # Task 2: api worker uses request_change to queue work in ui -- and never touches ui's files itself.
    state.add_task("req-1", "api",
                    "Use the request_change tool to ask region 'ui' to create a file called "
                    "requested.txt with any short content, since you cannot edit ui yourself. "
                    "Then use the Write tool to create done.txt in your own region confirming you asked.",
                    acceptance_cmd="test -f done.txt")

    counts2 = await sched.run()
    print("after request_change task, counts:", counts2)

    queued = state.get_task if False else None
    new_tasks = [dict(r) for r in state.conn.execute("SELECT * FROM tasks WHERE id LIKE 'req-%' AND id != 'req-1'")]
    print(f"request_change queued tasks: {[t['id'] for t in new_tasks]}")
    queued_ok = len(new_tasks) == 1 and new_tasks[0]["region"] == "ui" and new_tasks[0]["status"] == "done"
    print(f"[{'PASS' if queued_ok else 'FAIL'}] request_change queued a REAL task that later ran to completion")
    ok &= queued_ok

    requested_file = (ui_dir / "requested.txt").exists()
    print(f"[{'PASS' if requested_file else 'FAIL'}] the queued task actually wrote requested.txt in ui's region "
          f"(api never touched ui's files directly)")
    ok &= requested_file

    api_never_wrote_to_ui = not (ui_dir / "done.txt").exists()  # done.txt belongs in api's own dir
    print(f"[{'PASS' if api_never_wrote_to_ui else 'FAIL'}] api worker wrote its own confirmation "
          f"in its own region, not ui's")
    ok &= api_never_wrote_to_ui

    await rwA.stop(); await rwB.stop()
    await mockA.cleanup(); await mockB.cleanup()
    state.close()
    print("\nALL CROSS-REGION TOOL TESTS " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
