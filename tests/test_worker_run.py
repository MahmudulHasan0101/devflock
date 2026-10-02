import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web

from devflock.gateway import Gateway, Upstream
from devflock.worker import Worker, build_region_seed_prompt
from devflock.region import RegionMemory
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
    mock = await run_mock(18290, "sk-mock")
    gw = Gateway("w1", Upstream("http://127.0.0.1:18290", "sk-mock"), port=0)
    gw_port = await gw.start()

    project = Path(tempfile.mkdtemp(prefix="devflock-worker-"))
    region_dir = project / "backend"
    region_dir.mkdir()
    other_region = project / "frontend"
    other_region.mkdir()

    worker = Worker("w1", gw_port, "sk-mock", region_dir)

    # --- task 1: plain text task, establishes a session ---
    prompt1 = build_region_seed_prompt(region_dir, "backend", "Say hello.", None, {})
    r1 = await worker.run_task(prompt1, max_turns=1, allowed_tools=[])
    print(f"[{'PASS' if r1.ok else 'FAIL'}] task1 ok={r1.ok} session={r1.session_id} "
          f"cost=${r1.cost_usd} summary={r1.summary!r}")
    ok &= r1.ok
    session_after_1 = r1.session_id

    # --- task 2: resume same session ---
    r2 = await worker.run_task("Say hello again.", max_turns=1, allowed_tools=[], resume=True)
    print(f"[{'PASS' if r2.ok else 'FAIL'}] task2 ok={r2.ok} session={r2.session_id} "
          f"resumed_same_session={r2.session_id == session_after_1}")
    ok &= r2.ok

    # --- task 3: tool use INSIDE region (should succeed) ---
    r3 = await worker.run_task(
        "Use the Write tool to create notes.txt with some content.",
        max_turns=3, allowed_tools=["Write"], resume=False)
    created_inside = (region_dir / "notes.txt").exists() or (region_dir / "hello.py").exists()
    print(f"[{'PASS' if created_inside else 'FAIL'}] task3 write inside region: "
          f"tool_calls={r3.tool_calls} files={[p.name for p in region_dir.iterdir()]}")
    ok &= created_inside

    # --- task 4: path guard should block a write OUTSIDE the region ---
    escape_prompt = (
        f"Use the Write tool to create a file at the absolute path "
        f"{other_region / 'sneaky.txt'} with some content."
    )
    r4 = await worker.run_task(escape_prompt, max_turns=3, allowed_tools=["Write"], resume=False)
    escaped = (other_region / "sneaky.txt").exists()
    print(f"[{'PASS' if not escaped else 'FAIL'}] task4 path guard: "
          f"escaped={escaped} tool_calls={r4.tool_calls} ok={r4.ok}")
    ok &= not escaped

    print(f"\ngateway stats: {gw.stats}")
    await gw.stop()
    await mock.cleanup()
    print("\nALL WORKER TESTS " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
