"""Drive the REAL wizard (prompts, probes, packing, worktrees, workers, scheduler,
integration) against three mock servers. Only the LLM planner is stubbed, because
the mock can't write plan JSON. Run in a subprocess-free way via click's CliRunner."""
import asyncio, sys, tempfile, threading, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from aiohttp import web
from click.testing import CliRunner
from mock_llama_server import make_app
import devflock.cli as cli
from devflock.packer import ModuleSpec, Plan, TaskSpec

PORTS = {"mgr": (18601, "sk-mgr"), "w1": (18602, "sk-w1"), "w2": (18603, "sk-w2")}

def serve_mocks():
    ready = threading.Event()
    def run():
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        async def start():
            for port, key in PORTS.values():
                r = web.AppRunner(make_app(key, flaky=False, tool_capable=True)); await r.setup()
                await web.TCPSite(r, "127.0.0.1", port).start()
            ready.set()
        loop.run_until_complete(start()); loop.run_forever()
    threading.Thread(target=run, daemon=True).start(); ready.wait(10)

async def fake_plan(worker, mode, idea, n):
    write = "Use the Write tool to create hello.py containing a print statement."
    return Plan(
        modules=[ModuleSpec("alpha", path="mod_alpha", weight=2), ModuleSpec("beta", path="mod_beta", weight=1)],
        tasks=[TaskSpec("a1", "alpha", write, acceptance_cmd="test -f hello.py"),
               TaskSpec("b1", "beta", write, depends_on=["a1"], acceptance_cmd="test -f hello.py")]), "Two tiny modules."

def main():
    serve_mocks()
    cli.plan_fn = fake_plan
    project = Path(tempfile.mkdtemp(prefix="devflock-e2e-"))
    lines = [
        "http://127.0.0.1:18601 sk-mgr",   # manager: URL and key in one line
        "1",                                # mode 1: I choose the number of workers
        "2",                                #   ... 2 workers
        "",                                 #   ... no specific request
        "2",                                # verdict: accept 2 workers
        "http://127.0.0.1:18602|sk-w1",     # worker 1 (pipe-separated form)
        "http://127.0.0.1:18603 sk-w2",     # worker 2
        "y",                                # start
    ]
    res = CliRunner().invoke(cli.main, ["--project", str(project)], input="\n".join(lines) + "\n")
    print(res.output[-1800:])
    if res.exception and not isinstance(res.exception, SystemExit):
        import traceback; traceback.print_exception(res.exception)
    ok = True
    def check(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    check("wizard exited cleanly", res.exit_code == 0)
    check("all tasks done", "'done': 2" in res.output)
    def on_branch(path):
        return subprocess.run(["git", "cat-file", "-e", f"devflock/integration:{path}"], cwd=project).returncode == 0
    check("merged files are on the integration branch", on_branch("mod_alpha/hello.py") and on_branch("mod_beta/hello.py"))
    cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project, capture_output=True, text=True).stdout.strip()
    check(f"user's branch restored (on {cur!r}, not devflock/*)", not cur.startswith("devflock/"))
    tracked = subprocess.run(["git", "ls-files"], cwd=project, capture_output=True, text=True).stdout
    check("DevFlock state is not committed to git", ".devflock/state.db" not in tracked and "devflock.log" not in tracked)
    print("\nALL CLI E2E TESTS " + ("PASSED" if ok else "FAILED")); sys.exit(0 if ok else 1)

main()
