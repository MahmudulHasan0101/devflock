"""Drive the REAL wizard with Cloudflare as a provider choice:
   manager = Anthropic-compatible URL (mock llama), worker 1 = Cloudflare model picked by NUMBER,
   worker 2 = Cloudflare model picked by SEARCH TEXT. Real Claude Code runs both workers, each through
   the translating gateway to a mock Workers AI. Credentials come from env vars (token is never typed).
Run: python tests/test_cli_cloudflare_e2e.py
"""
import asyncio, os, sys, tempfile, threading, subprocess
from pathlib import Path

CF_PORT = 18810
os.environ["CLOUDFLARE_API_BASE"] = f"http://127.0.0.1:{CF_PORT}/client/v4"
os.environ["CLOUDFLARE_ACCOUNT_ID"] = "acct123"
os.environ["CLOUDFLARE_API_TOKEN"] = "cf-test-token"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from aiohttp import web
from click.testing import CliRunner
import mock_cloudflare_server as mcf
from mock_llama_server import make_app as make_llama
import devflock.cli as cli
from devflock.packer import ModuleSpec, Plan, TaskSpec

APPS = {}

def serve():
    ready = threading.Event()
    def run():
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        async def start():
            r = web.AppRunner(make_llama("sk-mgr", flaky=False, tool_capable=True)); await r.setup()
            await web.TCPSite(r, "127.0.0.1", 18809).start()
            APPS["cf"] = mcf.make_app(); r2 = web.AppRunner(APPS["cf"]); await r2.setup()
            await web.TCPSite(r2, "127.0.0.1", CF_PORT).start()
            ready.set()
        loop.run_until_complete(start()); loop.run_forever()
    threading.Thread(target=run, daemon=True).start(); ready.wait(10)

async def fake_plan(worker, mode, idea, n):
    write = "Use the Write tool to create hello.py containing a print statement."
    return Plan(
        modules=[ModuleSpec("alpha", path="mod_alpha", weight=2), ModuleSpec("beta", path="mod_beta", weight=1)],
        tasks=[TaskSpec("a1", "alpha", write, acceptance_cmd="test -f hello.py"),
               TaskSpec("b1", "beta", write, acceptance_cmd="test -f hello.py")]), "Two tiny modules."

def main():
    serve()
    cli.plan_fn = fake_plan
    project = Path(tempfile.mkdtemp(prefix="devflock-cfe2e-"))
    lines = [
        "1",                                # manager provider: 1) Anthropic-compatible endpoint
        "http://127.0.0.1:18809 sk-mgr",    #   manager URL + key
        "1", "2", "",                       # mode 1, two workers, no specific request
        "2",                                # accept 2 workers
        "2",                                # worker 1 provider: 2) Cloudflare Workers AI
        "3",                                #   model picked by number (sorted list: #3 = openai/gpt-oss-120b)
        "2",                                # worker 2 provider: Cloudflare
        "qwen",                             #   model picked by search text (unique match)
        "y",                                # start
    ]
    res = CliRunner().invoke(cli.main, ["--project", str(project)], input="\n".join(lines) + "\n")
    print(res.output[-2600:])
    if res.exception and not isinstance(res.exception, SystemExit):
        import traceback; traceback.print_exception(res.exception)
    ok = True
    def check(name, cond):
        nonlocal ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    check("wizard exited cleanly", res.exit_code == 0)
    check("credentials taken from env (never prompted)", "Cloudflare API token" not in res.output
          and "using Cloudflare credentials from environment" in res.output)
    check("token never printed", "cf-test-token" not in res.output)
    check("all tasks done", "'done': 2" in res.output)
    models_used = {r["model"] for r in APPS["cf"]["requests"]}
    check(f"each worker ran its own chosen model: {sorted(models_used)}",
          models_used == {"@cf/openai/gpt-oss-120b", "@cf/qwen/qwen3.8-27b"})
    def on_branch(path):
        return subprocess.run(["git", "cat-file", "-e", f"devflock/integration:{path}"], cwd=project).returncode == 0
    check("both Cloudflare workers' files merged onto the integration branch",
          on_branch("mod_alpha/hello.py") and on_branch("mod_beta/hello.py"))
    check("token not written to the project's log", "cf-test-token" not in (project / ".devflock" / "devflock.log").read_text())
    print("\nALL CLI CLOUDFLARE E2E TESTS " + ("PASSED" if ok else "FAILED")); sys.exit(0 if ok else 1)

main()
