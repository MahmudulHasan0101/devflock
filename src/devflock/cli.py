"""`devflock` -- interactive wizard.

  manager URL + key -> probe -> mode (fixed N / auto / idea) -> manager plans
  -> verdict (adjust N) -> worker URLs + keys (probed) -> run -> summary
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table

from . import cloudflare as cf
from . import gitops
from .config import Endpoint, PROVIDER_CLOUDFLARE, RunConfig
from .gateway import Gateway, Upstream
from .packer import Plan, max_parallelism, pack, recommend_n
from .planner import PlanningFailed, make_plan
from .probe import ProbeResult, probe
from .runner import run_flock
from .worker import Worker

console = Console()
plan_fn = make_plan  # indirection so tests can stub the (LLM-dependent) planner


def parse_conn(text: str) -> tuple[str, Optional[str]]:
    """Accept 'URL', 'URL KEY' or 'URL|KEY' (what a notebook can print in one line)."""
    parts = [p for p in re.split(r"[\s|]+", text.strip()) if p]
    url = parts[0].rstrip("/") if parts else ""
    return url, (parts[1] if len(parts) > 1 else None)


def ask_endpoint(label: str, name: str, allow_blank: bool = False) -> Optional[Endpoint]:
    hint = " (blank = reuse manager)" if allow_blank else ""
    raw = click.prompt(f"{label} URL{hint}", default="" if allow_blank else None,
                       show_default=False).strip()
    if not raw:
        return None
    url, key = parse_conn(raw)
    if not url.startswith("http"):
        console.print("[red]URL must start with http:// or https://[/red]")
        return ask_endpoint(label, name, allow_blank)
    if not key:
        key = click.prompt(f"{label} API key", hide_input=True).strip()
    return Endpoint(name=name, base_url=url, api_key=key)


def show_probe(name: str, r: ProbeResult):
    if not r.ok:
        console.print(f"  [red]✗ {name}: {r.error}[/red]")
        return
    bits = [f"{r.latency_s:.1f}s reply"]
    if r.n_ctx:
        bits.append(f"ctx {r.n_ctx:,}")
    bits.append("tools ✓" if r.tools_ok else ("tools ?" if r.tools_ok is None else "[yellow]tools ✗[/yellow]"))
    console.print(f"  [green]✓ {name}[/green]  " + ", ".join(bits))
    for w in r.warnings:
        console.print(f"    [yellow]! {w}[/yellow]")


# ---- Cloudflare Workers AI --------------------------------------------------
_cf_creds: Optional[cf.CfCreds] = None      # asked once per run, shared by every Cloudflare worker
_cf_models: Optional[list[cf.CfModel]] = None


def get_cf_creds() -> cf.CfCreds:
    """Account id + API token: env vars first (CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN),
    otherwise prompted once with the token hidden. Never echoed back in full."""
    global _cf_creds
    if _cf_creds:
        return _cf_creds
    acct, tok, gw = cf.creds_from_env()
    if acct and tok:
        console.print("  using Cloudflare credentials from environment variables")
    else:
        console.print("  Cloudflare needs your Account ID and an API token with "
                      "[bold]Account › Workers AI › Read[/bold] permission.")
        acct = acct or click.prompt("  Cloudflare Account ID").strip()
        tok = tok or click.prompt("  Cloudflare API token", hide_input=True).strip()
    _cf_creds = cf.CfCreds(acct, tok, gw)
    return _cf_creds


def _fmt_ctx(n: Optional[int]) -> str:
    return "" if not n else (f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k")


def show_cf_models(models: list[cf.CfModel], limit: int = 40):
    t = Table(show_header=True, header_style="bold")
    t.add_column("#", justify="right"); t.add_column("model"); t.add_column("ctx", justify="right"); t.add_column("tools")
    for i, m in enumerate(models[:limit], 1):
        tools = "✓" if m.function_calling else ("" if m.function_calling is None else "[dim]✗[/dim]")
        t.add_row(str(i), m.short, _fmt_ctx(m.context_window), tools)
    console.print(t)
    if len(models) > limit:
        console.print(f"  [dim]… {len(models) - limit} more; type part of a name to filter[/dim]")


def pick_cf_model(label: str, creds: cf.CfCreds) -> str:
    """Choose ONE Workers AI model for this worker: number, search text, or a pasted @cf/… id."""
    global _cf_models
    if _cf_models is None:
        try:
            with console.status("fetching Workers AI text-generation models…"):
                _cf_models = asyncio.run(cf.list_text_models(creds))
        except Exception as e:
            console.print(f"  [yellow]could not list models ({e}); paste a model id instead[/yellow]")
            _cf_models = []
    shown = _cf_models
    if shown:
        show_cf_models(shown)
    while True:
        raw = click.prompt(f"{label} model (number, search text, or @cf/… id)").strip()
        if raw.startswith(("@cf/", "@hf/")):
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(shown):
            return shown[int(raw) - 1].id
        hits = [m for m in _cf_models if raw.lower() in m.id.lower()]
        if len(hits) == 1:
            return hits[0].id
        if hits:
            shown = hits
            show_cf_models(shown)
        else:
            console.print("  [red]no match[/red]")


def ask_cloudflare_endpoint(label: str, name: str) -> Endpoint:
    creds = get_cf_creds()
    model = pick_cf_model(label, creds)
    return Endpoint(name=name, base_url=cf.ai_base_url(creds.account_id), api_key=creds.token,
                    auth_style="bearer", model_name=model, provider=PROVIDER_CLOUDFLARE,
                    gateway_id=creds.gateway_id)


def ask_provider(label: str, allow_reuse: bool) -> str:
    """'url' | 'cf' | 'reuse'"""
    console.print(f"\n[bold]{label}[/bold] runs on:\n"
                  "  1) Anthropic-compatible endpoint (llama-server / tunnel URL + key)\n"
                  "  2) Cloudflare Workers AI (pick a model)" +
                  ("\n  3) same as the manager" if allow_reuse else ""))
    choices = ["1", "2"] + (["3"] if allow_reuse else [])
    c = click.prompt("Choose", type=click.Choice(choices), default="3" if allow_reuse else "1")
    return {"1": "url", "2": "cf", "3": "reuse"}[c]


def connect(label: str, name: str, manager: Optional[Endpoint] = None) -> Endpoint:
    """Ask for an endpoint until it probes OK, or the person chooses to use it anyway."""
    while True:
        kind = ask_provider(label, allow_reuse=manager is not None)
        if kind == "reuse":
            console.print(f"  reusing the manager endpoint for {name} "
                          f"({'same Cloudflare model' if manager.provider == PROVIDER_CLOUDFLARE else 'they will share one GPU box'})")
            return dataclasses.replace(manager, name=name)
        ep = ask_cloudflare_endpoint(label, name) if kind == "cf" else ask_endpoint(label, name)
        with console.status(f"probing {name}…"):
            r = asyncio.run(probe(ep))
        show_probe(name, r)
        if r.ok:
            ep.auth_style = r.auth_style
            return ep
        choice = click.prompt("  [r]etry / [u]se anyway / [a]bort", type=click.Choice(["r", "u", "a"]),
                              default="r")
        if choice == "u":
            return ep
        if choice == "a":
            raise click.Abort()


async def plan_with_manager(ep: Endpoint, project: Path, mode: str, idea: Optional[str], n: Optional[int]):
    gw = Gateway("manager", Upstream.from_endpoint(ep))
    port = await gw.start()
    try:
        w = Worker("manager", port, ep.client_key, project, model_name=ep.client_model)
        return await plan_fn(w, mode, idea, n)
    finally:
        await gw.stop()


def show_verdict(plan: Plan, summary: str, rec: int):
    if summary:
        console.print(f"\n[bold]Manager's plan:[/bold] {summary}")
    t = Table(title="Modules")
    for c in ("module", "folder", "weight", "tasks"):
        t.add_column(c)
    for m in plan.modules:
        t.add_row(m.name, m.path or m.name, f"{m.weight:g}",
                  str(sum(1 for x in plan.tasks if x.module == m.name)))
    console.print(t)
    console.print(f"{len(plan.tasks)} tasks; up to {max_parallelism(plan)} can run at once. "
                  f"[bold]Recommended workers: {rec}[/bold]")


def show_assignment(assignments):
    t = Table(title="Region assignment")
    t.add_column("region"); t.add_column("modules")
    for a in assignments:
        t.add_row(a.region, ", ".join(a.modules))
    console.print(t)


def status_table(state, rws) -> Table:
    t = Table(title="DevFlock")
    for c in ("worker", "region", "doing"):
        t.add_column(c)
    running = {r["region"]: r["id"] for r in state.conn.execute("SELECT region, id FROM tasks WHERE status='running'")}
    for region, rw in rws.items():
        t.add_row(rw.worker_id, region, running.get(region, "[dim]idle[/dim]"))
    counts = state.counts_by_status()
    t.caption = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    return t


async def run_with_display(cfg, plan, assignments):
    holder: dict = {}
    task = asyncio.create_task(run_flock(cfg, plan, assignments,
                                         on_started=lambda st, rws: holder.update(state=st, rws=rws)))
    tty = console.is_terminal
    last = None
    with (Live(console=console, refresh_per_second=2) if tty else contextlib.nullcontext()) as live:
        while not task.done():
            await asyncio.wait({task}, timeout=1.5)
            if "state" not in holder:
                continue
            try:
                if tty:
                    live.update(status_table(holder["state"], holder["rws"]))
                else:
                    c = holder["state"].counts_by_status()
                    if c != last:
                        console.print(f"tasks: {c}")
                        last = c
            except Exception:
                pass  # display must never break a run
    return task.result()


@click.command()
@click.option("--project", "project_dir", default=".", show_default=True,
              help="Project folder (existing code, or where a new project will be created).")
def main(project_dir: str):
    """DevFlock: run many Claude Code workers, each on its own model endpoint."""
    project = Path(project_dir).resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / ".devflock").mkdir(exist_ok=True)
    logging.basicConfig(filename=str(project / ".devflock" / "devflock.log"), level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    console.print("[bold]DevFlock[/bold]  ·  many Claude Code workers, one project\n")

    manager = connect("Manager", "manager")

    console.print("\nHow should the workforce be sized?\n"
                  "  1) I choose the number of workers\n"
                  "  2) Let the manager decide, for my existing project\n"
                  "  3) I have a project idea; the manager designs it and splits the work")
    mode_no = click.prompt("Choose", type=click.Choice(["1", "2", "3"]), default="2")

    mode, idea, hint_n = "existing", None, None
    if mode_no == "3":
        mode = "idea"
        idea = click.prompt("Describe the project (or a path to a text file)").strip()
        if os.path.isfile(idea):
            idea = Path(idea).read_text()
    else:
        if mode_no == "1":
            hint_n = click.prompt("How many workers", type=click.IntRange(1, 64))
        idea = click.prompt("What should the workers do? (blank = document interfaces and add tests)",
                            default="", show_default=False).strip() or None

    if mode == "existing" and not gitops.is_git_repo(project) and any(
            p for p in project.iterdir() if p.name != ".devflock"):
        if not click.confirm("This folder is not a git repo. DevFlock will `git init` and commit its "
                             "current files (workers use git worktrees). Continue?", default=True):
            raise click.Abort()
    if gitops.is_dirty(project):
        click.confirm("You have uncommitted changes. Workers start from the last commit and will NOT "
                      "see them. Continue anyway?", abort=True)

    try:
        with console.status("manager is planning (this can take a few minutes)…"):
            plan, summary = asyncio.run(plan_with_manager(manager, project, mode, idea, hint_n))
    except PlanningFailed as e:
        console.print(f"[red]{e}[/red]")
        raise click.exceptions.Exit(1)

    rec = recommend_n(plan)
    show_verdict(plan, summary, rec)
    n = click.prompt("Number of workers to use (you can lower or raise it)",
                     type=click.IntRange(1, len(plan.modules)), default=min(hint_n or rec, len(plan.modules)))
    assignments = pack(plan, n)
    show_assignment(assignments)

    console.print(f"\nConnect {len(assignments)} worker endpoint(s):")
    workers = [connect(f"Worker {i}", f"worker-{i}", manager) for i in range(1, len(assignments) + 1)]

    click.confirm(f"\nStart {len(assignments)} workers on {len(plan.tasks)} tasks?", default=True, abort=True)
    cfg = RunConfig(project_dir=str(project), manager=manager, workers=workers, mode=mode, idea_prompt=idea)
    result = asyncio.run(run_with_display(cfg, plan, assignments))

    console.print(f"\n[bold]Finished.[/bold] tasks: {result['counts']}   "
                  f"tokens in/out: {result['tokens_in']:,}/{result['tokens_out']:,}")
    for f in result["failed"]:
        console.print(f"  [red]{f['status']}[/red] {f['id']} (region {f['region']})")
    if result["conflicts"]:
        console.print(f"[yellow]merge conflicts in regions: {result['conflicts']} — resolve on branch "
                      f"{result['branch']}[/yellow]")
    if result["integrated"]:
        console.print(f"Merged work is on branch [bold]{result['branch']}[/bold] "
                      f"(regions: {', '.join(result['integrated'])}). Review with: git diff {result['base_branch']}...{result['branch']}   (merge it when happy)")
