<p align="center">
  <img src="assets/title.png" alt="DevFlock" width="380">
</p>

# DevFlock v0.3.0

Run many Claude Code sessions in parallel, each pointed at its own
Anthropic-compatible endpoint (e.g. llama-server on Kaggle behind a tunnel).
A manager model splits the project into modules and tasks; one worker owns each
region; the orchestrator verifies work and merges it.

## Install and run
    pip install -e .
    devflock --project /path/to/project

Wizard: manager URL+key (`URL KEY` or `URL|KEY` on one line) -> probe -> choose
(1) your own worker count, (2) manager decides for an existing project, or
(3) project idea -> manager's plan and recommended N (you can change N) ->
worker URLs+keys (blank = reuse manager) -> run -> merged branch
`devflock/integration`. Your own branch is left untouched.

Before anything else, run `python tools/devflock_probe.py URL KEY --big --long`
against one real submachine (tunnel streaming, prefill speed, tool-call reliability).

## Layout
- planner.py    manager prompt; JSON parse, schema + cycle validation, retry with errors fed back
- packer.py     max parallelism, recommended N, packing modules into N regions
- runner.py     plan -> git worktrees -> workers -> scheduler -> integration
- scheduler.py  task DAG, per-region workers, verify + repair loop, blocks dependents of failures
- worker.py     Agent SDK session, write-path guard, session resume, hard timeout
- gateway.py    local per-worker proxy: retries, SSE passthrough, hot-swappable upstream
- tools.py      ask_region / request_change (in-process MCP tools) -- OFF by default, see below
- probe.py, state.py (SQLite), region.py (.devflock memory), gitops.py, verifier.py, cli.py

## Verified (mock server + the REAL bundled Claude Code binary)
gateway, worker (incl. the real path-guard hook blocking an out-of-region write),
scheduler, failure cascade (a failed task blocks its dependents, scheduler exits
cleanly), planner validation/retry, and the full wizard end to end:
`python tests/test_<name>.py` for gateway_live, worker_run, scheduler_live,
scheduler_failure, planner, cli_e2e. All six pass.

## NOT verified: your real Qwen endpoints
The mock is a scripted stand-in. Whether Qwen reliably drives Claude Code's tool
loop, produces valid plan JSON, and streams through a Cloudflare quick tunnel is
unknown until you run it. Expect to tune prompts in planner.py.

## Known gaps
- **ask_region / request_change are implemented but OFF by default**
  (`RunConfig.enable_cross_region_tools=False`). The mechanism works in
  isolation (a single Worker.run_task call offers and calls the tool
  correctly on turn 1), but driven through the Scheduler the MCP tool is
  consistently NOT offered on turn 1 of a fresh session -- not an
  intermittent race, every attempt hits it -- so tasks that rely on it
  exhaust their repair attempts and fail. Root cause not yet isolated,
  possibly an MCP handshake/session-startup ordering issue in this SDK
  version when combined with how Scheduler constructs sessions. See
  `tests/test_cross_region_tools.py` (currently failing; kept because it's
  the most useful repro for whoever debugs this next) and compare against
  the passing standalone debug pattern described in its docstring. Don't
  flip the flag on for a real run until this is fixed.
- The path guard covers Write/Edit only. A worker's Bash can still write
  anywhere. Run workers in a container/VM you don't mind losing.
  IS_SANDBOX=1 is set for root.
- No token-based session recycling yet (context_recycle_tokens is unused).
- Mode 2 ("auto") does no separate mapping phase; the manager just reads the repo.
- No mid-run tunnel swap UI (RegionWorker.swap_endpoint exists, unwired).
- No resume of an interrupted run from state.db (state is written, not re-read).
- Reported token counts come from Claude Code; any "cost" it computes is meaningless for Qwen.

## Lessons baked in (from real debugging against the real CLI)
- Running as root needs IS_SANDBOX=1 for bypassPermissions.
- `tools=` restricts what the model sees; `allowed_tools=` only auto-approves.
  MCP-provided tools (via mcp_servers=) are visible whenever configured,
  regardless of either list -- confirmed empirically against the real CLI.
- A failed task must block its dependents or the scheduler never exits.
- Region = folders inside a per-region worktree, so merges touch disjoint paths.
- Worktree paths nest under `.devflock/worktrees/...` by design, which means
  any naive regex looking for "a file path" in prompt/context text can
  mistake the directory segment `.devflock` for a file extension. (This bit
  the test mock repeatedly; it's a trap for anything else that tries to
  parse paths out of free text too.)

<p align="center">
  <img src="assets/logo.png" alt="DevFlock logo" width="140">
</p>
