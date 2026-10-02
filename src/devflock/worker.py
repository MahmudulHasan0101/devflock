"""Runs one task on one worker's Claude Code session.

Each Worker owns exactly one region directory (a git worktree) and talks to
Claude Code through its own local Gateway port. A path-guard PreToolUse hook
makes it physically unable to edit files outside its region, regardless of
what the model decides to do -- this is enforcement, not just a prompt
convention.

Sessions are resumed across tasks (so the model keeps region context warm)
up to `context_recycle_tokens`, then a fresh session is started, reseeded
from the region's `.devflock` file. See config.RunConfig.context_recycle_tokens
-- tune this against your own Phase-0 probe results, not this default.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
    HookMatcher,
    ProcessError,
    CLIConnectionError,
)

from .region import RegionMemory

logger = logging.getLogger("devflock.worker")


@dataclasses.dataclass
class TaskResult:
    ok: bool
    summary: str
    changed_interfaces: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    session_id: Optional[str] = None
    error: Optional[str] = None
    tool_calls: list[str] = dataclasses.field(default_factory=list)


def _path_guard_hook(write_roots: list[str], base_dir: str):
    """PreToolUse hook: deny any file-touching tool whose target resolves
    outside this worker's region directory. Applies to Edit/Write/MultiEdit
    (and Bash indirectly is NOT covered here -- see disallowed_tools/sandbox
    notes in the README for shelling out)."""
    roots_real = [os.path.realpath(r) for r in write_roots]

    async def guard(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        target = tool_input.get("file_path") or tool_input.get("path")
        if tool_name in ("Write", "Edit", "MultiEdit", "NotebookEdit") and target:
            resolved = os.path.realpath(os.path.join(base_dir, target)
                                         if not os.path.isabs(target) else target)
            if not any(resolved == r or resolved.startswith(r + os.sep) for r in roots_real):
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason":
                            f"DevFlock path guard: {target!r} is outside this "
                            f"worker's region ({', '.join(write_roots)}). Use ask_region/"
                            f"request_change to modify other regions.",
                    }
                }
        return {}

    return guard


class Worker:
    def __init__(self, worker_id: str, gateway_port: int, api_key: str,
                 region_dir: str | Path, model_name: str = "qwen",
                 extra_env: Optional[dict[str, str]] = None,
                 extra_write_roots: Optional[list[str | Path]] = None):
        self.worker_id = worker_id
        self.gateway_port = gateway_port
        self.api_key = api_key
        self.region_dir = str(region_dir)
        # cwd is region_dir; writes are allowed there plus any extra module folders
        self.write_roots = [self.region_dir] + [str(p) for p in (extra_write_roots or [])]
        self.model_name = model_name
        self.extra_env = extra_env or {}
        self._session_id: Optional[str] = None
        self._context_tokens_estimate = 0

    def _base_env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.gateway_port}",
            "ANTHROPIC_API_KEY": self.api_key,
            "ANTHROPIC_MODEL": self.model_name,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.model_name,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": self.model_name,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.model_name,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            # Required to use permission_mode=bypassPermissions when the
            # orchestrator runs as root (typical in a container). See
            # https://github.com/anthropics/claude-code -- this is the
            # documented escape hatch for disposable sandboxes, not a bypass
            # of anything the person hasn't already agreed to by running
            # headless workers in the first place.
            "IS_SANDBOX": "1",
            **self.extra_env,
        }

    def _options(self, resume: bool, max_turns: int,
                 allowed_tools: list[str], mcp_servers: Optional[dict] = None) -> ClaudeAgentOptions:
        # NOTE: tools/allowed_tools restrict only BUILT-IN tools (Read, Write, ...).
        # MCP-provided tools (mcp_servers=...) are visible whenever configured,
        # regardless of what's in this list -- confirmed against the real CLI.
        # So mcp_servers is the actual on/off switch for ask_region/request_change,
        # not this list.
        return ClaudeAgentOptions(
            env=self._base_env(),
            cwd=self.region_dir,
            model=self.model_name,
            permission_mode="bypassPermissions",
            tools=allowed_tools,
            allowed_tools=allowed_tools,     # auto-approves them (belt-and-braces with bypassPermissions)
            max_turns=max_turns,
            resume=self._session_id if resume else None,
            mcp_servers=mcp_servers or {},
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_path_guard_hook(self.write_roots, self.region_dir)])]},
        )

    async def run_task(self, prompt: str, *, max_turns: int = 20,
                        allowed_tools: Optional[list[str]] = None,
                        resume: bool = True, timeout_s: float = 180.0,
                        mcp_servers: Optional[dict] = None) -> TaskResult:
        """timeout_s guards against a hung submachine or dropped tunnel --
        the CLI subprocess talks to a remote box over a Cloudflare quick
        tunnel with no hard guarantee it ever comes back. On timeout we
        cancel the whole session; the caller sees ok=False and can retry
        against the same or a swapped-in worker."""
        if allowed_tools is None:  # note: [] legitimately means "no tools"
            allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        options = self._options(resume=resume and bool(self._session_id),
                                 max_turns=max_turns, allowed_tools=allowed_tools,
                                 mcp_servers=mcp_servers)

        tool_calls: list[str] = []
        text_out: list[str] = []
        result: Optional[ResultMessage] = None
        error: Optional[str] = None

        async def _run():
            nonlocal result
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                text_out.append(block.text)
                            elif isinstance(block, ToolUseBlock):
                                tool_calls.append(block.name)
                    elif isinstance(message, ResultMessage):
                        result = message

        try:
            async with asyncio.timeout(timeout_s):
                await _run()
        except TimeoutError:
            error = (f"worker[{self.worker_id}] task timed out after {timeout_s:.0f}s "
                     f"(submachine likely hung or tunnel dropped)")
            logger.warning(error)
            self._session_id = None  # session is presumed dead; force a fresh one next time
        except (ProcessError, CLIConnectionError) as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning("worker[%s] task failed: %s", self.worker_id, error)

        if result is not None:
            self._session_id = result.session_id or self._session_id

        ok = error is None and (result is None or result.subtype == "success")
        usage = (result.usage or {}) if result else {}
        return TaskResult(
            ok=ok,
            summary="\n".join(text_out).strip(),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=(result.total_cost_usd or 0.0) if result else 0.0,
            num_turns=result.num_turns if result else 0,
            session_id=self._session_id,
            error=error or (None if ok else f"result subtype={getattr(result, 'subtype', '?')}"),
            tool_calls=tool_calls,
        )

    def recycle_session(self):
        """Force the next run_task to start a fresh session (e.g. because
        estimated context has crossed the recycle threshold)."""
        self._session_id = None


def build_region_seed_prompt(region_dir: Path, region_name: str, task_description: str,
                              acceptance_cmd: Optional[str], neighbor_summaries: dict[str, str],
                              write_roots: Optional[list[str]] = None) -> str:
    """First prompt of a fresh session: region memory + task + how to
    verify + what neighboring regions publicly expose."""
    mem = RegionMemory.load_or_init(region_dir, region_name)
    neighbors = "\n\n".join(f"### {name}\n{summary}" for name, summary in neighbor_summaries.items())
    accept = f"\nWhen done, verify with: `{acceptance_cmd}`" if acceptance_cmd else ""
    roots = write_roots or [str(region_dir)]
    root_lines = "\n".join(f"  - {r}" for r in roots)
    return f"""You are the DevFlock worker owning the region `{region_name}`.
You may WRITE only inside these folders (writes anywhere else are blocked):
{root_lines}
You may read anything in the repository. To change code outside your folders,
describe the change you need in your final message instead of editing it.

## Your region's memory (.devflock)
{mem.render()}

## What neighboring regions publicly expose
{neighbors or "(none yet)"}

## Your task
{task_description}
{accept}

When finished, rewrite the `.devflock` file in your region: update Public
with any interface you added or changed, and Private with anything the next
worker on this region should know.
"""
