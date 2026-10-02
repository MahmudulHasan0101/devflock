"""In-process MCP tools that let a worker interact with OTHER regions without
ever touching their files directly.

  ask_region(region, question)      -- instant, answered from that region's
                                        published `.devflock` Public section.
                                        Never disturbs the other worker.
  request_change(region, description) -- queues a REAL task for that region's
                                        worker. Not immediate: the caller must
                                        check back later (e.g. with another
                                        ask_region) rather than assume it's done.

These are deliberately the only channel between regions. A worker can never
call another region's worker directly, which is what keeps this from turning
into the ask-loops / uncontrolled chatter problem multi-agent setups tend to
have -- see scheduler.py module docstring.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from .region import RegionMemory
from .state import State


def build_region_tools(state: State, region_dirs: dict[str, Path], caller_region: str,
                       caller_worker: str):
    """region_dirs: name -> that region's worktree root (where its `.devflock`
    file lives), for EVERY region in the run, not just the caller's own."""

    @tool("ask_region",
          "Ask about another region's public interface: what it exports, how to run its "
          "tests, recent changes. Answered instantly from that region's published docs -- "
          "does not interrupt its worker or guarantee the info is current.",
          {"region": str, "question": str})
    async def ask_region(args: dict) -> dict:
        region = args.get("region", "")
        if region == caller_region:
            return {"content": [{"type": "text",
                                 "text": "That's your own region -- read its files directly."}]}
        d = region_dirs.get(region)
        if not d:
            return {"content": [{"type": "text",
                                 "text": f"No such region {region!r}. Known regions: "
                                         f"{', '.join(sorted(region_dirs)) or '(none)'}"}],
                    "isError": True}
        mem = RegionMemory.load_or_init(d, region)
        state.log_event("ask_region", worker=caller_worker,
                        detail=f"{caller_region} -> {region}: {args.get('question', '')[:200]}")
        return {"content": [{"type": "text", "text": mem.public or "(nothing published yet)"}]}

    @tool("request_change",
          "Request a change in a region you cannot edit yourself. This QUEUES a task for "
          "that region's own worker; it does not happen now and you are not notified when "
          "it's done -- check back later with ask_region.",
          {"region": str, "description": str})
    async def request_change(args: dict) -> dict:
        region = args.get("region", "")
        if region == caller_region:
            return {"content": [{"type": "text",
                                 "text": "That's your own region -- edit it directly instead."}]}
        if region not in region_dirs:
            return {"content": [{"type": "text",
                                 "text": f"No such region {region!r}. Known regions: "
                                         f"{', '.join(sorted(region_dirs)) or '(none)'}"}],
                    "isError": True}
        task_id = f"req-{uuid.uuid4().hex[:8]}"
        state.add_task(task_id, region,
                       f"(Requested by region `{caller_region}`.) {args.get('description', '')}")
        state.log_event("request_change", worker=caller_worker, task=task_id,
                        detail=f"{caller_region} -> {region}: {args.get('description', '')[:200]}")
        return {"content": [{"type": "text",
                             "text": f"Queued as task {task_id} in region {region!r}. "
                                     f"It runs when that worker is next free."}]}

    return create_sdk_mcp_server("devflock", tools=[ask_region, request_change])
