"""Does the REAL bundled Claude Code CLI (via claude_agent_sdk) actually
complete a turn against our mock Anthropic-Messages server, through the
gateway? This is the riskiest integration point in the whole project.

Run: python tests/test_worker_live.py
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aiohttp import web

from devflock.gateway import Gateway, Upstream
from mock_llama_server import make_app


async def run_mock(port, key, tool_capable=True):
    app = make_app(key, flaky=False, tool_capable=tool_capable)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


async def main():
    from claude_agent_sdk import (
        query, ClaudeAgentOptions, AssistantMessage, TextBlock, ToolUseBlock,
        ResultMessage, UserMessage, ProcessError, CLIConnectionError,
    )

    mock = await run_mock(18190, "sk-mock")
    gw = Gateway("w-test", Upstream("http://127.0.0.1:18190", "sk-mock"), port=0)
    gw_port = await gw.start()

    workdir = tempfile.mkdtemp(prefix="devflock-smoke-")
    print(f"worker cwd: {workdir}")
    print(f"gateway: http://127.0.0.1:{gw_port}  (real Claude Code CLI will connect here)\n")

    options = ClaudeAgentOptions(
        env={
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{gw_port}",
            "ANTHROPIC_API_KEY": "sk-mock",
            "ANTHROPIC_MODEL": "qwen",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",  # required to use bypassPermissions when running as root
        },
        cwd=workdir,
        model="qwen",
        tools=[],  # no built-in tools -- isolate the plain text round trip first
        permission_mode="bypassPermissions",
        max_turns=1,
    )

    print("--- attempt 1: plain text round trip, no tools ---")
    saw_text = False
    result_msg = None
    try:
        async for message in query(prompt="Say the word BANANA and nothing else.", options=options):
            print(" ", type(message).__name__, getattr(message, "subtype", ""))
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print("    text:", repr(block.text))
                        if "BANANA" in block.text.upper() or "banana" in block.text.lower():
                            saw_text = True
            if isinstance(message, ResultMessage):
                result_msg = message
    except (ProcessError, CLIConnectionError) as e:
        print("  [FAIL] CLI raised:", repr(e))
        return 1

    if result_msg:
        print(f"  result: subtype={result_msg.subtype} "
              f"cost=${getattr(result_msg, 'total_cost_usd', None)} "
              f"turns={getattr(result_msg, 'num_turns', None)}")
    print(f"[{'PASS' if saw_text else 'FAIL'}] real CLI completed a plain-text turn via mock server\n")

    print("--- attempt 2: tool-use round trip (Write tool) ---")
    options2 = ClaudeAgentOptions(
        env=options.env, cwd=workdir, model="qwen",
        tools=["Write"], allowed_tools=["Write"],
        permission_mode="bypassPermissions", max_turns=3,
    )
    saw_tool = False
    try:
        async for message in query(
                prompt="Use the Write tool to create hello.py containing: print('hi')",
                options=options2):
            print(" ", type(message).__name__)
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        print("    tool_use:", block.name, block.input)
                        saw_tool = True
    except (ProcessError, CLIConnectionError) as e:
        print("  [WARN] tool-use turn raised:", repr(e))

    created = (Path(workdir) / "hello.py").exists()
    print(f"[{'PASS' if (saw_tool or created) else 'INFO'}] tool_use observed={saw_tool} "
          f"file written={created}")

    print(f"\ngateway stats: {gw.stats}")
    await gw.stop()
    await mock.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
