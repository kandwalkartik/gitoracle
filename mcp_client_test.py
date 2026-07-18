"""
mcp_client_test.py - GitOracle MCP Protocol Test
================================================
Verifies the full MCP stack without involving an LLM:
  1. Handshake  — initialize()
  2. Discovery  — list_tools(), list_resources(), list_prompts()
  3. Invocation — call_tool() on ALL 7 tools with realistic arguments
  4. Timing     — reports wall-clock time for each tool call

Run this first when setting up the project to confirm the MCP server
and GitHub API are both reachable before running any agent.
"""

import asyncio
import os
import time
import sys
import io

# Force UTF-8 on Windows terminals to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")

# Test calls: (tool_name, arguments_dict)
TEST_CALLS = [
    ("search_repositories",  {"query": "agentic ai framework language:python"}),
    ("get_repo_details",     {"owner": "langchain-ai", "repo": "langchain"}),
    ("list_open_issues",     {"owner": "langchain-ai", "repo": "langchain", "limit": 3}),
    ("list_contributors",    {"owner": "langchain-ai", "repo": "langchain", "limit": 3}),
    ("get_latest_release",   {"owner": "openai",       "repo": "openai-python"}),
    ("get_repo_commits",     {"owner": "openai",       "repo": "openai-python", "limit": 3}),
    ("compare_repos",        {"owner1": "langchain-ai", "repo1": "langchain",
                               "owner2": "microsoft",   "repo2": "autogen"}),
    ("get_user_profile",     {"username": "torvalds"}),
    ("get_org_details",      {"org": "huggingface"}),
]

PASS = "[PASS]"
FAIL = "[FAIL]"

async def main():
    print(f"\n{'='*65}")
    print(f" GitOracle MCP Client Test  —  All 7 Tools")
    print(f"{'='*65}")
    print(f"GitHub auth: {'YES (GITHUB_TOKEN set)' if os.environ.get('GITHUB_TOKEN') else 'NO  (60 req/hr limit)'}")
    print(f"{'='*65}\n")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # ── Step 1: Handshake ──────────────────────────────────────────
            t0 = time.perf_counter()
            await session.initialize()
            print(f"[1] Handshake complete ({(time.perf_counter()-t0)*1000:.0f} ms)\n")

            # ── Step 2: Discovery ──────────────────────────────────────────
            tools_result    = await session.list_tools()
            resources_result = await session.list_resources()
            prompts_result  = await session.list_prompts()

            print(f"[2] Discovery:")
            print(f"    Tools     ({len(tools_result.tools)}): {[t.name for t in tools_result.tools]}")
            print(f"    Resources : (URI-template based, listed as templates)")
            print(f"    Prompts   ({len(prompts_result.prompts)}): {[p.name for p in prompts_result.prompts]}\n")

            # ── Step 3: Tool invocations ───────────────────────────────────
            print(f"[3] Tool Invocations:")
            results_summary = []

            for i, (tool_name, args) in enumerate(TEST_CALLS, 1):
                print(f"\n  [{i}/{len(TEST_CALLS)}] {tool_name}({args})")
                t0 = time.perf_counter()
                try:
                    result = await session.call_tool(tool_name, arguments=args)
                    elapsed = (time.perf_counter() - t0) * 1000
                    texts = [item.text for item in result.content if hasattr(item, "text")]
                    output = "\n".join(texts) if texts else "(empty)"
                    # Indent output for readability
                    indented = "\n".join(f"      {line}" for line in output.splitlines())
                    print(f"  {PASS} ({elapsed:.0f} ms)")
                    print(indented)
                    results_summary.append((tool_name, True, elapsed))
                except Exception as e:
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"  {FAIL} ({elapsed:.0f} ms)  ERROR: {e}")
                    results_summary.append((tool_name, False, elapsed))

            # ── Summary ────────────────────────────────────────────────────
            print(f"\n{'='*65}")
            print(" SUMMARY")
            print(f"{'='*65}")
            passed = sum(1 for _, ok, _ in results_summary if ok)
            for name, ok, ms in results_summary:
                status = f"{PASS} PASS" if ok else f"{FAIL} FAIL"
                print(f"  {status}  {name:<30} ({ms:.0f} ms)")
            print(f"{'='*65}")
            print(f"  {passed}/{len(results_summary)} tools passed\n")


if __name__ == "__main__":
    asyncio.run(main())
