"""
mcp_bridge.py — GitOracle MCP Bridge
=====================================
Provides:
  - mcp_session()           : async context manager that launches the MCP server
                              subprocess and yields a ready ClientSession
  - discover_openai_tools() : converts MCP tool schemas → OpenAI function-calling format
  - call_mcp_tool()         : calls a tool and returns plain text

The bridge is the single connection point used by both agent_core.py and
langgraph_agent.py, so MCP lifecycle management lives in exactly one place.
"""

import os
import sys
import json
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from logger import get_logger

log = get_logger("MCPBridge")

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


@asynccontextmanager
async def mcp_session():
    """
    Launch mcp_server.py as a subprocess and yield a ready ClientSession.

    KEY: env=os.environ.copy() is required. The MCP stdio transport does NOT
    inherit the parent process environment by default — without it the server
    subprocess never sees GITHUB_TOKEN or any .env values.
    """
    if not os.path.exists(SERVER_PATH):
        log.error(f"MCP server not found at: {SERVER_PATH}")
        raise FileNotFoundError(f"mcp_server.py not found at {SERVER_PATH}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=os.environ.copy(),   # critical — pass parent env through
    )

    log.debug("Launching MCP server subprocess...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                log.debug("MCP session initialized.")
                yield session
    except Exception as e:
        log.error(f"MCP session failed: {e}")
        raise


async def discover_openai_tools(session: ClientSession) -> list[dict]:
    """
    Fetch tools from the live MCP server and convert them to OpenAI
    function-calling schema.

    MCP's tool.inputSchema is already valid JSON Schema, so this is a thin
    reshape — the tool definitions genuinely come from the running server,
    not from a hard-coded list here.
    """
    tools_result = await session.list_tools()
    openai_tools = []
    for t in tools_result.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        })
    log.debug(f"Discovered {len(openai_tools)} tools from MCP server.")
    return openai_tools


async def call_mcp_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """
    Execute a tool on the live MCP server and return the result as plain text.
    Handles both text content items and empty results gracefully.
    """
    try:
        result = await session.call_tool(name, arguments=arguments)
        texts = [item.text for item in result.content if hasattr(item, "text")]
        return "\n".join(texts) if texts else "(tool returned empty result)"
    except Exception as e:
        log.error(f"Tool call '{name}' failed: {e}")
        return f"ERROR: Tool '{name}' failed — {e}"
