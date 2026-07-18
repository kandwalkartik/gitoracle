"""
langgraph_agent.py — GitOracle Multi-Agent LangGraph Version
=============================================================
Architecture:
  User → classify_intent (Router) → conditional edge →
    [repo_info | issue_triage | release_notes | trending]
  Each specialist node calls the REAL MCP server with a scoped tool subset.

Fixes over v1:
  - asyncio.run() inside sync node functions → replaced with asyncio.new_event_loop()
    pattern to avoid "This event loop is already running" crash in Jupyter/IPython
  - Fragile persona.split()[1] label → explicit AGENT_NAME param per specialist
  - Unused `Optional` import removed
  - load_dotenv() + config.py integrated
  - Structured logger replaces all raw print() calls
  - Added 4th specialist: trending_node (searches for trending repos in a topic)
  - tc serialisation uses safe helper (same fix as agent_core.py)
"""

import json
import asyncio

from langgraph.graph import StateGraph
from typing import TypedDict

from config import CFG
from logger import get_logger
from model_client import get_client_and_model, check_connection
from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool

log = get_logger("LangGraph")
client, MODEL, BACKEND = get_client_and_model()


# ── State schema ──────────────────────────────────────────────────────────────

class GitOracleState(TypedDict):
    query: str
    history: list           # short-term memory across turns
    category: str           # repo_info | issue_triage | release_notes | trending
    agent_name: str         # display name of the specialist that ran
    tool_calls_log: list    # audit trail of every tool call made
    answer: str


# ── Router node ───────────────────────────────────────────────────────────────

def classify_intent(state: GitOracleState) -> GitOracleState:
    """Classify the user's query into one of 4 specialist categories."""
    prompt = (
        "Classify this GitHub query into exactly one category. "
        "Reply with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "  repo_info      – general repo stats, search, contributors, commits\n"
        "  issue_triage   – anything about open issues or bug reports\n"
        "  release_notes  – latest release, version, changelog\n"
        "  trending       – find trending or popular repos in a topic/language\n\n"
        f"Query: {state['query']}\n\nCategory:"
    )

    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    raw = result.choices[0].message.content.strip().lower()
    valid = {"repo_info", "issue_triage", "release_notes", "trending"}
    category = raw if raw in valid else "repo_info"

    log.router(category)
    return {**state, "category": category}


def route(state: GitOracleState) -> str:
    return state["category"]


# ── Safe tool call serialiser (fixes the .model_dump() crash) ─────────────────

def _serialise_tc(tc) -> dict:
    try:
        return tc.model_dump()
    except AttributeError:
        pass
    try:
        return tc.dict()
    except AttributeError:
        pass
    return {"id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}


# ── Shared specialist runner ──────────────────────────────────────────────────

def _run_specialist_sync(
    state: GitOracleState,
    agent_name: str,
    persona: str,
    allowed_tools: set[str],
) -> GitOracleState:
    """
    Synchronous wrapper for the async specialist logic.

    FIX (v1 bug): asyncio.run() raises RuntimeError if an event loop is
    already running (Jupyter, IPython, etc.). We create a NEW event loop
    explicitly to avoid this, then close it cleanly.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _run_specialist_async(state, agent_name, persona, allowed_tools)
        )
    finally:
        loop.close()


async def _run_specialist_async(
    state: GitOracleState,
    agent_name: str,
    persona: str,
    allowed_tools: set[str],
) -> GitOracleState:
    """
    One ReAct mini-loop restricted to the specialist's allowed tool subset.
    """
    async with mcp_session() as session:
        all_tools = await discover_openai_tools(session)
        scoped_tools = [t for t in all_tools if t["function"]["name"] in allowed_tools]

        messages = (
            [{"role": "system", "content": persona}]
            + state["history"]
            + [{"role": "user", "content": state["query"]}]
        )

        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=scoped_tools, tool_choice="auto",
            )
        except Exception as e:
            err_str = str(e)
            if "tool_use_failed" in err_str or "400" in err_str:
                log.warning(
                    f"[{agent_name}] Model generated malformed tool call — "
                    "retrying without tools."
                )
                try:
                    response = client.chat.completions.create(
                        model=MODEL, messages=messages, tool_choice="none",
                    )
                except Exception as e2:
                    return {**state, "agent_name": agent_name,
                            "answer": f"Sorry, {agent_name} failed: {e2}",
                            "tool_calls_log": list(state.get("tool_calls_log", []))}
            else:
                return {**state, "agent_name": agent_name,
                        "answer": f"Sorry, an error occurred: {e}",
                        "tool_calls_log": list(state.get("tool_calls_log", []))}

        msg = response.choices[0].message
        tool_calls_log = list(state.get("tool_calls_log", []))

        if response.choices[0].finish_reason == "tool_calls" and msg.tool_calls:
            # Execute all tool calls
            tool_results = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                log.info(f"[{agent_name}] → {tc.function.name}({args})")
                result_text = await call_mcp_tool(session, tc.function.name, args)
                log.tool_result(tc.function.name, result_text)
                tool_calls_log.append({"agent": agent_name, "tool": tc.function.name,
                                        "args": args, "result": result_text})
                tool_results.append((tc, result_text))

            # Synthesis pass: give LLM the tool results to produce final answer
            follow_up = messages + [
                {"role": "assistant", "content": msg.content,
                 "tool_calls": [_serialise_tc(tc) for tc in msg.tool_calls]}
            ] + [
                {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                for tc, result_text in tool_results
            ]

            final = client.chat.completions.create(model=MODEL, messages=follow_up)
            answer = final.choices[0].message.content
        else:
            answer = msg.content or "(no response)"

        # Update shared history for multi-turn memory
        new_history = list(state["history"])
        new_history.append({"role": "user", "content": state["query"]})
        new_history.append({"role": "assistant", "content": answer})

        # Trim history
        max_msgs = CFG.max_history_turns * 3
        if len(new_history) > max_msgs:
            new_history = new_history[-max_msgs:]

        return {**state, "answer": answer, "agent_name": agent_name,
                "history": new_history, "tool_calls_log": tool_calls_log}


# ── Specialist nodes ──────────────────────────────────────────────────────────

def repo_analyst_node(state: GitOracleState) -> GitOracleState:
    return _run_specialist_sync(
        state,
        agent_name="Chronicle",
        persona=(
            "You are Chronicle, the Repository Analyst of GitOracle. "
            "Use the available tools to answer questions about repository stats, "
            "contributors, and recent commits with real, current GitHub data."
        ),
        allowed_tools={"search_repositories", "get_repo_details",
                        "list_contributors", "get_repo_commits"},
    )


def issue_triage_node(state: GitOracleState) -> GitOracleState:
    return _run_specialist_sync(
        state,
        agent_name="Sentinel",
        persona=(
            "You are Sentinel, the Issue Triage specialist of GitOracle. "
            "Use list_open_issues to provide real, currently-open issues. "
            "Note probable priority (Critical/High/Medium/Low) for each."
        ),
        allowed_tools={"list_open_issues"},
    )


def release_notes_node(state: GitOracleState) -> GitOracleState:
    return _run_specialist_sync(
        state,
        agent_name="Herald",
        persona=(
            "You are Herald, the Release Notes specialist of GitOracle. "
            "Use get_latest_release to provide real, current version and changelog info."
        ),
        allowed_tools={"get_latest_release"},
    )


def trending_node(state: GitOracleState) -> GitOracleState:
    """Finds trending/popular repositories in a topic or language."""
    return _run_specialist_sync(
        state,
        agent_name="Pathfinder",
        persona=(
            "You are Pathfinder, the Trending Scout of GitOracle. "
            "Use search_repositories to find the most popular/trending repos "
            "for the topic or language the user mentioned. "
            "Sort by stars. Summarise the top 5 clearly."
        ),
        allowed_tools={"search_repositories"},
    )


# ── Build the LangGraph graph ─────────────────────────────────────────────────

builder = StateGraph(GitOracleState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("repo_info",       repo_analyst_node)
builder.add_node("issue_triage",    issue_triage_node)
builder.add_node("release_notes",   release_notes_node)
builder.add_node("trending",        trending_node)

builder.add_conditional_edges(
    "classify_intent", route,
    {
        "repo_info":      "repo_info",
        "issue_triage":   "issue_triage",
        "release_notes":  "release_notes",
        "trending":       "trending",
    },
)

builder.set_entry_point("classify_intent")
for terminal in ("repo_info", "issue_triage", "release_notes", "trending"):
    builder.set_finish_point(terminal)

graph = builder.compile()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    log.banner(
        "GitOracle  —  Multi-Agent LangGraph",
        "4 Specialists: Chronicle | Sentinel | Herald | Pathfinder",
    )

    if not check_connection(client, MODEL, BACKEND):
        return

    log.section(f"Ready  |  Model: {MODEL}  |  Type 'quit' to exit")
    history = []

    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            log.info("Exiting...")
            break

        if query.lower() == "quit":
            break
        if not query:
            continue

        result = graph.invoke({
            "query": query,
            "history": history,
            "category": "",
            "agent_name": "",
            "tool_calls_log": [],
            "answer": "",
        })

        history = result["history"]
        log.agent_response(result["agent_name"], result["answer"])


if __name__ == "__main__":
    main()
