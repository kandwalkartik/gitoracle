"""
agent_core.py — GitOracle Single ReAct Agent with Human-in-the-Loop
===================================================================
Architecture:
  User → LLM (Reason) → Select Tool → Human Approval Gate →
  MCP Tool → GitHub API → LLM synthesises → Response

Improvements over v1:
  - History trimmed to CFG.max_history_turns (prevents context overflow)
  - Session persistence via conversation_store.py
  - tc serialisation fixed (was using .model_dump() which is not universal)
  - load_dotenv() + config.py (env vars were never loaded from .env before)
  - Structured logger replaces all raw print() calls
"""

import json
import asyncio

from config import CFG
from logger import get_logger
from model_client import get_client_and_model, check_connection
from mcp_bridge import mcp_session, discover_openai_tools, call_mcp_tool
from conversation_store import ConversationStore

log = get_logger("Agent")

SYSTEM_PROMPT = (
    "You are GitOracle, a specialized GitHub intelligence assistant. "
    "Your primary purpose is to help users explore GitHub repositories, developers, "
    "organizations, and open-source software. "
    "You have access to live GitHub data through tools — always call tools to look up real facts instead of guessing.\n"
    "\n"
    "DETAILED REPOSITORY SUMMARIES:\n"
    "When a user asks about a repository (e.g., 'Tell me about owner/repo'), you MUST proactively call multiple tools "
    "(such as get_repo_details, get_repo_readme, list_contributors, and get_latest_release) to compile a rich, thorough summary. "
    "Do not ask the user if they want to know more; output the full summary immediately. Structure your summary as follows:\n"
    "1. Description & Purpose: What is the repository, what does it do, what problem does it solve, and what are its key features? (You must call get_repo_readme to read the project's README file and extract detailed project ideas and stack summaries!)\n"
    "2. Key Stats: Stars, forks, language, license, and open issues.\n"
    "3. Community & Activity: List the top contributors (with commit counts) and details of the latest release (version/date).\n"
    "\n"
    "CONVERSATIONAL BOUNDARIES & ZONAL CONTEXT:\n"
    "1. The active project/repository is permanently anchored to the VERY FIRST repository, developer, or organization queried at the start of the session (e.g. 'huggingface/transformers'). Subsequent follow-up topics must NEVER become the active project context.\n"
    "2. If the user's query is completely unrelated to technology, programming, or open-source (e.g., actors, sports, cooking, history, gossip):\n"
    "   - Refuse the question strictly (Level 1: Hard Rejection).\n"
    "   - Output exactly: 'I only answer questions related to software, developers, and repositories. Let's get back to [active project].'\n"
    "3. If the query is tech-related and has a logical correlation to the active project (Level 2: Connected Explanation):\n"
    "   - Provide a full answer and explain how it connects to the active project.\n"
    "4. If the query is tech-related but has NO correlation to the active project (Level 3: Uncorrelated Tech Follow-up):\n"
    "   - Provide only a very brief (2-sentence) basic summary of the topic.\n"
    "   - Append exactly: 'Note: This subject has no correlation to the active project ([active project]).'"
)


def _serialise_tool_call(tc) -> dict:
    """
    Safely serialise an OpenAI SDK ToolCall object to a plain dict.
    Handles both Pydantic v1 (.dict()) and v2 (.model_dump()) objects,
    and falls back to manual construction if neither is available.
    This was the source of the v1 crash on some SDK versions.
    """
    try:
        return tc.model_dump()
    except AttributeError:
        pass
    try:
        return tc.dict()
    except AttributeError:
        pass
    # Manual fallback — always safe
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


def _trim_history(history: list) -> list:
    """Keep only the most recent max_history_turns * 3 messages."""
    max_msgs = CFG.max_history_turns * 3
    if len(history) > max_msgs:
        trimmed = history[-max_msgs:]
        log.debug(f"History trimmed: {len(history)} → {len(trimmed)} messages")
        return trimmed
    return history


class MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class MockToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = MockFunction(name, arguments)


class GitOracleAgent:
    """Single ReAct agent with persistent memory and optional human approval gate."""

    def __init__(self, session_id: str | None = None, auto_approve: bool = False):
        self.client, self.model, self.backend = get_client_and_model()
        self.store = ConversationStore(session_id)
        self.history: list = []
        self.auto_approve = auto_approve
        self.store.load(self.history)   # restore previous session from disk

    async def ask(self, user_input: str, max_loops: int = 6) -> str:
        self.history.append({"role": "user", "content": user_input})

        async with mcp_session() as session:
            tools = await discover_openai_tools(session)

            for loop_num in range(max_loops):
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                    )
                except Exception as e:
                    # Groq tool_use_failed — model generated malformed tool call.
                    # Retry once without tools so we at least get a text answer.
                    err_str = str(e)
                    if "tool_use_failed" in err_str or "400" in err_str:
                        log.warning(
                            "Model generated a malformed tool call — "
                            "retrying without tools. "
                            "Tip: switch GROQ_MODEL=llama-3.3-70b-versatile in .env for best results."
                        )
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                tool_choice="none",
                            )
                        except Exception as e2:
                            return f"Sorry, the model failed to respond: {e2}"
                    else:
                        return f"Sorry, an error occurred: {e}"

                msg = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # --- Fallback Parser for raw text tool calls (e.g. Ollama/Qwen 'brtc') ---
                raw_calls = []
                content = msg.content or ""
                if "{" in content and ("name" in content or "arguments" in content):
                    import re
                    matches = re.findall(r'(\{\s*"name"\s*:.*?\})', content, re.DOTALL)
                    for match in matches:
                        try:
                            data = json.loads(match)
                            if "name" in data:
                                tc_id = f"call_raw_{loop_num}_{len(raw_calls)}"
                                args_val = data.get("arguments", {})
                                args_str = json.dumps(args_val) if isinstance(args_val, dict) else str(args_val)
                                raw_calls.append(MockToolCall(tc_id, data["name"], args_str))
                        except Exception as parse_err:
                            log.debug(f"Failed parsing raw tool call match: {parse_err}")
                
                if raw_calls:
                    log.info(f"Ollama/Qwen Fallback Parser: extracted {len(raw_calls)} raw tool call(s) from content.")
                    msg.tool_calls = raw_calls
                    finish_reason = "tool_calls"

                # ── Final answer: no tool needed ──────────────────────────
                if finish_reason != "tool_calls" or not msg.tool_calls:
                    answer = msg.content or "(no response)"
                    self.history.append({"role": "assistant", "content": answer})
                    self.history = _trim_history(self.history)
                    self.store.save(self.history)
                    return answer

                # ── Tool calls requested ──────────────────────────────────
                self.history.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [_serialise_tool_call(tc) for tc in msg.tool_calls],
                })

                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)

                    # Human-in-the-Loop approval gate (skipped when called from the web API)
                    if self.auto_approve:
                        approval = "y"
                    else:
                        log.tool_request(tc.function.name, args)
                        approval = input("  Approve? [y/n/skip all]: ").strip().lower()

                    if approval in ("n", "no"):
                        log.warning("Tool execution rejected by human.")
                        self.history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Tool execution was denied by the human reviewer. Do not retry this tool.",
                        })
                        continue

                    if approval in ("skip all", "s"):
                        # Auto-approve all remaining tools in this loop
                        log.info("Auto-approving all remaining tool calls.")

                    log.info(f"Executing tool: {tc.function.name} ...")
                    result_text = await call_mcp_tool(session, tc.function.name, args)
                    log.tool_result(tc.function.name, result_text)

                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

        self.history = _trim_history(self.history)
        self.store.save(self.history)
        return "I was unable to resolve your request within the reasoning step limit."


async def main():
    log.banner(
        "GitOracle  —  GitHub Intelligence Oracle",
        "Single ReAct Agent + Human-in-the-Loop  |  Type 'quit' to exit  |  'clear' to reset session",
    )

    agent = GitOracleAgent()

    if not check_connection(agent.client, agent.model, agent.backend):
        return

    log.section(f"Ready  |  Model: {agent.model}  |  Session: {agent.store.session_id}")
    log.info("Loaded session history with %d messages." % len(agent.history))

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            log.info("Exiting...")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            log.info("Session saved. Goodbye!")
            break

        if user_input.lower() == "clear":
            agent.store.clear()
            agent.history.clear()
            log.success("Session cleared.")
            continue

        answer = await agent.ask(user_input)
        log.agent_response("GitOracle", answer)


if __name__ == "__main__":
    asyncio.run(main())
