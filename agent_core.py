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
    "You have access to live GitHub data through tools — always call a tool "
    "to look up real facts instead of guessing. "
    "Remember context from earlier in the conversation so you can answer follow-up questions. "
    "You may answer general questions (like 'what is FastAPI' or 'who is Torvalds') ONLY IF "
    "they are follow-up questions related to repositories, technologies, or developers previously mentioned in the chat history. "
    "If the very first question is completely off-topic (e.g. cooking, general history, non-software math), "
    "politely decline and redirect them to ask about GitHub repos or developers."
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
