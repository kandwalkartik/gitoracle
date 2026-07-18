"""
api_server.py — GitOracle FastAPI Backend
==========================================
HTTP bridge between the web frontend and the GitOracle agents.

Endpoints:
  POST /chat          — Send a query to the single ReAct agent
  GET  /api/suggest   — Smart dropdown: username → repos sorted by stars
  GET  /health        — Simple liveness check

Run locally:
    uvicorn api_server:app --reload --port 8000

Environment variables (same .env as the rest of the project):
    GROQ_API_KEY, GITHUB_TOKEN, LLM_BACKEND, GROQ_MODEL
"""

import os
import json
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import CFG
from logger import get_logger

log = get_logger("APIServer")

app = FastAPI(
    title="GitOracle API",
    description="GitHub Intelligence Oracle — REST API",
    version="1.0.0",
)

# Allow frontend (any origin during dev; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Serve frontend static files ───────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list = []
    session_id: str | None = None


# ── Tool call capture (monkey-patch friendly) ─────────────────────────────────

_captured_tool_calls: list = []


def _capture_tool_calls_from_agent(agent) -> list:
    """Read tool_calls_log from the agent's conversation store if available."""
    try:
        history = agent.store.load_raw() if hasattr(agent.store, "load_raw") else []
        return []
    except Exception:
        return []


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send a query to the GitOracle single ReAct agent.
    Returns the answer and a log of any tool calls made.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"Chat request: {req.query[:80]}")

    try:
        # Import here so the module-level CFG is already loaded
        from agent_core import GitOracleAgent

        # Intercept tool calls by wrapping call_mcp_tool
        import mcp_bridge
        tool_calls_log: list = []
        _original_call = mcp_bridge.call_mcp_tool

        async def _instrumented_call(session, name, args):
            result = await _original_call(session, name, args)
            tool_calls_log.append({"tool": name, "args": args, "result": result})
            return result

        mcp_bridge.call_mcp_tool = _instrumented_call

        agent = GitOracleAgent(session_id=req.session_id)
        answer = await agent.ask(req.query)

        # Restore original
        mcp_bridge.call_mcp_tool = _original_call

        return ChatResponse(
            answer=answer,
            tool_calls=tool_calls_log,
            session_id=req.session_id,
        )

    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Suggest endpoint ──────────────────────────────────────────────────────────

GITHUB_API = os.environ.get("GITHUB_API", "https://api.github.com")


def _gh_headers() -> dict:
    h = {"User-Agent": "GitOracle-API/1.0", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


@app.get("/api/suggest")
async def suggest(q: str = ""):
    """
    Smart search suggestions.
    - If q looks like a username → return their repos sorted by stars + avatar
    - If q contains '/' → return nothing (user is typing owner/repo directly)
    - Otherwise → return nothing (free-form query)
    """
    q = q.strip()

    # Don't suggest for empty, spaces, or already-qualified paths
    if not q or " " in q or len(q) < 2:
        return {"type": "none", "results": []}

    if "/" in q:
        return {"type": "none", "results": []}

    async with httpx.AsyncClient(timeout=6) as client:
        # 1. Check if it's a valid user / org
        try:
            user_resp = await client.get(
                f"{GITHUB_API}/users/{q}", headers=_gh_headers()
            )
        except httpx.RequestError:
            return {"type": "none", "results": []}

        if user_resp.status_code != 200:
            return {"type": "none", "results": []}

        user_data = user_resp.json()
        avatar = user_data.get("avatar_url", "")
        account_type = user_data.get("type", "User")   # User | Organization
        display_name = user_data.get("name") or q

        # 2. Fetch their repos sorted by stars
        try:
            repos_resp = await client.get(
                f"{GITHUB_API}/users/{q}/repos",
                headers=_gh_headers(),
                params={"sort": "stars", "direction": "desc", "per_page": 6, "type": "public"},
            )
        except httpx.RequestError:
            return {"type": "none", "results": []}

        if repos_resp.status_code != 200:
            return {"type": "none", "results": []}

        repos = repos_resp.json()
        repo_list = [
            {
                "full_name": r["full_name"],
                "name": r["name"],
                "stars": r.get("stargazers_count", 0),
                "description": (r.get("description") or "")[:80],
                "language": r.get("language") or "",
            }
            for r in repos
            if not r.get("fork", False)   # skip forks by default
        ]

        return {
            "type": "user" if account_type == "User" else "org",
            "username": q,
            "display_name": display_name,
            "avatar": avatar,
            "results": repo_list,
        }


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": CFG.groq_model if CFG.llm_backend == "groq" else CFG.ollama_model,
        "backend": CFG.llm_backend,
        "github_auth": bool(os.environ.get("GITHUB_TOKEN")),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"GitOracle API Server starting on http://localhost:{port}")
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
