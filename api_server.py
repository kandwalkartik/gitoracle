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
import time
import re
import httpx
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import FastAPI, HTTPException, Request, Response
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

# CORS: Allow local dev origins and production website domain if specified
allowed_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
]
env_origin = os.environ.get("ALLOWED_ORIGIN")
if env_origin:
    allowed_origins.append(env_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if env_origin else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Simple sliding window rate limiting (IP-based, in-memory)
# Format: {ip_address: [timestamps_of_requests]}
request_history = defaultdict(list)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 15  # maximum requests per window

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Only rate limit API queries (/chat and /api/suggest)
    path = request.url.path
    if path in ["/chat"]:
        client_ip = request.client.host or "unknown"
        now = time.time()
        
        # Filter timestamps to keep only those within the window
        timestamps = request_history[client_ip]
        request_history[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
        
        if len(request_history[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            log.warning(f"Rate limit exceeded for IP: {client_ip} on path: {path}")
            raise HTTPException(status_code=429, detail="Too many requests. Please try again in a minute.")
            
        request_history[client_ip].append(now)
        
    return await call_next(request)

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
    # Basic input sanitization
    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    # Enforce reasonable length limit to prevent buffer / context overload abuses
    if len(clean_query) > 500:
        raise HTTPException(status_code=400, detail="Query exceeds maximum allowed length of 500 characters.")
        
    # Strip any control characters or shell patterns just to be safe
    clean_query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_query)
    
    log.info(f"Chat request (sanitized): {clean_query[:80]}")

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

        agent = GitOracleAgent(session_id=req.session_id, auto_approve=True)
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


# ── README endpoint ───────────────────────────────────────────────────────────

import base64

def _parse_readme(raw_md: str, owner: str, repo: str) -> dict:
    """Extract useful sections from raw README markdown."""
    import re as _re
    base_raw = f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD"

    def fix_url(url: str) -> str:
        url = url.strip()
        if url.startswith("http://") or url.startswith("https://"):
            if "github.com" in url and "/blob/" in url:
                url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            return url
        if url.startswith("./"):
            url = url[2:]
        return f"{base_raw}/{url}"

    # Images
    img_pattern = _re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    images = []
    for alt, src in img_pattern.findall(raw_md):
        src = fix_url(src)
        if any(x in src.lower() for x in ["shields.io", "badge", "travis", "circleci", "codecov", "gitter"]):
            continue
        images.append({"alt": alt or repo, "src": src})
        if len(images) >= 3:
            break

    # Tables
    table_pattern = _re.compile(r'(\|.+\|[\r\n]+\|[-| :]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)', _re.MULTILINE)
    tables_md = table_pattern.findall(raw_md)

    def md_table_to_html(md_tbl: str) -> str:
        lines = [l.strip() for l in md_tbl.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            return ""
        header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
        html = '<table class="readme-table"><thead><tr>'
        for h in header_cells:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"
        for row in lines[2:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            html += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
        html += "</tbody></table>"
        return html

    tables_html = [md_table_to_html(t) for t in tables_md[:2] if md_table_to_html(t)]

    # Intro paragraphs
    lines = raw_md.splitlines()
    intro_paras, current = [], []
    for line in lines:
        s = line.strip()
        if (s.startswith("#") or s.startswith("<") or s.startswith("```") or
                s.startswith("|") or s.startswith("![") or s.startswith("[![") or
                s.startswith("---") or not s):
            if current:
                para = " ".join(current).strip()
                if len(para) > 40:
                    intro_paras.append(para)
                current = []
        else:
            current.append(s)
        if len(intro_paras) >= 3:
            break
    if current:
        para = " ".join(current).strip()
        if len(para) > 40:
            intro_paras.append(para)

    intro = " ".join(intro_paras[:3])[:600]
    return {"intro": intro, "tables": tables_html, "images": images}


@app.get("/api/readme")
async def readme(owner: str = "", repo: str = ""):
    """Fetch and parse a GitHub repository README for rich content display."""
    if not owner or not repo:
        return {"intro": "", "tables": [], "images": []}
    async with httpx.AsyncClient(timeout=8) as client:
        try:
            r = await client.get(f"{GITHUB_API}/repos/{owner}/{repo}/readme", headers=_gh_headers())
        except httpx.RequestError:
            return {"intro": "", "tables": [], "images": []}
        if r.status_code != 200:
            return {"intro": "", "tables": [], "images": []}
        data = r.json()
        try:
            raw_md = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        except Exception:
            return {"intro": "", "tables": [], "images": []}
    return _parse_readme(raw_md, owner, repo)


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
