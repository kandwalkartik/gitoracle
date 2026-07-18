# GitOracle — GitHub Intelligence Oracle
### IBM AI Developer Capstone Project

GitOracle is a production-quality agentic AI system that answers questions about any public GitHub repository using natural language. Powered by a real MCP (Model Context Protocol) server, a ReAct agent with Human-in-the-Loop approval, and a LangGraph multi-agent router with four named specialists.

---

## Architecture

```
User Input
    │
    ├─► agent_core.py  (Single ReAct Agent + HITL)
    │       │
    │       └─► mcp_bridge.py ──► mcp_server.py ──► GitHub REST API
    │                                   │
    │                               cache.py (TTL)
    │
    └─► langgraph_agent.py  (Multi-Agent Router)
            │
            ├─► Chronicle   (Repo Analyst)
            ├─► Sentinel    (Issue Triage)
            ├─► Herald      (Release Notes)
            └─► Pathfinder  (Trending Scout)
```

**Supporting Infrastructure:**

| File | Purpose |
|---|---|
| `config.py` | Centralised env var loader with interactive fallback prompting |
| `logger.py` | Rich-based structured logger (stderr-only, MCP-safe) |
| `cache.py` | In-memory TTL cache for GitHub API responses |
| `conversation_store.py` | JSON session persistence across restarts |
| `health_check.py` | Pre-flight connectivity check |
| `cli.py` | Unified project entry point |
| `a2a_card.py` | A2A agent card for external discoverability |
| `param_test.py` | LLM parameter optimiser (temperature + top_p grid search) |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Open `.env` and fill in your values:
```
GITHUB_TOKEN=ghp_your_token_here      # optional but recommended (5,000 req/hr)
LLM_BACKEND=groq                      # "groq" (cloud) or "ollama" (local)
GROQ_API_KEY=gsk_your_key_here        # required if LLM_BACKEND=groq
```
Get a free Groq key at: **console.groq.com** → API Keys

Get a free GitHub token at: **github.com** → Settings → Developer settings → Personal access tokens → Tokens (classic) → tick `public_repo`

### 3. Run the health check
```bash
python cli.py health
```

### 4. Test the MCP server (no LLM required)
```bash
python cli.py test
```

### 5. Run GitOracle
```bash
python cli.py          # interactive menu
python cli.py agent    # Single ReAct Agent with Human-in-the-Loop
python cli.py multi    # Multi-Agent LangGraph (4 specialists)
```

---

## MCP Tools (7 total)

| Tool | Description |
|---|---|
| `search_repositories` | Search GitHub repos by keyword, topic, or language |
| `get_repo_details` | Stars, forks, language, license, topics, homepage |
| `list_open_issues` | Currently open issues with labels and assignees |
| `list_contributors` | Top contributors by commit count |
| `get_latest_release` | Latest release tag, date, and notes |
| `get_repo_commits` | Recent commit history with author and message |
| `compare_repos` | Side-by-side metric comparison of two repositories |

---

## LangGraph Specialists (4 nodes)

| Specialist | Role | Tools Available |
|---|---|---|
| **Chronicle** | Repository Analyst | search, details, contributors, commits |
| **Sentinel** | Issue Triage | list_open_issues |
| **Herald** | Release Notes | get_latest_release |
| **Pathfinder** | Trending Scout | search_repositories |

---

## Project File Structure

```
IBM cert/
├── cli.py                  ← Start here
├── health_check.py         ← Run first
├── config.py               ← Env var management + interactive prompting
├── logger.py               ← Structured logging (stderr-safe)
├── cache.py                ← GitHub API TTL cache
├── conversation_store.py   ← Session persistence
│
├── mcp_server.py           ← Phase 1: MCP server (7 tools, 2 prompts)
├── mcp_client_test.py      ← Phase 1: Protocol test
│
├── model_client.py         ← Phase 2: LLM connector (Ollama/Groq)
├── mcp_bridge.py           ← Phase 2: MCP bridge
│
├── agent_core.py           ← Phase 3: Single ReAct agent + HITL
├── langgraph_agent.py      ← Phase 4: Multi-agent LangGraph
├── a2a_card.py             ← Phase 5: A2A agent card
├── param_test.py           ← Phase 6: LLM parameter optimiser
│
├── agent_card.json         ← Generated: exported agent card
├── sessions/               ← Auto-created: conversation history
├── param_test_results.json ← Generated: parameter test results
├── requirements.txt
├── .env.example
└── .env                    ← Your local config (never commit this)
```

---

## Key Design Decisions

- **MCP-safe logger**: Rich logger outputs to `stderr` exclusively. The MCP stdio transport owns `stdout` for JSON-RPC — mixing them corrupts the protocol.
- **Option C config**: `.env` file used silently if present; falls back to interactive terminal prompting for missing keys and offers to save them.
- **TTL caching**: GitHub API responses cached 60s. Repeated questions about the same repo cost zero rate-limit quota.
- **History trimming**: Capped at 20 turns to prevent context window overflow on long sessions.
- **Backend agnostic**: Swap Ollama ↔ Groq by changing one line in `.env`.
- **Safe serialisation**: Tool call objects use a multi-fallback helper supporting both Pydantic v1 and v2.

---

## Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: groq` | `pip install -r requirements.txt` |
| GitHub returns 403 | Add `GITHUB_TOKEN` to `.env` |
| Ollama connection refused | Run `ollama serve` first |
| Groq 401 Unauthorized | Check `GROQ_API_KEY` in `.env` |
| History lost on restart | Saved automatically to `sessions/` |
