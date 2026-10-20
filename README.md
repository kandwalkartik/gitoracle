# GitOracle — GitHub Intelligence Oracle
### IBM AI Developer Capstone Project

GitOracle answers questions about public GitHub repositories in natural language. It combines a custom MCP (Model Context Protocol) server, a ReAct agent with human approval for selected actions, and a LangGraph router with four specialist agents.

**Problem:** Finding repository details across README files, issues, releases, and contributor activity takes several separate GitHub searches. GitOracle brings those sources into one question-and-answer workflow.

---

## Architecture

```
User Input
    │
    ├─► api_server.py (FastAPI Web App + CORS + Auto-Search + Reset Context)
    │
    ├─► agent_core.py (Single ReAct Agent + HITL + Zonal Anchoring)
    │       │
    │       ├─► model_client.py (LLMLoadBalancer) ──► Groq API Client Pool (O(1) Failover)
    │       │
    │       └─► mcp_bridge.py ──► mcp_server.py ──► GitHub REST API
    │                                   │
    │                               cache.py (TTL)
    │
    └─► langgraph_agent.py (Multi-Agent Router)
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
| `model_client.py` | LLMLoadBalancer with O(1) scheduling, monotonic cooldown failover, and metrics |
| `api_server.py` | FastAPI backend with autocomplete suggestion API and diagnostic key metrics |
| `a2a_card.py` | A2A agent card for external discoverability |
| `param_test.py` | LLM parameter optimiser (temperature + top_p grid search) |

---

## Quick Start

### 1. Install dependencies
```bash
python -m venv .venv
# Activate .venv for your shell, then:
pip install -r requirements.txt
```

### 2. Configure environment
Copy `.env.example` to `.env`, then fill in your values:
```
GITHUB_TOKEN=ghp_your_token_here      # optional but recommended (5,000 req/hr)
LLM_BACKEND=groq                      # "groq" (cloud) or "ollama" (local)
GROQ_API_KEY=gsk_key1,gsk_key2,...    # Comma-separated list of Groq keys for O(1) failover load balancing
```
Get free Groq keys at: **console.groq.com** → API Keys (you can add multiple keys separated by commas for failover support).

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

### Web demo

Run `python api_server.py` and open `http://localhost:8000` to use the local web interface. This demo still needs a configured LLM backend for agent answers.

## MCP Tools (10 total)

| Tool | Description |
|---|---|
| `search_repositories` | Search GitHub repos by keyword, topic, or language |
| `get_repo_details` | Stars, forks, language, license, topics, homepage |
| `get_repo_readme` | README content for a public repository |
| `list_open_issues` | Currently open issues with labels and assignees |
| `list_contributors` | Top contributors by commit count |
| `get_latest_release` | Latest release tag, date, and notes |
| `get_repo_commits` | Recent commit history with author and message |
| `compare_repos` | Side-by-side metric comparison of two repositories |
| `get_user_profile` | Public GitHub user details |
| `get_org_details` | Public GitHub organization details |

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
├── mcp_server.py           ← Phase 1: MCP server (10 tools, 2 prompts)
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

## Verification and limitations

- `python cli.py health` checks service connectivity. `python cli.py test` exercises the MCP protocol and nine tool calls; it does not test every agent response or the web interface.
- Answers depend on the availability and rate limits of the GitHub API and the configured LLM backend. Review generated answers against the linked repository before relying on them.
- GitHub responses are cached in memory for 60 seconds by default, so recent repository changes can take a short time to appear.
- The project does not include a committed screenshot gallery or a hosted demo. Use the local web interface above to inspect it.
