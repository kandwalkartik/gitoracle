"""
health_check.py — GitOracle Pre-flight Health Check
====================================================
Run this before anything else to confirm all required services are reachable.
Checks:
  1. Python version (>=3.10 required for union type hints)
  2. Required packages installed
  3. .env file exists and key variables are set
  4. GitHub API reachable (with and without token)
  5. LLM backend reachable (Ollama or Groq)
  6. MCP server can launch and complete a handshake

Usage:
    python health_check.py
"""

import os
import sys
import time
# Force UTF-8 on Windows terminals to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import asyncio
import importlib
import importlib.util

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = []

def check(label: str, passed: bool, detail: str = "", is_warning: bool = False):
    if passed:
        tag = PASS
        color = "\033[92m"  # green
    elif is_warning:
        tag = WARN
        color = "\033[93m"  # yellow
    else:
        tag = FAIL
        color = "\033[91m"  # red
    reset = "\033[0m"
    suffix = f"  — {detail}" if detail else ""
    print(f"  {color}{tag}{reset} {label}{suffix}")
    results.append((label, passed or is_warning))


# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n[1] Python Version")
major, minor = sys.version_info[:2]
check(
    f"Python {major}.{minor}",
    major >= 3 and minor >= 10,
    detail=f"(need >=3.10, found {major}.{minor})" if minor < 10 else "",
)

# ── 2. Required packages ──────────────────────────────────────────────────────
print("\n[2] Required Packages")
REQUIRED = ["mcp", "httpx", "langgraph", "openai", "dotenv", "rich", "pydantic"]
OPTIONAL = ["groq"]
for pkg in REQUIRED:
    spec = importlib.util.find_spec(pkg)
    check(pkg, spec is not None, detail="pip install " + pkg if spec is None else "")
for pkg in OPTIONAL:
    spec = importlib.util.find_spec(pkg)
    check(pkg + " (optional, Groq backend)", spec is not None,
          detail="pip install groq  - only needed if LLM_BACKEND=groq", is_warning=spec is None)

# ── 3. .env file & variables ──────────────────────────────────────────────────
print("\n[3] Environment Variables")
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
check(".env file exists", os.path.exists(env_path),
      detail="Copy .env.example → .env and fill in your values" if not os.path.exists(env_path) else "")

github_token = os.environ.get("GITHUB_TOKEN", "")
check("GITHUB_TOKEN", bool(github_token),
      detail="Not set — limited to 60 req/hr. Set token for 5,000/hr.", is_warning=not github_token)

llm_backend = os.environ.get("LLM_BACKEND", "ollama").lower()
check(f"LLM_BACKEND={llm_backend}", llm_backend in ("ollama", "groq"),
      detail=f"Unknown backend '{llm_backend}' — must be 'ollama' or 'groq'")

if llm_backend == "groq":
    groq_key = os.environ.get("GROQ_API_KEY", "")
    check("GROQ_API_KEY", bool(groq_key), detail="Required when LLM_BACKEND=groq")

# ── 4. GitHub API reachability ────────────────────────────────────────────────
print("\n[4] GitHub API")
try:
    import httpx
    headers = {"User-Agent": "GitOracle-HealthCheck"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    t0 = time.perf_counter()
    r = httpx.get("https://api.github.com/rate_limit", headers=headers, timeout=8)
    ms = (time.perf_counter() - t0) * 1000
    if r.status_code == 200:
        data = r.json()
        core = data["resources"]["core"]
        remaining = core["remaining"]
        limit = core["limit"]
        check(f"GitHub API reachable ({ms:.0f} ms)", True,
              detail=f"Rate limit: {remaining}/{limit} remaining")
    else:
        check(f"GitHub API", False, detail=f"HTTP {r.status_code}")
except Exception as e:
    check("GitHub API reachable", False, detail=str(e))

# ── 5. LLM backend ────────────────────────────────────────────────────────────
print(f"\n[5] LLM Backend ({llm_backend})")
try:
    if llm_backend == "groq":
        from groq import Groq
        c = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    else:
        from openai import OpenAI
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        c = OpenAI(base_url=base_url, api_key="ollama")
        model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

    t0 = time.perf_counter()
    c.chat.completions.create(
        model=model, messages=[{"role": "user", "content": "Say OK"}], max_tokens=5
    )
    ms = (time.perf_counter() - t0) * 1000
    check(f"{llm_backend} ({model}) reachable ({ms:.0f} ms)", True)
except Exception as e:
    hint = ""
    if llm_backend == "ollama":
        hint = "Start Ollama: `ollama serve` then `ollama pull " + model + "`"
    check(f"{llm_backend} ({model}) reachable", False, detail=hint or str(e))

# ── 6. MCP server handshake ───────────────────────────────────────────────────
print("\n[6] MCP Server Handshake")

async def _test_mcp():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    if not os.path.exists(server_path):
        return False, "mcp_server.py not found"

    server_params = StdioServerParameters(
        command=sys.executable, args=[server_path], env=os.environ.copy()
    )
    try:
        t0 = time.perf_counter()
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                ms = (time.perf_counter() - t0) * 1000
                return True, f"{len(tools.tools)} tools discovered in {ms:.0f} ms"
    except Exception as e:
        return False, str(e)

mcp_ok, mcp_detail = asyncio.run(_test_mcp())
check("MCP server launches and handshakes", mcp_ok, detail=mcp_detail)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
passed = sum(1 for _, ok in results if ok)
failed = len(results) - passed
print(f"  Health check: {passed}/{len(results)} checks passed", end="")
if failed:
    print(f"  ({failed} failed/warned — review items above)")
else:
    print("  — all systems go! ✓")
print(f"{'='*55}\n")
