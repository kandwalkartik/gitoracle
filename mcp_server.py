"""
mcp_server.py — GitOracle MCP Server
=====================================
Exposes 9 live GitHub tools, 1 resource template, and 2 reusable prompts
via the FastMCP stdio transport.

Tools       : search_repositories, get_repo_details, list_open_issues,
              list_contributors, get_latest_release,
              get_repo_commits, compare_repos,          (v2)
              get_user_profile, get_org_details         (v3 — new)
Resource    : github://repo/{owner}/{repo}/summary
Prompts     : issue_triage_prompt, repo_analysis_prompt

Improvements over v1:
  - load_dotenv() called at startup (was silently ignored before)
  - GitHub token deferred to request time (not frozen at import)
  - Response caching via cache.py (saves rate-limit quota)
  - Two new tools: get_repo_commits, compare_repos
  - One new prompt: repo_analysis_prompt
  - Structured logger replaces raw print() calls
"""

import os
import sys
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load .env before anything reads env vars
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from cache import cached_get as _cached_get
from logger import get_logger

log = get_logger("MCPServer")

mcp = FastMCP("GitOracle — GitHub Intelligence Oracle")

GITHUB_API = os.environ.get("GITHUB_API", "https://api.github.com")

# ── HTTP helper (token read at call-time, not import-time) ──────────────────

def _headers() -> dict:
    """Build request headers fresh each call so token rotation works."""
    h = {"User-Agent": "GitOracle-MCP-Server/2.0"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(path: str, params: dict | None = None) -> tuple[int, dict]:
    """Real HTTP GET against the GitHub REST API."""
    try:
        r = httpx.get(
            f"{GITHUB_API}{path}", headers=_headers(), params=params,
            timeout=10, follow_redirects=True,
        )
        return r.status_code, (r.json() if r.content else {})
    except httpx.RequestError as e:
        return -1, {"message": f"Network error: {e}"}


def _cached(path: str, params: dict | None = None) -> tuple[int, dict]:
    """Cached wrapper around _get."""
    return _cached_get(path, params=params, fetch_fn=_get)


def _err(status: int, data: dict, context: str) -> str:
    """Convert a GitHub error response into a friendly, actionable message."""
    msg = str(data.get("message", ""))
    if status == 403 and "rate limit" in msg.lower():
        return (
            f"ERROR: GitHub API rate limit exceeded for {context}. "
            "Unauthenticated requests are capped at 60/hour. "
            "Set GITHUB_TOKEN in your .env file to raise this to 5,000/hour."
        )
    if status == 404:
        return f"ERROR: {context} was not found on GitHub (404). Check the owner/repo spelling."
    if status == 401:
        return f"ERROR: GitHub authentication failed for {context}. Check your GITHUB_TOKEN."
    if status == -1:
        return f"ERROR: Could not reach GitHub. {msg}"
    return f"ERROR: GitHub returned status {status} for {context}: {msg or 'unknown error'}"


# ── TOOLS ────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_repositories(query: str) -> str:
    """Search public GitHub repositories by keyword, topic, or language.

    Args:
        query: Search string, e.g. 'machine learning language:python stars:>1000'
    """
    status, data = _cached("/search/repositories", {"q": query, "sort": "stars", "per_page": 5})
    if status != 200:
        return _err(status, data, f"search '{query}'")

    items = data.get("items", [])
    if not items:
        return f"No repositories found for: {query}"

    lines = [f"Top results for '{query}':"]
    for r in items:
        desc = (r.get("description") or "No description")[:80]
        lines.append(f"- {r['full_name']}  ★{r['stargazers_count']:,}  —  {desc}")
    return "\n".join(lines)


@mcp.tool()
def get_repo_details(owner: str, repo: str) -> str:
    """Get live details for a GitHub repository: stars, forks, language, license.

    Args:
        owner: Repository owner, e.g. 'facebook'
        repo:  Repository name, e.g. 'react'
    """
    status, data = _cached(f"/repos/{owner}/{repo}")
    if status != 200:
        return _err(status, data, f"{owner}/{repo}")

    license_name = (data.get("license") or {}).get("name", "No license")
    topics = ", ".join(data.get("topics", [])) or "none"
    return (
        f"{data['full_name']}: {data.get('description', 'No description')}\n"
        f"★ {data['stargazers_count']:,} stars | {data['forks_count']:,} forks | "
        f"Language: {data.get('language', 'N/A')} | License: {license_name}\n"
        f"Open issues: {data['open_issues_count']} | "
        f"Default branch: {data['default_branch']} | Topics: {topics}\n"
        f"Homepage: {data.get('homepage') or 'N/A'}"
    )


@mcp.tool()
def get_repo_readme(owner: str, repo: str) -> str:
    """Get the README.md content of a GitHub repository to read detailed project summaries and instructions.

    Args:
        owner: Repository owner, e.g. 'facebook'
        repo:  Repository name, e.g. 'react'
    """
    status, data = _cached(f"/repos/{owner}/{repo}/readme")
    if status != 200:
        return f"README file not found or couldn't be loaded for {owner}/{repo}."

    download_url = data.get("download_url")
    if not download_url:
        return "README download url not available."

    try:
        r = httpx.get(download_url, timeout=10, follow_redirects=True)
        if r.status_code == 200:
            content = r.text
            # Limit characters to avoid model context overflow
            if len(content) > 6000:
                return content[:6000] + "\n\n... (README content truncated to 6,000 characters) ..."
            return content
        return f"Failed to download README content (HTTP {r.status_code})."
    except Exception as e:
        return f"Error downloading README: {e}"


@mcp.tool()
def list_open_issues(owner: str, repo: str, limit: int = 5) -> str:
    """List currently open issues for a GitHub repository.

    Args:
        owner: Repository owner
        repo:  Repository name
        limit: Max number of issues to return (default 5, max 30)
    """
    limit = min(max(1, int(limit)), 30)
    status, data = _cached(f"/repos/{owner}/{repo}/issues", {"state": "open", "per_page": limit})
    if status != 200:
        return _err(status, data, f"{owner}/{repo} issues")

    if not data:
        return f"{owner}/{repo} has no open issues — great news!"

    lines = [f"Open issues for {owner}/{repo} (showing {len(data)}):"]
    for issue in data:
        labels = ", ".join(l["name"] for l in issue.get("labels", [])) or "no labels"
        assignee = (issue.get("assignee") or {}).get("login", "unassigned")
        lines.append(f"- #{issue['number']}: {issue['title']}  [{labels}]  @{assignee}")
    return "\n".join(lines)


@mcp.tool()
def list_contributors(owner: str, repo: str, limit: int = 5) -> str:
    """List the top contributors to a GitHub repository by commit count.

    Args:
        owner: Repository owner
        repo:  Repository name
        limit: Max contributors to return (default 5)
    """
    status, data = _cached(f"/repos/{owner}/{repo}/contributors", {"per_page": int(limit)})
    if status != 200:
        return _err(status, data, f"{owner}/{repo} contributors")

    if not data:
        return f"No contributor data available for {owner}/{repo}."

    lines = [f"Top {len(data)} contributors to {owner}/{repo}:"]
    for c in data:
        lines.append(f"- {c['login']}: {c['contributions']:,} commits")
    return "\n".join(lines)


@mcp.tool()
def get_latest_release(owner: str, repo: str) -> str:
    """Get the latest published release for a GitHub repository.

    Args:
        owner: Repository owner
        repo:  Repository name
    """
    status, data = _cached(f"/repos/{owner}/{repo}/releases/latest")
    if status != 200:
        return _err(status, data, f"{owner}/{repo} latest release")

    body = (data.get("body") or "No release notes provided.")[:400]
    return (
        f"Latest release of {owner}/{repo}: {data['tag_name']}  ({data.get('name', '')})\n"
        f"Published: {data['published_at']}\n"
        f"Pre-release: {data.get('prerelease', False)}\n"
        f"Notes:\n{body}"
    )


@mcp.tool()
def get_repo_commits(owner: str, repo: str, limit: int = 5) -> str:
    """Get the most recent commits for a GitHub repository.

    Args:
        owner: Repository owner
        repo:  Repository name
        limit: Max commits to return (default 5, max 20)
    """
    limit = min(max(1, limit), 20)
    status, data = _cached(f"/repos/{owner}/{repo}/commits", {"per_page": limit})
    if status != 200:
        return _err(status, data, f"{owner}/{repo} commits")

    if not data:
        return f"No commits found for {owner}/{repo}."

    lines = [f"Recent commits to {owner}/{repo}:"]
    for c in data:
        sha = c["sha"][:7]
        msg = c["commit"]["message"].split("\n")[0][:80]
        author = c["commit"]["author"]["name"]
        date = c["commit"]["author"]["date"][:10]
        lines.append(f"- [{sha}] {date}  {author}: {msg}")
    return "\n".join(lines)


@mcp.tool()
def compare_repos(owner1: str, repo1: str, owner2: str, repo2: str) -> str:
    """Compare two GitHub repositories side-by-side on key metrics.

    Args:
        owner1: Owner of the first repository
        repo1:  Name of the first repository
        owner2: Owner of the second repository
        repo2:  Name of the second repository
    """
    s1, d1 = _cached(f"/repos/{owner1}/{repo1}")
    s2, d2 = _cached(f"/repos/{owner2}/{repo2}")

    if s1 != 200:
        return _err(s1, d1, f"{owner1}/{repo1}")
    if s2 != 200:
        return _err(s2, d2, f"{owner2}/{repo2}")

    def _row(label, v1, v2):
        winner = " ←" if str(v1) >= str(v2) else ""
        winner2 = " ←" if str(v2) > str(v1) else ""
        return f"  {label:<18} {str(v1):<20}{winner:<3}  {str(v2):<20}{winner2}"

    lines = [
        f"Repository Comparison",
        f"  {'Metric':<18} {d1['full_name']:<23}  {d2['full_name']}",
        "  " + "-" * 70,
        _row("Stars",     f"{d1['stargazers_count']:,}", f"{d2['stargazers_count']:,}"),
        _row("Forks",     f"{d1['forks_count']:,}",      f"{d2['forks_count']:,}"),
        _row("Open Issues", d1['open_issues_count'],     d2['open_issues_count']),
        _row("Language",  d1.get('language','N/A'),      d2.get('language','N/A')),
        _row("License",   (d1.get('license') or {}).get('name','None'), (d2.get('license') or {}).get('name','None')),
        _row("Last Push", d1['pushed_at'][:10],          d2['pushed_at'][:10]),
    ]
    return "\n".join(lines)


# ── NEW v3 TOOLS ─────────────────────────────────────────────────────────────

@mcp.tool()
def get_user_profile(username: str) -> str:
    """Get the public GitHub profile of a user or bot account.

    Args:
        username: GitHub username (e.g. 'torvalds', 'gvanrossum')

    Requires: read:user scope on the GitHub token (public profiles only).
    """
    status, data = _cached(f"/users/{username}")
    if status == 404:
        return f"User '{username}' not found on GitHub."
    if status != 200:
        return _err(status, data, f"user/{username}")

    name        = data.get("name") or username
    bio         = data.get("bio")  or "No bio provided."
    company     = data.get("company") or "N/A"
    location    = data.get("location") or "N/A"
    blog        = data.get("blog") or "N/A"
    pub_repos   = data.get("public_repos", 0)
    followers   = data.get("followers", 0)
    following   = data.get("following", 0)
    joined      = (data.get("created_at") or "")[:10]
    account_type = data.get("type", "User")

    return (
        f"GitHub {account_type}: {name} (@{username})\n"
        f"Bio      : {bio}\n"
        f"Company  : {company}\n"
        f"Location : {location}\n"
        f"Website  : {blog}\n"
        f"Joined   : {joined}\n"
        f"Public Repos: {pub_repos:,} | Followers: {followers:,} | Following: {following:,}\n"
        f"Profile  : https://github.com/{username}"
    )


@mcp.tool()
def get_org_details(org: str) -> str:
    """Get the public details of a GitHub organisation.

    Args:
        org: GitHub organisation login (e.g. 'microsoft', 'huggingface')

    Requires: read:org scope on the GitHub token (public orgs only).
    """
    status, data = _cached(f"/orgs/{org}")
    if status == 404:
        return f"Organisation '{org}' not found on GitHub."
    if status != 200:
        return _err(status, data, f"org/{org}")

    name        = data.get("name") or org
    description = data.get("description") or "No description."
    blog        = data.get("blog") or "N/A"
    location    = data.get("location") or "N/A"
    email       = data.get("email") or "N/A"
    pub_repos   = data.get("public_repos", 0)
    followers   = data.get("followers", 0)
    created     = (data.get("created_at") or "")[:10]
    verified    = data.get("is_verified", False)

    return (
        f"GitHub Organisation: {name} (@{org}){' [Verified]' if verified else ''}\n"
        f"Description : {description}\n"
        f"Location    : {location}\n"
        f"Website     : {blog}\n"
        f"Email       : {email}\n"
        f"Founded     : {created}\n"
        f"Public Repos: {pub_repos:,} | Followers: {followers:,}\n"
        f"Profile     : https://github.com/{org}"
    )


# ── RESOURCE ─────────────────────────────────────────────────────────────────

@mcp.resource("github://repo/{owner}/{repo}/summary")
def repo_summary_resource(owner: str, repo: str) -> str:
    """Read-only snapshot of a repository's key stats, addressed by URI."""
    status, data = _cached(f"/repos/{owner}/{repo}")
    if status != 200:
        return _err(status, data, f"{owner}/{repo}")

    return (
        f"REPO SUMMARY: {data['full_name']}\n"
        f"Stars: {data['stargazers_count']:,} | Forks: {data['forks_count']:,} | "
        f"Open Issues: {data['open_issues_count']}\n"
        f"Language: {data.get('language', 'N/A')} | "
        f"Created: {data['created_at'][:10]} | Last push: {data['pushed_at'][:10]}"
    )


# ── PROMPTS ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def issue_triage_prompt() -> str:
    """Structured workflow for triaging a GitHub issue."""
    return (
        "Review this GitHub issue and output exactly 4 lines:\n"
        "1. Type: Bug | Feature Request | Question | Documentation\n"
        "2. Priority: Critical | High | Medium | Low  (justify in ≤5 words)\n"
        "3. Suggested labels: (comma-separated, use existing GH label names)\n"
        "4. Dashboard summary: one sentence, ≤15 words\n"
        "Be concise. Do not add extra commentary."
    )


@mcp.prompt()
def repo_analysis_prompt() -> str:
    """Structured workflow for producing a health report on a repository."""
    return (
        "Using the available tools, produce a structured repository health report with:\n"
        "1. Overview: name, description, primary language, license\n"
        "2. Activity: recent commits, last release, open issue count\n"
        "3. Community: top 3 contributors and their commit counts\n"
        "4. Assessment: 2-3 sentences on the repo's apparent health and activity level\n"
        "Call tools as needed. Keep each section brief."
    )


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token_status = "authenticated" if os.environ.get("GITHUB_TOKEN") else "unauthenticated (60 req/hr limit)"
    print(f"GitOracle MCP Server v3.0  |  GitHub: {token_status}", file=sys.stderr)
    print("Tools (9): search_repositories, get_repo_details, list_open_issues,", file=sys.stderr)
    print("           list_contributors, get_latest_release, get_repo_commits,", file=sys.stderr)
    print("           compare_repos, get_user_profile, get_org_details", file=sys.stderr)
    print("Prompts: issue_triage_prompt, repo_analysis_prompt", file=sys.stderr)
    print("Ready — waiting for MCP client connections...\n", file=sys.stderr)
    mcp.run()
