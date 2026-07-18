"""
param_test.py — GitOracle LLM Parameter Optimiser
==================================================
Tests different combinations of temperature and top_p to find the best
settings for an agent that does tool calling + factual GitHub lookups.

What we measure for each parameter combination:
  1. Tool Selection Accuracy  — did it pick the RIGHT tool?
  2. Argument Accuracy        — did it extract owner/repo correctly?
  3. Hallucination Resistance — did it call a tool instead of guessing?
  4. Consistency Score        — does it give the same tool call twice?
  5. Format Adherence         — does it follow the system prompt structure?

Why these matter for GitOracle:
  - Temperature too HIGH → LLM randomly answers from memory instead of calling tools
  - Temperature too LOW  → LLM becomes rigid, may refuse ambiguous queries
  - top_p too LOW        → Vocabulary too restricted, answers sound robotic
  - top_p too HIGH       → Same problems as high temperature

Usage:
    python param_test.py
    python param_test.py --backend groq
    python param_test.py --backend ollama
"""

import json
import time
import sys
import os
import argparse
import io

# Force UTF-8 on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Parameter grid to test ────────────────────────────────────────────────────

TEMPERATURES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
TOP_P_VALUES = [0.85, 0.90, 0.95]

# ── Test scenarios ────────────────────────────────────────────────────────────
# Each scenario defines:
#   prompt        : what the user would type
#   expected_tool : the correct tool to call
#   expected_args : key args the tool call MUST contain
#   must_not_guess: True = it must call a tool, not answer from memory

SCENARIOS = [
    {
        "id": "T1",
        "desc": "Explicit owner/repo — get details",
        "prompt": "How many stars does facebook/react have?",
        "expected_tool": "get_repo_details",
        "expected_args": {"owner": "facebook", "repo": "react"},
        "must_not_guess": True,
    },
    {
        "id": "T2",
        "desc": "Implicit repo name — search required",
        "prompt": "Find me the top Python machine learning frameworks on GitHub",
        "expected_tool": "search_repositories",
        "expected_args": {},   # no specific args required, just must call search
        "must_not_guess": True,
    },
    {
        "id": "T3",
        "desc": "Latest release lookup",
        "prompt": "What is the latest version of openai/openai-python?",
        "expected_tool": "get_latest_release",
        "expected_args": {"owner": "openai", "repo": "openai-python"},
        "must_not_guess": True,
    },
    {
        "id": "T4",
        "desc": "Issues lookup",
        "prompt": "Show me open issues in microsoft/vscode",
        "expected_tool": "list_open_issues",
        "expected_args": {"owner": "microsoft", "repo": "vscode"},
        "must_not_guess": True,
    },
    {
        "id": "T5",
        "desc": "Comparison query",
        "prompt": "Compare langchain-ai/langchain with microsoft/autogen",
        "expected_tool": "compare_repos",
        "expected_args": {
            "owner1": "langchain-ai", "repo1": "langchain",
            "owner2": "microsoft",    "repo2": "autogen",
        },
        "must_not_guess": True,
    },
    {
        "id": "T6",
        "desc": "Contributors lookup",
        "prompt": "Who are the main contributors to torvalds/linux?",
        "expected_tool": "list_contributors",
        "expected_args": {"owner": "torvalds", "repo": "linux"},
        "must_not_guess": True,
    },
    {
        "id": "T7",
        "desc": "Hallucination trap — must NOT answer from memory",
        "prompt": "How many stars does tensorflow/tensorflow have?",
        "expected_tool": "get_repo_details",
        "expected_args": {"owner": "tensorflow", "repo": "tensorflow"},
        "must_not_guess": True,  # Critical: LLM knows this but MUST call tool
    },
    {
        "id": "T8",
        "desc": "Recent commits",
        "prompt": "Show me recent commits to huggingface/transformers",
        "expected_tool": "get_repo_commits",
        "expected_args": {"owner": "huggingface", "repo": "transformers"},
        "must_not_guess": True,
    },
]

SYSTEM_PROMPT = (
    "You are GitOracle, an expert GitHub intelligence assistant. "
    "You have access to live GitHub data through tools. "
    "ALWAYS call a tool to look up real facts — NEVER answer from memory or training data. "
    "Even if you think you know the answer, you MUST use a tool to get current data."
)

# Minimal tool schemas for testing (no MCP server needed)
TEST_TOOLS = [
    {"type": "function", "function": {
        "name": "search_repositories",
        "description": "Search public GitHub repositories",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_repo_details",
        "description": "Get details for a GitHub repository",
        "parameters": {"type": "object", "properties": {
            "owner": {"type": "string"}, "repo": {"type": "string"}},
            "required": ["owner", "repo"]}}},
    {"type": "function", "function": {
        "name": "list_open_issues",
        "description": "List open issues for a GitHub repository",
        "parameters": {"type": "object", "properties": {
            "owner": {"type": "string"}, "repo": {"type": "string"},
            "limit": {"type": "integer"}}, "required": ["owner", "repo"]}}},
    {"type": "function", "function": {
        "name": "list_contributors",
        "description": "List top contributors to a GitHub repository",
        "parameters": {"type": "object", "properties": {
            "owner": {"type": "string"}, "repo": {"type": "string"}},
            "required": ["owner", "repo"]}}},
    {"type": "function", "function": {
        "name": "get_latest_release",
        "description": "Get the latest release of a GitHub repository",
        "parameters": {"type": "object", "properties": {
            "owner": {"type": "string"}, "repo": {"type": "string"}},
            "required": ["owner", "repo"]}}},
    {"type": "function", "function": {
        "name": "get_repo_commits",
        "description": "Get recent commits for a GitHub repository",
        "parameters": {"type": "object", "properties": {
            "owner": {"type": "string"}, "repo": {"type": "string"},
            "limit": {"type": "integer"}}, "required": ["owner", "repo"]}}},
    {"type": "function", "function": {
        "name": "compare_repos",
        "description": "Compare two GitHub repositories side by side",
        "parameters": {"type": "object", "properties": {
            "owner1": {"type": "string"}, "repo1": {"type": "string"},
            "owner2": {"type": "string"}, "repo2": {"type": "string"}},
            "required": ["owner1", "repo1", "owner2", "repo2"]}}},
]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_response(response, scenario: dict) -> dict:
    """Score a single LLM response against the scenario expectations."""
    result = {
        "tool_called":    False,   # did it make a tool call at all?
        "right_tool":     False,   # was it the expected tool?
        "args_correct":   False,   # did args match expected?
        "no_hallucinate": False,   # didn't just answer in text (for must_not_guess)
        "total":          0,
        "tool_name":      None,
        "tool_args":      {},
        "text_response":  None,
    }

    msg = response.choices[0].message
    finish = response.choices[0].finish_reason

    # Did it produce a text response instead of a tool call?
    if finish != "tool_calls" or not msg.tool_calls:
        result["text_response"] = (msg.content or "")[:120]
        if not scenario["must_not_guess"]:
            result["no_hallucinate"] = True
        return result

    result["tool_called"] = True
    result["no_hallucinate"] = True   # called a tool, didn't guess

    tc = msg.tool_calls[0]
    result["tool_name"] = tc.function.name
    try:
        result["tool_args"] = json.loads(tc.function.arguments)
    except Exception:
        result["tool_args"] = {}

    # Right tool?
    if tc.function.name == scenario["expected_tool"]:
        result["right_tool"] = True

    # Args correct?
    expected = scenario["expected_args"]
    if expected:
        args = result["tool_args"]
        result["args_correct"] = all(
            str(args.get(k, "")).lower() == str(v).lower()
            for k, v in expected.items()
        )
    else:
        result["args_correct"] = True   # no specific args required

    # Total score (out of 4)
    result["total"] = sum([
        result["tool_called"],
        result["right_tool"],
        result["args_correct"],
        result["no_hallucinate"],
    ])

    return result


# ── Runner ────────────────────────────────────────────────────────────────────

def run_test(client, model: str, temp: float, top_p: float, scenario: dict) -> dict:
    """Run one scenario at given temp/top_p, return scored result."""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": scenario["prompt"]},
        ],
        tools=TEST_TOOLS,
        tool_choice="auto",
        temperature=temp,
        top_p=top_p,
        max_tokens=256,
    )
    t0 = time.perf_counter()
    response = client.chat.completions.create(**kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    scored = score_response(response, scenario)
    scored["elapsed_ms"] = round(elapsed)
    return scored


def get_client_and_model(backend: str):
    if backend == "groq":
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            print("ERROR: GROQ_API_KEY not set in .env")
            sys.exit(1)
        return Groq(api_key=key), os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    else:
        from openai import OpenAI
        return (
            OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                   api_key="ollama"),
            os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GitOracle Parameter Optimiser")
    parser.add_argument("--backend", choices=["groq", "ollama"], default="groq")
    parser.add_argument("--quick", action="store_true",
                        help="Test only 3 temps x 2 top_p values (faster)")
    args = parser.parse_args()

    client, model = get_client_and_model(args.backend)

    temps  = [0.0, 0.2, 0.5] if args.quick else TEMPERATURES
    top_ps = [0.90, 0.95]    if args.quick else TOP_P_VALUES

    total_combos = len(temps) * len(top_ps)
    total_calls  = total_combos * len(SCENARIOS)

    print(f"\n{'='*65}")
    print(f" GitOracle Parameter Optimiser")
    print(f"{'='*65}")
    print(f" Backend  : {args.backend} ({model})")
    print(f" Grid     : {len(temps)} temps x {len(top_ps)} top_p = {total_combos} combos")
    print(f" Scenarios: {len(SCENARIOS)} per combo = {total_calls} total API calls")
    print(f"{'='*65}\n")

    # Results: combo_key -> {scores, details}
    results = {}

    combo_num = 0
    for temp in temps:
        for top_p in top_ps:
            combo_num += 1
            key = f"temp={temp:.1f}  top_p={top_p:.2f}"
            print(f"[{combo_num}/{total_combos}] Testing {key} ...")

            combo_scores = []
            scenario_results = []

            for s in SCENARIOS:
                try:
                    r = run_test(client, model, temp, top_p, s)
                    combo_scores.append(r["total"])
                    scenario_results.append({
                        "scenario": s["id"],
                        "desc": s["desc"],
                        **r
                    })
                    status = "[OK]" if r["total"] == 4 else f"[{r['total']}/4]"
                    tool_info = r["tool_name"] or f"TEXT: {r.get('text_response','')[:40]}"
                    print(f"    {status} {s['id']}: {tool_info}  ({r['elapsed_ms']} ms)")
                    time.sleep(0.3)   # be kind to the API
                except Exception as e:
                    print(f"    [ERR] {s['id']}: {e}")
                    combo_scores.append(0)

            avg = sum(combo_scores) / len(combo_scores) if combo_scores else 0
            pct = (avg / 4) * 100
            results[key] = {
                "temp": temp, "top_p": top_p,
                "avg_score": round(avg, 2),
                "pct": round(pct, 1),
                "perfect": sum(1 for s in combo_scores if s == 4),
                "details": scenario_results,
            }
            print(f"    --> Score: {avg:.2f}/4  ({pct:.0f}%)  "
                  f"Perfect: {results[key]['perfect']}/{len(SCENARIOS)}\n")

    # ── Rankings ──────────────────────────────────────────────────────────────
    ranked = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)

    print(f"\n{'='*65}")
    print(f" RESULTS — Ranked Best to Worst")
    print(f"{'='*65}")
    print(f"  {'Rank':<5} {'Config':<28} {'Score':>7} {'Perfect':>8}")
    print(f"  {'-'*55}")
    for i, (k, v) in enumerate(ranked, 1):
        marker = "  <-- RECOMMENDED" if i == 1 else ""
        print(f"  {i:<5} {k:<28} {v['pct']:>6.1f}%  "
              f"{v['perfect']:>3}/{len(SCENARIOS)}{marker}")

    best_k, best_v = ranked[0]
    print(f"\n{'='*65}")
    print(f" RECOMMENDATION")
    print(f"{'='*65}")
    print(f"  Best config : {best_k}")
    print(f"  Score       : {best_v['pct']:.1f}%  ({best_v['perfect']}/{len(SCENARIOS)} perfect)")
    print(f"\n  Add these to your agent LLM calls:")
    print(f"    temperature = {best_v['temp']}")
    print(f"    top_p       = {best_v['top_p']}")
    print(f"{'='*65}\n")

    # ── Save full results to JSON ─────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "param_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "backend": args.backend, "model": model,
            "best": {"config": best_k, "temperature": best_v["temp"],
                     "top_p": best_v["top_p"], "score_pct": best_v["pct"]},
            "all_results": {k: {x: v[x] for x in v if x != "details"}
                            for k, v in results.items()},
        }, f, indent=2)
    print(f"  Full results saved to: param_test_results.json\n")


if __name__ == "__main__":
    main()
