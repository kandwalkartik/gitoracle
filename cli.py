"""
cli.py — GitOracle Unified Entry Point
======================================
Single command to launch any part of the GitOracle project.

Usage:
    python cli.py                    # shows interactive menu
    python cli.py health             # run health check
    python cli.py test               # run MCP client test (all 7 tools)
    python cli.py agent              # run single ReAct agent (HITL)
    python cli.py multi              # run multi-agent LangGraph version
    python cli.py a2a                # run A2A card demo + export
    python cli.py session <name>     # start agent with a named session
"""

import sys
import os
import asyncio

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_menu():
    print("""
╔════════════════════════════════════════════════════╗
║        GitOracle — GitHub Intelligence Oracle      ║
║                 IBM Capstone v2.0                 ║
╠════════════════════════════════════════════════════╣
║  h  / health   →  Pre-flight health check            ║
║  t  / test     →  MCP protocol test (all 7 tools)    ║
║  a  / agent    →  Single ReAct agent + HITL          ║
║  m  / multi    →  Multi-agent LangGraph (4 nodes)    ║
║  c  / a2a      →  A2A agent card demo + JSON export  ║
║  q  / quit     →  Exit                               ║
╚════════════════════════════════════════════════════╝
""")


def run_health():
    import runpy
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "health_check.py"),
        run_name="__main__",
    )


def run_test():
    from mcp_client_test import main as mcp_test_main
    asyncio.run(mcp_test_main())


def run_agent(session_id: str | None = None):
    from agent_core import main as agent_main, GitOracleAgent
    if session_id:
        import agent_core
        _original = agent_core.GitOracleAgent.__init__

        def _patched_init(self, sid=None):
            _original(self, session_id)

        agent_core.GitOracleAgent.__init__ = _patched_init
    asyncio.run(agent_main())


def run_multi():
    from langgraph_agent import main as lg_main
    lg_main()


def run_a2a():
    from a2a_card import demo
    demo()


def dispatch(command: str, args: list):
    command = command.lower()
    if command in ("h", "health"):
        run_health()
    elif command in ("t", "test"):
        run_test()
    elif command in ("a", "agent"):
        session_id = args[0] if args else None
        run_agent(session_id)
    elif command in ("m", "multi"):
        run_multi()
    elif command in ("c", "a2a"):
        run_a2a()
    elif command in ("q", "quit", "exit"):
        print("Goodbye!")
        sys.exit(0)
    else:
        print(f"Unknown command: '{command}'")
        print_menu()


def main():
    # Command-line mode
    if len(sys.argv) > 1:
        dispatch(sys.argv[1], sys.argv[2:])
        return

    # Interactive menu mode
    print_menu()
    while True:
        try:
            choice = input("Select option: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not choice:
            continue

        parts = choice.split()
        dispatch(parts[0], parts[1:])


if __name__ == "__main__":
    main()
