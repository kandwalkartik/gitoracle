"""
logger.py — GitOracle Structured Logger
========================================
Rich-based logger that replaces scattered print() calls throughout the project.

Provides:
  - Timestamped, colour-coded output for all log levels
  - Structured panels for tool calls, tool results, and agent responses
  - A single `get_logger()` factory so every module shares the same instance
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich import box
import datetime

# CRITICAL: Rich must write to stderr, NOT stdout.
# The MCP stdio transport uses stdout exclusively for JSON-RPC protocol messages.
# Any non-JSON bytes written to stdout corrupts the MCP session.
_console = Console(stderr=True, highlight=False)


class GitOracleLogger:
    """Thin wrapper that adds context-aware formatting on top of Rich."""

    def __init__(self, name: str = "GitOracle"):
        self.name = name
        self._console = _console

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    # ── Level-based logging ─────────────────────────────────────────────────

    def info(self, msg: str):
        self._console.print(f"[dim]{self._ts()}[/dim] [bold cyan][{self.name}][/bold cyan] {msg}")

    def debug(self, msg: str):
        self._console.print(f"[dim]{self._ts()}[/dim] [dim][{self.name}][/dim] [dim]{msg}[/dim]")

    def warning(self, msg: str):
        self._console.print(f"[dim]{self._ts()}[/dim] [bold yellow][{self.name} WARN][/bold yellow] {msg}")

    def error(self, msg: str):
        self._console.print(f"[dim]{self._ts()}[/dim] [bold red][{self.name} ERROR][/bold red] {msg}")

    def success(self, msg: str):
        self._console.print(f"[dim]{self._ts()}[/dim] [bold green][{self.name}][/bold green] ✓ {msg}")

    # ── Structured panels ───────────────────────────────────────────────────

    def tool_request(self, tool_name: str, args: dict):
        """Display a Human-in-the-Loop tool approval panel."""
        body = Text()
        body.append("Tool:  ", style="bold white")
        body.append(f"{tool_name}\n", style="bold yellow")
        body.append("Args:  ", style="bold white")
        body.append(str(args), style="cyan")
        self._console.print(
            Panel(body, title="[bold magenta]⚡ TOOL REQUEST — HUMAN APPROVAL REQUIRED[/bold magenta]",
                  border_style="magenta", box=box.ROUNDED, expand=False)
        )

    def tool_result(self, tool_name: str, result: str):
        """Display the raw tool result."""
        self._console.print(
            Panel(result, title=f"[bold green]✓ TOOL RESULT: {tool_name}[/bold green]",
                  border_style="green", box=box.ROUNDED)
        )

    def agent_response(self, agent_name: str, response: str):
        """Display the final LLM answer."""
        self._console.print(
            Panel(response, title=f"[bold blue]🤖 {agent_name}[/bold blue]",
                  border_style="blue", box=box.ROUNDED)
        )

    def router(self, category: str):
        """Log a routing decision."""
        self._console.print(
            f"[dim]{self._ts()}[/dim] [bold magenta][Router][/bold magenta] "
            f"Intent classified → [bold yellow]{category}[/bold yellow]"
        )

    def section(self, title: str):
        """Print a visual section divider."""
        self._console.print(Rule(f"[bold]{title}[/bold]", style="dim"))

    def cached(self, path: str):
        self._console.print(
            f"[dim]{self._ts()}[/dim] [dim cyan][Cache HIT][/dim cyan] {path}"
        )

    def banner(self, title: str, subtitle: str = ""):
        """Print a startup banner."""
        body = Text(justify="center")
        body.append(title, style="bold cyan")
        if subtitle:
            body.append(f"\n{subtitle}", style="dim white")
        self._console.print(Panel(body, border_style="cyan", box=box.DOUBLE_EDGE))


# Module-level singleton
_default_logger = GitOracleLogger()

def get_logger(name: str = "GitOracle") -> GitOracleLogger:
    """Return a named logger instance."""
    if name == "GitOracle":
        return _default_logger
    return GitOracleLogger(name)
