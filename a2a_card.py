"""
a2a_card.py — GitOracle Agent-to-Agent (A2A) Card
=================================================
Publishes GitOracle as a discoverable, versioned agent card so external
orchestrators or peer agents can find it, understand its capabilities,
and route requests to it — without ever seeing the MCP server, LangGraph
graph, or GitHub API underneath.

Improvements over v1:
  - AgentCard now includes author, contact, created_at, and input_modes fields
  - Skill now includes an examples field (concrete usage hints for peer agents)
  - AgentRegistry gains find_by_skill_id() and export_all_json() methods
  - Updated card includes the new 'trending' specialist skill
  - Card is exported to agent_card.json for external consumption
"""

import json
import os
import datetime
from typing import List, Optional
from pydantic import BaseModel


# ── Schema ────────────────────────────────────────────────────────────────────

class Skill(BaseModel):
    id: str
    description: str
    tags: List[str]
    examples: List[str] = []           # concrete example queries for this skill


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    endpoint: str
    author: str = ""
    contact: str = ""
    created_at: str = ""
    input_modes: List[str] = ["text"]  # modalities the agent accepts
    skills: List[Skill]


# ── GitOracle self-description ─────────────────────────────────────────────────

GITORACLE_CARD = AgentCard(
    name="GitOracle",
    description=(
        "Agentic GitHub intelligence assistant. "
        "Answers questions about repositories, issues, releases, contributors, "
        "commits, and trending projects using a live MCP server backed by the "
        "real GitHub REST API."
    ),
    version="2.0.0",
    endpoint="http://localhost:9001/gitoracle",
    author="GitOracle",
    contact="gitoracle@example.com",
    created_at=datetime.date.today().isoformat(),
    input_modes=["text"],
    skills=[
        Skill(
            id="repo_info",
            description=(
                "Look up live stats, search results, contributors, and recent "
                "commits for any public GitHub repository."
            ),
            tags=["github", "repository", "stars", "forks", "contributors",
                  "commits", "search", "compare"],
            examples=[
                "How many stars does facebook/react have?",
                "Who are the top contributors to langchain-ai/langchain?",
                "Compare pytorch/pytorch with tensorflow/tensorflow",
                "Show me the last 5 commits to openai/openai-python",
            ],
        ),
        Skill(
            id="issue_triage",
            description="List and summarise open issues for a GitHub repository.",
            tags=["github", "issues", "triage", "bugs", "open issues"],
            examples=[
                "What are the open issues in langchain-ai/langchain?",
                "Show me the top 5 open bugs in microsoft/vscode",
                "Are there any critical issues open in huggingface/transformers?",
            ],
        ),
        Skill(
            id="release_notes",
            description=(
                "Get the latest published release version and notes for a "
                "GitHub repository."
            ),
            tags=["github", "release", "version", "changelog", "latest"],
            examples=[
                "What is the latest version of openai/openai-python?",
                "Show me the release notes for langchain-ai/langchain",
                "Is there a new release of microsoft/autogen?",
            ],
        ),
        Skill(
            id="trending",
            description=(
                "Find the most popular or trending public GitHub repositories "
                "for a given topic or programming language."
            ),
            tags=["github", "trending", "popular", "discover", "topic", "language"],
            examples=[
                "What are the most popular Python AI frameworks on GitHub?",
                "Find trending TypeScript repositories",
                "Show me the top GitHub projects for Rust web development",
            ],
        ),
    ],
)


# ── Registry ──────────────────────────────────────────────────────────────────

class AgentRegistry:
    """In-memory registry of agent cards for A2A discovery."""

    def __init__(self):
        self.agents: List[AgentCard] = []

    def register(self, card: AgentCard):
        """Add an agent card to the registry."""
        self.agents.append(card)

    def discover(self, tag: str) -> List[str]:
        """Return names of all agents that have a skill matching the tag."""
        return [
            a.name
            for a in self.agents
            for s in a.skills
            if tag in s.tags
        ]

    def find_by_skill_id(self, skill_id: str) -> Optional[AgentCard]:
        """Return the first agent that exposes the given skill ID, or None."""
        for agent in self.agents:
            for skill in agent.skills:
                if skill.id == skill_id:
                    return agent
        return None

    def export_all_json(self, path: str):
        """Write all registered agent cards to a JSON file."""
        data = [json.loads(a.model_dump_json()) for a in self.agents]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Registry exported to: {path}")


# ── Demo / export entry point ─────────────────────────────────────────────────

def demo():
    registry = AgentRegistry()
    registry.register(GITORACLE_CARD)

    print("=" * 60)
    print(" GitOracle A2A Agent Card Demo")
    print("=" * 60)

    print("\nRegistered agents:")
    for a in registry.agents:
        print(f"  - {a.name} v{a.version}  [{len(a.skills)} skills]")

    print("\nDiscover agents that can handle 'triage':")
    print(f"  discover('triage') → {registry.discover('triage')}")

    print("\nDiscover agents that can handle 'trending':")
    print(f"  discover('trending') → {registry.discover('trending')}")

    print("\nFind agent by skill ID 'release_notes':")
    found = registry.find_by_skill_id("release_notes")
    print(f"  find_by_skill_id('release_notes') → {found.name if found else None}")

    # Export the card to agent_card.json alongside this file
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_card.json")
    registry.export_all_json(export_path)

    print(f"\nFull card JSON (also saved to agent_card.json):")
    print(GITORACLE_CARD.model_dump_json(indent=2))


if __name__ == "__main__":
    demo()
