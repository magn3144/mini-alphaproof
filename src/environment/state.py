"""
Data structures for representing proof states and results in Lean 4.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class Goal:
    """Represents a single proof goal in Lean."""
    id: int
    hypotheses: List[str]  # Context/assumptions
    target: str  # Goal to prove

    def to_string(self, format: str = "human") -> str:
        """
        Convert goal to string representation.

        Args:
            format: Output format ("human" or "llm")

        Returns:
            Formatted string representation of the goal
        """
        if format == "human":
            lines = []
            if self.hypotheses:
                lines.append("  Hypotheses:")
                for hyp in self.hypotheses:
                    lines.append(f"    {hyp}")
            lines.append("  Target:")
            # Only add turnstile if not already present
            if self.target.strip().startswith("⊢"):
                lines.append(f"    {self.target}")
            else:
                lines.append(f"    ⊢ {self.target}")
            return "\n".join(lines)
        elif format == "llm":
            parts = []
            if self.hypotheses:
                parts.append("Hypotheses: " + ", ".join(self.hypotheses))
            parts.append(f"Target: {self.target}")
            return " | ".join(parts)
        else:
            return json.dumps({
                "id": self.id,
                "hypotheses": self.hypotheses,
                "target": self.target
            })


@dataclass
class ProofState:
    """Represents the complete state of a proof attempt."""
    goals: List[Goal]
    env_id: Optional[int] = None
    proof_state_id: Optional[int] = None
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    proof_status: Optional[str] = None
    sorries: List = field(default_factory=list)

    def is_complete(self) -> bool:
        """Check if the proof is complete (no remaining goals)."""
        return len(self.goals) == 0

    def num_goals(self) -> int:
        """Return the number of remaining goals."""
        return len(self.goals)

    def has_cheating_tactics(self) -> bool:
        """
        Check if the proof state contains cheating tactics.

        Returns:
            True if proof uses sorry, admit, or other invalid tactics
        """
        # Check if we have sorries
        if self.sorries:
            return True

        # Check proof status
        if self.proof_status:
            proof_status_lower = self.proof_status.lower()
            if "sorry" in proof_status_lower or "admit" in proof_status_lower:
                return True

        return False

    def to_string(self, format: str = "human") -> str:
        """
        Convert proof state to string representation.

        Args:
            format: Output format ("human", "llm", or "json")

        Returns:
            Formatted string representation of the proof state
        """
        if format == "human":
            lines = []

            # Show proof status if available
            if self.proof_status:
                lines.append(f"Proof Status: {self.proof_status}")

            # Status line
            if self.is_complete():
                lines.append("Status: Complete ✓")
                lines.append("Proof finished!")
            else:
                goal_text = "goal" if self.num_goals() == 1 else "goals"
                lines.append(f"Status: In progress ({self.num_goals()} {goal_text} remaining)")
                lines.append("")

                # Add each goal
                for i, goal in enumerate(self.goals, 1):
                    lines.append(f"Goal {i}:")
                    lines.append(goal.to_string(format="human"))
                    if i < len(self.goals):
                        lines.append("")

            # Add messages if any
            if self.messages:
                lines.append("")
                lines.append("Messages:")
                for msg in self.messages:
                    lines.append(f"  {msg}")

            # Add errors if any
            if self.errors:
                lines.append("")
                lines.append("Errors:")
                for err in self.errors:
                    lines.append(f"  {err}")

            return "\n".join(lines)

        elif format == "llm":
            lines = []

            # Include proof status
            if self.proof_status:
                lines.append(f"Proof Status: {self.proof_status}")

            if self.is_complete():
                lines.append("Proof complete. No remaining goals.")
                return "\n".join(lines) if lines else "Proof complete. No remaining goals."

            lines.append(f"Goals: {self.num_goals()}")
            for i, goal in enumerate(self.goals, 1):
                lines.append(f"Goal {i}: {goal.to_string(format='llm')}")

            if self.errors:
                lines.append("Errors: " + "; ".join(self.errors))

            return "\n".join(lines)

        else:  # json
            return json.dumps({
                "goals": [
                    {
                        "id": g.id,
                        "hypotheses": g.hypotheses,
                        "target": g.target
                    }
                    for g in self.goals
                ],
                "complete": self.is_complete(),
                "num_goals": self.num_goals(),
                "proof_status": self.proof_status,
                "has_cheating": self.has_cheating_tactics(),
                "messages": self.messages,
                "errors": self.errors,
                "env_id": self.env_id,
                "proof_state_id": self.proof_state_id
            }, indent=2)


@dataclass
class TacticResult:
    """Represents the result of applying a tactic."""
    success: bool
    new_state: Optional[ProofState]
    error_message: Optional[str] = None
    proof_complete: bool = False

    def __bool__(self) -> bool:
        """Allow using TacticResult in boolean context."""
        return self.success
