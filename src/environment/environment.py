"""
Main Lean 4 environment class for interactive theorem proving.
"""
from typing import Optional, List, Tuple
import logging

from .lean_interface import LeanInterface
from .state import ProofState, TacticResult
from .exceptions import Lean4Exception, TacticFailedException
from .utils import format_state_with_colors


logger = logging.getLogger(__name__)


class Lean4Environment:
    """
    Main interface for interacting with the Lean 4 theorem prover.

    Provides both a simple interface for human interaction and structured
    API for RL agents.
    """

    def __init__(
        self,
        theorem_statement: str,
        lean_version: str = "v4.27.0-rc1",
        timeout: int = 30,
        verbose: bool = False
    ):
        """
        Initialize a Lean 4 environment with a theorem to prove.

        Args:
            theorem_statement: Theorem to prove (e.g., "theorem ex : 1 + 1 = 2 := by sorry")
            lean_version: Lean version to use
            timeout: Timeout in seconds for each tactic
            verbose: Enable verbose logging

        Example:
            >>> env = Lean4Environment("theorem ex : 1 + 1 = 2 := by sorry")
            >>> print(env.get_state_string())
            >>> env.apply_tactic("norm_num")
            >>> print(env.is_complete())
        """
        self.theorem_statement = theorem_statement
        self.lean_version = lean_version
        self.timeout = timeout
        self.verbose = verbose

        # Initialize Lean interface
        self.interface = LeanInterface(
            lean_version=lean_version,
            timeout=timeout,
            verbose=verbose
        )

        # State management
        self.current_state: Optional[ProofState] = None
        self.initial_theorem: str = theorem_statement

        # History stack for backtracking: (tactic, proof_state_id, state)
        self.history: List[Tuple[str, Optional[int], ProofState]] = []

        # Statistics
        self.steps_taken: int = 0
        self.tactics_applied: List[str] = []

        # Initialize the theorem
        self.reset(theorem_statement)

    def reset(self, theorem_statement: Optional[str] = None) -> ProofState:
        """
        Reset the environment to a new theorem or the initial theorem.

        Args:
            theorem_statement: New theorem statement, or None to reset to initial

        Returns:
            Initial proof state

        Raises:
            TheoremSyntaxException: If theorem syntax is invalid
        """
        if theorem_statement is not None:
            self.theorem_statement = theorem_statement

        logger.info(f"Resetting environment with theorem: {self.theorem_statement}")

        # Clear state
        self.current_state = None
        self.history = []
        self.steps_taken = 0
        self.tactics_applied = []

        # Initialize the theorem with Lean
        try:
            self.current_state = self.interface.initialize_theorem(self.theorem_statement)
            logger.info(f"Initial state has {self.current_state.num_goals()} goals")
            return self.current_state
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise

    def step(self, tactic: str) -> TacticResult:
        """
        Apply a tactic to the current proof state.

        This is the core method for RL agents and structured interaction.

        Args:
            tactic: Tactic to apply (e.g., "intro", "rfl", "norm_num")

        Returns:
            TacticResult containing success status, new state, and error info

        Example:
            >>> result = env.step("intro n")
            >>> if result.success:
            ...     print("Tactic succeeded!")
            ...     print(result.new_state.to_string())
        """
        if self.current_state is None:
            return TacticResult(
                success=False,
                new_state=None,
                error_message="No proof state initialized",
                proof_complete=False
            )

        if self.current_state.is_complete():
            return TacticResult(
                success=False,
                new_state=self.current_state,
                error_message="Proof is already complete",
                proof_complete=True
            )

        try:
            # Save current state to history
            self.history.append((
                tactic,
                self.current_state.proof_state_id,
                self.current_state
            ))

            # Execute the tactic
            logger.debug(f"Applying tactic: {tactic}")
            response = self.interface.execute_tactic(
                tactic=tactic,
                proof_state_id=self.current_state.proof_state_id
            )

            # Parse the new state
            new_state = self.interface._parse_proof_state(response)

            # Update statistics
            self.steps_taken += 1
            self.tactics_applied.append(tactic)
            self.current_state = new_state

            # Check if proof is complete
            proof_complete = new_state.is_complete()

            if proof_complete:
                logger.info(f"Proof complete after {self.steps_taken} steps!")

            return TacticResult(
                success=True,
                new_state=new_state,
                error_message=None,
                proof_complete=proof_complete
            )

        except TacticFailedException as e:
            # Tactic failed, but we can continue
            logger.warning(f"Tactic failed: {e.error_message}")

            # Remove the failed attempt from history
            if self.history and self.history[-1][0] == tactic:
                self.history.pop()

            return TacticResult(
                success=False,
                new_state=self.current_state,
                error_message=e.error_message,
                proof_complete=False
            )

        except Lean4Exception as e:
            # More serious error
            logger.error(f"Lean error: {e}")

            # Remove the failed attempt from history
            if self.history and self.history[-1][0] == tactic:
                self.history.pop()

            return TacticResult(
                success=False,
                new_state=self.current_state,
                error_message=str(e),
                proof_complete=False
            )

    def is_complete(self) -> bool:
        """
        Check if the current proof is complete.

        Returns:
            True if proof is complete (no remaining goals), False otherwise
        """
        if self.current_state is None:
            return False
        return self.current_state.is_complete()

    def get_state_string(self) -> str:
        """
        Get a human/LLM readable string representation of the current state.

        Returns:
            Formatted string showing current goals and status

        Example output:
            Status: In progress (2 goals remaining)

            Goal 1:
              Hypotheses:
                n : Nat
              Target:
                ⊢ n + 0 = 0 + n

            Goal 2:
              Hypotheses:
                n m : Nat
                IH : n + m = m + n
              Target:
                ⊢ n + Nat.succ m = Nat.succ m + n
        """
        if self.current_state is None:
            return "No proof state initialized"

        return self.current_state.to_string(format="human")

    def apply_tactic(self, tactic: str) -> bool:
        """
        Simple interface to apply a tactic.

        This is a convenience wrapper around step() for interactive use.

        Args:
            tactic: Tactic to apply

        Returns:
            True if tactic succeeded, False otherwise

        Example:
            >>> if env.apply_tactic("intro n"):
            ...     print("Success!")
            ... else:
            ...     print("Tactic failed")
        """
        result = self.step(tactic)
        return result.success

    def render(self, mode: str = "human", use_color: bool = None) -> str:
        """
        Pretty-print the current proof state with optional ANSI colors.

        Args:
            mode: Rendering mode ("human", "llm", or "json")
            use_color: Enable ANSI colors (None = auto-detect terminal)

        Returns:
            Formatted string representation

        Example:
            >>> print(env.render())  # Auto-detects color support
            >>> print(env.render(use_color=True))  # Force colors
        """
        if self.current_state is None:
            return "No proof state initialized"

        output_lines = []

        # Add theorem name if we can extract it
        theorem_name = self._extract_theorem_name()
        if theorem_name:
            output_lines.append(f"Theorem: {theorem_name}")

        # Add statistics
        output_lines.append(f"Steps taken: {self.steps_taken}")

        # Add current state
        output_lines.append("")
        output_lines.append(self.current_state.to_string(format=mode))

        result = "\n".join(output_lines)

        # Apply colors if in human mode
        if mode == "human":
            result = format_state_with_colors(result, use_color=use_color)

        return result

    def backtrack(self, steps: int = 1) -> ProofState:
        """
        Undo previous tactics.

        Args:
            steps: Number of steps to backtrack (default: 1)

        Returns:
            Proof state after backtracking

        Raises:
            ValueError: If trying to backtrack more steps than available

        Example:
            >>> env.apply_tactic("intro")  # Wrong tactic
            >>> env.backtrack()  # Undo it
            >>> env.apply_tactic("trivial")  # Try correct tactic
        """
        if steps <= 0:
            raise ValueError("Steps must be positive")

        if steps > len(self.history):
            raise ValueError(
                f"Cannot backtrack {steps} steps, only {len(self.history)} available"
            )

        # Remove the last 'steps' entries from history
        for _ in range(steps):
            if self.history:
                tactic, proof_state_id, state = self.history.pop()
                self.current_state = state
                self.steps_taken -= 1
                if self.tactics_applied:
                    self.tactics_applied.pop()

        logger.info(f"Backtracked {steps} steps")
        return self.current_state

    def _extract_theorem_name(self) -> Optional[str]:
        """Extract the theorem name from the theorem statement."""
        try:
            # Simple parsing: "theorem name : ..." -> "name"
            if "theorem" in self.theorem_statement:
                parts = self.theorem_statement.split()
                theorem_idx = parts.index("theorem")
                if theorem_idx + 1 < len(parts):
                    name = parts[theorem_idx + 1]
                    # Remove any trailing colon
                    return name.rstrip(":")
        except Exception:
            pass
        return None

    def get_stats(self) -> dict:
        """
        Get statistics about the current proof session.

        Returns:
            Dictionary with statistics
        """
        return {
            "steps_taken": self.steps_taken,
            "tactics_applied": self.tactics_applied.copy(),
            "num_goals": self.current_state.num_goals() if self.current_state else 0,
            "proof_complete": self.is_complete(),
            "history_length": len(self.history)
        }

    def close(self):
        """
        Close the environment and cleanup resources.

        Should be called when done with the environment.
        """
        logger.info("Closing Lean4Environment")
        if self.interface:
            self.interface.shutdown()
        self.current_state = None
        self.history = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass
