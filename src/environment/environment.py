"""
Main Lean 4 environment class for interactive theorem proving.
"""
from typing import Optional
import logging

from .lean_interface import LeanInterface
from .state import ProofState, TacticResult
from .exceptions import Lean4Exception, TacticFailedException, InvalidProofException
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
        verbose: bool = False,
        allow_sorry: bool = False
    ):
        """
        Initialize a Lean 4 environment with a theorem to prove.

        Args:
            theorem_statement: Theorem to prove (e.g., "theorem ex : 1 + 1 = 2 := by sorry")
            lean_version: Lean version to use
            timeout: Timeout in seconds for each tactic
            verbose: Enable verbose logging
            allow_sorry: If False, reject tactics that use 'sorry' (default: False for production use)

        Example:
            >>> env = Lean4Environment("theorem ex : 1 + 1 = 2 := by sorry")
            >>> print(env.get_state_string())
            >>> env.apply_tactic("norm_num")
            >>> print(env.is_complete())
        """
        self.lean_version = lean_version
        self.timeout = timeout
        self.verbose = verbose
        self.allow_sorry = allow_sorry

        # Initialize Lean interface
        self.interface = LeanInterface(
            lean_version=lean_version,
            timeout=timeout,
            verbose=verbose,
            allow_sorry=allow_sorry
        )

        # State management
        self.current_state: Optional[ProofState] = None

        # Statistics
        self.steps_taken: int = 0

        # Initialize the theorem with Lean
        logger.info(f"Initializing environment with theorem: {theorem_statement}")

        try:
            self.current_state = self.interface.initialize_theorem(theorem_statement)
            logger.info(f"Initial state has {self.current_state.num_goals()} goals")
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
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
            self.current_state = new_state

            # Validate that proof doesn't use cheating tactics
            if not self.allow_sorry and new_state.has_cheating_tactics():
                # Rollback the state change
                logger.warning(f"Tactic '{tactic}' introduced cheating tactics, rejecting")
                return TacticResult(
                    success=False,
                    new_state=self.current_state,
                    error_message=f"Invalid proof: {new_state.proof_status}",
                    proof_complete=False
                )

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

            return TacticResult(
                success=False,
                new_state=self.current_state,
                error_message=e.error_message,
                proof_complete=False
            )

        except InvalidProofException as e:
            # Proof uses cheating tactics
            logger.error(f"Invalid proof: {e.reason}")

            return TacticResult(
                success=False,
                new_state=self.current_state,
                error_message=f"Invalid proof: {e.reason}",
                proof_complete=False
            )

        except Lean4Exception as e:
            # More serious error
            logger.error(f"Lean error: {e}")

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

        # Get current state
        result = self.current_state.to_string(format=mode)

        # Apply colors if in human mode
        if mode == "human":
            result = format_state_with_colors(result, use_color=use_color)

        return result

    def get_stats(self) -> dict:
        """
        Get statistics about the current proof session.

        Returns:
            Dictionary with statistics
        """
        return {
            "steps_taken": self.steps_taken,
            "num_goals": self.current_state.num_goals() if self.current_state else 0,
            "proof_complete": self.is_complete()
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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        del exc_type, exc_val, exc_tb  # Unused
        self.close()
        return False

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass
