"""
Low-level interface for communicating with the Lean 4 server.
Wraps the LeanInteract library for server management.
"""
from typing import Dict, Any, Optional, List
import logging

try:
    from lean_interact import LeanProject, AutoLeanServer
except ImportError:
    raise ImportError(
        "lean-interact is required for Lean 4 support. "
        "Install it with: pip install lean-interact"
    )

from .exceptions import (
    ServerCrashException,
    TheoremSyntaxException,
    TacticFailedException,
    TimeoutException
)
from .state import Goal, ProofState


logger = logging.getLogger(__name__)


class LeanInterface:
    """
    Manages the Lean server lifecycle and provides methods for
    executing commands and tactics.
    """

    def __init__(
        self,
        lean_version: str = "v4.27.0-rc1",
        timeout: int = 30,
        verbose: bool = False
    ):
        """
        Initialize the Lean interface.

        Args:
            lean_version: Lean version to use (e.g., "v4.27.0-rc1")
            timeout: Timeout in seconds for tactic execution
            verbose: Enable verbose logging
        """
        self.lean_version = lean_version
        self.timeout = timeout
        self.verbose = verbose
        self.server: Optional[AutoLeanServer] = None
        self._current_env_id: Optional[int] = None

        # Configure logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)

        # Initialize the server
        self._initialize_server()

    def _initialize_server(self):
        """Initialize the Lean server with auto-recovery."""
        try:
            # Create a Lean project
            # LeanInteract handles project setup automatically
            logger.info(f"Initializing Lean server with version {self.lean_version}")

            # AutoLeanServer handles crash recovery automatically
            self.server = AutoLeanServer(
                lean_version=self.lean_version,
                verbose=self.verbose
            )

            logger.info("Lean server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Lean server: {e}")
            raise ServerCrashException(f"Could not start Lean server: {e}")

    def initialize_theorem(self, theorem: str) -> ProofState:
        """
        Initialize a new theorem for proving.

        Args:
            theorem: Theorem statement (e.g., "theorem ex : 1 + 1 = 2 := by sorry")

        Returns:
            Initial proof state

        Raises:
            TheoremSyntaxException: If theorem syntax is invalid
        """
        if self.server is None:
            raise ServerCrashException("Server not initialized")

        try:
            # Create a new environment with the theorem
            # The theorem should be in the form: "theorem name : statement := by sorry"
            logger.debug(f"Initializing theorem: {theorem}")

            # Send the theorem to Lean
            response = self.server.run_code(theorem)

            # Check for errors
            if response.get("error"):
                error_msg = response.get("error", "Unknown error")
                raise TheoremSyntaxException(theorem, error_msg)

            # Extract the initial proof state
            # LeanInteract returns the proof state when encountering a sorry
            proof_state = self._parse_proof_state(response)

            # Store environment ID if available
            if "env_id" in response:
                self._current_env_id = response["env_id"]

            return proof_state

        except TheoremSyntaxException:
            raise
        except Exception as e:
            logger.error(f"Error initializing theorem: {e}")
            raise TheoremSyntaxException(theorem, str(e))

    def execute_tactic(
        self,
        tactic: str,
        proof_state_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute a tactic on the current proof state.

        Args:
            tactic: Tactic to execute (e.g., "intro", "rfl", "norm_num")
            proof_state_id: Optional proof state ID to apply tactic to

        Returns:
            Dictionary containing the response from Lean

        Raises:
            TacticFailedException: If tactic application fails
            TimeoutException: If tactic times out
        """
        if self.server is None:
            raise ServerCrashException("Server not initialized")

        try:
            logger.debug(f"Executing tactic: {tactic}")

            # Execute the tactic
            # LeanInteract handles tactic execution through run_tactic
            response = self.server.run_tactic(
                tactic=tactic,
                proof_state_id=proof_state_id,
                timeout=self.timeout
            )

            # Check for timeout
            if response.get("timeout"):
                raise TimeoutException(tactic, self.timeout)

            # Check for errors
            if response.get("error"):
                error_msg = response.get("error", "Unknown error")
                raise TacticFailedException(tactic, error_msg)

            return response

        except (TacticFailedException, TimeoutException):
            raise
        except Exception as e:
            logger.error(f"Error executing tactic '{tactic}': {e}")
            raise TacticFailedException(tactic, str(e))

    def _parse_proof_state(self, response: Dict[str, Any]) -> ProofState:
        """
        Parse a Lean response into a ProofState object.

        Args:
            response: Response dictionary from Lean

        Returns:
            Parsed ProofState object
        """
        goals = []
        messages = []
        errors = []

        # Extract goals from response
        if "goals" in response and response["goals"]:
            for i, goal_data in enumerate(response["goals"]):
                # Parse goal structure
                hypotheses = goal_data.get("hypotheses", [])
                target = goal_data.get("target", goal_data.get("conclusion", ""))

                # Format hypotheses
                formatted_hyps = []
                if isinstance(hypotheses, list):
                    formatted_hyps = [str(h) for h in hypotheses]
                elif isinstance(hypotheses, str):
                    formatted_hyps = [hypotheses]

                goals.append(Goal(
                    id=i + 1,
                    hypotheses=formatted_hyps,
                    target=target
                ))

        # Extract messages
        if "messages" in response:
            messages = response["messages"]
            if isinstance(messages, str):
                messages = [messages]

        # Extract errors
        if "error" in response and response["error"]:
            errors = [response["error"]]

        # Create proof state
        proof_state = ProofState(
            goals=goals,
            env_id=response.get("env_id"),
            proof_state_id=response.get("proof_state_id"),
            messages=messages,
            errors=errors
        )

        return proof_state

    def shutdown(self):
        """Shutdown the Lean server."""
        if self.server is not None:
            logger.info("Shutting down Lean server")
            try:
                # LeanInteract's AutoLeanServer cleanup
                self.server.close()
            except Exception as e:
                logger.warning(f"Error during server shutdown: {e}")
            finally:
                self.server = None
                self._current_env_id = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
