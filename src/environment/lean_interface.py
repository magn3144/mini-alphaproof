"""
Low-level interface for communicating with the Lean 4 server.
Wraps the LeanInteract library for server management.
"""
from typing import Dict, Any, Optional
import logging

try:
    from lean_interact import AutoLeanServer, LeanREPLConfig
    from lean_interact.interface import Command, ProofStep, ProofStepResponse, LeanError
except ImportError:
    raise ImportError(
        "lean-interact is required for Lean 4 support. "
        "Install it with: pip install lean-interact"
    )

from .exceptions import (
    ServerCrashException,
    TheoremSyntaxException,
    TacticFailedException,
    TimeoutException,
    InvalidProofException
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
        verbose: bool = False,
        allow_sorry: bool = False
    ):
        """
        Initialize the Lean interface.

        Args:
            lean_version: Lean version to use (e.g., "v4.27.0-rc1")
            timeout: Timeout in seconds for tactic execution
            verbose: Enable verbose logging
            allow_sorry: If False, reject tactics that use 'sorry' or other cheating tactics
        """
        self.lean_version = lean_version
        self.timeout = timeout
        self.verbose = verbose
        self.allow_sorry = allow_sorry
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
            # Create a Lean REPL configuration
            logger.info(f"Initializing Lean server with version {self.lean_version}")

            # Create config with the specified Lean version
            config = LeanREPLConfig(
                lean_version=self.lean_version,
                verbose=self.verbose
            )

            # AutoLeanServer handles crash recovery automatically
            self.server = AutoLeanServer(config=config)

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

            # Send the theorem to Lean using Command
            response = self.server.run(Command(cmd=theorem, rootGoals=True))

            # Check if response is an error
            if isinstance(response, LeanError):
                raise TheoremSyntaxException(theorem, response.message)

            # Extract the initial proof state from sorries
            # LeanInteract returns the proof state when encountering a sorry
            if not response.sorries:
                raise TheoremSyntaxException(
                    theorem,
                    "No 'sorry' found in theorem. Theorem must contain 'sorry' to create initial proof state."
                )

            # Use the first sorry's proof state
            first_sorry = response.sorries[0]

            # Parse the goal string to extract hypotheses and target
            goal_str = first_sorry.goal
            hypotheses = []
            target = goal_str

            # Split by turnstile to separate hypotheses from target
            if '⊢' in goal_str:
                parts = goal_str.split('⊢')
                if len(parts) == 2:
                    hyp_part = parts[0].strip()
                    target = parts[1].strip()

                    # Parse hypotheses (each on a separate line)
                    if hyp_part:
                        hypotheses = [
                            line.strip()
                            for line in hyp_part.split('\n')
                            if line.strip()
                        ]

            proof_state = ProofState(
                goals=[Goal(
                    id=1,
                    hypotheses=hypotheses,
                    target=target
                )],
                env_id=response.env,
                proof_state_id=first_sorry.proof_state,
                messages=[],
                errors=[]
            )

            # Store environment ID
            self._current_env_id = response.env

            return proof_state

        except TheoremSyntaxException:
            raise
        except Exception as e:
            logger.error(f"Error initializing theorem: {e}")
            raise TheoremSyntaxException(theorem, str(e))

    def _validate_proof_response(
        self,
        response: Any,  # ProofStepResponse
        tactic: str
    ) -> None:
        """
        Validate that a proof response doesn't use cheating tactics.

        Args:
            response: ProofStepResponse from lean-interact
            tactic: The tactic that was applied

        Raises:
            InvalidProofException: If proof uses sorry or other invalid tactics
        """
        if self.allow_sorry:
            return  # Skip validation if sorry is allowed

        # Check proof_status field
        proof_status = getattr(response, 'proof_status', '')
        if proof_status:
            proof_status_lower = proof_status.lower()

            # Check for "sorry" in status
            if "sorry" in proof_status_lower and proof_status_lower.startswith("incomplete"):
                raise InvalidProofException(
                    tactic,
                    f"Proof contains 'sorry': {proof_status}"
                )

            # Check for "admit" in status
            if "admit" in proof_status_lower:
                raise InvalidProofException(
                    tactic,
                    f"Proof uses 'admit' tactic: {proof_status}"
                )

        # Check sorries list
        sorries = getattr(response, 'sorries', [])
        if sorries:
            raise InvalidProofException(
                tactic,
                f"Proof introduced {len(sorries)} sorry(ies)"
            )

        # Check messages for sorry-related warnings
        messages = getattr(response, 'messages', [])
        for message in messages:
            message_data = getattr(message, 'data', '')
            if "declaration uses 'sorry'" in message_data:
                raise InvalidProofException(
                    tactic,
                    "Proof declaration uses 'sorry'"
                )

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

        if proof_state_id is None:
            raise TacticFailedException(tactic, "No proof_state_id provided")

        try:
            logger.debug(f"Executing tactic: {tactic}")

            # Execute the tactic using ProofStep
            response = self.server.run(
                ProofStep(tactic=tactic, proofState=proof_state_id),
                timeout=self.timeout
            )

            # Check if response is an error
            if isinstance(response, LeanError):
                raise TacticFailedException(tactic, response.message)

            # Validate that proof doesn't use cheating tactics
            self._validate_proof_response(response, tactic)

            # Convert ProofStepResponse to dictionary format
            result = {
                "goals": response.goals,
                "proof_state_id": response.proof_state,
                "proof_status": getattr(response, 'proof_status', None),
                "sorries": getattr(response, 'sorries', []),
                "messages": getattr(response, 'messages', []),
                "error": None
            }

            return result

        except (TacticFailedException, TimeoutException):
            raise
        except Exception as e:
            logger.error(f"Error executing tactic '{tactic}': {e}")
            raise TacticFailedException(tactic, str(e))

    def _parse_proof_state(self, response: Dict[str, Any]) -> ProofState:
        """
        Parse a Lean response into a ProofState object.

        Args:
            response: Response dictionary from Lean (from execute_tactic)

        Returns:
            Parsed ProofState object
        """
        goals_list = []
        messages = []
        errors = []

        # Extract goals from response
        # The response["goals"] is a list of goal strings from lean-interact
        # Each goal string has the format:
        #   hypothesis1
        #   hypothesis2
        #   ...
        #   ⊢ target
        if "goals" in response and response["goals"]:
            for i, goal_data in enumerate(response["goals"]):
                # goal_data is typically a string
                if isinstance(goal_data, str):
                    # Split by the turnstile symbol
                    parts = goal_data.split('⊢')
                    if len(parts) == 2:
                        # We have hypotheses and target
                        hyp_part = parts[0].strip()
                        target = parts[1].strip()

                        # Parse hypotheses (each on a separate line)
                        formatted_hyps = []
                        if hyp_part:
                            # Split by newlines and filter out empty lines
                            formatted_hyps = [
                                line.strip()
                                for line in hyp_part.split('\n')
                                if line.strip()
                            ]
                    else:
                        # No turnstile, the whole thing is the target
                        formatted_hyps = []
                        target = goal_data.strip()
                        # Add turnstile if missing
                        if not target.startswith('⊢'):
                            target = f'⊢ {target}'
                else:
                    # Fallback for unexpected format
                    formatted_hyps = []
                    target = str(goal_data)

                goals_list.append(Goal(
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

        # Extract proof status and sorries
        proof_status = response.get("proof_status", None)
        sorries = response.get("sorries", [])

        # Create proof state
        proof_state = ProofState(
            goals=goals_list,
            env_id=response.get("env_id"),
            proof_state_id=response.get("proof_state_id"),
            messages=messages,
            errors=errors,
            proof_status=proof_status,
            sorries=sorries
        )

        return proof_state

    def shutdown(self):
        """Shutdown the Lean server."""
        if self.server is not None:
            logger.info("Shutting down Lean server")
            try:
                # LeanInteract's AutoLeanServer cleanup
                self.server.kill()
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
        del exc_type, exc_val, exc_tb  # Unused
        self.shutdown()
        return False
