"""
Custom exception classes for Lean 4 environment.
"""


class Lean4Exception(Exception):
    """Base exception for all Lean 4 related errors."""
    pass


class TacticFailedException(Lean4Exception):
    """Raised when a tactic application fails."""
    def __init__(self, tactic: str, error_message: str):
        self.tactic = tactic
        self.error_message = error_message
        super().__init__(f"Tactic '{tactic}' failed: {error_message}")


class TheoremSyntaxException(Lean4Exception):
    """Raised when theorem syntax is invalid."""
    def __init__(self, theorem: str, error_message: str):
        self.theorem = theorem
        self.error_message = error_message
        super().__init__(f"Invalid theorem syntax: {error_message}")


class ServerCrashException(Lean4Exception):
    """Raised when the Lean server crashes unexpectedly."""
    def __init__(self, message: str = "Lean server crashed"):
        super().__init__(message)


class TimeoutException(Lean4Exception):
    """Raised when a tactic execution times out."""
    def __init__(self, tactic: str, timeout: int):
        self.tactic = tactic
        self.timeout = timeout
        super().__init__(f"Tactic '{tactic}' timed out after {timeout} seconds")
