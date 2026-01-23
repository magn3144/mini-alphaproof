"""
Lean 4 environment for interactive theorem proving.

This module provides a high-level interface for interacting with the Lean 4
theorem prover, supporting both human interaction and RL agent training.
"""

from .environment import Lean4Environment
from .state import Goal, ProofState, TacticResult
from .exceptions import (
    Lean4Exception,
    TacticFailedException,
    TheoremSyntaxException,
    ServerCrashException,
    TimeoutException
)

__all__ = [
    # Main environment class
    "Lean4Environment",

    # Data structures
    "Goal",
    "ProofState",
    "TacticResult",

    # Exceptions
    "Lean4Exception",
    "TacticFailedException",
    "TheoremSyntaxException",
    "ServerCrashException",
    "TimeoutException",
]

__version__ = "0.1.0"
