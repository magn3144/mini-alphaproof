from typing import Any

Params = Any


class SharedStorage:
    """In-memory checkpoint storage shared by learner and actors."""

    def __init__(self):
        """Initialize the parameter checkpoint table."""
        self._params = {}

    def latest_params(self) -> Params:
        """Return the most recent network parameters."""
        return self._params[max(self._params.keys())]

    def save_params(self, step: int, params: Params):
        """Save parameters for a training step."""
        self._params[step] = params
