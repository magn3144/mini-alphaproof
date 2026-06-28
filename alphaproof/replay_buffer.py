import random
from typing import Any

import torch

from alphaproof.config import Config
from alphaproof.game import Game, extract_transitions


class ReplayBuffer:
    """Replay storage for state-action-value training triples."""

    def __init__(self, config: Config):
        """Initialize replay limits from the configuration."""
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.buffer = []

    def save_game(self, game: Game):
        """Add solved-game transitions to the replay window."""
        transitions = extract_transitions(game.root)
        self.buffer.extend(transitions)
        self.buffer = self.buffer[-self.window_size:]

    def sample_batch(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Sample a batch of tokenized replay transitions."""
        return [self.sample_transition() for _ in range(self.batch_size)]

    def sample_transition(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Sample and tokenize one replay transition."""
        if not self.buffer:
            raise ValueError('Cannot sample from an empty replay buffer.')

        observation, action, value = random.choice(self.buffer)
        tokenized_observation = self.tokenize(observation)
        tokenized_action = self.tokenize(action)
        return (tokenized_observation, tokenized_action, value)

    def tokenize(self, input_value: Any) -> torch.Tensor:
        """Tokenize text as padded UTF-8 byte IDs."""
        encoded = str(input_value).encode('utf-8')[:self.sequence_length]
        tokens = torch.zeros(self.sequence_length, dtype=torch.int32)
        if encoded:
            tokens[:len(encoded)] = torch.tensor(list(encoded), dtype=torch.int32) + 1
        return tokens
