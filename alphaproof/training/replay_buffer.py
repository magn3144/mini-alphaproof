import random
from typing import Any

from transformers import AutoTokenizer
import torch

from alphaproof.core.config import Config
from alphaproof.core.game import Game, extract_transitions


class ReplayBuffer:
    """Replay storage for state-action-value training triples."""

    def __init__(self, config: Config):
        """Initialize replay limits from the configuration."""
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.max_state_length = config.max_state_length
        self.max_action_length = config.max_action_length
        self.buffer = []
        self.tokenizer_model = config.tokenizer_model
        self.tokenizer = self._load_tokenizer()

    def save_game(self, game: Game):
        """Add solved-game transitions to the replay window."""
        transitions = extract_transitions(game.root)
        self.buffer.extend(transitions)
        self.buffer = self.buffer[-self.window_size:]

    def __len__(self) -> int:
        """Return the number of replay transitions."""
        return len(self.buffer)

    def sample_batch(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Sample a batch of tokenized replay transitions."""
        return [self.sample_transition() for _ in range(self.batch_size)]

    def sample_transition(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Sample and tokenize one replay transition."""
        if not self.buffer:
            raise ValueError('Cannot sample from an empty replay buffer.')

        observation, action, value = random.choice(self.buffer)
        tokenized_observation = self.tokenize(
            observation, self.max_state_length
        )
        tokenized_action = self.tokenize(action, self.max_action_length)
        return (tokenized_observation, tokenized_action, value)

    def tokenize(self, input_value: Any, max_length: int) -> torch.Tensor:
        """Tokenize text with the CodeT5+ tokenizer."""
        encoded = self.tokenizer(
            str(input_value),
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return encoded.input_ids.squeeze(0).long()

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_model)
