import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from alphaproof.core.config import Config
from alphaproof.core.game import Game, extract_transitions


Transition = tuple[str, str, float]
TokenizedTransition = tuple[torch.Tensor, torch.Tensor, float]


class ReplayBuffer:
    """Persistent replay storage for state-action-value training triples."""

    def __init__(self, config: Config, path: Path):
        """Initialize replay limits and restore existing transitions."""
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.max_state_length = config.max_state_length
        self.max_action_length = config.max_action_length
        if not 0 < config.validation_fraction < 1:
            raise ValueError('validation_fraction must be between zero and one.')
        self.validation_stride = round(1 / config.validation_fraction)
        self.validation_window_size = max(
            1, round(self.window_size * config.validation_fraction)
        )
        self.path = path
        self.path.touch(exist_ok=True)
        self.buffer: list[Transition] = []
        self.validation_buffer: list[Transition] = []
        self.transition_count = 0
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        self._load()

    def save_game(self, game: Game) -> None:
        """Add solved-game transitions to the replay window and JSONL file."""
        for observation, action, value in extract_transitions(game.root):
            self._save_transition((str(observation), str(action), value))

    def __len__(self) -> int:
        """Return the number of train transitions in the replay window."""
        return len(self.buffer)

    def sample_batch(self) -> list[TokenizedTransition]:
        """Sample a uniformly random training batch."""
        return [self.sample_transition() for _ in range(self.batch_size)]

    def sample_transition(self) -> TokenizedTransition:
        """Sample and tokenize one training transition."""
        if not self.buffer:
            raise ValueError('Cannot sample from an empty replay buffer.')
        return self._tokenize_transition(random.choice(self.buffer))

    def validation_batch(self, batch_size: int) -> list[TokenizedTransition]:
        """Return a fixed held-out replay batch."""
        return [
            self._tokenize_transition(transition)
            for transition in self.validation_buffer[:batch_size]
        ]

    def _save_transition(self, transition: Transition) -> None:
        """Assign and persist one transition."""
        self.transition_count += 1
        validation = self.transition_count % self.validation_stride == 0
        target = self.validation_buffer if validation else self.buffer
        target.append(transition)
        if validation:
            self.validation_buffer = self.validation_buffer[
                -self.validation_window_size:
            ]
        else:
            self.buffer = self.buffer[-self.window_size:]

        state, action, value = transition
        record = {
            'state': state,
            'action': action,
            'value': value,
            'validation': validation,
        }
        with self.path.open('a', encoding='utf-8') as replay_file:
            replay_file.write(json.dumps(record) + '\n')

    def _load(self) -> None:
        """Restore the replay window from JSONL."""
        if not self.path.exists():
            return
        with self.path.open(encoding='utf-8') as replay_file:
            for line in replay_file:
                record = json.loads(line)
                transition = (
                    str(record['state']),
                    str(record['action']),
                    float(record['value']),
                )
                if record['validation']:
                    self.validation_buffer.append(transition)
                else:
                    self.buffer.append(transition)
                self.transition_count += 1
        self.buffer = self.buffer[-self.window_size:]
        self.validation_buffer = self.validation_buffer[
            -self.validation_window_size:
        ]

    def _tokenize_transition(
        self,
        transition: Transition,
    ) -> TokenizedTransition:
        state, action, value = transition
        return (
            self._tokenize(state, self.max_state_length),
            self._tokenize(action, self.max_action_length),
            value,
        )

    def _tokenize(self, text: str, max_length: int) -> torch.Tensor:
        """Tokenize text with fixed padding for replay batches."""
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return encoded.input_ids.squeeze(0).long()
