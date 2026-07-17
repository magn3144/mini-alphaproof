import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from torch import nn

from alphaproof.core.config import Config
from alphaproof.core.environment import Observation, State
from alphaproof.core.network import Network, NetworkSamplingOutput
from alphaproof.inference.infer import (
    load_network_checkpoint,
    make_config,
    prove,
)


class FakeTokenizer:
    """Tokenize one state and decode generated action IDs."""

    def __call__(self, observation, **kwargs):
        del observation, kwargs
        return SimpleNamespace(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
        )

    def batch_decode(self, sequences, **kwargs):
        del kwargs
        return [f'action_{index}' for index in range(len(sequences))]


class FakeModel:
    """Expand encoder outputs like Transformers generation does."""

    config = SimpleNamespace(pad_token_id=0)

    def get_encoder(self):
        def encode(**kwargs):
            del kwargs
            return SimpleNamespace(last_hidden_state=torch.ones(1, 2, 2))

        return encode

    def generate(self, encoder_outputs, **kwargs):
        del kwargs
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.repeat(
            8, 1, 1
        )
        return SimpleNamespace(
            sequences=torch.tensor(
                [[0, 4, 2, 0]] + [[0, 4, 5, 2]] * 7,
                dtype=torch.long,
            ),
            scores=(
                torch.zeros(8, 2),
                torch.zeros(8, 2),
                torch.zeros(8, 2),
            ),
        )

    def compute_transition_scores(self, sequences, scores, **kwargs):
        del scores, kwargs
        transition_scores = torch.tensor(
            [[-1.0, -2.0, float('-inf')]] + [[-1.0, -2.0, -3.0]] * 7
        )
        return transition_scores[:sequences.shape[0]]


class FakeEnvironment:
    """Solve a theorem after one tactic."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args

    def initial_state(self, theorem):
        del theorem
        return State(0, 0.0, Observation([]), False, 1)

    def step(self, state_id, action, tactic_timeout=1.0):
        del state_id, action, tactic_timeout
        return State(1, 0.0, Observation([]), True, 0)


class FakeSearchNetwork:
    """Always propose one successful tactic."""

    def sample(self, observation):
        del observation
        return NetworkSamplingOutput({'trivial': 0.0}, -1.0)


class FakeCheckpointNetwork:
    """Store parameters loaded by the checkpoint helper."""

    def __init__(self):
        self.params = {}


class NetworkSamplingTest(unittest.TestCase):
    def test_value_is_computed_before_generation_expands_encoder_batch(self):
        network = Network.__new__(Network)
        nn.Module.__init__(network)
        network.device = torch.device('cpu')
        network.max_state_length = 2
        network.max_action_length = 2
        network.num_sampled_actions = 8
        network.tokenizer = cast(Any, FakeTokenizer())
        network.model = cast(Any, FakeModel())
        network.value_head = nn.Linear(2, 2)
        network.register_buffer('value_bins', torch.tensor([-1.0, 0.0]))

        output = network.sample('state')

        self.assertIsInstance(output.value, float)
        self.assertEqual(len(output.action_logprobs), 8)
        self.assertEqual(output.action_logprobs['action_0'], -3.0)


class InferenceTest(unittest.TestCase):
    def test_prove_returns_verified_game(self):
        config = SimpleNamespace(
            num_simulations=1,
            prior_temperature=200,
            tactic_timeout=1.0,
            no_legal_actions_value=-40,
            ps_c=0.01,
            ps_alpha=0.6,
            environment_ctor=FakeEnvironment,
        )
        with patch('alphaproof.inference.infer.final_check', return_value=True):
            game = prove(
                'theorem inference_test : True := by sorry',
                cast(Config, config),
                cast(Network, FakeSearchNetwork()),
            )

        self.assertTrue(game.root.is_optimal)

    def test_rl_run_restores_search_config_and_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            sft_run_dir = run_dir / 'sft'
            saved_config = {
                'sft_run_dir': str(sft_run_dir),
                'lr': 1e-5,
                'max_state_length': 32,
                'max_action_length': 16,
                'pb_c_base': 123,
            }
            (run_dir / 'config.json').write_text(
                json.dumps({'config': saved_config}),
                encoding='utf-8',
            )
            checkpoints_dir = run_dir / 'checkpoints'
            checkpoints_dir.mkdir()
            torch.save(
                {'network_params': {'value': torch.tensor(1)}},
                checkpoints_dir / 'step_0000001.pt',
            )
            latest_path = checkpoints_dir / 'step_0000002.pt'
            torch.save(
                {'network_params': {'value': torch.tensor(2)}},
                latest_path,
            )
            args = SimpleNamespace(
                run_dir=run_dir,
                num_simulations=7,
                tactic_timeout=2.0,
            )
            config = make_config(cast(Any, args))
            network = FakeCheckpointNetwork()
            loaded_path = load_network_checkpoint(
                run_dir,
                cast(Network, network),
            )

        self.assertEqual(config.pb_c_base, 123)
        self.assertEqual(config.num_simulations, 7)
        self.assertEqual(config.tactic_timeout, 2.0)
        self.assertEqual(loaded_path, latest_path)
        self.assertEqual(network.params['value'].item(), 2)


if __name__ == '__main__':
    unittest.main()
