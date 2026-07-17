import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch

from alphaproof.core.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_GAMES,
    DEFAULT_SFT_RUN_DIR,
    DEFAULT_TRAINING_STEPS,
    Config,
)
from alphaproof.core.environment import NodeType, Observation
from alphaproof.core.game import Game, Node, compute_value_target, select_optimal_action
from alphaproof.training.replay_buffer import ReplayBuffer
from alphaproof.training.shared_storage import SharedStorage
from alphaproof.training import train as training
from alphaproof.training.train import RunLogger


class FakeTokenizer:
    """Minimal tokenizer for replay persistence tests."""

    def __call__(self, text: str, max_length: int, **kwargs):
        del text, kwargs
        return SimpleNamespace(
            input_ids=torch.arange(max_length).reshape(1, max_length)
        )


class FakeWandbRun:
    """Collect W&B calls without contacting the service."""

    def __init__(self):
        self.metrics = []

    def log(self, metrics):
        self.metrics.append(metrics)

    def finish(self):
        return None


class TinyNetwork:
    """Small network implementing the checkpoint interface."""

    def __init__(self):
        self.device = torch.device('cpu')
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    @property
    def params(self):
        return {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }

    @params.setter
    def params(self, params):
        self.model.load_state_dict(params)


class FakeLearner:
    """Learner with deterministic losses for orchestration tests."""

    def __init__(self):
        self.updates = 0

    def update(self, batch):
        del batch
        self.updates += 1
        return float(self.updates)

    def evaluate(self, batch):
        del batch
        return 0.5


class FakeTrainingReplay:
    """Non-empty replay buffer with a held-out batch."""

    def __len__(self):
        return 3

    def sample_batch(self):
        return []

    def validation_batch(self, batch_size):
        del batch_size
        return [('state', 'action', -1.0)]


class FakeTrainingLogger:
    """Collect learner metric calls."""

    def __init__(self):
        self.metrics = []

    def log_training(self, *metrics):
        self.metrics.append(metrics)


class FakeStorage:
    """Collect learner checkpoint steps."""

    def __init__(self):
        self.steps = []

    def save_checkpoint(self, step, network):
        del network
        self.steps.append(step)


class ReplayBufferTest(unittest.TestCase):
    def test_persists_train_and_validation_transitions(self):
        config = Config(1, 2, 1, 1, 1e-4, validation_fraction=0.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.jsonl'
            with patch(
                'alphaproof.training.replay_buffer.AutoTokenizer.from_pretrained',
                return_value=FakeTokenizer(),
            ):
                replay = ReplayBuffer(config, path)
                for index in range(8):
                    replay._save_transition(
                        (f'state {index}', f'action {index}', -float(index))
                    )
                restored = ReplayBuffer(config, path)

            self.assertEqual(len(restored), 6)
            self.assertEqual(len(restored.validation_buffer), 2)
            state, action, _ = restored.sample_transition()
            self.assertEqual(tuple(state.shape), (640,))
            self.assertEqual(tuple(action.shape), (128,))


class OptimalActionTest(unittest.TestCase):
    def test_selects_shortest_of_multiple_proven_actions(self):
        terminal = Node(
            action='short',
            observation=Observation([]),
            prior=0.1,
            state_id=1,
            node_type=NodeType.OR,
            reward=0,
            is_optimal=True,
            is_terminal=True,
        )
        longer = Node(
            action='long',
            observation=Observation([]),
            prior=0.9,
            state_id=2,
            node_type=NodeType.OR,
            reward=0,
            is_optimal=True,
        )
        longer.children['finish'] = Node(
            action='finish',
            observation=Observation([]),
            prior=1.0,
            state_id=3,
            node_type=NodeType.OR,
            reward=0,
            is_optimal=True,
            is_terminal=True,
        )
        root = Node(
            action=None,
            observation=Observation([]),
            prior=1.0,
            state_id=0,
            node_type=NodeType.OR,
            reward=0,
            is_optimal=True,
        )
        root.children = {'short': terminal, 'long': longer}

        self.assertEqual(select_optimal_action(root), 'short')
        self.assertEqual(compute_value_target(root), -1)


class SharedStorageTest(unittest.TestCase):
    def test_checkpoint_round_trip_restores_step_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = SharedStorage(Path(directory))
            network = TinyNetwork()
            expected = network.params
            storage.save_checkpoint(12, cast(Any, network))
            with torch.no_grad():
                network.model.weight.zero_()

            step = storage.load_latest_checkpoint(cast(Any, network))

            self.assertEqual(step, 12)
            self.assertTrue(
                torch.equal(network.params['weight'], expected['weight'])
            )


class RunLoggerTest(unittest.TestCase):
    def test_logs_reward_and_rolling_average(self):
        with tempfile.TemporaryDirectory() as directory:
            wandb_run = FakeWandbRun()
            logger = RunLogger(Path(directory), 2, wandb_run)
            failed = Game('theorem failed : False := by sorry', False, 1)
            solved = Game('theorem solved : True := by sorry', False, 1)
            solved.root.is_optimal = True
            solved.root.value_target = -3

            logger.log_game(failed, 0)
            logger.log_game(solved, 1)

            self.assertEqual(
                wandb_run.metrics[-1]['actor/rolling_success_rate'], 0.5
            )
            self.assertEqual(wandb_run.metrics[-1]['actor/episode_reward'], -3)
            records = [
                json.loads(line)
                for line in (Path(directory) / 'results.jsonl').read_text().splitlines()
            ]
            self.assertEqual(
                [record['episode_reward'] for record in records], [None, -3]
            )


class TrainingCliTest(unittest.TestCase):
    def test_defaults_are_sized_for_a_full_training_run(self):
        with patch.object(sys, 'argv', ['train', 'rl_test']):
            args = training.parse_args()

        config = training.make_config(args)

        self.assertEqual(args.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(args.num_games, DEFAULT_NUM_GAMES)
        self.assertEqual(args.training_steps, DEFAULT_TRAINING_STEPS)
        self.assertEqual(config.sft_run_dir, DEFAULT_SFT_RUN_DIR)

    def test_empty_replay_fails_training(self):
        config = Config(1, 1, training_steps=1)

        with self.assertRaisesRegex(
            ValueError, 'no actor game was solved'
        ):
            training.train_network(
                config,
                cast(Any, FakeLearner()),
                cast(Any, FakeStorage()),
                cast(Any, []),
                0,
                cast(Any, FakeTrainingLogger()),
            )

    def test_resume_restores_saved_cli_and_algorithm_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sft_run = root / 'sft'
            (sft_run / 'model_source').mkdir(parents=True)
            (sft_run / 'network_params.pt').touch()
            dataset = root / 'theorems.jsonl'
            dataset.write_text('{"theorem": "theorem t : True := by sorry"}\n')

            new_argv = [
                'train',
                'rl_test',
                '--training-steps',
                '3',
            ]
            with patch.object(sys, 'argv', new_argv):
                new_args = training.parse_args()
            config = Config(
                800,
                32,
                dataset_path=dataset,
                sft_run_dir=sft_run,
                training_steps=3,
            )
            config.pb_c_base = 123
            with (
                patch.object(training, 'RUNS_DIR', root),
                patch.object(training, 'make_config', return_value=config),
            ):
                new_args, run_dir, saved = training.prepare_run(new_args)
            self.assertIsNone(saved)
            self.assertTrue(new_args.wandb_run_id)
            wandb_run_id = new_args.wandb_run_id
            training.save_run_config(run_dir, new_args, config)

            with patch.object(sys, 'argv', ['train', 'rl_test', '--resume']):
                resume_args = training.parse_args()
            with patch.object(training, 'RUNS_DIR', root):
                resume_args, _, saved = training.prepare_run(resume_args)
            restored = training.make_config(resume_args, saved)

            self.assertEqual(restored.training_steps, 3)
            self.assertEqual(restored.pb_c_base, 123)
            self.assertEqual(resume_args.wandb_run_id, wandb_run_id)

    def test_learner_logs_losses_and_saves_resumable_checkpoints(self):
        config = Config(
            1,
            1,
            1,
            1,
            1e-4,
            training_steps=3,
            checkpoint_interval=2,
            validation_batch_size=1,
            validation_interval=2,
            log_interval=1,
        )
        network = FakeLearner()
        replay = FakeTrainingReplay()
        storage = FakeStorage()
        logger = FakeTrainingLogger()

        training.train_network(
            config,
            cast(Any, network),
            cast(Any, storage),
            cast(Any, replay),
            0,
            cast(Any, logger),
        )

        self.assertEqual(network.updates, 3)
        self.assertEqual(storage.steps, [2, 3])
        self.assertEqual(logger.metrics[1][2], 0.5)


if __name__ == '__main__':
    unittest.main()
