import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, cast

import torch

from alphaproof.core.actors import run_mcts
from alphaproof.core.config import DEFAULT_MODEL_PATH, Config
from alphaproof.core.environment import NodeType
from alphaproof.core.game import Game, Node, extract_proof_script, final_check
from alphaproof.core.helper import replace_sorry_proof, theorem_for_game
from alphaproof.core.network import Network, Params


def load_run_config(run_dir: Path) -> dict[str, Any]:
    """Load optional settings saved alongside a network checkpoint."""
    config_path = run_dir / 'config.json'
    if not config_path.exists():
        return {}
    with config_path.open(encoding='utf-8') as file:
        return json.load(file)


def make_config(args: argparse.Namespace) -> Config:
    """Build search configuration for an SFT or RL run."""
    run_data = load_run_config(args.run_dir)
    saved_config = run_data.get('config', {})
    if saved_config:
        model_run_dir = Path(saved_config['sft_run_dir'])
        learning_rate = float(saved_config['lr'])
    else:
        model_run_dir = args.run_dir
        learning_rate = float(run_data.get('learning_rate', 5e-5))

    config = Config(
        num_simulations=args.num_simulations,
        batch_size=1,
        num_actors=1,
        num_games=1,
        lr=learning_rate,
        sft_run_dir=model_run_dir,
        max_state_length=int(
            saved_config.get(
                'max_state_length', run_data.get('max_state_length', 640)
            )
        ),
        max_action_length=int(
            saved_config.get(
                'max_action_length', run_data.get('max_action_length', 128)
            )
        ),
    )
    for name in (
        'pb_c_base',
        'pb_c_init',
        'value_discount',
        'prior_temperature',
        'no_legal_actions_value',
        'ps_c',
        'ps_alpha',
        'num_value_bins',
    ):
        if name in saved_config:
            setattr(config, name, saved_config[name])
    config.tactic_timeout = args.tactic_timeout
    return config


def load_network_checkpoint(run_dir: Path, network: Network) -> Path:
    """Load the latest RL checkpoint or the SFT network parameters."""
    checkpoints = sorted((run_dir / 'checkpoints').glob('step_*.pt'))
    if checkpoints:
        checkpoint_path = checkpoints[-1]
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )
        checkpoint = cast(dict[str, Any], checkpoint)
        network.params = cast(Params, checkpoint['network_params'])
        return checkpoint_path

    checkpoint_path = run_dir / 'network_params.pt'
    network.load_params(checkpoint_path)
    return checkpoint_path


def prove(
    theorem: str,
    config: Config,
    network: Network,
    disprove: bool = False,
) -> Game:
    """Search for and verify a proof of one theorem."""
    game = Game(theorem, disprove, config.num_simulations)
    with config.environment_ctor() as environment:
        state = environment.initial_state(theorem)
        if disprove:
            state = environment.step(state.id, 'disprove')
        game.root = Node(
            action=None,
            observation=state.observation,
            prior=1.0,
            state_id=state.id,
            node_type=NodeType.OR,
            reward=state.reward,
            is_optimal=state.terminal,
            is_terminal=state.terminal,
        )
        run_mcts(config, game, network, environment)

    if game.root.is_optimal:
        game.root.is_optimal = final_check(game)
    return game


def read_theorem(args: argparse.Namespace) -> str:
    """Read the requested theorem from the command line or a file."""
    if args.theorem is not None:
        theorem = args.theorem.strip()
    else:
        if args.theorem_file is None:
            raise ValueError('A theorem or theorem file is required.')
        theorem = args.theorem_file.read_text(encoding='utf-8').strip()
    if theorem.count('sorry') != 1:
        raise ValueError('The theorem must contain exactly one `sorry`.')
    return theorem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse inference command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Search for a verified Lean proof with AlphaProof.'
    )
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            'SFT or RL run containing trained network parameters '
            f'(default: {DEFAULT_MODEL_PATH}).'
        ),
    )
    theorem_source = parser.add_mutually_exclusive_group(required=True)
    theorem_source.add_argument('--theorem', help='Lean theorem containing `sorry`.')
    theorem_source.add_argument('--theorem-file', type=Path)
    parser.add_argument('--num-simulations', type=int, default=800)
    parser.add_argument('--num-sampled-actions', type=int, default=8)
    parser.add_argument('--tactic-timeout', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--disprove', action='store_true')
    args = parser.parse_args(argv)

    if not args.run_dir.is_dir():
        parser.error(f'Run does not exist: {args.run_dir}')
    has_sft_params = (args.run_dir / 'network_params.pt').is_file()
    has_rl_params = any((args.run_dir / 'checkpoints').glob('step_*.pt'))
    if not has_sft_params and not has_rl_params:
        parser.error(f'Run contains no network parameters: {args.run_dir}')
    if args.theorem_file is not None and not args.theorem_file.is_file():
        parser.error(f'Theorem file does not exist: {args.theorem_file}')
    if args.num_simulations < 1:
        parser.error('--num-simulations must be positive')
    if args.num_sampled_actions < 1:
        parser.error('--num-sampled-actions must be positive')
    if args.tactic_timeout <= 0:
        parser.error('--tactic-timeout must be positive')
    return args


def main() -> None:
    """Run AlphaProof inference for one theorem."""
    args = parse_args()
    try:
        theorem = read_theorem(args)
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = make_config(args)
    network = Network(config)
    load_network_checkpoint(args.run_dir, network)
    network.num_sampled_actions = args.num_sampled_actions

    game = prove(theorem, config, network, disprove=args.disprove)
    if not game.root.is_optimal:
        print('No verified proof found.', file=sys.stderr)
        raise SystemExit(1)

    proof_lines = extract_proof_script(game.root)
    declaration = replace_sorry_proof(
        theorem_for_game(theorem, args.disprove),
        proof_lines,
    )
    print(f'import Mathlib\n\n{declaration}')


if __name__ == '__main__':
    main()
