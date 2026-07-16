"""Supervised fine-tuning of AlphaProof's CodeT5 policy and value network."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, RobertaTokenizer

from alphaproof.core.config import Config
from alphaproof.core.network import Network
from alphaproof.core.paths import DATASET_DIR, MODELS_DIR, RUNS_DIR


DEFAULT_TRAIN_INPUT = (
    DATASET_DIR / 'leantree_mathlib_state_action_pairs.train.jsonl'
)
DEFAULT_VALIDATION_INPUT = (
    DATASET_DIR / 'leantree_mathlib_state_action_pairs.validation.jsonl'
)
DEFAULT_MODEL_PATH = MODELS_DIR / 'Salesforce--codet5p-220m'


@dataclass(frozen=True)
class TrainingExample:
    """One LeanTree transition with an AlphaProof value target."""

    state: str
    action: str
    value_target: float


@dataclass(frozen=True)
class DatasetStats:
    """Counts collected while validating and filtering an input JSONL."""

    records_read: int
    examples_kept: int
    states_too_long: int
    actions_too_long: int


class TransitionDataset(Dataset[TrainingExample]):
    """In-memory LeanTree transitions used for shuffled SFT batches."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


class TransitionCollator:
    """Tokenize already length-checked states and tactics with dynamic padding."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(
        self, batch: list[TrainingExample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        states = [example.state for example in batch]
        actions = [example.action for example in batch]
        observations = self.tokenizer(
            states,
            padding=True,
            truncation=False,
            return_tensors='pt',
        ).input_ids
        action_tokens = self.tokenizer(
            text_target=actions,
            padding=True,
            truncation=False,
            return_tensors='pt',
        ).input_ids
        value_targets = torch.tensor(
            [example.value_target for example in batch],
            dtype=torch.float32,
        )
        return observations.long(), action_tokens.long(), value_targets


def token_length(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count tokens without allowing the tokenizer to truncate the text."""
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    return len(encoded['input_ids'])


def load_examples(
    path: Path,
    tokenizer: PreTrainedTokenizerBase,
    max_state_length: int,
    max_action_length: int,
    record_limit: int | None,
) -> tuple[list[TrainingExample], DatasetStats]:
    """Validate records and reject examples that would require truncation."""
    examples = []
    records_read = 0
    states_too_long = 0
    actions_too_long = 0

    with path.open(encoding='utf-8') as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if record_limit is not None and records_read >= record_limit:
                break
            records_read += 1
            record = json.loads(line)
            state = record.get('state')
            action = record.get('action')
            proof_depth = record.get('proof_depth')
            if not isinstance(state, str) or not state.strip():
                raise ValueError(
                    f'Expected a non-empty string state on line {line_number} of {path}.'
                )
            if not isinstance(action, str) or not action.strip():
                raise ValueError(
                    f'Expected a non-empty string action on line {line_number} of {path}.'
                )
            if (
                not isinstance(proof_depth, int)
                or isinstance(proof_depth, bool)
                or proof_depth < 1
            ):
                raise ValueError(
                    f'Expected a positive integer proof_depth on line '
                    f'{line_number} of {path}.'
                )

            state_length = token_length(tokenizer, state)
            action_length = token_length(tokenizer, action)
            if state_length > max_state_length:
                states_too_long += 1
                continue
            if action_length > max_action_length:
                actions_too_long += 1
                continue
            examples.append(
                TrainingExample(
                    state=state.strip(),
                    action=action.strip(),
                    value_target=-float(proof_depth),
                )
            )

    return examples, DatasetStats(
        records_read=records_read,
        examples_kept=len(examples),
        states_too_long=states_too_long,
        actions_too_long=actions_too_long,
    )


def resolve_device(name: str) -> torch.device:
    """Resolve auto to CUDA, then Apple Silicon MPS, then CPU."""
    if name != 'auto':
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS was requested but is not available.')
    return device


def make_network(args: argparse.Namespace, device: torch.device) -> Network:
    """Construct the existing AlphaProof network with SFT hyperparameters."""
    config = Config(
        num_simulations=1,
        batch_size=args.batch_size,
        num_actors=1,
        num_games=1,
        lr=args.learning_rate,
        tokenizer_model=str(args.model),
    )
    config.sequence_length = args.max_state_length
    config.value_weight = args.value_weight
    tokenizer = RobertaTokenizer(
        vocab=str(args.model / 'vocab.json'),
        merges=str(args.model / 'merges.txt'),
        model_max_length=args.max_state_length,
    )
    with patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer):
        network = Network(config)
    network.device = device
    network.to(device=device, dtype=torch.bfloat16)
    return network


def batch_losses(
    network: Network,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """Compute joint policy and proof-depth value losses."""
    observations, actions, value_targets = batch
    observations = observations.to(network.device)
    actions = actions.to(network.device)
    value_targets = value_targets.to(network.device)
    output = network(observations, actions)
    policy_loss = output.policy_loss.float()
    value_loss = network.value_loss(
        output.value_logits.float(),
        value_targets.float(),
    )
    total_loss = policy_loss + value_weight * value_loss
    return total_loss, policy_loss, value_loss, output


def train_epoch(
    network: Network,
    data_loader: DataLoader[Any],
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    """Train for one epoch and return mean component losses."""
    network.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    examples_seen = 0

    for step, batch in enumerate(data_loader, start=1):
        loss, policy_loss, value_loss, _ = batch_losses(
            network, batch, args.value_weight
        )
        network.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
        network.optimizer.step()

        batch_size = batch[0].shape[0]
        examples_seen += batch_size
        total_loss += loss.detach().item() * batch_size
        total_policy_loss += policy_loss.detach().item() * batch_size
        total_value_loss += value_loss.detach().item() * batch_size
        if step % args.log_every == 0:
            print(
                f'Epoch {epoch}/{args.epochs}, step {step}/{len(data_loader)}, '
                f'loss {total_loss / examples_seen:.4f}',
                flush=True,
            )

    return {
        'loss': total_loss / examples_seen,
        'policy_loss': total_policy_loss / examples_seen,
        'value_loss': total_value_loss / examples_seen,
    }


def validate(
    network: Network,
    data_loader: DataLoader[Any],
    value_weight: float,
) -> dict[str, float]:
    """Evaluate losses, teacher-forced tactic accuracy, and value error."""
    network.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_value_error = 0.0
    exact_tactics = 0
    examples_seen = 0
    pad_token_id = network.model.config.pad_token_id

    with torch.no_grad():
        for batch in data_loader:
            loss, policy_loss, value_loss, output = batch_losses(
                network, batch, value_weight
            )
            actions = batch[1].to(network.device)
            value_targets = batch[2].to(network.device)
            predictions = output.policy_logits.argmax(dim=-1)
            if pad_token_id is None:
                correct = predictions.eq(actions).all(dim=-1)
            else:
                token_correct = predictions.eq(actions) | actions.eq(pad_token_id)
                correct = token_correct.all(dim=-1)

            value_probabilities = torch.softmax(
                output.value_logits.float(),
                dim=-1,
            )
            predicted_values = (
                value_probabilities * network.value_bins.float()
            ).sum(dim=-1)
            batch_size = actions.shape[0]
            examples_seen += batch_size
            exact_tactics += int(correct.sum().item())
            total_value_error += (
                torch.abs(predicted_values - value_targets).sum().item()
            )
            total_loss += loss.item() * batch_size
            total_policy_loss += policy_loss.item() * batch_size
            total_value_loss += value_loss.item() * batch_size

    return {
        'loss': total_loss / examples_seen,
        'policy_loss': total_policy_loss / examples_seen,
        'value_loss': total_value_loss / examples_seen,
        'teacher_forced_tactic_accuracy': exact_tactics / examples_seen,
        'value_mae': total_value_error / examples_seen,
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert paths in an argparse namespace to JSON-compatible strings."""
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Append one epoch of training and validation metrics."""
    with path.open('a', encoding='utf-8') as metrics_file:
        metrics_file.write(json.dumps(metrics) + '\n')


def save_checkpoint(
    run_dir: Path,
    network: Network,
    epoch: int,
    args: argparse.Namespace,
) -> Path:
    """Save resumable training state and AlphaProof-compatible parameters."""
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(
        {
            'epoch': epoch,
            'network_params': network.params,
            'optimizer_state_dict': network.optimizer.state_dict(),
            'args': serializable_args(args),
        },
        checkpoint_path,
    )
    parameters_path = run_dir / 'network_params.pt'
    temporary_path = run_dir / 'network_params.tmp'
    torch.save(network.params, temporary_path)
    temporary_path.replace(parameters_path)
    return checkpoint_path


def save_network_source(
    run_dir: Path,
    network: Network,
    base_model_dir: Path,
) -> Path:
    """Save lightweight model metadata for constructing the trained Network."""
    model_source_dir = run_dir / 'model_source'
    model_source_dir.mkdir(exist_ok=True)
    network.tokenizer.save_pretrained(model_source_dir)

    config = json.loads(
        (base_model_dir / 'config.json').read_text(encoding='utf-8')
    )
    config.pop('torch_dtype', None)
    config['dtype'] = 'bfloat16'
    (model_source_dir / 'config.json').write_text(
        json.dumps(config, indent=2) + '\n',
        encoding='utf-8',
    )

    source_weights = (base_model_dir / 'pytorch_model.bin').resolve()
    linked_weights = model_source_dir / 'pytorch_model.bin'
    if linked_weights.exists() or linked_weights.is_symlink():
        linked_weights.unlink()
    linked_weights.symlink_to(source_weights)
    return model_source_dir


def load_latest_checkpoint(run_dir: Path, network: Network) -> int:
    """Restore the most recent complete SFT epoch."""
    checkpoints = sorted((run_dir / 'checkpoints').glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        raise FileNotFoundError(f'No SFT checkpoints found under {run_dir}.')
    checkpoint = torch.load(
        checkpoints[-1],
        map_location=network.device,
        weights_only=False,
    )
    network.params = checkpoint['network_params']
    network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return int(checkpoint['epoch'])


def print_dataset_stats(name: str, stats: DatasetStats) -> None:
    """Report exactly how many overlength records were rejected."""
    print(
        f'{name}: read {stats.records_read:,}, kept {stats.examples_kept:,}, '
        f'skipped {stats.states_too_long:,} long states and '
        f'{stats.actions_too_long:,} long actions',
        flush=True,
    )


def train(args: argparse.Namespace) -> Path:
    """Run joint supervised policy and value training."""
    run_dir = RUNS_DIR / args.run_name
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f'SFT run does not exist: {run_dir}')
    elif run_dir.exists():
        raise FileExistsError(f'SFT run already exists: {run_dir}')
    else:
        run_dir.mkdir(parents=True)
        (run_dir / 'config.json').write_text(
            json.dumps(serializable_args(args), indent=2) + '\n',
            encoding='utf-8',
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    network = make_network(args, device)
    train_examples, train_stats = load_examples(
        args.train_input,
        network.tokenizer,
        args.max_state_length,
        args.max_action_length,
        args.num_pairs,
    )
    validation_examples, validation_stats = load_examples(
        args.validation_input,
        network.tokenizer,
        args.max_state_length,
        args.max_action_length,
        args.num_validation_pairs,
    )
    print_dataset_stats('Train', train_stats)
    print_dataset_stats('Validation', validation_stats)
    if not train_examples:
        raise ValueError('No training examples remain after length filtering.')
    if not validation_examples:
        raise ValueError('No validation examples remain after length filtering.')

    collator = TransitionCollator(network.tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TransitionDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        pin_memory=device.type == 'cuda',
    )
    validation_loader = DataLoader(
        TransitionDataset(validation_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == 'cuda',
    )

    first_epoch = 1
    if args.resume:
        first_epoch = load_latest_checkpoint(run_dir, network) + 1
    if first_epoch > args.epochs:
        model_source_dir = save_network_source(run_dir, network, args.model)
        print(
            f'Run already completed epoch {first_epoch - 1}; '
            f'model metadata is available at {model_source_dir}',
            flush=True,
        )
        return run_dir

    metrics_path = run_dir / 'metrics.jsonl'
    print(f'Training on {device}', flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        training_metrics = train_epoch(network, train_loader, args, epoch)
        validation_metrics = validate(
            network, validation_loader, args.value_weight
        )
        metrics = {
            'epoch': epoch,
            'train': training_metrics,
            'validation': validation_metrics,
        }
        append_metrics(metrics_path, metrics)
        checkpoint_path = save_checkpoint(run_dir, network, epoch, args)
        print(
            f"Finished epoch {epoch}/{args.epochs}: train loss "
            f"{training_metrics['loss']:.4f}, validation loss "
            f"{validation_metrics['loss']:.4f}; saved {checkpoint_path}",
            flush=True,
        )

    model_source_dir = save_network_source(run_dir, network, args.model)
    print(
        f'Saved AlphaProof model metadata to {model_source_dir}',
        flush=True,
    )
    return run_dir


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse supervised fine-tuning command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaProof on LeanTree transitions.'
    )
    parser.add_argument('run_name', help='Directory name under data/runs.')
    parser.add_argument('--train-input', type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument(
        '--validation-input', type=Path, default=DEFAULT_VALIDATION_INPUT
    )
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--epochs', type=positive_int, required=True)
    parser.add_argument('--num-pairs', type=positive_int)
    parser.add_argument('--num-validation-pairs', type=positive_int)
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--value-weight', type=float, default=0.001)
    parser.add_argument('--max-state-length', type=positive_int, default=640)
    parser.add_argument('--max-action-length', type=positive_int, default=128)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--log-every', type=positive_int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if Path(args.run_name).name != args.run_name:
        parser.error('run_name must be a single directory name')
    if not args.train_input.is_file():
        parser.error(f'training JSONL does not exist: {args.train_input}')
    if not args.validation_input.is_file():
        parser.error(
            f'validation JSONL does not exist: {args.validation_input}'
        )
    if not args.model.is_dir():
        parser.error(f'model directory does not exist: {args.model}')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be positive')
    if args.value_weight < 0:
        parser.error('--value-weight cannot be negative')
    if args.max_grad_norm <= 0:
        parser.error('--max-grad-norm must be positive')
    return args


def main() -> None:
    """Run supervised fine-tuning."""
    run_dir = train(parse_args())
    print(f'Training complete. Outputs saved under {run_dir}', flush=True)


if __name__ == '__main__':
    main()
