"""Supervised fine-tuning of CodeT5+ on Lean state-action pairs."""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedTokenizerBase,
    RobertaTokenizer,
    T5ForConditionalGeneration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / 'data'
    / 'dataset'
    / 'nemotron_math_proofs_v1_state_action_pairs.jsonl'
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / 'models' / 'Salesforce--codet5p-220m'
RUNS_DIR = PROJECT_ROOT / 'data' / 'runs'


class StateActionDataset(Dataset):
    """In-memory state-action pairs used for shuffled SFT batches."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.pairs[index]


class StateActionCollator:
    """Tokenize state inputs and tactic targets with dynamic padding."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_state_length: int,
        max_action_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_state_length = max_state_length
        self.max_action_length = max_action_length
        if tokenizer.pad_token_id is None:
            raise ValueError('Tokenizer must define a padding token.')
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        states, actions = zip(*batch)
        state_tokens = self.tokenizer(
            list(states),
            max_length=self.max_state_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        action_tokens = self.tokenizer(
            text_target=list(actions),
            max_length=self.max_action_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        labels = action_tokens.input_ids
        labels = labels.masked_fill(labels == self.pad_token_id, -100)
        return {
            'input_ids': state_tokens.input_ids,
            'attention_mask': state_tokens.attention_mask,
            'labels': labels,
        }


def load_state_action_pairs(
    input_path: Path,
    num_pairs: int,
) -> list[tuple[str, str]]:
    """Load exactly num_pairs validated records from the input JSONL."""
    pairs = []
    with input_path.open(encoding='utf-8') as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            state = record.get('state')
            action = record.get('action')
            if not isinstance(state, str) or not isinstance(action, str):
                raise ValueError(
                    f'Expected string state and action on line {line_number}.'
                )
            pairs.append((state, action))
            if len(pairs) == num_pairs:
                break

    if len(pairs) != num_pairs:
        raise ValueError(
            f'Requested {num_pairs:,} pairs, but {input_path} contains only '
            f'{len(pairs):,}.'
        )
    return pairs


def load_tokenizer(model_path: Path) -> RobertaTokenizer:
    """Load the local CodeT5 tokenizer from its vocabulary and merges."""
    return RobertaTokenizer(
        vocab=str(model_path / 'vocab.json'),
        merges=str(model_path / 'merges.txt'),
        model_max_length=512,
    )


def save_training_config(run_dir: Path, args: argparse.Namespace) -> None:
    """Save the command-line training configuration as JSON."""
    config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    (run_dir / 'config.json').write_text(
        json.dumps(config, indent=2) + '\n',
        encoding='utf-8',
    )


def append_metrics(metrics_path: Path, metrics: dict[str, Any]) -> None:
    """Append one epoch's metrics to the run JSONL."""
    with metrics_path.open('a', encoding='utf-8') as metrics_file:
        metrics_file.write(json.dumps(metrics) + '\n')


def resolve_device(device_name: str) -> torch.device:
    """Resolve auto to CUDA when available, otherwise CPU."""
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return device


def train(args: argparse.Namespace) -> Path:
    """Fine-tune CodeT5+ to predict tactics from Lean states."""
    if not args.input.is_file():
        raise FileNotFoundError(f'State-action JSONL does not exist: {args.input}')
    if not args.model.is_dir():
        raise FileNotFoundError(f'Model directory does not exist: {args.model}')

    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists():
        raise FileExistsError(f'Run directory already exists: {run_dir}')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    pairs = load_state_action_pairs(args.input, args.num_pairs)
    dataset = StateActionDataset(pairs)
    tokenizer = load_tokenizer(args.model)
    collator = StateActionCollator(
        tokenizer,
        max_state_length=args.max_state_length,
        max_action_length=args.max_action_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    device = resolve_device(args.device)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        pin_memory=device.type == 'cuda',
    )

    model = T5ForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_dir.mkdir(parents=True)
    save_training_config(run_dir, args)
    metrics_path = run_dir / 'metrics.jsonl'

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        examples_seen = 0

        for step, batch in enumerate(data_loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError('Model did not return a training loss.')
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = batch['input_ids'].shape[0]
            total_loss += outputs.loss.item() * batch_size
            examples_seen += batch_size
            if step % args.log_every == 0:
                print(
                    f'Epoch {epoch}/{args.epochs}, step {step}/{len(data_loader)}, '
                    f'loss {total_loss / examples_seen:.4f}',
                    flush=True,
                )

        epoch_loss = total_loss / examples_seen
        checkpoint_dir = run_dir / f'checkpoint_epoch_{epoch:03d}'
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        append_metrics(metrics_path, {'epoch': epoch, 'loss': epoch_loss})
        print(
            f'Finished epoch {epoch}/{args.epochs}: loss {epoch_loss:.4f}; '
            f'saved {checkpoint_dir}',
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
    """Parse SFT command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune CodeT5+ on Lean state-action pairs.'
    )
    parser.add_argument('run_name', help='Directory name under data/runs.')
    parser.add_argument('--epochs', type=positive_int, required=True)
    parser.add_argument('--num-pairs', type=positive_int, required=True)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--max-state-length', type=positive_int, default=512)
    parser.add_argument('--max-action-length', type=positive_int, default=128)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--log-every', type=positive_int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    if Path(args.run_name).name != args.run_name:
        parser.error('run_name must be a single directory name.')
    if args.learning_rate <= 0:
        parser.error('--learning-rate must be positive.')
    if args.weight_decay < 0:
        parser.error('--weight-decay cannot be negative.')
    if args.max_grad_norm <= 0:
        parser.error('--max-grad-norm must be positive.')
    return args


def main() -> None:
    """Run supervised fine-tuning."""
    args = parse_args()
    run_dir = train(args)
    print(f'Training complete. Checkpoints saved under {run_dir}')


if __name__ == '__main__':
    main()
