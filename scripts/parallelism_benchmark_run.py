import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

from alphaproof.formalize.data_cleaning.data_cleaning import (
        argument_parser,
        clean_rows,
)
from alphaproof.formalize.data_cleaning.parallel import ParallelContext
from alphaproof.formalize.data_cleaning.pipeline import Timers
from alphaproof.formalize.qwen3 import Qwen3


def warmup_records(input_path: Path, rows: int) -> list[dict]:
    """Read a distinct, same-type warmup batch for a benchmark run."""
    if rows == 0:
        return []
    records = []
    seen_ids = set()
    question_type = None
    with input_path.open(encoding='utf-8') as input_file:
        for line in input_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('id') in seen_ids:
                continue
            if question_type is None:
                question_type = record.get('question_type')
            if record.get('question_type') != question_type:
                continue
            seen_ids.add(record.get('id'))
            records.append(record)
            if len(records) == rows:
                break
    return records


def run_warmup(
        context: ParallelContext,
        model: Qwen3,
        input_path: Path,
        rows: int,
) -> dict[str, float]:
    """Warm up one benchmark process before measured cleaning starts."""
    start = perf_counter()
    records = warmup_records(input_path, rows) if context.is_main else None
    records = context.broadcast(records)
    if records:
        context.clean_batch(records, model, Timers())
    return {'warmup_seconds': perf_counter() - start}


def parse_args() -> argparse.Namespace:
    """Parse data-cleaning arguments plus benchmark warmup settings."""
    parser = argument_parser()
    parser.description = 'Run one measured data-cleaning benchmark configuration.'
    parser.add_argument('--warmup-rows', type=int, default=2)
    return parser.parse_args()


def run_benchmark_configuration(args: argparse.Namespace) -> None:
    """Warm up and run one measured parallelism configuration."""
    if args.warmup_rows < 0:
        raise ValueError('warmup_rows cannot be negative.')
    clean_rows(
            args.rows_to_clean,
            input_path=args.input_path,
            output_path=args.output_path,
            metrics_path=args.metrics_path,
            print_summary=args.summary,
            batch_size=args.batch_size,
            max_model_batch_size=args.max_model_batch_size,
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            parallelism=args.parallelism,
            seed=args.seed,
            wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            wandb_run_name=args.wandb_run_name,
            before_measurement=partial(
                    run_warmup,
                    input_path=args.input_path,
                    rows=args.warmup_rows,
            ),
    )


if __name__ == '__main__':
    run_benchmark_configuration(parse_args())
