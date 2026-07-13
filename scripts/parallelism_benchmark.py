import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from alphaproof.formalize.data_cleaning.filter_problems import (
        FILTERED_NUMINA_MATH_1_5_PATH,
)
from alphaproof.formalize.data_cleaning.metrics import wandb_is_configured
from alphaproof.formalize.qwen3 import PARALLELISM_MODES


DEFAULT_BATCH_SIZES = [8, 16, 32, 64, 128, 256]
DEFAULT_MODES = ['balanced', 'tensor', 'data']
BENCHMARK_RUNNER = Path(__file__).with_name('parallelism_benchmark_run.py').resolve()


def comma_list(value: str) -> list[str]:
    """Split a comma-separated environment or CLI value."""
    return [item.strip() for item in value.split(',') if item.strip()]


def load_cohort(
        input_path: Path,
        question_type: str,
        size: int,
        seed: int,
) -> list[dict]:
    """Select deterministic, distinct source rows for every benchmark batch."""
    candidates = []
    seen_ids = set()
    with input_path.open(encoding='utf-8') as input_file:
        for line in input_file:
            record = json.loads(line)
            if record.get('question_type') != question_type:
                continue
            if record['id'] in seen_ids:
                continue
            seen_ids.add(record['id'])
            candidates.append(record)
    if len(candidates) < size:
        raise ValueError(
                f'Need {size} distinct {question_type} rows, found {len(candidates)}.'
        )
    selected = random.Random(seed).sample(candidates, size)
    cohort = []
    for benchmark_ix, record in enumerate(selected):
        benchmark_record = dict(record)
        benchmark_record['benchmark'] = {
                'benchmark_id': f'benchmark_{benchmark_ix:05d}',
                'original_id': record['id'],
        }
        benchmark_record['id'] = f'benchmark_{benchmark_ix:05d}'
        cohort.append(benchmark_record)
    return cohort


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as one fixed JSONL benchmark input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def is_oom(return_code: int, metrics: dict | None, log_text: str) -> bool:
    """Classify caught and fatal accelerator out-of-memory failures."""
    if metrics is not None and metrics.get('out_of_memory'):
        return True
    lowered = log_text.lower()
    return return_code != 0 and (
            'out of memory' in lowered or 'failed to allocate' in lowered
    )


def cleaner_command(
        mode: str,
        batch_size: int,
        input_path: Path,
        output_path: Path,
        metrics_path: Path,
        seed: int,
        max_model_batch_size: int | None,
        wandb: bool,
        wandb_project: str,
        group: str,
) -> list[str]:
    """Build a fresh Python or torchrun cleaner subprocess command."""
    run_args = [
            str(BENCHMARK_RUNNER),
            str(batch_size),
            '--input-path',
            str(input_path),
            '--output-path',
            str(output_path),
            '--metrics-path',
            str(metrics_path),
            '--batch-size',
            str(batch_size),
            '--parallelism',
            mode,
            '--seed',
            str(seed),
            '--warmup-rows',
            '2',
            '--model',
            'qwen3.6-27b',
            '--device',
            'cuda',
            '--torch-dtype',
            'auto',
    ]
    if max_model_batch_size is not None:
        run_args.extend(['--max-model-batch-size', str(max_model_batch_size)])
    if wandb:
        run_args.extend(
                [
                        '--wandb',
                        '--wandb-project',
                        wandb_project,
                        '--wandb-group',
                        group,
                        '--wandb-run-name',
                        f'{mode}-batch-{batch_size}',
                ]
        )
    if mode in {'tensor', 'data'}:
        torchrun = str(Path(sys.executable).with_name('torchrun'))
        return [
                torchrun,
                '--standalone',
                '--nproc-per-node=2',
                *run_args,
        ]
    return [sys.executable, *run_args]


def comparison_row(
        mode: str,
        batch_size: int,
        return_code: int,
        status: str,
        metrics: dict | None,
) -> dict[str, Any]:
    """Build one compact JSON/CSV comparison row."""
    model_metrics = metrics.get('model', {}) if metrics else {}
    memory = metrics.get('peak_memory', {}) if metrics else {}
    return {
            'parallelism': mode,
            'global_batch_size': batch_size,
            'status': status,
            'return_code': return_code,
            'source_rows_per_minute': (
                    metrics.get('source_rows_per_minute') if metrics else None
            ),
            'output_rows_per_minute': (
                    metrics.get('output_rows_per_minute') if metrics else None
            ),
            'generated_tokens_per_second': model_metrics.get(
                    'generated_tokens_per_second'
            ),
            'total_tokens_per_second': model_metrics.get('total_tokens_per_second'),
            'max_model_call_batch_size': model_metrics.get(
                    'max_model_call_batch_size'
            ),
            'peak_allocated_bytes': max(
                    (
                            gpu_metrics['peak_allocated_bytes']
                            for gpu_metrics in memory.values()
                    ),
                    default=None,
            ),
            'peak_reserved_bytes': max(
                    (
                            gpu_metrics['peak_reserved_bytes']
                            for gpu_metrics in memory.values()
                    ),
                    default=None,
            ),
            'peak_memory': memory or None,
            'metrics_path': (
                    f'metrics/{mode}_batch_{batch_size}.json' if metrics else None
            ),
    }


def write_comparison(output_dir: Path, rows: list[dict]) -> None:
    """Write combined detailed JSON and compact CSV artifacts."""
    largest_successful = {}
    first_oom = {}
    for row in rows:
        mode = row['parallelism']
        if row['status'] == 'success':
            largest_successful[mode] = row['global_batch_size']
        elif row['status'] == 'oom' and mode not in first_oom:
            first_oom[mode] = row['global_batch_size']
    comparison = {
            'configurations': rows,
            'largest_successful_batch': largest_successful,
            'first_oom_batch': first_oom,
    }
    with (output_dir / 'comparison.json').open('w', encoding='utf-8') as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write('\n')

    fieldnames = [key for key in rows[0] if key != 'peak_memory'] if rows else []
    with (output_dir / 'comparison.csv').open(
            'w',
            encoding='utf-8',
            newline='',
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def run_benchmark(args: argparse.Namespace) -> None:
    """Run every mode and batch size in an isolated subprocess."""
    if any(mode not in PARALLELISM_MODES for mode in args.modes):
        raise ValueError(f'Unknown mode in {args.modes}.')
    if args.cohort_size < max(args.batch_sizes):
        raise ValueError('cohort-size must cover the largest batch without copying rows.')

    output_dir = args.output_dir
    for directory in ['inputs', 'outputs', 'metrics', 'logs']:
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(
            args.input_path,
            args.question_type,
            args.cohort_size,
            args.seed,
    )
    input_paths = {}
    for batch_size in args.batch_sizes:
        input_path = output_dir / 'inputs' / f'batch_{batch_size}.jsonl'
        write_jsonl(input_path, cohort[:batch_size])
        input_paths[batch_size] = input_path

    rows = []
    for mode in args.modes:
        mode_oom = False
        for batch_size in args.batch_sizes:
            if mode_oom:
                continue
            output_path = output_dir / 'outputs' / f'{mode}_batch_{batch_size}.jsonl'
            metrics_path = output_dir / 'metrics' / f'{mode}_batch_{batch_size}.json'
            log_path = output_dir / 'logs' / f'{mode}_batch_{batch_size}.log'
            command = cleaner_command(
                    mode,
                    batch_size,
                    input_paths[batch_size],
                    output_path,
                    metrics_path,
                    args.seed,
                    args.max_model_batch_size,
                    args.wandb,
                    args.wandb_project,
                    args.group,
            )
            environment = dict(os.environ)
            environment['OMP_NUM_THREADS'] = '4' if mode in {'tensor', 'data'} else '8'
            with log_path.open('w', encoding='utf-8') as log_file:
                completed = subprocess.run(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        check=False,
                )
            metrics = None
            if metrics_path.exists():
                with metrics_path.open(encoding='utf-8') as metrics_file:
                    metrics = json.load(metrics_file)
            log_text = log_path.read_text(encoding='utf-8', errors='replace')
            oom = is_oom(completed.returncode, metrics, log_text)
            status = 'oom' if oom else (
                    'success'
                    if completed.returncode == 0 and metrics is not None
                    else 'error'
            )
            rows.append(
                    comparison_row(
                            mode,
                            batch_size,
                            completed.returncode,
                            status,
                            metrics,
                    )
            )
            write_comparison(output_dir, rows)
            mode_oom = oom


def parse_args() -> argparse.Namespace:
    """Parse CLI and environment overrides for the benchmark sweep."""
    job_id = os.environ.get('LSB_JOBID', 'local')
    default_output = Path(
            os.environ.get(
                    'BENCHMARK_OUTPUT_DIR',
                    f'runs/parallelism_benchmark/{job_id}',
            )
    )
    default_batches = [
            int(value)
            for value in comma_list(
                    os.environ.get('BENCHMARK_BATCH_SIZES', '8,16,32,64,128,256')
            )
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=Path, default=FILTERED_NUMINA_MATH_1_5_PATH)
    parser.add_argument('--output-dir', type=Path, default=default_output)
    parser.add_argument(
            '--modes',
            nargs='+',
            default=comma_list(os.environ.get('BENCHMARK_MODES', 'balanced,tensor,data')),
    )
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=default_batches)
    parser.add_argument(
            '--cohort-size',
            type=int,
            default=int(
                    os.environ.get('BENCHMARK_COHORT_SIZE', max(default_batches))
            ),
    )
    parser.add_argument(
            '--question-type',
            default=os.environ.get('BENCHMARK_QUESTION_TYPE', 'proof'),
    )
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '0')))
    parser.add_argument(
            '--max-model-batch-size',
            type=int,
            default=(
                    int(os.environ['MAX_MODEL_BATCH_SIZE'])
                    if 'MAX_MODEL_BATCH_SIZE' in os.environ
                    else None
            ),
    )
    parser.add_argument(
            '--wandb',
            action=argparse.BooleanOptionalAction,
            default=wandb_is_configured(),
    )
    parser.add_argument(
            '--wandb-project',
            default=os.environ.get('WANDB_PROJECT', 'alphaproof-data-cleaning'),
    )
    parser.add_argument('--group', default=job_id)
    return parser.parse_args()


if __name__ == '__main__':
    run_benchmark(parse_args())
