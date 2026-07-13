import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from alphaproof.core.paths import PROJECT_ROOT

try:
    import wandb as wandb_module
except ImportError:
    wandb_module = None


def project_env_value(name: str) -> str | None:
    """Read one value from the environment or the project-root .env file."""
    environment_value = os.environ.get(name)
    if environment_value:
        return environment_value

    env_path = PROJECT_ROOT / '.env'
    if not env_path.exists():
        return None
    with env_path.open(encoding='utf-8') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip().removeprefix('export ').strip()
            if key != name:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            return value or None
    return None


def wandb_is_configured() -> bool:
    """Return whether a W&B API key is available for automatic logging."""
    return project_env_value('WANDB_API_KEY') is not None


def reset_peak_memory(distributed: bool) -> None:
    """Reset CUDA peak-memory counters immediately before measurement."""
    if not torch.cuda.is_available():
        return
    device_indices = [torch.cuda.current_device()] if distributed else range(
            torch.cuda.device_count()
    )
    for device_ix in device_indices:
        torch.cuda.reset_peak_memory_stats(device_ix)


def peak_memory(rank: int, distributed: bool) -> dict[str, dict[str, int]]:
    """Return peak allocated and reserved CUDA bytes by GPU."""
    if not torch.cuda.is_available():
        return {}
    device_indices = [torch.cuda.current_device()] if distributed else list(
            range(torch.cuda.device_count())
    )
    return {
            f'gpu_{device_ix}': {
                    'rank': rank,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(device_ix),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(device_ix),
            }
            for device_ix in device_indices
    }


def aggregate_model_metrics(
        mode: str,
        rank_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate logical token throughput across execution modes."""
    if mode != 'data':
        metrics = dict(rank_metrics[0])
        metrics['rank_metrics'] = rank_metrics
        return metrics

    prompt_tokens = sum(metrics['prompt_tokens'] for metrics in rank_metrics)
    generated_tokens = sum(metrics['generated_tokens'] for metrics in rank_metrics)
    generation_seconds = max(
            (metrics['generation_seconds'] for metrics in rank_metrics),
            default=0.0,
    )
    stages: dict[str, dict[str, float | int]] = {}
    for metrics in rank_metrics:
        for name, stage_metrics in metrics['stages'].items():
            stage = stages.setdefault(name, {'calls': 0, 'seconds': 0.0})
            stage['calls'] += stage_metrics['calls']
            stage['seconds'] = max(stage['seconds'], stage_metrics['seconds'])
    return {
            'calls': sum(metrics['calls'] for metrics in rank_metrics),
            'prompt_tokens': prompt_tokens,
            'generated_tokens': generated_tokens,
            'generation_seconds': generation_seconds,
            'generated_tokens_per_second': (
                    generated_tokens / generation_seconds if generation_seconds else 0.0
            ),
            'total_tokens_per_second': (
                    (prompt_tokens + generated_tokens) / generation_seconds
                    if generation_seconds else 0.0
            ),
            'max_model_call_batch_size': max(
                    (
                            metrics['max_model_call_batch_size']
                            for metrics in rank_metrics
                    ),
                    default=0,
            ),
            'stages': stages,
            'model_calls': [
                    call
                    for metrics in rank_metrics
                    for call in metrics['model_calls']
            ],
            'rank_metrics': rank_metrics,
    }


def local_run_metrics(
        context: Any,
        model: Any,
        run: Any,
        model_load_seconds: float,
        measured_cleaning_seconds: float,
        total_job_seconds: float,
        preparation_metrics: dict[str, float],
) -> dict[str, Any]:
    """Collect metrics owned by one process after cleaning finishes."""
    return {
            'model_load_seconds': model_load_seconds,
            'measured_cleaning_seconds': measured_cleaning_seconds,
            'total_job_seconds': total_job_seconds,
            'max_source_batch_size': context.max_source_batch_size,
            'model': model.metrics(),
            'timers': dict(run.timers.total),
            'memory': peak_memory(context.rank, context.initialized),
            'preparation_metrics': preparation_metrics,
    }


def cohort_hash(input_path: Path) -> str:
    """Return a stable hash of the exact cleaning input file."""
    digest = hashlib.sha256()
    with input_path.open('rb') as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_had_out_of_memory(run: Any) -> bool:
    """Return whether any source row recorded an out-of-memory error."""
    return any(
            summary.get('error')
            and (
                    'out of memory' in summary['error'].lower()
                    or 'failed to allocate' in summary['error'].lower()
            )
            for summary in run.row_summaries
    )


def aggregate_run_metrics(
        parallelism: str,
        batch_size: int,
        max_model_batch_size: int | None,
        seed: int,
        model: Any,
        torch_dtype: str,
        input_path: Path,
        run: Any,
        rank_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build final rank-0 metrics from the completed cleaning run."""
    measured_seconds = max(
            item['measured_cleaning_seconds'] for item in rank_metrics
    )
    model_metrics = aggregate_model_metrics(
            parallelism,
            [item['model'] for item in rank_metrics],
    )
    metrics = {
            'configuration': {
                    'parallelism': parallelism,
                    'global_batch_size': batch_size,
                    'per_rank_source_batch_sizes': [
                            item['max_source_batch_size'] for item in rank_metrics
                    ],
                    'max_model_batch_size': max_model_batch_size,
                    'seed': seed,
                    'model': model.model_name,
                    'dtype': torch_dtype,
                    'cohort_hash': cohort_hash(input_path),
                    'lsf_job_id': os.environ.get('LSB_JOBID'),
            },
            'source_rows': run.rows_read,
            'successful_rows': (
                    run.rows_read
                    - run.errored_rows
                    - run.missing_information_rows
            ),
            'failed_rows': run.errored_rows,
            'missing_information_rows': run.missing_information_rows,
            'produced_rows': len(run.output_rows),
            'out_of_memory': run_had_out_of_memory(run),
            'source_rows_per_minute': (
                    run.rows_read * 60 / measured_seconds if measured_seconds else 0.0
            ),
            'output_rows_per_minute': (
                    len(run.output_rows) * 60 / measured_seconds
                    if measured_seconds else 0.0
            ),
            'model_load_seconds': max(
                    item['model_load_seconds'] for item in rank_metrics
            ),
            'measured_cleaning_seconds': measured_seconds,
            'total_job_seconds': max(
                    item['total_job_seconds'] for item in rank_metrics
            ),
            'model': model_metrics,
            'stage_timers_by_rank': [item['timers'] for item in rank_metrics],
            'peak_memory': {
                    gpu: values
                    for item in rank_metrics
                    for gpu, values in item['memory'].items()
            },
    }
    preparation_names = {
            name
            for item in rank_metrics
            for name in item['preparation_metrics']
    }
    metrics.update(
            {
                    name: max(
                            item['preparation_metrics'].get(name, 0.0)
                            for item in rank_metrics
                    )
                    for name in preparation_names
            }
    )
    return metrics


def write_metrics(metrics_path: Path | None, metrics: dict[str, Any]) -> None:
    """Write detailed run metrics when an output path was requested."""
    if metrics_path is None:
        return
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)
        metrics_file.write('\n')


def log_wandb(
        enabled: bool,
        project: str,
        group: str | None,
        run_name: str | None,
        metrics: dict[str, Any],
) -> None:
    """Log rank-0 application metrics to W&B when explicitly enabled."""
    if not enabled:
        return
    if wandb_module is None:
        raise RuntimeError('Install the wandb package before using --wandb.')
    api_key = project_env_value('WANDB_API_KEY')
    if api_key is None:
        raise RuntimeError(
                'Set WANDB_API_KEY in the project-root .env file before using W&B.'
        )
    os.environ.setdefault('WANDB_API_KEY', api_key)
    if not wandb_module.login():
        raise RuntimeError('W&B login failed.')
    wandb_run = wandb_module.init(
            project=project,
            group=group,
            name=run_name,
            config=metrics['configuration'],
    )
    wandb_run.log(
            {
                    'source_rows_per_minute': metrics['source_rows_per_minute'],
                    'output_rows_per_minute': metrics['output_rows_per_minute'],
                    'generated_tokens_per_second': metrics['model'][
                            'generated_tokens_per_second'
                    ],
                    'total_tokens_per_second': metrics['model'][
                            'total_tokens_per_second'
                    ],
                    'max_model_call_batch_size': metrics['model'][
                            'max_model_call_batch_size'
                    ],
                    'successful_rows': metrics['successful_rows'],
                    'failed_rows': metrics['failed_rows'],
                    'missing_information_rows': metrics[
                            'missing_information_rows'
                    ],
                    'produced_rows': metrics['produced_rows'],
            }
    )
    wandb_run.summary.update(metrics)
    wandb_run.finish()
