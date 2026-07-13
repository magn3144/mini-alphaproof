import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from alphaproof.formalize.data_cleaning.cleaning_run import (
        clean_dataset,
        data_cleaning_summary,
)
from alphaproof.formalize.data_cleaning.filter_problems import (
        FILTERED_NUMINA_MATH_1_5_PATH,
)
from alphaproof.formalize.data_cleaning.metrics import (
        aggregate_run_metrics,
        local_run_metrics,
        log_wandb,
        reset_peak_memory,
        wandb_is_configured,
        write_metrics,
)
from alphaproof.formalize.data_cleaning.model import (
        CLEANING_MODEL_ALIASES,
        DEFAULT_CLEANING_MODEL,
        load_cleaning_model,
)
from alphaproof.formalize.data_cleaning.parallel import (
        ParallelContext,
        initialize_parallelism,
)
from alphaproof.formalize.data_cleaning.paths import CLEANED_NUMINA_MATH_1_5_PATH
from alphaproof.formalize.qwen3 import PARALLELISM_MODES, Qwen3


def load_model_for_run(
        context: ParallelContext,
        model_name: str,
        device: str | None,
        torch_dtype: str,
        quantization: str | None,
        seed: int,
        max_model_batch_size: int | None,
) -> tuple[Qwen3, float]:
    """Load the cleaning model on the device assigned to this process."""
    if context.initialized:
        device = f'cuda:{context.local_rank}'
    start = perf_counter()
    model = load_cleaning_model(
            model_name,
            device,
            torch_dtype,
            quantization,
            context.mode,
            seed,
            max_model_batch_size,
    )
    load_seconds = perf_counter() - start
    if context.is_main:
        print(
                f'Loaded {model.model_name} from {model.model_dir} '
                f'with parallelism={context.mode}.'
        )
    return model, load_seconds


def clean_rows(
        rows_to_clean: int,
        input_path: Path = FILTERED_NUMINA_MATH_1_5_PATH,
        output_path: Path = CLEANED_NUMINA_MATH_1_5_PATH,
        print_summary: bool = False,
        batch_size: int = 4,
        model: str = DEFAULT_CLEANING_MODEL,
        device: str | None = None,
        torch_dtype: str = 'auto',
        quantization: str | None = None,
        parallelism: str = 'none',
        seed: int = 0,
        max_model_batch_size: int | None = None,
        metrics_path: Path | None = None,
        wandb: bool | None = None,
        wandb_project: str = 'alphaproof-data-cleaning',
        wandb_group: str | None = None,
        wandb_run_name: str | None = None,
        before_measurement: Callable[
                [ParallelContext, Qwen3],
                dict[str, float],
        ] | None = None,
) -> dict[str, Any] | None:
    """Load a model and clean source rows with the requested parallelism."""
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1.')
    if rows_to_clean < 0:
        raise ValueError('rows_to_clean cannot be negative.')

    job_start = perf_counter()
    context = initialize_parallelism(parallelism)
    try:
        qwen, model_load_seconds = load_model_for_run(
                context,
                model,
                device,
                torch_dtype,
                quantization,
                seed,
                max_model_batch_size,
        )
        preparation_metrics = (
                before_measurement(context, qwen)
                if before_measurement is not None
                else {}
        )
        qwen.reset_metrics()
        context.max_source_batch_size = 0
        reset_peak_memory(context.initialized)

        run, measured_seconds = clean_dataset(
                context,
                qwen,
                rows_to_clean,
                input_path,
                output_path,
                batch_size,
        )
        local_metrics = local_run_metrics(
                context,
                qwen,
                run,
                model_load_seconds,
                measured_seconds,
                perf_counter() - job_start,
                preparation_metrics,
        )
        rank_metrics = context.gather(local_metrics)
        if not context.is_main:
            return None
        assert rank_metrics is not None

        metrics = aggregate_run_metrics(
                parallelism,
                batch_size,
                max_model_batch_size,
                seed,
                qwen,
                torch_dtype,
                input_path,
                run,
                rank_metrics,
        )
        write_metrics(metrics_path, metrics)
        log_wandb(
                wandb if wandb is not None else wandb_is_configured(),
                wandb_project,
                wandb_group,
                wandb_run_name,
                metrics,
        )
        if print_summary:
            print(data_cleaning_summary(run))
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics
    finally:
        context.close()


def argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for data cleaning."""
    parser = argparse.ArgumentParser()
    parser.add_argument('rows_to_clean', type=int)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--input-path', type=Path, default=FILTERED_NUMINA_MATH_1_5_PATH)
    parser.add_argument('--output-path', type=Path, default=CLEANED_NUMINA_MATH_1_5_PATH)
    parser.add_argument('--metrics-path', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-model-batch-size', type=int)
    parser.add_argument('--parallelism', choices=sorted(PARALLELISM_MODES), default='none')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
            '--model',
            choices=sorted(CLEANING_MODEL_ALIASES),
            default=DEFAULT_CLEANING_MODEL,
    )
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--quantization', choices=['4bit', '8bit'])
    parser.add_argument(
            '--wandb',
            action=argparse.BooleanOptionalAction,
            default=wandb_is_configured(),
    )
    parser.add_argument('--wandb-project', default='alphaproof-data-cleaning')
    parser.add_argument('--wandb-group')
    parser.add_argument('--wandb-run-name')
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data cleaning."""
    return argument_parser().parse_args()


if __name__ == '__main__':
    args = parse_args()
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
    )
