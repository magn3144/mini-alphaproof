import argparse
from pathlib import Path

from alphaproof.formalize.data_cleaning.cleaning_run import (
        clean_dataset,
        data_cleaning_summary,
)
from alphaproof.formalize.data_cleaning.filter_problems import (
        FILTERED_NUMINA_MATH_1_5_PATH,
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
) -> Qwen3:
    """Load the cleaning model on the device assigned to this process."""
    if context.initialized:
        device = f'cuda:{context.local_rank}'
    model = load_cleaning_model(
            model_name,
            device,
            torch_dtype,
            quantization,
            context.mode,
            seed,
            max_model_batch_size,
    )
    if context.is_main:
        print(
                f'Loaded {model.model_name} from {model.model_dir} '
                f'with parallelism={context.mode}.'
        )
    return model


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
) -> None:
    """Load a model and clean source rows with the requested parallelism."""
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1.')
    if rows_to_clean < 0:
        raise ValueError('rows_to_clean cannot be negative.')

    context = initialize_parallelism(parallelism)
    try:
        qwen = load_model_for_run(
                context,
                model,
                device,
                torch_dtype,
                quantization,
                seed,
                max_model_batch_size,
        )
        run = clean_dataset(
                context,
                qwen,
                rows_to_clean,
                input_path,
                output_path,
                batch_size,
        )
        if context.is_main and print_summary:
            print(data_cleaning_summary(run))
    finally:
        context.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for data cleaning."""
    parser = argparse.ArgumentParser()
    parser.add_argument('rows_to_clean', type=int)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--input-path', type=Path, default=FILTERED_NUMINA_MATH_1_5_PATH)
    parser.add_argument('--output-path', type=Path, default=CLEANED_NUMINA_MATH_1_5_PATH)
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    clean_rows(
            args.rows_to_clean,
            input_path=args.input_path,
            output_path=args.output_path,
            print_summary=args.summary,
            batch_size=args.batch_size,
            max_model_batch_size=args.max_model_batch_size,
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            parallelism=args.parallelism,
            seed=args.seed,
    )
