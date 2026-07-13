import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist

from alphaproof.formalize.data_cleaning.model import load_cleaning_model
from alphaproof.formalize.data_cleaning.parallel import (
        ParallelContext,
        initialize_parallelism,
)


MODES = ['none', 'balanced', 'tensor', 'data']
BATCH_SIZES = [8, 16, 32, 64, 128, 256]
WARMUP_TOKENS = 8
MODULE_NAME = 'scripts.parallelism_benchmark'

TARGET_FILLER = (
        'Mathematics studies patterns in numbers, shapes, functions, and logical '
        'arguments. A careful proof states its assumptions, explains each deduction, '
        'and reaches a conclusion without relying on missing information. '
)
TARGET_TEXT = (TARGET_FILLER * 6)[:1000]
PROMPT_START = (
        'Output exactly the text between <text> and </text>. Preserve every character, '
        'including punctuation and spaces. Do not add an introduction, explanation, or '
        'closing sentence. Begin with the first character inside <text> and finish with '
        'the last character inside </text>. '
)
PROMPT_PADDING_TEXT = (
        'Copy the supplied passage verbatim. Accuracy is more important than style. '
)
PROMPT_END = '\n<text>' + TARGET_TEXT + '</text>'
PROMPT_PADDING = (
        PROMPT_PADDING_TEXT
        * ((2000 - len(PROMPT_START) - len(PROMPT_END)) // len(PROMPT_PADDING_TEXT) + 1)
)[:2000 - len(PROMPT_START) - len(PROMPT_END)]
PROMPT = PROMPT_START + PROMPT_PADDING + PROMPT_END
assert len(TARGET_TEXT) == 1000
assert len(PROMPT) == 2000
assert TARGET_TEXT in PROMPT


def synchronize_cuda(context: ParallelContext) -> None:
    """Wait for CUDA work used by this process to finish."""
    if context.initialized:
        torch.cuda.synchronize(context.local_rank)
        return
    for device_ix in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_ix)


def generate_target(
        model: Any,
        encoded: dict[str, torch.Tensor],
        tokenizer: Any,
        target_token_ids: list[int],
) -> torch.Tensor:
    """Generate the supplied target tokens for every encoded prompt."""
    prompt_length = encoded['input_ids'].shape[-1]

    def allowed_tokens(_batch_ix: int, input_ids: torch.Tensor) -> list[int]:
        generated_length = input_ids.shape[-1] - prompt_length
        return [target_token_ids[generated_length]]

    with torch.inference_mode():
        return model.generate(
                **encoded,
                min_new_tokens=len(target_token_ids),
                max_new_tokens=len(target_token_ids),
                do_sample=False,
                pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                ),
                prefix_allowed_tokens_fn=allowed_tokens,
        )


def aggregate_measurement(
        context: ParallelContext,
        elapsed_seconds: float,
        generated_tokens: int,
) -> tuple[float, int]:
    """Return global elapsed time and logical generated-token count."""
    if not context.initialized:
        return elapsed_seconds, generated_tokens

    elapsed = torch.tensor(
            elapsed_seconds,
            dtype=torch.float64,
            device=f'cuda:{context.local_rank}',
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    if context.mode == 'data':
        tokens = torch.tensor(
                generated_tokens,
                dtype=torch.int64,
                device=f'cuda:{context.local_rank}',
        )
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
        generated_tokens = int(tokens.item())

    return float(elapsed.item()), generated_tokens


def collect_completions(
        context: ParallelContext,
        completions: list[str],
        batch_size: int,
) -> None:
    """Verify and collect decoded completions like the cleaning pipeline."""
    if context.mode == 'tensor':
        gathered: list[list[str] | None] = [None] * context.world_size
        dist.all_gather_object(gathered, completions)
        if any(rank_completions != completions for rank_completions in gathered):
            raise RuntimeError('Tensor-parallel ranks produced different completions.')
        return

    if context.mode != 'data':
        return

    indexed_completions = list(zip(
            range(context.rank, batch_size, context.world_size),
            completions,
    ))
    gathered: list[list[tuple[int, str]] | None] | None = None
    if context.is_main:
        gathered = [None] * context.world_size
    dist.gather_object(indexed_completions, gathered, dst=0)
    if context.is_main:
        assert gathered is not None
        merged: list[str | None] = [None] * batch_size
        for rank_completions in gathered:
            assert rank_completions is not None
            for global_ix, completion in rank_completions:
                merged[global_ix] = completion
        if any(completion is None for completion in merged):
            raise RuntimeError('Missing gathered completion.')


def run_worker(mode: str, batch_size: int, result_path: Path) -> None:
    """Run one isolated benchmark configuration."""
    context = initialize_parallelism(mode)
    try:
        device = f'cuda:{context.local_rank}' if context.initialized else 'cuda'
        qwen = load_cleaning_model(
                'qwen3.6-27b',
                device,
                'auto',
                None,
                parallelism=mode,
                seed=0,
        )
        tokenizer = qwen.tokenizer
        model = qwen.model
        assert tokenizer is not None
        assert model is not None

        local_batch_size = (
                len(range(context.rank, batch_size, context.world_size))
                if mode == 'data'
                else batch_size
        )
        target_token_ids: list[int] = tokenizer.encode(
                TARGET_TEXT,
                add_special_tokens=False,
        )
        if tokenizer.decode(target_token_ids) != TARGET_TEXT:
            raise RuntimeError('Target text does not round-trip through the tokenizer.')

        warmup_encoded = qwen.encode_prompts(
                [PROMPT] * local_batch_size,
                tokenizer,
        )
        warmup_encoded = {
                name: tensor.to(qwen.device)
                for name, tensor in warmup_encoded.items()
        }
        generate_target(
                model,
                warmup_encoded,
                tokenizer,
                target_token_ids[:WARMUP_TOKENS],
        )
        synchronize_cuda(context)
        if context.initialized:
            dist.barrier()

        end_to_end_start = perf_counter()
        encoded = qwen.encode_prompts([PROMPT] * local_batch_size, tokenizer)
        encoded = {
                name: tensor.to(qwen.device)
                for name, tensor in encoded.items()
        }
        prompt_length = encoded['input_ids'].shape[-1]
        prompt_tokens_per_item = int(encoded['attention_mask'][0].sum().item())

        synchronize_cuda(context)
        generation_start = perf_counter()
        generated = generate_target(model, encoded, tokenizer, target_token_ids)
        synchronize_cuda(context)
        generation_seconds = perf_counter() - generation_start

        completion_ids = generated[:, prompt_length:]
        expected_shape = (local_batch_size, len(target_token_ids))
        if tuple(completion_ids.shape) != expected_shape:
            raise RuntimeError(
                    f'Expected completion shape {expected_shape}, '
                    f'got {tuple(completion_ids.shape)}.'
            )
        expected_ids = torch.tensor(
                target_token_ids,
                dtype=completion_ids.dtype,
                device=completion_ids.device,
        ).expand_as(completion_ids)
        if not torch.equal(completion_ids, expected_ids):
            raise RuntimeError('Generated output did not match the target text.')

        completions = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
        )
        if any(completion != TARGET_TEXT for completion in completions):
            raise RuntimeError('Decoded output did not match the target text.')
        collect_completions(context, completions, batch_size)
        end_to_end_seconds = perf_counter() - end_to_end_start

        generation_seconds, generated_tokens = aggregate_measurement(
                context,
                generation_seconds,
                int(completion_ids.numel()),
        )
        end_to_end_seconds, _ = aggregate_measurement(
                context,
                end_to_end_seconds,
                int(completion_ids.numel()),
        )
        if context.is_main:
            generated_characters = batch_size * len(TARGET_TEXT)
            result = {
                    'parallelism': mode,
                    'global_batch_size': batch_size,
                    'per_rank_batch_size': local_batch_size,
                    'prompt_characters': len(PROMPT),
                    'prompt_tokens_per_item': prompt_tokens_per_item,
                    'generated_characters_per_item': len(TARGET_TEXT),
                    'generated_tokens_per_item': len(target_token_ids),
                    'generated_characters': generated_characters,
                    'generated_tokens': generated_tokens,
                    'generation_seconds': generation_seconds,
                    'generation_characters_per_second': (
                            generated_characters / generation_seconds
                    ),
                    'generation_tokens_per_second': (
                            generated_tokens / generation_seconds
                    ),
                    'end_to_end_seconds': end_to_end_seconds,
                    'end_to_end_characters_per_second': (
                            generated_characters / end_to_end_seconds
                    ),
                    'end_to_end_tokens_per_second': (
                            generated_tokens / end_to_end_seconds
                    ),
                    'status': 'success',
            }
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8',
            )
    finally:
        context.close()


def worker_command(
        mode: str,
        batch_size: int,
        result_path: Path,
) -> list[str]:
    """Build the isolated command for one benchmark configuration."""
    worker_args = [
            MODULE_NAME,
            '--worker',
            '--mode',
            mode,
            '--batch-size',
            str(batch_size),
            '--result-path',
            str(result_path),
    ]
    if mode in {'tensor', 'data'}:
        torchrun = str(Path(sys.executable).with_name('torchrun'))
        return [
                torchrun,
                '--standalone',
                '--nproc-per-node=2',
                '--max-restarts=0',
                '--module',
                *worker_args,
        ]
    return [sys.executable, '-m', *worker_args]


def log_is_oom(log_text: str) -> bool:
    """Return whether worker output reports an accelerator OOM."""
    lowered = log_text.lower()
    return 'out of memory' in lowered or 'failed to allocate' in lowered


def run_configuration(
        mode: str,
        batch_size: int,
        output_dir: Path,
) -> dict[str, Any]:
    """Run and collect one isolated benchmark configuration."""
    result_path = output_dir / 'worker_results' / f'{mode}_{batch_size}.json'
    log_path = output_dir / 'logs' / f'{mode}_{batch_size}.log'
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)

    environment = dict(os.environ)
    environment['OMP_NUM_THREADS'] = '4' if mode in {'tensor', 'data'} else '8'
    with log_path.open('w', encoding='utf-8') as log_file:
        completed = subprocess.run(
                worker_command(mode, batch_size, result_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
        )

    log_text = log_path.read_text(encoding='utf-8', errors='replace')
    if completed.returncode != 0:
        return {
                'parallelism': mode,
                'global_batch_size': batch_size,
                'status': 'oom' if log_is_oom(log_text) else 'error',
                'return_code': completed.returncode,
                'log_path': str(log_path.relative_to(output_dir)),
        }
    if not result_path.exists():
        return {
                'parallelism': mode,
                'global_batch_size': batch_size,
                'status': 'error',
                'return_code': completed.returncode,
                'log_path': str(log_path.relative_to(output_dir)),
        }

    result = json.loads(result_path.read_text(encoding='utf-8'))
    result['return_code'] = completed.returncode
    result['log_path'] = str(log_path.relative_to(output_dir))
    return result


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    """Persist all completed configurations as JSONL and CSV."""
    with (output_dir / 'results.jsonl').open('w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + '\n')

    fieldnames = [
            'parallelism',
            'global_batch_size',
            'per_rank_batch_size',
            'prompt_characters',
            'prompt_tokens_per_item',
            'generated_characters_per_item',
            'generated_tokens_per_item',
            'generated_characters',
            'generated_tokens',
            'generation_seconds',
            'generation_characters_per_second',
            'generation_tokens_per_second',
            'end_to_end_seconds',
            'end_to_end_characters_per_second',
            'end_to_end_tokens_per_second',
            'status',
            'return_code',
            'log_path',
    ]
    with (output_dir / 'results.csv').open(
            'w',
            encoding='utf-8',
            newline='',
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({name: result.get(name) for name in fieldnames})


def run_sweep(output_dir: Path) -> list[dict[str, Any]]:
    """Benchmark all modes and stop a mode after its first OOM."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    write_results(output_dir, results)
    for mode in MODES:
        for batch_size in BATCH_SIZES:
            print(f'Running parallelism={mode}, batch_size={batch_size}...', flush=True)
            result = run_configuration(mode, batch_size, output_dir)
            results.append(result)
            write_results(output_dir, results)

            if result['status'] == 'success':
                print(
                        f"Generation: {result['generation_seconds']:.3f}s, "
                        f"{result['generation_tokens_per_second']:.1f} tokens/s, "
                        f"{result['generation_characters_per_second']:.1f} "
                        f"characters/s. End-to-end: "
                        f"{result['end_to_end_seconds']:.3f}s, "
                        f"{result['end_to_end_tokens_per_second']:.1f} tokens/s, "
                        f"{result['end_to_end_characters_per_second']:.1f} "
                        f"characters/s.",
                        flush=True,
                )
            elif result['status'] == 'oom':
                print('OOM; skipping larger batches for this mode.', flush=True)
                break
            else:
                raise RuntimeError(
                        f'Benchmark failed for parallelism={mode}, '
                        f'batch_size={batch_size}. See '
                        f'{output_dir / result["log_path"]}.'
                )
    return results


def parse_args() -> argparse.Namespace:
    """Parse sweep and internal worker arguments."""
    job_id = os.environ.get('LSB_JOBID', 'local')
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--output-dir',
            type=Path,
            default=Path(f'runs/parallelism_benchmark/{job_id}'),
    )
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--mode', choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument('--batch-size', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--result-path', type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.worker:
        if args.mode is None or args.batch_size is None or args.result_path is None:
            raise ValueError('Worker mode requires mode, batch-size, and result-path.')
        run_worker(args.mode, args.batch_size, args.result_path)
    else:
        run_sweep(args.output_dir)
