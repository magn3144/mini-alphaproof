"""Flatten the LeanTree Mathlib dataset into AlphaProof SFT transitions.

The input is the serialized ``leantree_mathlib.jsonl`` dataset used by
NanoProof. Each output record contains the stringified Lean state, its tactic,
and the remaining proof-tree depth. The last 10% of Lean files form the
validation split, exactly as in NanoProof.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leantree import LeanFile, StoredError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / 'data' / 'dataset' / 'leantree_mathlib.jsonl'
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / 'data'
    / 'dataset'
    / 'leantree_mathlib_state_action_pairs.jsonl'
)
STATE_VERSION = 1


@dataclass(frozen=True)
class FileResult:
    """Transitions, stored errors, and timing from one serialized Lean file."""

    source_line: int
    split: str
    pairs: tuple[dict[str, str | int], ...]
    errors: tuple[str, ...]
    seconds: float


def derived_path(path: Path, label: str) -> Path:
    """Insert a label before a path's JSONL suffix."""
    return path.with_name(f'{path.stem}.{label}{path.suffix}')


def output_paths(output_path: Path) -> dict[str, Path]:
    """Return every append-only output participating in checkpoints."""
    return {
        name: derived_path(output_path, name)
        for name in ('train', 'validation', 'errors', 'timings')
    }


def count_lines(path: Path) -> int:
    """Count serialized Lean files without loading the dataset into memory."""
    with path.open('rb') as input_file:
        return sum(1 for _ in input_file)


def selected_counts(
    total_files: int,
    validation_fraction: float,
    limit: int | None,
) -> tuple[int, int, int]:
    """Return validation boundary and selected train/validation file counts."""
    validation_files = int(total_files * validation_fraction)
    validation_start = total_files - validation_files
    if limit is None or limit >= total_files:
        return validation_start, validation_start, validation_files

    train_files = min(validation_start, max(1, int(limit * (1 - validation_fraction))))
    selected_validation = min(validation_files, limit - train_files)
    return validation_start, train_files, selected_validation


def selected_line(
    line_index: int,
    validation_start: int,
    train_files: int,
    validation_files: int,
) -> bool:
    """Return whether a source file belongs to the requested subset."""
    return line_index < train_files or (
        validation_start
        <= line_index
        < validation_start + validation_files
    )


def flatten_lean_file(source_line: int, split: str, line: bytes) -> FileResult:
    """Deserialize one LeanTree file and reproduce NanoProof's transitions."""
    started = time.perf_counter()
    try:
        data = json.loads(line)
        root_error = data.get('error')
        if isinstance(root_error, str):
            return FileResult(
                source_line=source_line,
                split=split,
                pairs=(),
                errors=(root_error,),
                seconds=time.perf_counter() - started,
            )

        lean_file = LeanFile.deserialize(data)
        pairs: list[dict[str, str | int]] = []
        errors: list[str] = []
        for theorem in lean_file.theorems:
            if isinstance(theorem, StoredError):
                errors.append(theorem.error)
                continue
            for by_block in theorem.by_blocks:
                tree = by_block.tree
                if isinstance(tree, StoredError):
                    errors.append(tree.error)
                    continue
                if tree is None:
                    errors.append('LeanTree record contains an empty proof tree.')
                    continue
                for node in tree.get_nodes():
                    if node.state is None or node.tactic is None:
                        continue
                    action = str(node.tactic.tactic).strip()
                    if 'sorry' in action or 'admit' in action or action == 'bound':
                        continue
                    pairs.append(
                        {
                            'state': str(node.state),
                            'action': action,
                            'proof_depth': node.proof_depth,
                        }
                    )

        return FileResult(
            source_line=source_line,
            split=split,
            pairs=tuple(pairs),
            errors=tuple(errors),
            seconds=time.perf_counter() - started,
        )
    except Exception as error:
        return FileResult(
            source_line=source_line,
            split=split,
            pairs=(),
            errors=(str(error),),
            seconds=time.perf_counter() - started,
        )


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Append records and make them durable before advancing the checkpoint."""
    if not records:
        return
    with path.open('ab') as output_file:
        for record in records:
            output_file.write(
                (json.dumps(record, ensure_ascii=False) + '\n').encode('utf-8')
            )
        output_file.flush()
        os.fsync(output_file.fileno())


def current_offsets(paths: dict[str, Path]) -> dict[str, int]:
    """Return byte offsets for all append-only outputs."""
    return {name: path.stat().st_size for name, path in paths.items()}


def write_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace the extraction checkpoint."""
    temporary_path = path.with_suffix('.tmp')
    temporary_path.write_text(
        json.dumps(state, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(path)


def truncate_to_checkpoint(
    paths: dict[str, Path], offsets: dict[str, int]
) -> None:
    """Discard output written after the last committed checkpoint."""
    for name, path in paths.items():
        with path.open('r+b') as output_file:
            output_file.truncate(offsets[name])


def input_identity(path: Path) -> dict[str, Any]:
    """Return metadata used to reject unsafe resume attempts."""
    stat = path.stat()
    return {
        'input_path': str(path.resolve()),
        'input_size': stat.st_size,
        'input_mtime_ns': stat.st_mtime_ns,
    }


def new_state(
    args: argparse.Namespace,
    total_files: int,
    selected_total: int,
) -> dict[str, Any]:
    """Construct the initial resumable extraction state."""
    return {
        'version': STATE_VERSION,
        **input_identity(args.input),
        'output_path': str(args.output.resolve()),
        'validation_fraction': args.validation_fraction,
        'limit': args.limit,
        'total_files': total_files,
        'selected_files': selected_total,
        'next_input_offset': 0,
        'next_line_index': 0,
        'records_seen': 0,
        'files_with_pairs': 0,
        'files_with_errors': 0,
        'pairs_written': 0,
        'train_pairs': 0,
        'validation_pairs': 0,
        'output_offsets': {},
    }


def validate_state(
    state: dict[str, Any],
    args: argparse.Namespace,
    total_files: int,
    selected_total: int,
) -> None:
    """Ensure a checkpoint belongs to this exact conversion run."""
    expected = {
        'version': STATE_VERSION,
        **input_identity(args.input),
        'output_path': str(args.output.resolve()),
        'validation_fraction': args.validation_fraction,
        'limit': args.limit,
        'total_files': total_files,
        'selected_files': selected_total,
    }
    mismatches = [
        name for name, value in expected.items() if state.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            'Cannot resume because these settings changed: '
            + ', '.join(mismatches)
        )


def commit_batch(
    results: list[FileResult],
    paths: dict[str, Path],
    state: dict[str, Any],
    state_path: Path,
    next_input_offset: int,
    next_line_index: int,
) -> None:
    """Append a completed batch and atomically advance its checkpoint."""
    outputs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in paths
    }
    for result in results:
        outputs[result.split].extend(result.pairs)
        outputs['timings'].append(
            {
                'source_line': result.source_line,
                'split': result.split,
                'seconds': result.seconds,
                'pairs': len(result.pairs),
                'errors': len(result.errors),
            }
        )
        if result.errors:
            outputs['errors'].append(
                {
                    'source_line': result.source_line,
                    'split': result.split,
                    'errors': list(result.errors),
                }
            )

        state['records_seen'] += 1
        state['pairs_written'] += len(result.pairs)
        state[f'{result.split}_pairs'] += len(result.pairs)
        if result.pairs:
            state['files_with_pairs'] += 1
        if result.errors:
            state['files_with_errors'] += 1

    for name, records in outputs.items():
        append_jsonl(paths[name], records)
    state['next_input_offset'] = next_input_offset
    state['next_line_index'] = next_line_index
    state['output_offsets'] = current_offsets(paths)
    write_state(state_path, state)


def print_progress(
    state: dict[str, Any],
    started: float,
    starting_records: int,
) -> None:
    """Print cumulative conversion counts and current-run throughput."""
    elapsed = time.perf_counter() - started
    converted = state['records_seen'] - starting_records
    rate = converted / elapsed if elapsed > 0 else 0.0
    print(
        f"Converted {state['records_seen']:,}/{state['selected_files']:,} Lean "
        f"files; wrote {state['pairs_written']:,} transitions; "
        f"{state['files_with_errors']:,} files had errors; "
        f'{rate:.1f} files/s',
        flush=True,
    )


def run_conversion(args: argparse.Namespace) -> dict[str, Any]:
    """Run or resume conversion of serialized LeanTree proof trees."""
    started = time.perf_counter()
    total_files = count_lines(args.input)
    validation_start, train_files, validation_files = selected_counts(
        total_files,
        args.validation_fraction,
        args.limit,
    )
    selected_total = train_files + validation_files
    print(
        f'Found {total_files:,} Lean files; converting {train_files:,} train '
        f'and {validation_files:,} validation files',
        flush=True,
    )

    paths = output_paths(args.output)
    work_dir = args.work_dir or args.output.with_name(f'{args.output.stem}.work')
    state_path = work_dir / 'state.json'
    work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f'Resume checkpoint does not exist: {state_path}')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        validate_state(state, args, total_files, selected_total)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f'Resume output does not exist: {path}')
        truncate_to_checkpoint(paths, state['output_offsets'])
    else:
        existing = [path for path in paths.values() if path.exists()]
        if state_path.exists() or existing:
            names = ', '.join(str(path) for path in [state_path, *existing])
            raise FileExistsError(
                f'Outputs already exist; use --resume or remove them: {names}'
            )
        for path in paths.values():
            path.touch()
        state = new_state(args, total_files, selected_total)
        state['output_offsets'] = current_offsets(paths)
        write_state(state_path, state)

    starting_records = state['records_seen']
    last_progress = state['records_seen']
    results: list[FileResult] = []
    with args.input.open('rb') as input_file:
        input_file.seek(state['next_input_offset'])
        line_index = state['next_line_index']
        while state['records_seen'] + len(results) < selected_total:
            line = input_file.readline()
            if not line:
                break
            current_index = line_index
            line_index += 1
            if not selected_line(
                current_index,
                validation_start,
                train_files,
                validation_files,
            ):
                continue
            split = 'train' if current_index < validation_start else 'validation'
            results.append(flatten_lean_file(current_index + 1, split, line))

            if len(results) == args.batch_size:
                commit_batch(
                    results,
                    paths,
                    state,
                    state_path,
                    input_file.tell(),
                    line_index,
                )
                if args.timings:
                    for result in results:
                        print(
                            f'[file {result.source_line}] {result.seconds:.3f}s, '
                            f'{len(result.pairs)} transitions, '
                            f'{len(result.errors)} errors',
                            flush=True,
                        )
                results = []
                if state['records_seen'] - last_progress >= args.progress_every:
                    print_progress(state, started, starting_records)
                    last_progress = state['records_seen']

        if results:
            commit_batch(
                results,
                paths,
                state,
                state_path,
                input_file.tell(),
                line_index,
            )

    print_progress(state, started, starting_records)
    for name, path in paths.items():
        print(f'Wrote {name} data to {path}', flush=True)
    return state


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('value must be positive')
    return parsed


def fraction(value: str) -> float:
    """Parse a fraction strictly between zero and one."""
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError('value must be between zero and one')
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse LeanTree dataset conversion arguments."""
    parser = argparse.ArgumentParser(
        description='Convert the LeanTree Mathlib dataset for AlphaProof SFT.'
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--limit', type=positive_int)
    parser.add_argument('--batch-size', type=positive_int, default=100)
    parser.add_argument('--progress-every', type=positive_int, default=1000)
    parser.add_argument('--validation-fraction', type=fraction, default=0.1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timings', action='store_true')
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f'LeanTree JSONL does not exist: {args.input}')
    return args


def main() -> None:
    """Convert the LeanTree Mathlib dataset."""
    run_conversion(parse_args())


if __name__ == '__main__':
    main()
