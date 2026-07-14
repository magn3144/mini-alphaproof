"""Convert completed Lean proof scripts into AlphaProof SFT state-action pairs.

Each proof is parsed and replayed by LeanTree. Every solved proof-tree node
produces one pair whose state is Lean's pretty-printed tactic state and whose
action is the tactic applied at that state.
"""

import argparse
import cProfile
import io
import json
import os
import pstats
import random
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

from leantree import LeanFile, LeanProject, ProofTree, StoredError


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / 'data'
    / 'dataset'
    / 'nemotron_math_proofs_v1_finished_lean_proofs.jsonl'
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / 'data'
    / 'dataset'
    / 'nemotron_math_proofs_v1_state_action_pairs.jsonl'
)
DEFAULT_PROJECT_PATH = REPO_ROOT / 'lean_project'
SPLIT_NAMES = ('train', 'validation', 'test')
DEFAULT_SPLIT = (0.6, 0.2, 0.2)


def print_profile(profile: cProfile.Profile, proof_number: int) -> None:
    """Print the slowest calls made while LeanTree loaded one proof."""
    output = io.StringIO()
    pstats.Stats(profile, stream=output).strip_dirs().sort_stats(
        'cumulative'
    ).print_stats(40)
    print(f'[proof {proof_number}] LeanTree cumulative profile:', flush=True)
    print(output.getvalue(), end='', flush=True)


def proof_tree_to_pairs(tree: ProofTree) -> list[dict[str, str]]:
    """Return the state-action pair at every node of a solved proof tree."""
    if not tree.is_solved():
        raise ValueError('LeanTree produced an unsolved proof tree.')

    pairs = []
    for node in tree.get_nodes():
        if node.state is None or node.tactic is None:
            raise ValueError('A solved proof-tree node has no state or tactic.')
        pairs.append({
            'state': str(node.state),
            'action': node.tactic.tactic.tactic,
        })
    return pairs


def lean_file_to_pairs(lean_file: LeanFile) -> tuple[list[dict[str, str]], list[str]]:
    """Extract pairs and extraction errors from all tactic blocks in a file."""
    pairs = []
    errors = []
    for theorem_index, theorem in enumerate(lean_file.theorems):
        if isinstance(theorem, StoredError):
            errors.append(f'theorem {theorem_index}: {theorem.error}')
            continue

        for block_index, block in enumerate(theorem.by_blocks):
            if block.tree is None:
                errors.append(
                    f'theorem {theorem_index}, block {block_index}: missing proof tree'
                )
                continue
            if isinstance(block.tree, StoredError):
                errors.append(
                    f'theorem {theorem_index}, block {block_index}: '
                    f'{block.tree.error}'
                )
                continue

            try:
                pairs.extend(proof_tree_to_pairs(block.tree))
            except ValueError as error:
                errors.append(
                    f'theorem {theorem_index}, block {block_index}: {error}'
                )
    return pairs, errors


def lean_script_to_pairs(
    project: LeanProject,
    lean_script: str,
    proof_number: int | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Parse and replay one Lean script, then return its state-action pairs."""
    total_started = time.perf_counter()
    if proof_number is not None:
        print(
            f'[proof {proof_number}] started ({len(lean_script):,} characters)',
            flush=True,
        )

    write_started = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        'w',
        dir=project.path,
        prefix='NemotronProof_',
        suffix='.lean',
        encoding='utf-8',
        delete=False,
    ) as temporary_file:
        temporary_file.write(lean_script)
        temporary_path = Path(temporary_file.name)
    if proof_number is not None:
        print(
            f'[proof {proof_number}] wrote temporary file in '
            f'{time.perf_counter() - write_started:.3f}s',
            flush=True,
        )

    try:
        load_started = time.perf_counter()
        profile = cProfile.Profile() if proof_number is not None else None
        if proof_number is not None:
            print(f'[proof {proof_number}] LeanTree load started', flush=True)
        if profile is not None:
            profile.enable()
        try:
            lean_file = project.load_file(temporary_path, use_cache=False)
        except Exception:
            if proof_number is not None:
                print(
                    f'[proof {proof_number}] LeanTree load failed after '
                    f'{time.perf_counter() - load_started:.3f}s',
                    flush=True,
                )
            raise
        finally:
            if profile is not None:
                profile.disable()
                assert proof_number is not None
                print_profile(profile, proof_number)

        if proof_number is not None:
            block_count = sum(
                len(theorem.by_blocks)
                for theorem in lean_file.theorems
                if not isinstance(theorem, StoredError)
            )
            print(
                f'[proof {proof_number}] LeanTree load finished in '
                f'{time.perf_counter() - load_started:.3f}s '
                f'({len(lean_file.theorems)} theorems, {block_count} blocks)',
                flush=True,
            )

        pairs_started = time.perf_counter()
        pairs, errors = lean_file_to_pairs(lean_file)
        if proof_number is not None:
            print(
                f'[proof {proof_number}] converted {len(pairs)} pairs in '
                f'{time.perf_counter() - pairs_started:.3f}s',
                flush=True,
            )
        return pairs, errors
    finally:
        temporary_path.unlink(missing_ok=True)
        if proof_number is not None:
            print(
                f'[proof {proof_number}] total: '
                f'{time.perf_counter() - total_started:.3f}s',
                flush=True,
            )


def write_json_line(output_file: TextIO, record: dict[str, Any]) -> None:
    """Write one UTF-8 JSON object followed by a newline."""
    output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def split_output_paths(output_path: Path) -> dict[str, Path]:
    """Return train, validation, and test paths derived from an output path."""
    return {
        name: output_path.with_name(f'{output_path.stem}.{name}{output_path.suffix}')
        for name in SPLIT_NAMES
    }


def convert_proofs(
    input_path: Path,
    output_path: Path,
    errors_path: Path,
    project_path: Path,
    limit: int | None = None,
    progress_every: int = 100,
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = 0,
    timings: bool = False,
) -> dict[str, int]:
    """Extract pairs and split proof scripts across three dataset files."""
    if not input_path.is_file():
        raise FileNotFoundError(f'Input JSONL does not exist: {input_path}')
    output_paths = split_output_paths(output_path)
    resolved_paths = {
        input_path.resolve(),
        errors_path.resolve(),
        *(path.resolve() for path in output_paths.values()),
    }
    if len(resolved_paths) != 5:
        raise ValueError('Input, outputs, and errors paths must be different.')
    if not (project_path / '.lake').is_dir():
        raise FileNotFoundError(
            f'Lean project is not built: {project_path}. '
            'Run `lake update` and `lake build` in that directory first.'
        )

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_paths = {
        name: path.with_suffix(path.suffix + '.tmp')
        for name, path in output_paths.items()
    }
    temporary_errors_path = errors_path.with_suffix(errors_path.suffix + '.tmp')
    setup_started = time.perf_counter()
    if timings:
        os.environ['LEANTREE_TIMINGS'] = '1'
    project = LeanProject(project_path)
    if timings:
        print(
            f'[setup] initialized LeanProject in '
            f'{time.perf_counter() - setup_started:.3f}s',
            flush=True,
        )
    rng = random.Random(seed)
    train_threshold = split[0]
    validation_threshold = split[0] + split[1]

    records_seen = 0
    scripts_with_pairs = 0
    pairs_written = 0
    split_pairs_written = {name: 0 for name in SPLIT_NAMES}
    scripts_with_errors = 0

    with (
        input_path.open(encoding='utf-8') as input_file,
        temporary_output_paths['train'].open('w', encoding='utf-8') as train_file,
        temporary_output_paths['validation'].open(
            'w', encoding='utf-8'
        ) as validation_file,
        temporary_output_paths['test'].open('w', encoding='utf-8') as test_file,
        temporary_errors_path.open('w', encoding='utf-8') as errors_file,
    ):
        output_files = {
            'train': train_file,
            'validation': validation_file,
            'test': test_file,
        }
        for source_line, line in enumerate(input_file, start=1):
            if limit is not None and records_seen >= limit:
                break
            records_seen += 1
            record_started = time.perf_counter()

            try:
                parse_started = time.perf_counter()
                record = json.loads(line)
                lean_script = record['lean_proof']
                if not isinstance(lean_script, str):
                    raise TypeError('lean_proof must be a string.')
                if timings:
                    print(
                        f'[proof {source_line}] parsed JSON in '
                        f'{time.perf_counter() - parse_started:.3f}s',
                        flush=True,
                    )
                pairs, errors = lean_script_to_pairs(
                    project,
                    lean_script,
                    proof_number=source_line if timings else None,
                )
                if not pairs and not errors:
                    errors = ['No tactic proof trees were found.']
            except Exception as error:
                pairs = []
                errors = [f'{type(error).__name__}: {error}']

            output_started = time.perf_counter()
            if pairs:
                scripts_with_pairs += 1
                split_value = rng.random()
                if split_value < train_threshold:
                    split_name = 'train'
                elif split_value < validation_threshold:
                    split_name = 'validation'
                else:
                    split_name = 'test'
                for pair in pairs:
                    write_json_line(output_files[split_name], pair)
                pairs_written += len(pairs)
                split_pairs_written[split_name] += len(pairs)

            if errors:
                scripts_with_errors += 1
                write_json_line(
                    errors_file,
                    {'source_line': source_line, 'errors': errors},
                )

            if timings:
                print(
                    f'[proof {source_line}] wrote results in '
                    f'{time.perf_counter() - output_started:.3f}s; '
                    f'record total: {time.perf_counter() - record_started:.3f}s',
                    flush=True,
                )

            if records_seen % progress_every == 0:
                print(
                    f'Read {records_seen:,} proofs; '
                    f'wrote {pairs_written:,} state-action pairs; '
                    f'{scripts_with_errors:,} proofs had errors',
                    flush=True,
                )

    for name, path in output_paths.items():
        temporary_output_paths[name].replace(path)
    temporary_errors_path.replace(errors_path)
    return {
        'records_seen': records_seen,
        'scripts_with_pairs': scripts_with_pairs,
        'pairs_written': pairs_written,
        **{
            f'{name}_pairs_written': split_pairs_written[name]
            for name in SPLIT_NAMES
        },
        'scripts_with_errors': scripts_with_errors,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Replay completed Lean proofs and extract AlphaProof SFT '
            'state-action pairs.'
        )
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            'Base output path. Split names are inserted before the suffix, for '
            'example pairs.train.jsonl.'
        ),
    )
    parser.add_argument(
        '--errors',
        type=Path,
        help='Error JSONL path. Defaults to <output stem>.errors.jsonl.',
    )
    parser.add_argument(
        '--project',
        type=Path,
        default=DEFAULT_PROJECT_PATH,
        help='Lean project used to compile and replay the proof scripts.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process at most this many proof scripts.',
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=100,
        help='Print progress after this many proof scripts.',
    )
    parser.add_argument(
        '--split',
        type=float,
        nargs=3,
        metavar=('TRAIN', 'VALIDATION', 'TEST'),
        default=DEFAULT_SPLIT,
        help='Dataset split ratios (default: 0.6 0.2 0.2).',
    )
    parser.add_argument('--seed', type=int, default=0, help='Dataset split seed.')
    parser.add_argument(
        '--timings',
        action='store_true',
        help='Print detailed timing information for every proof.',
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.progress_every < 1:
        parser.error('--progress-every must be positive.')
    if any(ratio < 0 for ratio in args.split):
        parser.error('--split ratios cannot be negative.')
    if abs(sum(args.split) - 1.0) > 1e-9:
        parser.error('--split ratios must sum to 1.')
    return args


def main() -> None:
    """Run the state-action conversion."""
    args = parse_args()
    errors_path = args.errors or args.output.with_suffix('.errors.jsonl')
    stats = convert_proofs(
        input_path=args.input,
        output_path=args.output,
        errors_path=errors_path,
        project_path=args.project,
        limit=args.limit,
        progress_every=args.progress_every,
        split=tuple(args.split),
        seed=args.seed,
        timings=args.timings,
    )
    output_paths = split_output_paths(args.output)
    print(json.dumps(stats, indent=2))
    for name, path in output_paths.items():
        print(f'Wrote {name} state-action pairs to {path}')
    print(f'Wrote extraction errors to {errors_path}')


if __name__ == '__main__':
    main()
