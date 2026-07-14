"""Convert completed Lean proof scripts into AlphaProof SFT state-action pairs.

Each proof is parsed and replayed by LeanTree. Every solved proof-tree node
produces one pair whose state is Lean's pretty-printed tactic state and whose
action is the tactic applied at that state.
"""

import argparse
import json
import tempfile
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
            if isinstance(block, StoredError):
                errors.append(
                    f'theorem {theorem_index}, block {block_index}: {block.error}'
                )
                continue
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
) -> tuple[list[dict[str, str]], list[str]]:
    """Parse and replay one Lean script, then return its state-action pairs."""
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

    try:
        lean_file = project.load_file(temporary_path, use_cache=False)
        return lean_file_to_pairs(lean_file)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_line(output_file: TextIO, record: dict[str, Any]) -> None:
    """Write one UTF-8 JSON object followed by a newline."""
    output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def convert_proofs(
    input_path: Path,
    output_path: Path,
    errors_path: Path,
    project_path: Path,
    limit: int | None = None,
    progress_every: int = 100,
) -> dict[str, int]:
    """Stream completed proofs through LeanTree and write state-action pairs."""
    if not input_path.is_file():
        raise FileNotFoundError(f'Input JSONL does not exist: {input_path}')
    resolved_paths = {
        input_path.resolve(),
        output_path.resolve(),
        errors_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError('Input, output, and errors paths must be different.')
    if not (project_path / '.lake').is_dir():
        raise FileNotFoundError(
            f'Lean project is not built: {project_path}. '
            'Run `lake update` and `lake build` in that directory first.'
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_errors_path = errors_path.with_suffix(errors_path.suffix + '.tmp')
    project = LeanProject(project_path)

    records_seen = 0
    scripts_with_pairs = 0
    pairs_written = 0
    scripts_with_errors = 0

    with (
        input_path.open(encoding='utf-8') as input_file,
        temporary_output_path.open('w', encoding='utf-8') as output_file,
        temporary_errors_path.open('w', encoding='utf-8') as errors_file,
    ):
        for source_line, line in enumerate(input_file, start=1):
            if limit is not None and records_seen >= limit:
                break
            records_seen += 1

            try:
                record = json.loads(line)
                lean_script = record['lean_proof']
                if not isinstance(lean_script, str):
                    raise TypeError('lean_proof must be a string.')
                pairs, errors = lean_script_to_pairs(project, lean_script)
                if not pairs and not errors:
                    errors = ['No tactic proof trees were found.']
            except Exception as error:
                pairs = []
                errors = [f'{type(error).__name__}: {error}']

            if pairs:
                scripts_with_pairs += 1
                for pair in pairs:
                    write_json_line(output_file, pair)
                pairs_written += len(pairs)

            if errors:
                scripts_with_errors += 1
                write_json_line(
                    errors_file,
                    {'source_line': source_line, 'errors': errors},
                )

            if records_seen % progress_every == 0:
                print(
                    f'Read {records_seen:,} proofs; '
                    f'wrote {pairs_written:,} state-action pairs; '
                    f'{scripts_with_errors:,} proofs had errors',
                    flush=True,
                )

    temporary_output_path.replace(output_path)
    temporary_errors_path.replace(errors_path)
    return {
        'records_seen': records_seen,
        'scripts_with_pairs': scripts_with_pairs,
        'pairs_written': pairs_written,
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
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
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
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.progress_every < 1:
        parser.error('--progress-every must be positive.')
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
    )
    print(json.dumps(stats, indent=2))
    print(f'Wrote state-action pairs to {args.output}')
    print(f'Wrote extraction errors to {errors_path}')


if __name__ == '__main__':
    main()
