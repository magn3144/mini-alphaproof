"""Trace completed Lean proofs into AlphaProof SFT state-action pairs.

The input JSONL must contain a ``lean_proof`` string in every record. Proofs are
materialized as independent modules in a generated Lake workspace and traced in
batches with LeanDojo-v2. Completed batches are checkpointed so interrupted runs
can resume without repeating earlier work.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any


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
DEFAULT_BATCH_SIZE = 100
DEFAULT_SPLIT = (0.6, 0.2, 0.2)
GENERATED_LIBRARY = 'LeanProject'
GENERATED_DIRECTORY = Path(GENERATED_LIBRARY) / 'Generated'
SPLIT_NAMES = ('train', 'validation', 'test')
STATE_VERSION = 1


def lean_dojo_extractor_path() -> Path:
    """Locate LeanDojo-v2's repository-tracing program without importing it."""
    try:
        package = distribution('lean-dojo-v2')
    except PackageNotFoundError as error:
        raise RuntimeError(
            'LeanDojo-v2 is not installed. Run `uv sync --extra lean-tracing`.'
        ) from error
    installed_path = package.locate_file(
        'lean_dojo_v2/lean_dojo/data_extraction/ExtractData.lean'
    )
    path = Path(str(installed_path))
    if not path.is_file():
        raise FileNotFoundError(f'LeanDojo-v2 extractor not found: {path}')
    return path


@dataclass(frozen=True)
class SourceRecord:
    """One source JSONL record selected for the current batch."""

    source_line: int
    lean_proof: str | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordResult:
    """Extracted pairs and errors for one source record."""

    source_line: int
    pairs: tuple[dict[str, str], ...]
    errors: tuple[str, ...]


def output_paths(output_path: Path) -> dict[str, Path]:
    """Return split output paths derived from a base output path."""
    return {
        name: output_path.with_name(
            f'{output_path.stem}.{name}{output_path.suffix}'
        )
        for name in SPLIT_NAMES
    }


def errors_path(output_path: Path) -> Path:
    """Return the extraction-error path derived from the base output path."""
    return output_path.with_suffix('.errors.jsonl')


def default_work_path(output_path: Path) -> Path:
    """Return the default persistent work directory."""
    return output_path.with_name(f'{output_path.stem}.work')


def run_command(
    command: list[str],
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output."""
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(
            f'Command failed ({result.returncode}): {" ".join(command)}\n{output}'
        )
    return result


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace a JSON file."""
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    temporary_path.replace(path)


def append_json_lines(path: Path, records: list[dict[str, Any]]) -> None:
    """Append JSON records and make them durable before checkpointing."""
    if not records:
        return
    with path.open('ab') as output_file:
        for record in records:
            line = json.dumps(record, ensure_ascii=False) + '\n'
            output_file.write(line.encode('utf-8'))
        output_file.flush()
        os.fsync(output_file.fileno())


def file_offsets(paths: dict[str, Path]) -> dict[str, int]:
    """Return the current byte length of every transactional output."""
    return {name: path.stat().st_size for name, path in paths.items()}


def truncate_files(paths: dict[str, Path], offsets: dict[str, int]) -> None:
    """Roll output files back to a checkpointed set of byte offsets."""
    for name, path in paths.items():
        with path.open('r+b') as output_file:
            output_file.truncate(offsets[name])


def project_fingerprint(project_path: Path) -> str:
    """Hash the Lean project files that define the tracing environment."""
    digest = hashlib.sha256()
    for name in ('lean-toolchain', 'lakefile.toml', 'lake-manifest.json'):
        path = project_path / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing Lean project file: {path}')
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def initial_state(
    input_path: Path,
    output_path: Path,
    project_path: Path,
    split: tuple[float, float, float],
    seed: int,
) -> dict[str, Any]:
    """Create a new resumable-run state."""
    stat = input_path.stat()
    return {
        'version': STATE_VERSION,
        'input_path': str(input_path.resolve()),
        'input_size': stat.st_size,
        'input_mtime_ns': stat.st_mtime_ns,
        'output_path': str(output_path.resolve()),
        'project_path': str(project_path.resolve()),
        'project_fingerprint': project_fingerprint(project_path),
        'split': list(split),
        'seed': seed,
        'next_source_line': 1,
        'next_input_offset': 0,
        'records_seen': 0,
        'scripts_with_pairs': 0,
        'pairs_written': 0,
        'train_pairs_written': 0,
        'validation_pairs_written': 0,
        'test_pairs_written': 0,
        'scripts_with_errors': 0,
        'pending_batch': None,
        'output_offsets': {},
    }


def validate_state(
    state: dict[str, Any],
    input_path: Path,
    output_path: Path,
    project_path: Path,
    split: tuple[float, float, float],
    seed: int,
) -> None:
    """Reject resume attempts made with incompatible inputs or options."""
    stat = input_path.stat()
    expected = {
        'version': STATE_VERSION,
        'input_path': str(input_path.resolve()),
        'input_size': stat.st_size,
        'input_mtime_ns': stat.st_mtime_ns,
        'output_path': str(output_path.resolve()),
        'project_path': str(project_path.resolve()),
        'project_fingerprint': project_fingerprint(project_path),
        'split': list(split),
        'seed': seed,
    }
    mismatches = [
        name
        for name, value in expected.items()
        if state.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            'Cannot resume because these settings changed: '
            + ', '.join(mismatches)
        )


def initialize_workspace(workspace: Path, project_path: Path) -> None:
    """Create the persistent generated Lean workspace if necessary."""
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / 'lakefile.toml').exists():
        shutil.copyfile(
            project_path / 'lean-toolchain',
            workspace / 'lean-toolchain',
        )
        shutil.copyfile(project_path / 'lakefile.toml', workspace / 'lakefile.toml')
        shutil.copyfile(
            project_path / 'lake-manifest.json',
            workspace / 'lake-manifest.json',
        )
        library_path = workspace / GENERATED_LIBRARY
        library_path.mkdir(parents=True, exist_ok=True)
        (workspace / f'{GENERATED_LIBRARY}.lean').write_text(
            '/- Persistent root module for LeanDojo tracing. -/\n',
            encoding='utf-8',
        )

    source_packages = (project_path / '.lake' / 'packages').resolve()
    if not source_packages.is_dir():
        raise FileNotFoundError(
            f'Lean packages are not built: {source_packages}. Run `lake update` '
            'and `lake build` in the source Lean project first.'
        )
    lake_path = workspace / '.lake'
    lake_path.mkdir(exist_ok=True)
    packages_path = lake_path / 'packages'
    packages_path.mkdir(exist_ok=True)
    for source_package in source_packages.iterdir():
        package_path = packages_path / source_package.name
        if not package_path.exists():
            package_path.symlink_to(source_package.resolve(), target_is_directory=True)

    lean_prefix = Path(run_command(
        ['lake', 'env', 'lean', '--print-prefix'],
        workspace,
    ).stdout.strip())
    lean4_path = packages_path / 'lean4'
    if not lean4_path.exists():
        lean4_path.symlink_to(lean_prefix, target_is_directory=True)

    root_olean = (
        workspace
        / '.lake'
        / 'build'
        / 'lib'
        / 'lean'
        / f'{GENERATED_LIBRARY}.olean'
    )
    if not root_olean.is_file():
        run_command(['lake', 'build'], workspace)


def clear_generated_batch(workspace: Path) -> None:
    """Remove artifacts owned by the previous generated batch."""
    paths = (
        workspace / GENERATED_DIRECTORY,
        workspace / '.lake' / 'build' / 'lib' / 'lean' / GENERATED_DIRECTORY,
        workspace / '.lake' / 'build' / 'ir' / GENERATED_DIRECTORY,
    )
    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def split_lean_script(lean_script: str) -> tuple[list[str], list[str]]:
    """Separate module-header commands from the script body."""
    header = []
    body = []
    for line in lean_script.splitlines():
        stripped = line.strip()
        if (
            stripped == 'prelude'
            or stripped.startswith('import ')
            or stripped.startswith('public import ')
            or stripped.startswith('private import ')
        ):
            header.append(line)
        else:
            body.append(line)
    return header, body


def generated_source(record: SourceRecord) -> str:
    """Wrap one standalone proof script in an isolated namespace."""
    assert record.lean_proof is not None
    header, body = split_lean_script(record.lean_proof)
    namespace = f'AlphaProofTrace.Proof{record.source_line}'
    parts = [*header, '', f'namespace {namespace}', *body, '', f'end {namespace}']
    return '\n'.join(parts).strip() + '\n'


def module_path(workspace: Path, source_line: int) -> Path:
    """Return the generated Lean module path for a source line."""
    return workspace / GENERATED_DIRECTORY / f'Proof_{source_line:09d}.lean'


def olean_path(workspace: Path, source_line: int) -> Path:
    """Return the expected compiled module path for a source line."""
    return (
        workspace
        / '.lake'
        / 'build'
        / 'lib'
        / 'lean'
        / GENERATED_DIRECTORY
        / f'Proof_{source_line:09d}.olean'
    )


def ast_path(workspace: Path, source_line: int) -> Path:
    """Return the expected LeanDojo trace path for a source line."""
    return (
        workspace
        / '.lake'
        / 'build'
        / 'ir'
        / GENERATED_DIRECTORY
        / f'Proof_{source_line:09d}.ast.json'
    )


def write_root_module(workspace: Path, records: list[SourceRecord]) -> None:
    """Make Lake build every generated proof module in the current batch."""
    imports = [
        f'import {GENERATED_LIBRARY}.Generated.Proof_{record.source_line:09d}'
        for record in records
        if record.lean_proof is not None
    ]
    (workspace / f'{GENERATED_LIBRARY}.lean').write_text(
        '\n'.join(imports) + '\n',
        encoding='utf-8',
    )


def prepare_batch(workspace: Path, records: list[SourceRecord]) -> None:
    """Materialize valid JSONL records as independent Lean modules."""
    clear_generated_batch(workspace)
    generated_directory = workspace / GENERATED_DIRECTORY
    generated_directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.lean_proof is not None:
            module_path(workspace, record.source_line).write_text(
                generated_source(record),
                encoding='utf-8',
            )
    write_root_module(workspace, records)


def build_batch(
    workspace: Path,
    records: list[SourceRecord],
) -> tuple[list[SourceRecord], dict[int, str]]:
    """Build all modules and isolate records that Lean rejects."""
    candidates = [record for record in records if record.lean_proof is not None]
    if not candidates:
        return [], {}

    result = run_command(['lake', 'build'], workspace, check=False)
    valid = [
        record
        for record in candidates
        if olean_path(workspace, record.source_line).is_file()
    ]
    failed = [record for record in candidates if record not in valid]
    build_output = (result.stdout + result.stderr).strip()
    error_text = build_output[-8000:] or 'Lean did not produce an .olean file.'
    build_errors = {record.source_line: error_text for record in failed}

    for record in failed:
        module_path(workspace, record.source_line).unlink(missing_ok=True)
    if valid and result.returncode != 0:
        write_root_module(workspace, valid)
        run_command(['lake', 'build'], workspace)
    return valid, build_errors


def trace_batch(
    workspace: Path,
    records: list[SourceRecord],
    threads: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Run LeanDojo-v2's repository tracer and load generated traces."""
    extractor_path = lean_dojo_extractor_path()
    shutil.copyfile(
        extractor_path,
        workspace / extractor_path.name,
    )

    def trace_record(
        record: SourceRecord,
    ) -> tuple[int, dict[str, Any] | None, str | None]:
        result = run_command(
            [
                'lake',
                'env',
                'lean',
                '--run',
                extractor_path.name,
                str(module_path(workspace, record.source_line).resolve()),
            ],
            workspace,
            check=False,
        )
        path = ast_path(workspace, record.source_line)
        if path.is_file():
            return (
                record.source_line,
                json.loads(path.read_text(encoding='utf-8')),
                None,
            )
        output = (result.stdout + result.stderr).strip()
        return (
            record.source_line,
            None,
            output[-8000:] or 'LeanDojo did not produce an AST trace.',
        )

    with ThreadPoolExecutor(max_workers=threads) as executor:
        traced = list(executor.map(trace_record, records))
    traces = {}
    errors = {}
    for source_line, trace, error in traced:
        if trace is not None:
            traces[source_line] = trace
        if error is not None:
            errors[source_line] = error
    return traces, errors


def position_byte_index(position: Any, source: bytes) -> int:
    """Convert a Lean JSON position to a UTF-8 byte index."""
    if isinstance(position, int):
        return position
    if not isinstance(position, dict):
        raise ValueError(f'Unsupported Lean position: {position!r}')
    if isinstance(position.get('byteIdx'), int):
        return position['byteIdx']
    if isinstance(position.get('raw'), dict):
        return position_byte_index(position['raw'], source)
    if isinstance(position.get('1'), int):
        return position['1']

    line = position.get('line')
    column = position.get('column', position.get('character'))
    if not isinstance(line, int) or not isinstance(column, int):
        raise ValueError(f'Unsupported Lean position: {position!r}')
    lines = source.splitlines(keepends=True)
    return sum(len(item) for item in lines[:line]) + column


def tactic_text(source: bytes, tactic: dict[str, Any]) -> str:
    """Slice a tactic from its LeanDojo byte positions."""
    start = position_byte_index(tactic['pos'], source)
    end = position_byte_index(tactic['endPos'], source)
    text = source[start:end].decode('utf-8').strip()
    lines = text.splitlines()
    if len(lines) < 2:
        return text

    line_start = source.rfind(b'\n', 0, start) + 1
    indent = len(source[line_start:start].decode('utf-8'))
    fixed = [lines[0]]
    for line in lines[1:]:
        offset = min(indent, len(line) - len(line.lstrip()))
        fixed.append(line[offset:])
    return '\n'.join(fixed)


def pairs_from_trace(
    traces: dict[int, dict[str, Any]],
    valid_records: list[SourceRecord],
    workspace: Path,
) -> dict[int, RecordResult]:
    """Extract single-goal pairs in AlphaProof's JSONL format."""
    pairs_by_line: dict[int, list[dict[str, str]]] = {
        record.source_line: [] for record in valid_records
    }
    rejected_by_line = {record.source_line: 0 for record in valid_records}

    for record in valid_records:
        source_line = record.source_line
        source = module_path(workspace, source_line).read_bytes()
        signatures: set[tuple[str, str, str]] = set()
        tactics = traces[source_line]['tactics']
        ranges = [
            (
                position_byte_index(tactic['pos'], source),
                position_byte_index(tactic['endPos'], source),
            )
            for tactic in tactics
        ]
        for tactic, (start, end) in zip(tactics, ranges, strict=True):
            contains_tactic = any(
                start <= child_start
                and child_end <= end
                and (start, end) != (child_start, child_end)
                for child_start, child_end in ranges
            )
            if contains_tactic:
                rejected_by_line[source_line] += 1
                continue
            state = tactic['stateBefore'].strip()
            state_after = tactic['stateAfter'].strip()
            action = tactic_text(source, tactic)
            signature = (state, action, state_after)
            if signature in signatures:
                continue
            signatures.add(signature)
            if state.count('⊢') != 1 or not action or '·' in action:
                rejected_by_line[source_line] += 1
                continue
            pairs_by_line[source_line].append({
                'state': state,
                'action': action,
            })

    results = {}
    for record in valid_records:
        source_line = record.source_line
        pairs = pairs_by_line[source_line]
        errors = []
        rejected = rejected_by_line[source_line]
        if not pairs:
            errors.append(
                'LeanDojo found no usable atomic single-goal tactics '
                f'({rejected} traced tactics rejected).'
            )
        results[source_line] = RecordResult(
            source_line=source_line,
            pairs=tuple(pairs),
            errors=tuple(errors),
        )
    return results


def process_batch(
    workspace: Path,
    records: list[SourceRecord],
    threads: int,
) -> tuple[list[RecordResult], dict[str, float]]:
    """Build, trace, and convert one resumable batch."""
    batch_started = time.perf_counter()
    prepare_started = time.perf_counter()
    prepare_batch(workspace, records)
    prepare_seconds = time.perf_counter() - prepare_started

    build_started = time.perf_counter()
    valid_records, build_errors = build_batch(workspace, records)
    build_seconds = time.perf_counter() - build_started

    trace_seconds = 0.0
    parse_seconds = 0.0
    traced_results: dict[int, RecordResult] = {}
    if valid_records:
        trace_started = time.perf_counter()
        traces, trace_errors = trace_batch(workspace, valid_records, threads)
        trace_seconds = time.perf_counter() - trace_started

        parse_started = time.perf_counter()
        traced_records = [
            record for record in valid_records if record.source_line in traces
        ]
        traced_results = pairs_from_trace(traces, traced_records, workspace)
        parse_seconds = time.perf_counter() - parse_started
    else:
        trace_errors = {}

    results = []
    for record in records:
        errors = list(record.errors)
        if record.source_line in build_errors:
            errors.append(f'Lean build failed:\n{build_errors[record.source_line]}')
        if record.source_line in trace_errors:
            errors.append(f'LeanDojo trace failed:\n{trace_errors[record.source_line]}')
        traced_result = traced_results.get(record.source_line)
        pairs = traced_result.pairs if traced_result is not None else ()
        if traced_result is not None:
            errors.extend(traced_result.errors)
        results.append(RecordResult(record.source_line, pairs, tuple(errors)))

    timings = {
        'prepare_seconds': prepare_seconds,
        'build_seconds': build_seconds,
        'trace_seconds': trace_seconds,
        'parse_seconds': parse_seconds,
        'total_seconds': time.perf_counter() - batch_started,
    }
    return results, timings


def split_name_for(
    source_line: int,
    seed: int,
    split: tuple[float, float, float],
) -> str:
    """Assign a source proof deterministically to a dataset split."""
    digest = hashlib.sha256(f'{seed}:{source_line}'.encode()).digest()
    value = int.from_bytes(digest[:8], 'big') / 2**64
    if value < split[0]:
        return 'train'
    if value < split[0] + split[1]:
        return 'validation'
    return 'test'


def read_batch(
    input_file,
    next_source_line: int,
    batch_size: int,
) -> tuple[list[SourceRecord], int, int]:
    """Read and validate one batch from the current binary input position."""
    records = []
    source_line = next_source_line
    while len(records) < batch_size:
        raw_line = input_file.readline()
        if not raw_line:
            break
        try:
            data = json.loads(raw_line)
            lean_proof = data['lean_proof']
            if not isinstance(lean_proof, str):
                raise TypeError('lean_proof must be a string.')
            record = SourceRecord(source_line, lean_proof)
        except Exception as error:
            record = SourceRecord(
                source_line,
                None,
                (f'{type(error).__name__}: {error}',),
            )
        records.append(record)
        source_line += 1
    return records, source_line, input_file.tell()


def transactional_paths(output_path: Path, work_path: Path) -> dict[str, Path]:
    """Return every append-only file covered by the resume transaction."""
    paths = output_paths(output_path)
    paths['errors'] = errors_path(output_path)
    paths['timings'] = work_path / 'timings.jsonl'
    return paths


def write_batch_results(
    paths: dict[str, Path],
    results: list[RecordResult],
    timings: dict[str, Any],
    split: tuple[float, float, float],
    seed: int,
) -> dict[str, int]:
    """Append a processed batch and return its statistics."""
    split_records = {name: [] for name in SPLIT_NAMES}
    error_records = []
    stats = {
        'scripts_with_pairs': 0,
        'pairs_written': 0,
        'train_pairs_written': 0,
        'validation_pairs_written': 0,
        'test_pairs_written': 0,
        'scripts_with_errors': 0,
    }
    for result in results:
        if result.pairs:
            name = split_name_for(result.source_line, seed, split)
            split_records[name].extend(result.pairs)
            stats['scripts_with_pairs'] += 1
            stats['pairs_written'] += len(result.pairs)
            stats[f'{name}_pairs_written'] += len(result.pairs)
        if result.errors:
            error_records.append({
                'source_line': result.source_line,
                'errors': list(result.errors),
            })
            stats['scripts_with_errors'] += 1

    for name in SPLIT_NAMES:
        append_json_lines(paths[name], split_records[name])
    append_json_lines(paths['errors'], error_records)
    append_json_lines(paths['timings'], [timings])
    return stats


def restore_pending_batch(
    state: dict[str, Any],
    paths: dict[str, Path],
    state_path: Path,
) -> None:
    """Roll back partial appends left by an interrupted batch."""
    pending = state.get('pending_batch')
    if pending is None:
        return
    truncate_files(paths, pending['output_offsets'])
    state['pending_batch'] = None
    state['output_offsets'] = pending['output_offsets']
    write_json_atomic(state_path, state)
    print(
        f'Restored outputs before unfinished batch beginning at source line '
        f'{pending["start_source_line"]:,}.',
        flush=True,
    )


def convert_proofs(args: argparse.Namespace) -> dict[str, Any]:
    """Run the resumable LeanDojo extraction pipeline."""
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    project_path = args.project.resolve()
    work_path = (args.work_dir or default_work_path(output_path)).resolve()
    state_path = work_path / 'state.json'
    workspace = work_path / 'repository'
    split = tuple(args.split)

    if not input_path.is_file():
        raise FileNotFoundError(f'Input JSONL does not exist: {input_path}')
    work_path.mkdir(parents=True, exist_ok=True)
    paths = transactional_paths(output_path, work_path)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f'Resume state does not exist: {state_path}')
        state = json.loads(state_path.read_text(encoding='utf-8'))
        validate_state(
            state,
            input_path,
            output_path,
            project_path,
            split,
            args.seed,
        )
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f'Resume output does not exist: {path}')
        restore_pending_batch(state, paths, state_path)
    else:
        if state_path.exists():
            raise FileExistsError(
                f'Run state already exists: {state_path}. Use --resume or choose '
                'another --work-dir.'
            )
        state = initial_state(
            input_path,
            output_path,
            project_path,
            split,
            args.seed,
        )
        for path in paths.values():
            path.write_bytes(b'')
        state['output_offsets'] = file_offsets(paths)
        write_json_atomic(state_path, state)

    setup_started = time.perf_counter()
    initialize_workspace(workspace, project_path)
    print(
        f'[setup] workspace ready in {time.perf_counter() - setup_started:.3f}s; '
        f'resuming at source line {state["next_source_line"]:,}',
        flush=True,
    )

    run_started = time.perf_counter()
    with input_path.open('rb') as input_file:
        input_file.seek(state['next_input_offset'])
        while args.limit is None or state['records_seen'] < args.limit:
            remaining = (
                args.batch_size
                if args.limit is None
                else min(args.batch_size, args.limit - state['records_seen'])
            )
            records, next_source_line, next_input_offset = read_batch(
                input_file,
                state['next_source_line'],
                remaining,
            )
            if not records:
                break

            batch_number = state['records_seen'] // args.batch_size + 1
            offsets_before = file_offsets(paths)
            state['pending_batch'] = {
                'start_source_line': records[0].source_line,
                'end_source_line': records[-1].source_line,
                'output_offsets': offsets_before,
            }
            write_json_atomic(state_path, state)

            print(
                f'[batch {batch_number}] tracing source lines '
                f'{records[0].source_line:,}-{records[-1].source_line:,}',
                flush=True,
            )
            results, timings = process_batch(workspace, records, args.threads)
            timings.update({
                'batch': batch_number,
                'start_source_line': records[0].source_line,
                'end_source_line': records[-1].source_line,
                'records': len(records),
            })
            batch_stats = write_batch_results(
                paths,
                results,
                timings,
                split,
                args.seed,
            )

            state['next_source_line'] = next_source_line
            state['next_input_offset'] = next_input_offset
            state['records_seen'] += len(records)
            for name, value in batch_stats.items():
                state[name] += value
            state['pending_batch'] = None
            state['output_offsets'] = file_offsets(paths)
            write_json_atomic(state_path, state)

            print(
                f'[batch {batch_number}] {timings["total_seconds"]:.3f}s total '
                f'(prepare {timings["prepare_seconds"]:.3f}s, '
                f'build {timings["build_seconds"]:.3f}s, '
                f'trace {timings["trace_seconds"]:.3f}s, '
                f'parse {timings["parse_seconds"]:.3f}s); '
                f'wrote {batch_stats["pairs_written"]:,} pairs',
                flush=True,
            )

    state['run_seconds'] = state.get('run_seconds', 0.0) + (
        time.perf_counter() - run_started
    )
    write_json_atomic(state_path, state)
    return state


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Trace completed Lean proofs with LeanDojo-v2 and extract atomic '
            'single-goal AlphaProof SFT state-action pairs.'
        )
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT_PATH)
    parser.add_argument(
        '--work-dir',
        type=Path,
        help='Persistent workspace and checkpoint directory.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Process at most this many source scripts in total.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Proof modules per resumable batch (default: {DEFAULT_BATCH_SIZE}).',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=os.cpu_count() or 1,
        help='Maximum proof files traced concurrently.',
    )
    parser.add_argument(
        '--split',
        type=float,
        nargs=3,
        metavar=('TRAIN', 'VALIDATION', 'TEST'),
        default=DEFAULT_SPLIT,
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the checkpoint in --work-dir.',
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    if args.threads < 1:
        parser.error('--threads must be positive.')
    if any(ratio < 0 for ratio in args.split):
        parser.error('--split ratios cannot be negative.')
    if abs(sum(args.split) - 1.0) > 1e-9:
        parser.error('--split ratios must sum to 1.')
    return args


def main() -> None:
    """Run state-action extraction."""
    args = parse_args()
    state = convert_proofs(args)
    print(json.dumps({
        name: state[name]
        for name in (
            'records_seen',
            'scripts_with_pairs',
            'pairs_written',
            'train_pairs_written',
            'validation_pairs_written',
            'test_pairs_written',
            'scripts_with_errors',
            'run_seconds',
        )
    }, indent=2))
    for name, path in output_paths(args.output.resolve()).items():
        print(f'Wrote {name} state-action pairs to {path}')
    print(f'Wrote extraction errors to {errors_path(args.output.resolve())}')


if __name__ == '__main__':
    main()
