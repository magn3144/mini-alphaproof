import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


TIMINGS_FILE = 'timings.jsonl'


def parse_args() -> argparse.Namespace:
    """Parse the run directory to summarize."""
    parser = argparse.ArgumentParser(description='Summarize RL run timings.')
    parser.add_argument('run_dir', type=Path)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load game timing records from JSONL."""
    with path.open(encoding='utf-8') as timing_file:
        records = [json.loads(line) for line in timing_file if line.strip()]
    if not records:
        raise ValueError(f'No timing records found in {path}.')
    return records


def print_distribution(name: str, values: list[float]) -> None:
    """Print a compact duration distribution."""
    print(
        f'{name}: count={len(values)}, mean={mean(values):.3f}s, '
        f'median={median(values):.3f}s, max={max(values):.3f}s'
    )


def main() -> None:
    """Print component and per-tactic timing summaries for one RL run."""
    args = parse_args()
    records = load_records(args.run_dir / TIMINGS_FILE)

    game_seconds = [float(record['total_seconds']) for record in records]
    generation_samples = [
        float(expansion['seconds'])
        for record in records
        for expansion in record['tactic_generation']['expansions']
    ]
    verification_samples = [
        float(record['final_verification']['seconds'])
        for record in records
        if record['final_verification'] is not None
    ]

    setup_seconds = sum(float(record['setup_seconds']) for record in records)
    generation_seconds = sum(
        float(record['tactic_generation']['total_seconds'])
        for record in records
    )
    tactic_seconds = sum(
        float(record['tactic_execution']['total_seconds'])
        for record in records
    )
    internal_seconds = sum(
        float(record['internal_actions']['total_seconds'])
        for record in records
    )
    verification_seconds = sum(verification_samples)
    verifier_startup_seconds = sum(
        float(record['final_verification']['verifier_startup_seconds'])
        for record in records
        if record['final_verification'] is not None
    )
    total_seconds = sum(game_seconds)
    other_seconds = total_seconds - sum((
        setup_seconds,
        generation_seconds,
        tactic_seconds,
        internal_seconds,
        verification_seconds,
    ))

    print_distribution('Games', game_seconds)
    print_distribution('Node tactic generation', generation_samples)
    if verification_samples:
        print_distribution('Final proof verification', verification_samples)
        print(f'Verifier startup total: {verifier_startup_seconds:.3f}s')

    print('\nTime by component')
    print(f'{"Component":<28} {"Seconds":>12} {"Percent":>10}')
    components = (
        ('Setup', setup_seconds),
        ('Tactic generation', generation_seconds),
        ('Tactic execution', tactic_seconds),
        ('Internal actions', internal_seconds),
        ('Final proof verification', verification_seconds),
        ('Other game work', other_seconds),
    )
    for name, seconds in sorted(components, key=lambda item: item[1], reverse=True):
        print(f'{name:<28} {seconds:>12.3f} {seconds / total_seconds:>9.1%}')

    tactic_totals: dict[str, dict[str, int | float]] = {}
    for record in records:
        for tactic in record['tactic_execution']['tactics']:
            name = str(tactic['tactic'])
            if name not in tactic_totals:
                tactic_totals[name] = {
                    'count': 0,
                    'successful_count': 0,
                    'total_seconds': 0.0,
                    'min_seconds': float(tactic['min_seconds']),
                    'max_seconds': float(tactic['max_seconds']),
                }
            total = tactic_totals[name]
            total['count'] += int(tactic['count'])
            total['successful_count'] += int(tactic['successful_count'])
            total['total_seconds'] += float(tactic['total_seconds'])
            total['min_seconds'] = min(
                float(total['min_seconds']),
                float(tactic['min_seconds']),
            )
            total['max_seconds'] = max(
                float(total['max_seconds']),
                float(tactic['max_seconds']),
            )

    print('\nTactics by total execution time')
    print(
        f'{"Tactic":<50} {"Count":>8} {"Success":>8} '
        f'{"Total":>10} {"Average":>10} {"Min":>10} {"Max":>10}'
    )
    ranked_tactics = sorted(
        tactic_totals.items(),
        key=lambda item: float(item[1]['total_seconds']),
        reverse=True,
    )
    for tactic, timing in ranked_tactics:
        count = int(timing['count'])
        successful_count = int(timing['successful_count'])
        total = float(timing['total_seconds'])
        minimum = float(timing['min_seconds'])
        maximum = float(timing['max_seconds'])
        display_tactic = tactic.replace('\n', '\\n')
        print(
            f'{display_tactic:<50} {count:>8} {successful_count:>8} '
            f'{total:>9.3f}s {total / count:>9.3f}s '
            f'{minimum:>9.3f}s {maximum:>9.3f}s'
        )


if __name__ == '__main__':
    main()
