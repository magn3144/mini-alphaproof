from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaproof.core.config import Config


_PROOF_SEPARATOR = ':= by sorry'


def theorem_for_game(theorem: str, disprove: bool) -> str:
    """Return the Lean theorem declaration used for this game objective."""
    if disprove:
        return negate_theorem(theorem)
    return theorem


def negate_theorem(theorem: str) -> str:
    """Create a theorem whose goal is the negation of the original goal."""
    before_proof, proof = _split_sorry_proof(theorem)
    goal_start = _find_goal_colon(before_proof)
    header = before_proof[:goal_start + 1]
    goal = before_proof[goal_start + 1:].strip()
    header = _rename_decl(header, '_disproof')
    return f'{header} ¬ ({goal}) {proof}'


def replace_goal_with_false(theorem: str) -> str:
    """Create a theorem whose hypotheses imply False."""
    before_proof, proof = _split_sorry_proof(theorem)
    goal_start = _find_goal_colon(before_proof)
    header = before_proof[:goal_start + 1]
    header = _rename_decl(header, '_exfalso')
    return f'{header} False {proof}'


def theorem_name(theorem: str) -> str | None:
    """Extract the declaration name if the theorem is named."""
    match = re.match(r'\s*(?:theorem|lemma)\s+([^\s(:]+)', theorem)
    if match is None:
        return None
    return match.group(1)


def replace_sorry_proof(theorem: str, proof_lines: list[str]) -> str:
    """Replace the theorem's single sorry with a generated tactic proof."""
    if theorem.count('sorry') != 1:
        raise ValueError('Expected theorem to contain exactly one sorry.')
    proof = '\n' + '\n'.join(proof_lines)
    if ' sorry' in theorem:
        return theorem.replace(' sorry', proof, 1)
    return theorem.replace('sorry', proof, 1)


def _split_sorry_proof(theorem: str) -> tuple[str, str]:
    if theorem.count('sorry') != 1:
        raise ValueError('Expected theorem to contain exactly one sorry.')
    separator_index = theorem.rfind(_PROOF_SEPARATOR)
    if separator_index == -1:
        raise ValueError(f'Expected theorem to end with `{_PROOF_SEPARATOR}`.')
    return (
            theorem[:separator_index].rstrip(),
            theorem[separator_index:],
    )


def _find_goal_colon(header: str) -> int:
    depth = 0
    goal_colon = -1
    opening = '([{'
    closing = ')]}'

    for index, char in enumerate(header):
        if char in opening:
            depth += 1
        elif char in closing:
            depth -= 1
        elif char == ':' and depth == 0:
            goal_colon = index

    if goal_colon == -1:
        raise ValueError('Could not find theorem goal.')
    return goal_colon


def _rename_decl(header: str, suffix: str) -> str:
    match = re.match(r'(\s*(?:theorem|lemma)\s+)([^\s(:]+)(.*)', header, re.S)
    if match is None:
        return header
    prefix, name, rest = match.groups()
    return f'{prefix}{name}{suffix}{rest}'


def make_config() -> 'Config':
    """Create the default pseudocode training configuration."""
    from alphaproof.core.config import Config

    return Config(
            num_simulations=800,
            batch_size=2048,
            num_actors=3000,
            num_games=1,
            lr=1.0,
    )
