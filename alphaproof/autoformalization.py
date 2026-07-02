from __future__ import annotations

import enum
import re
from collections import Counter
from pathlib import Path

from alphaproof.environment import Environment
from alphaproof.goedel_prover import GoedelProver
from leantree import LeanProject


THEOREM_NAME = 'generated_problem'
LEAN_PROJECT_DIR = Path(__file__).resolve().parent.parent / 'lean_project'
_goedel_prover: GoedelProver | None = None


def sample_auto_formalization(nl_problem: str) -> str:
    """Samples a Lean formalization using Goedel Prover."""
    global _goedel_prover
    if _goedel_prover is None:
        _goedel_prover = GoedelProver().load()

    prompt = f"""<task>
Translate the natural language math problem into one Lean 4 theorem statement.
</task>

<instructions>
Return only the Lean theorem statement.
Do not include imports, a proof, ":= by", "sorry", comments, markdown, or explanation.
</instructions>

<example>
<problem>
For every natural number n, n plus zero equals n.
</problem>
<output>
theorem problem_name (n : Nat) : n + 0 = n
</output>
</example>

<problem>
{nl_problem}
</problem>

<output>
"""
    return _goedel_prover.sample(prompt, num_samples=1)[0]


def extract_lean_code(sample: str) -> str:
    """Extracts the Lean code from a sample."""
    statement = re.sub(
        r'^\s*theorem\s+\S+',
        f'theorem {THEOREM_NAME}',
        sample,
        count=1,
    )
    return statement + ' := by\n  sorry'


def lean_is_valid_syntax(
        lean_statement: str,
        environment: Environment | None = None,
) -> bool:
    """Validates Lean code for syntax and common linting errors."""
    if environment is not None:
        try:
            environment.initial_state(lean_statement)
            return True
        except Exception:
            return False

    with Environment(LeanProject(str(LEAN_PROJECT_DIR))) as environment:
        return lean_is_valid_syntax(lean_statement, environment)


def lean_is_complete_proof(lean_code: str) -> bool:
    """Checks if Lean accepts the code as a full proof."""
    raise NotImplementedError()


def lean_replace_goal_with_false(lean_code: str) -> str:
    """Creates a new statement where the goal is to prove a contradiction is among the hypotheses."""
    raise NotImplementedError()


def lean_negate_statement(lean_code: str) -> str:
    """Creates a new statement where the goal is to disprove the original statement."""
    raise NotImplementedError()


def is_provable(lean_statement: str) -> bool:
    """Runs Alphaproof to check if the Lean statement is provable."""
    raise NotImplementedError()


def has_trivial_counterexample(lean_statement: str) -> bool:
    """Run a modified version of Lean's `plausible` tactic with extra support for real numbers."""
    raise NotImplementedError()


def is_easily_provable(lean_statement: str) -> bool:
    """Checks if the statement can be easily decided by an ad-hoc set of simple tactics."""
    # try to prove the statement
    for tactic in [
            "simp",
            "norm_num",
            "abel",
            "nlinarith",
            "linarith",
            "ring",
            "aesop",
            "trivial",
    ]:
        if lean_is_complete_proof(lean_statement + " := by " + tactic):
            return True

    return False


def deformalize_lean(lean_statement: str) -> str:
    """Deformalizes a Lean statement into a natural language statement."""
    # Uses an off-the-shelf, publicly available model.
    raise NotImplementedError()


def check_cycle_consistency(
        original_statement: str,
        deformalized_statement: str,
) -> bool:
    """Checks if the original and deformalized statements are equivalent."""
    # Uses an off-the-shelf, publicly available model.
    raise NotImplementedError()


def auto_formalize_problem(nl_problem: str, n_samples: int) -> str | None:
    """Translates for a natural language statement into a Lean statement."""

    samples = [sample_auto_formalization(nl_problem) for _ in range(n_samples)]
    lean_problems = [extract_lean_code(sample) for sample in samples]
    vote_counter = Counter(lean_problems)  # Deduplicate and count votes.

    problems_with_votes = [
            (votes, problem) for problem, votes in vote_counter.items()
    ]
    problems_with_votes.sort(reverse=True)  # Order by votes (most to least).

    with Environment(LeanProject(str(LEAN_PROJECT_DIR))) as environment:
        # Find the most-voted candidate that passes sanity checking.
        for _, lean_problem in problems_with_votes:
            # Remove samples that do not have a valid Lean syntax.
            if not lean_is_valid_syntax(lean_problem, environment):
                continue

            # Create two new Lean statements: one where the goal is to disprove the
            # original statement, and one where the goal is to prove the hypotheses
            # are contradictory.
            lean_negated = lean_negate_statement(lean_problem)
            lean_exfalso = lean_replace_goal_with_false(lean_problem)

            # Discard statements that have a single-tactic proof.
            if (
                    is_easily_provable(lean_problem)
                    or is_easily_provable(lean_negated)
                    or is_easily_provable(lean_exfalso)
            ):
                continue

            # Discard statements that have a trivial counterexamples.
            if has_trivial_counterexample(lean_problem):
                continue

            # Check cycle consistency: ask a model to deformalize a statement,
            # then ask if the original and deformalized statements are equivalent.
            deformalized_stmt = deformalize_lean(lean_problem)
            if not check_cycle_consistency(nl_problem, deformalized_stmt):
                continue

            # Use small-budget Alphaproof to check if the statement is disprovable or
            # the hypotheses are contradictory.
            if is_provable(lean_negated) or is_provable(
                    lean_exfalso
            ):
                continue

            return lean_problem

    # All samples failed.
    return None
