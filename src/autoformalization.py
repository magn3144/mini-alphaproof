"""Autoformalization: Natural language to Lean translation for AlphaProof."""

# pylint: disable=all

import collections
import enum
import re

from nobodywho import Chat
from lean_interact import LeanREPLConfig, LeanServer, Command
from lean_interact.interface import LeanError

Counter = collections.Counter

herald_chat = Chat("../models/herald_q4_k_m.gguf")
_lean_server = LeanServer(LeanREPLConfig())


class ComputeBudget(enum.Enum):
  LOW = "low"
  HIGH = "high"


def sample_auto_formalization(nl_problem: str) -> str:
  """Samples a Lean formalization using an LLM."""
  prompt = f"Instruction: Translate to Lean 4.\nInput: {nl_problem}\nOutput:"
  response = herald_chat.ask(prompt).completed()
  return response.strip()


def extract_lean_code(sample: str) -> str:
  """Extracts the Lean code from a sample."""
  stripped = sample.strip()
  for keyword in ("theorem", "lemma", "def"):
    idx = stripped.find(keyword)
    if idx != -1:
      return stripped[idx:]
  return stripped


def lean_is_valid_syntax(lean_statement: str) -> bool:
  """Validates Lean code for syntax and common linting errors."""
  result = _lean_server.run(Command(cmd=lean_statement))
  return not isinstance(result, LeanError)


def lean_is_complete_proof(lean_code: str) -> bool:
  """Checks if Lean accepts the code as a full proof."""
  result = _lean_server.run(Command(cmd=lean_code))
  if isinstance(result, LeanError):
    return False
  return len(result.sorries) == 0


def lean_replace_goal_with_false(lean_code: str) -> str:
  """Creates a new statement where the goal is to prove a contradiction is among the hypotheses."""
  # Match ": <goal> :=" and replace the goal with False
  return re.sub(r':\s*([^:=]+?)\s*:=', ': False :=', lean_code, count=1)


def lean_negate_statement(lean_code: str) -> str:
  """Creates a new statement where the goal is to disprove the original statement."""
  # Match ": <goal> :=" and wrap the goal in ¬(...)
  def _negate(m):
    goal = m.group(1).strip()
    return f': ¬({goal}) :='
  return re.sub(r':\s*([^:=]+?)\s*:=', _negate, lean_code, count=1)


def is_provable(lean_statement: str, budget: ComputeBudget) -> bool:
  """Runs Alphaproof to check if the Lean statement is provable."""
  # Stub: full AlphaProof search not yet available.
  return False


def has_trivial_counterexample(lean_statement: str) -> bool:
  """Run a modified version of Lean's `plausible` tactic with extra support for real numbers."""
  # Build a version of the statement that uses the plausible tactic
  code = re.sub(r':= by\s+sorry', ':= by plausible', lean_statement)
  if code == lean_statement:
    # No substitution happened, append manually
    code = lean_statement + " := by plausible"
  result = _lean_server.run(Command(cmd=code))
  # If plausible succeeds (no error), it found a counterexample
  return not isinstance(result, LeanError)


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
  herald_chat.system(
      "You are an expert at reading Lean 4 code and explaining it in plain "
      "natural language. Given a Lean 4 theorem statement, output a clear "
      "natural language description of what it states. Output only the "
      "natural language description, no code or extra commentary."
  )
  return herald_chat.chat(lean_statement)


def check_cycle_consistency(
    original_statement: str,
    deformalized_statement: str,
) -> bool:
  """Checks if the original and deformalized statements are equivalent."""
  herald_chat.system(
      "You are a math expert. You will be given two mathematical statements. "
      "Determine if they are mathematically equivalent. "
      "Answer with exactly 'yes' or 'no'."
  )
  prompt = (
      f"Statement 1: {original_statement}\n\n"
      f"Statement 2: {deformalized_statement}\n\n"
      "Are these two statements mathematically equivalent?"
  )
  response = herald_chat.chat(prompt).strip().lower()
  return response.startswith("yes")


def auto_formalize_problem(nl_problem: str, n_samples: int) -> str | None:
  """Translates for a natural language statement into a Lean statement."""

  samples = [sample_auto_formalization(nl_problem) for _ in range(n_samples)]
  lean_problems = [extract_lean_code(sample) for sample in samples]
  vote_counter = Counter(lean_problems)  # Deduplicate and count votes.

  problems_with_votes = [
      (votes, problem) for problem, votes in vote_counter.items()
  ]
  problems_with_votes.sort(reverse=True)  # Order by votes (most to least).

  # Find the most-voted candidate that passes sanity checking.
  for _, lean_problem in problems_with_votes:
    # Remove samples that do not have a valid Lean syntax.
    if not lean_is_valid_syntax(lean_problem):
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

    # Check cycle consistency: ask a public model to deformalize a statement,
    # then ask if the original and deformalized statements are equivalent.
    deformalized_stmt = deformalize_lean(lean_problem)
    if not check_cycle_consistency(nl_problem, deformalized_stmt):
      continue

    # Use small-budget Alphaproof to check if the statement is disprovable or
    # the hypotheses are contradictory.
    if is_provable(lean_negated, ComputeBudget.LOW) or is_provable(
        lean_exfalso, ComputeBudget.LOW
    ):
      continue

    return lean_problem

  # All samples failed.
  return None
