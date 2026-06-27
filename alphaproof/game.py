import subprocess
import tempfile
from pathlib import Path

from alphaproof.helper import replace_sorry_proof, theorem_for_game, theorem_name
from alphaproof.environment import Action, Node, NodeType, Observation, Theorem
from leantree import LeanTactic


LEAN_PROJECT_DIR = Path('lean_project')


class Game:
    """A single episode of interaction with the environment."""

    def __init__(self, theorem: Theorem, disprove: bool, num_simulations: int):
        """Create an episode around one theorem objective."""
        self.theorem = theorem
        # Whether to try to prove or disprove the theorem.
        self.disprove = disprove
        # Number of simulations to run. Provided by the matchmaker.
        self.num_simulations = num_simulations
        # Dummy node for the type checker.
        self.root = Node(
                action=None,
                observation=Observation([]),
                prior=1.0,
                state_id=0,
                and_or=NodeType.OR,
                reward=0.0,
        )


def compute_value_target(node: Node) -> float:
    """Computes the actual value for a node, to be used as a target in learning."""
    if node.is_terminal:
        node.value_target = 0
        return 0
    elif node.node_type == NodeType.OR:
        action = select_optimal_action(node)
        child_value = compute_value_target(node.children[action])
        value = -1 + child_value
        node.value_target = value
        return value
    elif node.node_type == NodeType.AND:
        value = min(compute_value_target(child) for child in node.children.values())
        node.value_target = value
        return value
    else:
        raise ValueError(f'Unknown to_play: {node.node_type}')


def extract_transitions(node: Node) -> list[tuple[Observation, Action, float]]:
    """Extracts transitions from the game."""
    if not node.is_optimal:
        return []
    assert node.node_type == NodeType.OR
    transitions = []
    while node.node_type == NodeType.OR and not node.is_terminal:
        action = select_optimal_action(node)
        transitions.append((node.observation, action, node.value_target))
        node = node.children[action]
    if node.node_type == NodeType.AND:
        for _, child in node.children.items():
            transitions.extend(extract_transitions(child))
    return transitions


def select_optimal_action(node: Node) -> Action:
    """Selects the optimal action from the node."""
    assert node.node_type == NodeType.OR
    [(action, _)] = [
            (action, child)
            for action, child in node.children.items()
            if child.is_optimal
    ]
    return action


def final_check(game: Game) -> bool:
    """Checks that the proof found is actually valid."""
    theorem = theorem_for_game(game.theorem, game.disprove)
    try:
        proof_lines = extract_proof_script(game.root)
        lean_code = build_lean_check(theorem, proof_lines)
        return run_lean_check(lean_code)
    except Exception:
        return False


def extract_proof_script(node: Node, indent: int = 2) -> list[str]:
    """Extract a Lean tactic script from the optimal proof tree."""
    if node.is_terminal:
        return []

    if node.node_type == NodeType.OR:
        action = select_optimal_action(node)
        tactic = action_to_tactic(action)
        child = node.children[action]

        if _is_internal_action(tactic):
            return extract_proof_script(child, indent)
        return [
                _indent(tactic, indent),
                *extract_proof_script(child, indent),
        ]

    if node.node_type == NodeType.AND:
        lines = []
        for child in node.children.values():
            lines.append(' ' * indent + '·')
            lines.extend(extract_proof_script(child, indent + 2))
        return lines

    raise ValueError(f'Unknown node type: {node.node_type}')


def action_to_tactic(action: Action) -> str:
    """Convert an AlphaProof action into tactic text."""
    if isinstance(action, LeanTactic):
        return action.tactic
    return action


def build_lean_check(theorem: Theorem, proof_lines: list[str]) -> str:
    """Build the Lean file used for final proof checking."""
    declaration = replace_sorry_proof(theorem, proof_lines)
    footer = ''
    name = theorem_name(declaration)
    if name is not None:
        footer = f'\n\n#print axioms {name}'
    return f'import Mathlib\nset_option warningAsError true\n\n{declaration}{footer}\n'


def run_lean_check(lean_code: str) -> bool:
    """Run Lean on generated code and reject sorry-backed proofs."""
    LEAN_PROJECT_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        dir=LEAN_PROJECT_DIR,
        suffix='.lean',
        prefix='AlphaProofFinalCheck',
        delete=False,
    ) as file:
        file.write(lean_code)
        file_path = Path(file.name)

    try:
        result = subprocess.run(
                ['lake', 'env', 'lean', file_path.name],
                cwd=LEAN_PROJECT_DIR,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
        )
    finally:
        file_path.unlink(missing_ok=True)

    return result.returncode == 0


def _indent(text: str, spaces: int) -> str:
    prefix = ' ' * spaces
    return '\n'.join(prefix + line if line else line for line in text.splitlines())


def _is_internal_action(tactic: str) -> bool:
    return tactic == 'disprove' or tactic.startswith('focus_goal ')
