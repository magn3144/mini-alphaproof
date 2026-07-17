import subprocess
import tempfile
from pathlib import Path

from alphaproof.core.environment import Action, NodeType, Observation, Theorem
from alphaproof.core.helper import replace_sorry_proof, theorem_for_game, theorem_name
from alphaproof.core.paths import LEAN_PROJECT_DIR
from leantree import LeanTactic


class Node:
    """Node in the search tree."""

    def __init__(
        self,
        action: Action | None,
        observation: Observation,
        prior: float,
        state_id: int,
        node_type: NodeType,
        reward: float,
        is_optimal: bool = False,
        is_terminal: bool = False,
    ):
        """Initialize a search node reached by an optional incoming action."""
        # Action that was taken to reach this node.
        self.action = action
        # Observation after the action has been applied.
        self.observation = observation
        # Environment state ID after the action has been applied.
        self.state_id = state_id
        # Whether the node is an OR or AND node.
        self.node_type = node_type
        # Whether the action closed the proof of the previous goal.
        self.is_terminal = is_terminal
        # Whether the node is part of an optimal path.
        self.is_optimal = is_optimal
        # Per-step reward obtained after applying the action.
        self.reward = reward
        # Prior probability of the node according to the policy.
        self.prior = prior

        self.visit_count: int = 0
        self.evaluations: int = 0
        self.value_sum: float = 0.0
        self.children: dict[Action, Node] = {}

        # Not used in search, but used as a regression target in RL.
        self.value_target: float = 0.0

    def expanded(self) -> bool:
        """Return whether this node has children."""
        return len(self.children) > 0

    def value(self) -> float:
        """Return the average backed-up value."""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def prior_sum(self) -> float:
        """Return the sum of child policy priors."""
        return sum(child.prior for child in self.children.values())


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
                node_type=NodeType.OR,
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
    """Select the proven action with the shortest solution."""
    assert node.node_type == NodeType.OR
    optimal_children = [
            (action, child)
            for action, child in node.children.items()
            if child.is_optimal
    ]
    if not optimal_children:
        raise ValueError('Node has no proven action.')
    action, _ = min(
            optimal_children,
            key=lambda item: (
                    solution_length(item[1]),
                    -item[1].prior,
                    str(item[0]),
            ),
    )
    return action


def solution_length(node: Node) -> int:
    """Return the number of steps in the shortest proven solution."""
    if node.is_terminal:
        return 0
    if node.node_type == NodeType.OR:
        action = select_optimal_action(node)
        return 1 + solution_length(node.children[action])
    if node.node_type == NodeType.AND:
        return max(solution_length(child) for child in node.children.values())
    raise ValueError(f'Unknown node type: {node.node_type}')


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
            child_lines = extract_proof_script(child, indent + 2)
            lines.extend(_bullet_lines(child_lines, indent))
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


def run_lean(
        lean_code: str,
        prefix: str = 'AlphaProofCheck',
) -> subprocess.CompletedProcess[str]:
    """Run Lean on generated code and return the completed process."""
    LEAN_PROJECT_DIR.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        'w',
        dir=LEAN_PROJECT_DIR,
        suffix='.lean',
        prefix=prefix,
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

    return result


def run_lean_check(lean_code: str) -> bool:
    """Run Lean on generated code and reject sorry-backed proofs."""
    result = run_lean(lean_code, prefix='AlphaProofFinalCheck')
    return result.returncode == 0


def _indent(text: str, spaces: int) -> str:
    prefix = ' ' * spaces
    return '\n'.join(prefix + line if line else line for line in text.splitlines())


def _bullet_lines(child_lines: list[str], indent: int) -> list[str]:
    bullet = ' ' * indent + '·'
    if not child_lines:
        return [bullet]

    first_line = child_lines[0]
    child_indent = ' ' * (indent + 2)
    if '\n' in first_line or not first_line.startswith(child_indent):
        return [bullet, *child_lines]

    return [
            f'{bullet} {first_line.removeprefix(child_indent)}',
            *child_lines[1:],
    ]


def _is_internal_action(tactic: str) -> bool:
    return tactic == 'disprove' or tactic.startswith('focus_goal ')
