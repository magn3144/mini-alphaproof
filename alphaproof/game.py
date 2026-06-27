from alphaproof.environment import Action, Node, NodeType, Observation, Theorem


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
    # Extract tactics from the tree, write the statement and its proof to a file,
    # add a footer checking the axioms, and then run the `lean` binary.
    # Properly handle the case where we attempt to disprove a theorem.
    return True
