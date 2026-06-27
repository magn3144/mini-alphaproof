import typing
from typing import Any, Callable, List, Dict
import enum

from alphaproof.helper import negate_theorem
from leantree import LeanProject, LeanTactic, LeanProofState


# Observations in AlphaProof are the tactic state.
Observation = LeanProofState

# Actions in AlphaProof are Lean tactics (except for special actions, to start a
# disproof, or to focus on a goal).
Action = LeanTactic | str

Theorem = str


class NodeType(enum.Enum):
    """Node type used by the AND-OR search tree."""
    OR = 1
    AND = 2


class State(typing.NamedTuple):
    """Environment tactic state returned after applying an action."""
    id: int
    reward: float
    observation: Observation
    terminal: bool
    num_goals: int


class Environment:
    """Lean environment."""

    def __init__(
        self,
        project: LeanProject,
        imports: tuple[str, ...] = ('Mathlib',),
    ):
        """Create a LeanTree-backed proof environment."""
        self.project = project
        self.imports = imports
        self._env = self.project.environment()
        self._env.__enter__()
        self._sent_imports = False
        self._next_state_id = -1
        self._branches: dict[int, Any] = {}
        self._theorems: dict[int, Theorem] = {}

    def close(self) -> None:
        """Stop the underlying Lean process."""
        self._env.__exit__(None, None, None)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def get_next_state_id(self):
        self._next_state_id += 1
        return self._next_state_id

    def _send_imports(self) -> None:
        """Send configured imports once per Lean process."""
        if self._sent_imports:
            return
        for module in self.imports:
            self._env.send_command(f'import {module}')
        self._sent_imports = True

    def _state_from_branch(
        self,
        branch: Any,
        reward: float = 0.0,
        theorem: Theorem | None = None,
    ) -> State:
        """Store a LeanTree proof branch and expose it as an AlphaProof state."""
        state_id = self.get_next_state_id()
        self._branches[state_id] = branch
        if theorem is not None:
            self._theorems[state_id] = theorem

        observation = branch.state
        terminal = observation.is_solved()
        return State(
            id=state_id,
            reward=reward,
            observation=observation,
            terminal=terminal,
            num_goals=len(observation.goals),
        )

    def _state_from_branches(self, branches: list[Any], reward: float = 0.0) -> State:
        """Store LeanTree's factorized branches as one AlphaProof state."""
        if not branches:
            state_id = self.get_next_state_id()
            self._branches[state_id] = None
            return State(
                id=state_id,
                reward=reward,
                observation=LeanProofState([]),
                terminal=True,
                num_goals=0,
            )

        if len(branches) == 1:
            return self._state_from_branch(branches[0], reward=reward)

        state_id = self.get_next_state_id()
        self._branches[state_id] = branches
        observation = LeanProofState([
            goal
            for branch in branches
            for goal in branch.state.goals
        ])
        return State(
            id=state_id,
            reward=reward,
            observation=observation,
            terminal=False,
            num_goals=len(branches),
        )

    def initial_state(self, theorem: Theorem) -> State:
        """Returns the initial tactic state."""
        self._send_imports()
        branch = self._env.proof_from_sorry(theorem)
        return self._state_from_branch(branch, theorem=theorem)

    def step(self, state_id: int, action: Action) -> State:
        """Applies the action in the given state, returns the new state."""
        if state_id not in self._branches:
            raise ValueError(f'Unknown state id: {state_id}')

        branch = self._branches[state_id]
        tactic = action.tactic if isinstance(action, LeanTactic) else action

        if tactic == 'disprove':
            if state_id not in self._theorems:
                raise ValueError('Can only disprove from an initial theorem state.')
            theorem = negate_theorem(self._theorems[state_id])
            branch = self._env.proof_from_sorry(theorem)
            return self._state_from_branch(branch, theorem=theorem)

        if tactic.startswith('focus_goal '):
            branches = branch
            if not isinstance(branches, list):
                raise ValueError('Can only focus a state with multiple goals.')
            try:
                goal_index = int(tactic.removeprefix('focus_goal ').strip())
            except ValueError as exc:
                raise ValueError(f'Invalid focus action: {tactic}') from exc
            try:
                return self._state_from_branch(branches[goal_index])
            except IndexError as exc:
                raise ValueError(f'Goal index out of range: {goal_index}') from exc

        if branch is None:
            raise ValueError('Cannot apply a tactic to a terminal state.')
        if isinstance(branch, list):
            raise ValueError('Use focus_goal <i> before applying a tactic.')

        try:
            branches = branch.apply_tactic(action)
        except Exception as exc:
            raise ValueError(f'Invalid tactic {tactic!r}: {exc}') from exc
        return self._state_from_branches(branches)


class Config:
    """Hyperparameters and constructors used by the pseudocode pipeline."""

    def __init__(
        self,
        num_simulations: int,
        batch_size: int,
        num_actors: int,
        lr: float,
        environment_ctor: Callable[[], Environment] = (
            lambda: Environment(LeanProject('lean_project'))
        ),
    ):
        """Populate acting, search, training, and matchmaker settings."""
        ### Acting
        self.environment_ctor = environment_ctor
        self.num_actors = num_actors

        self.num_simulations = num_simulations

        # UCB formula
        self.pb_c_base = 3200
        self.pb_c_init = 0.001
        self.value_discount = 0.99
        self.prior_temperature = 200

        # Other MCTS parameters
        self.no_legal_actions_value = -40

        # Progressive sampling parameters
        self.ps_c = 0.01
        self.ps_alpha = 0.6

        # Value predictions
        self.num_value_bins = 64

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.sequence_length = 32
        self.lr = lr
        self.value_weight = 0.001

        # Matchmaker
        self.mm_disprove_rate = 0.5
        self.mm_trust_count = 8
        self.mm_fully_decided_trust_count = 12
        self.mm_proved_weight = 1e-3
        self.mm_undecided_weight = 0.1


class Node:
    """Node in the search tree."""

    def __init__(
        self,
        action: Action | None,
        observation: Observation,
        prior: float,
        state_id: int,
        and_or: NodeType,
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
        self.node_type = and_or
        # Whether the action closed the proof of the previous goal.
        self.is_terminal = is_terminal
        # Whether the node is part of an optimal path.
        self.is_optimal = is_optimal
        # Per-step reward obtained after applying the action.
        self.reward = reward
        # Prior probability of the node according to the policy.
        self.prior = prior

        self.visit_count = 0
        self.evaluations = 0
        self.value_sum = 0
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
