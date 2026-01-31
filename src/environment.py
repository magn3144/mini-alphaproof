"""Lean environment interface for AlphaProof."""

# pylint: disable=all

import enum
import typing
from typing import Any, Callable

from lean_interact import LeanREPLConfig, LeanServer, Command, ProofStep
from lean_interact.interface import LeanError


# Observations in AlphaProof are the tactic state.
Observation = str

# Actions in AlphaProof are Lean tactics (except for special actions, to start a
# disproof, or to focus on a goal).
Action = str

# Network parameters.
Params = Any


class Player(enum.Enum):
  OR = 1
  AND = 2


class State(typing.NamedTuple):
  id: int
  reward: float
  observation: Observation
  terminal: bool
  num_goals: int


class Theorem(typing.NamedTuple):
  """A theorem to be proved."""
  header: str
  statement: str


class Environment:
  """Lean environment."""

  def __init__(self):
    self.server = LeanServer(LeanREPLConfig())
    self._next_id = 0
    # state_id -> (proof_state_int, env_int)
    self._states: dict[int, tuple[State, int]] = {}

  def _alloc(self, proof_state: State, env: int) -> int:
    sid = self._next_id
    self._next_id += 1
    self._states[sid] = (proof_state, env)
    return sid

  def initial_state(self, theorem: Theorem, disprove: bool) -> State:
    """Returns the initial tactic state."""
    if disprove:
      statement = f"¬ ({theorem.statement})"
    else:
      statement = theorem.statement
    cmd = f"{theorem.header}\ntheorem _target : {statement} := by sorry"
    result = self.server.run(Command(cmd=cmd))
    if isinstance(result, LeanError):
      raise ValueError(result.message)
    sorry = result.sorries[0]
    # Get full tactic state (hypotheses + goal) by running skip.
    skip_result = self.server.run(ProofStep(tactic="skip", proof_state=sorry.proof_state))
    if isinstance(skip_result, LeanError):
      raise ValueError(f"Failed to get tactic state: {skip_result.message}")
    sid = self._alloc(sorry.proof_state, result.env)
    return State(
        id=sid,
        reward=0.0,
        observation=skip_result.goals[0],
        terminal=False,
        num_goals=1,
    )

  def _isolate_goal(self, proof_state: State, goal_index: int, num_goals: int) -> tuple[State, str]:
    """Isolate a single goal by sorry-ing all others (internalSorry).

    Returns (isolated_proof_state, goal_string).
    """
    if num_goals <= 1:
      result = self.server.run(ProofStep(tactic="skip", proof_state=proof_state))
      if isinstance(result, LeanError):
        raise RuntimeError(f"Failed to skip: {result.message}")
      return proof_state, result.goals[0]
    ps = proof_state
    result = self.server.run(ProofStep(tactic=f"rotate_left {goal_index + 1}", proof_state=ps))
    if isinstance(result, LeanError):
      raise RuntimeError(f"Failed to rotate: {result.message}")
    ps = result.proof_state
    for _ in range(num_goals - 1):
      result = self.server.run(ProofStep(tactic="sorry", proof_state=ps))
      if isinstance(result, LeanError):
        raise RuntimeError(f"Failed to sorry: {result.message}")
      ps = result.proof_state
    # Get the full tactic state (hypotheses + goal) from the isolated proof state.
    result = self.server.run(ProofStep(tactic="skip", proof_state=ps))
    if isinstance(result, LeanError):
      raise RuntimeError(f"Failed to skip after isolation: {result.message}")
    return ps, result.goals[0]

  def step(self, state_id: int, action: Action) -> list[State] | None:
    """Applies the action in the given state, returns a list of states (one per goal)."""
    proof_state, env = self._states[state_id]
    result = self.server.run(ProofStep(tactic=action, proof_state=proof_state))
    if isinstance(result, LeanError):
      raise ValueError(result.message)
    goals = result.goals
    n = len(goals)
    if n == 0:
      return [State(
          id=self._alloc(result.proof_state, env),
          reward=-1.0,
          observation=None,
          terminal=True,
          num_goals=0,
      )]
    states = []
    for i in range(n):
      isolated_ps, tactic_state = self._isolate_goal(result.proof_state, i, n)
      states.append(State(
          id=self._alloc(isolated_ps, env),
          reward=-1.0,
          observation=tactic_state,
          terminal=False,
          num_goals=1,
      ))
    return states


class Node:
  """Node in the search tree."""

  def __init__(
      self,
      action: Action | None,
      observation: Observation,
      prior: float,
      state_id: int,
      to_play: Player,
      reward: float,
      is_optimal: bool = False,
      is_terminal: bool = False,
  ):
    # Action that was taken to reach this node.
    self.action = action
    # Observation after the action has been applied.
    self.observation = observation
    # Environment state ID after the action has been applied.
    self.state_id = state_id
    # Whether the node is an OR or AND node.
    self.to_play = to_play
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
    self.value_target = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def prior_sum(self) -> float:
    return sum(child.prior for child in self.children.values())


class Game:
  """A single episode of interaction with the environment."""

  def __init__(self, theorem: Theorem, disprove: bool, num_simulations: int):
    self.theorem = theorem
    # Whether to try to prove or disprove the theorem.
    self.disprove = disprove
    # Number of simulations to run. Provided by the matchmaker.
    self.num_simulations = num_simulations
    # Dummy node for the type checker.
    self.root = Node(
        action=None,
        observation='',
        prior=1.0,
        state_id=0,
        to_play=Player.OR,
        reward=0.0,
    )


class Config:

  def __init__(
      self,
      num_simulations: int,
      batch_size: int,
      num_actors: int,
      lr: float,
      environment_ctor: Callable[[], Environment] = None,
  ):
    from src.environment import Environment
    if environment_ctor is None:
      environment_ctor = lambda: Environment()
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
