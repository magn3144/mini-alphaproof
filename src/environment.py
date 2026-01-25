"""Lean environment interface for AlphaProof."""

# pylint: disable=all

import enum
import typing
from typing import Any


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

  def initial_state(self, theorem: Theorem) -> State:
    """Returns the initial tactic state."""
    raise NotImplementedError()

  def step(self, state_id: int, action: Action) -> State:
    """Applies the action in the given state, returns the new state."""
    raise NotImplementedError()
  

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
