"""Network, training infrastructure, actors, and matchmaker for AlphaProof."""

# pylint: disable=all

import dataclasses
import random
import typing
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from src.environment import (
    Environment, Config, Node, Game, Theorem, Player, Observation, Action, Params
)
from lean_interact import LeanREPLConfig, LeanServer, Command
from lean_interact.interface import LeanError
from mcts import run_mcts


##### Helpers #####


def compute_value_target(node: Node) -> float:
  """Computes the actual value for a node, to be used as a target in learning."""
  if node.is_terminal:
    node.value_target = 0
    return 0
  elif node.to_play == Player.OR:
    action = select_optimal_action(node)
    child_value = compute_value_target(node.children[action])
    value = -1 + child_value
    node.value_target = value
    return value
  elif node.to_play == Player.AND:
    value = min(compute_value_target(child) for child in node.children.values())
    node.value_target = value
    return value
  else:
    raise ValueError(f'Unknown to_play: {node.to_play}')


def extract_transitions(node: Node) -> list[tuple[Observation, Action, float]]:
  """Extracts transitions from the game."""
  if not node.is_optimal:
    return []
  assert node.to_play == Player.OR
  transitions = []
  while node.to_play == Player.OR and not node.is_terminal:
    action = select_optimal_action(node)
    transitions.append((node.observation, action, node.value_target))
    node = node.children[action]
  if node.to_play == Player.AND:
    for _, child in node.children.items():
      transitions.extend(extract_transitions(child))
  return transitions


def select_optimal_action(node: Node) -> Action:
  """Selects the optimal action from the node."""
  assert node.to_play == Player.OR
  [(action, _)] = [
      (action, child)
      for action, child in node.children.items()
      if child.is_optimal
  ]
  return action


def _extract_tactics(node: Node) -> list[str]:
  """Extract the sequence of tactics from the optimal path in the tree."""
  tactics = []
  while not node.is_terminal:
    if node.to_play == Player.OR:
      action = select_optimal_action(node)
      tactics.append(action)
      node = node.children[action]
    elif node.to_play == Player.AND:
      # AND node: all children must be solved. Recurse into each,
      # wrapping sub-proofs with `· ` (Lean focusing syntax).
      for child in node.children.values():
        child_tactics = _extract_tactics(child)
        if child_tactics:
          tactics.extend([f"· {t}" for t in child_tactics])
      break
  return tactics


def final_check(game: Game) -> bool:
  """Checks that the proof found is actually valid."""
  tactics = _extract_tactics(game.root)
  if not tactics:
    return False

  theorem = game.theorem
  proof_body = "\n  ".join(tactics)

  code = (
      f"{theorem.header}\n"
      f"theorem _final_check : {theorem.statement} := by\n"
      f"  {proof_body}\n"
  )

  server = LeanServer(LeanREPLConfig())
  # Check the proof compiles without errors or sorries.
  result = server.run(Command(cmd=code))
  if isinstance(result, LeanError):
    return False
  if result.sorries:
    return False

  # Check axioms to ensure no sorry-based axioms were used.
  axiom_result = server.run(Command(cmd="#print axioms _final_check", env=result.env))
  if isinstance(axiom_result, LeanError):
    return False
  for msg in axiom_result.messages:
    if "sorry" in msg.data.lower():
      return False

  return True


def value_loss(value_logits: torch.Tensor, value_targets: float) -> float:
  # Calculate the categorical cross-entropy loss.
  return 0.0


def make_config() -> Config:
  return Config(
      num_simulations=800,
      batch_size=2048,
      num_actors=3000,
      lr=1.0,
  )


def launch_job(f, *args):
  f(*args)


##### End Helpers #####


class NetworkTrainingOutput(typing.NamedTuple):
  """Output of the network during training."""
  value_logits: torch.Tensor
  policy_logits: torch.Tensor


class NetworkSamplingOutput(typing.NamedTuple):
  """Output of the network when sampling actions."""
  action_logprobs: Dict[Action, float]
  value: float


class Network:
  def __init__(self, config: Config):
    self.params = {'weights': torch.tensor([0.0], requires_grad=True)}

    self.num_value_bins = config.num_value_bins
    self.value_weight = config.value_weight
    self.optimizer = torch.optim.Adam([self.params['weights']], lr=config.lr)

  def _compute_loss(self, batch):
    loss = torch.tensor(0.0, requires_grad=True)
    for observations, actions, value_targets in batch:
      network_output = self.forward(self.params, observations, actions)
      # Policy loss
      policy_loss = F.cross_entropy(
          network_output.policy_logits, actions
      )
      # Value loss
      v_loss = value_loss(network_output.value_logits, value_targets)
      loss = loss + policy_loss + self.value_weight * v_loss

    return loss

  def forward(
      self, params: Params, observation: torch.Tensor, action: torch.Tensor
  ) -> NetworkTrainingOutput:
    # Predict value logits and policy logits from given observation and action.
    # observation and action are passed to the network.
    value_logits = torch.zeros(self.num_value_bins)
    policy_logits = torch.tensor([0.0])
    return NetworkTrainingOutput(
        value_logits=value_logits, policy_logits=policy_logits
    )

  def sample(self, observation: str) -> NetworkSamplingOutput:
    # Predict value and sample actions from a given observation.
    # observation is tokenized and passed to the network to produce value
    # logits. The value is then calcualated from value logits and bin locations.
    value = 0.
    return NetworkSamplingOutput(action_logprobs={'placeholder_action': -2.},
                                 value=value)

  def update(self, batch: list[tuple[torch.Tensor, torch.Tensor, float]]):
    # Update the network weights.
    self.optimizer.zero_grad()
    loss = self._compute_loss(batch)
    loss.backward()
    self.optimizer.step()


class ReplayBuffer:

  def __init__(self, config: Config):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.sequence_length = config.sequence_length
    self.buffer = []

  def save_game(self, game):
    transitions = extract_transitions(game.root)
    self.buffer.extend(transitions)
    self.buffer = self.buffer[-self.window_size:]

  def sample_batch(self) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    return [self.sample_transition() for _ in range(self.batch_size)]

  def sample_transition(self) -> tuple[torch.Tensor, torch.Tensor, float]:
    # Sample transition from buffer either uniformly or according to some
    # priority.
    observation, action, value = self.buffer[0]
    tokenized_observation = self.tokenize(observation)
    tokenized_action = self.tokenize(action)
    return (tokenized_observation, tokenized_action, value)

  def tokenize(self, input_string: str) -> torch.Tensor:
    return torch.zeros((self.batch_size, self.sequence_length), dtype=torch.int32)


class SharedStorage:

  def __init__(self):
    self._params = {}

  def latest_params(self) -> Params:
    return self._params[max(self._params.keys())]

  def save_params(self, step: int, params: Params):
    self._params[step] = params


class Matchmaker:

  @dataclasses.dataclass
  class Stats:
    """Statistics for a theorem."""
    # List of (disprove, result) tuples:
    # Disprove is True iff this was an attempt to disprove the theorem.
    # Result is True iff the attempt was successful.
    attempts: list[tuple[bool, bool]]

    def update(self, game: Game):
      """Update statistics with the results of a game."""
      self.attempts.append((game.disprove, game.root.is_optimal))

    def weight(self, config: Config) -> float:
      """Compute weight of this theorem."""
      if not self.attempts:
        return 1.0
      disproved = any(
          disprove and success for (disprove, success) in self.attempts
      )
      proved = any(
          (not disprove) and success for (disprove, success) in self.attempts
      )
      if disproved:
        return 0.0
      elif len(self.attempts) < config.mm_trust_count:
        return 1.0
      elif not disproved and not proved:
        # Never managed to prove or disprove.
        return config.mm_undecided_weight
      else:
        latest = self.attempts[-config.mm_fully_decided_trust_count :]
        if all((not disprove) and success for (disprove, success) in latest):
          # Consistently proved.
          return config.mm_proved_weight
      return 1.0

  def __init__(self, config: Config):
    self.config = config
    # Load theorems and their stats from the database.
    self.theorem_stats: dict[Theorem, Matchmaker.Stats] = {}

  def compute_num_simulations(self, theorem: Theorem, stats: Stats) -> int:
    """Compute number of simulations to run for a theorem."""
    return 1000

  def get_start_position(self) -> Game:
    """Get a start position for a new game to be played."""
    # Get a theorem to be proved or disproved based on the per-theorem stats.
    # Prefer interesting theorems.
    weights = [
        stats.weight(self.config) for stats in self.theorem_stats.values()
    ]
    [(theorem, stats)] = random.choices(
        list(self.theorem_stats.items()), weights, k=1
    )
    disprove = random.random() < self.config.mm_disprove_rate
    num_simulations = self.compute_num_simulations(theorem, stats)
    return Game(
        theorem=theorem, disprove=disprove, num_simulations=num_simulations
    )

  def send_game(self, game: Game):
    """Send completed game to matchmaker."""
    self.theorem_stats[game.theorem].update(game)


##### RL part 1: Actors #####


# Each acting job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the learner by
# writing it to a shared replay buffer.
def run_actor(config: Config, storage: SharedStorage,
              replay_buffer: ReplayBuffer, matchmaker: Matchmaker):
  network = Network(config)
  while True:
    network.params = storage.latest_params()
    game = play_game(config, network, matchmaker)
    if game.root.is_optimal:
      replay_buffer.save_game(game)
    matchmaker.send_game(game)


# Each game is produced by starting from the initial Lean state, and executing
# Monte Carlo tree search to find a proof. If one is found, we extract from the
# search tree the state-tactic-value transitions in the proof, which are added
# to a replay buffer for training.
def play_game(config: Config, network: Network, matchmaker: Matchmaker) -> Game:
  game = matchmaker.get_start_position()
  environment = config.environment_ctor()

  state = environment.initial_state(game.theorem, game.disprove)
  game.root = Node(
      action=None,
      observation=state.observation,
      prior=1.0,
      to_play=Player.OR,
      state_id=state.id,
      is_optimal=state.terminal,
      is_terminal=state.terminal,
      reward=state.reward,
  )
  assert game.root.to_play == Player.OR

  # Run Monte Carlo tree search to find a proof.
  run_mcts(config, game, network, environment)
  if game.root.is_optimal:
    # Perform final check to ensure the proof is valid.
    game.root.is_optimal = final_check(game)
    # Compute value targets for the proof.
    compute_value_target(game.root)

  return game


##### End Actors #####

##### RL part 2: Learning #####


def train_network(config: Config, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):

  network = Network(config)

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_params(i, network.params)
    batch = replay_buffer.sample_batch()
    network.update(batch)
  storage.save_params(config.training_steps, network.params)


# AlphaProof training is split into two independent parts: A learner which
# updates the network, and actors which play games to generate data.
# These two parts only communicate by transferring the latest network checkpoint
# from the learner to the actor, and the finished games from the actor
# to the learner.
def alphaproof_train(config: Config) -> Network:
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)
  matchmaker = Matchmaker(config)

  for _ in range(config.num_actors):
    launch_job(run_actor, config, storage, replay_buffer, matchmaker)

  train_network(config, storage, replay_buffer)

  return storage.latest_params()
