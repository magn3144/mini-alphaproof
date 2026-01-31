"""Network, training infrastructure, actors, and matchmaker for AlphaProof."""

# pylint: disable=all

import dataclasses
import random
import typing
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.environment import (
    Environment, Config, Node, Game, Theorem, Player, Observation, Action, Params
)
from src.models.heads import PolicyHead, ValueHead
from src.models.tokenizer import SimpleTokenizer
from src.models.transformer import TransformerEncoderDecoder

from lean_interact import LeanREPLConfig, LeanServer, Command
from lean_interact.interface import LeanError
from src.mcts import run_mcts

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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
  """Encoder-decoder network with policy and value heads.

  Supports three initialization modes:
  1. Random initialization from config
  2. Loading from checkpoint
  3. Loading pretrained model from HuggingFace
  """

  def __init__(
      self,
      config: Config,
      model_path: Optional[str] = None,
      pretrained_model_name: Optional[str] = None,
      hidden_dim: int = 512,
      num_layers: int = 6,
      num_heads: int = 8,
      dropout: float = 0.1,
      num_action_samples: int = 16,
  ):
    """Initialize network.

    Args:
        config: AlphaProof config
        model_path: Path to checkpoint (for loading saved model)
        pretrained_model_name: HuggingFace model name (e.g., "google/byt5-small")
        hidden_dim: Hidden dimension (for random init)
        num_layers: Number of encoder/decoder layers (for random init)
        num_heads: Number of attention heads (for random init)
        dropout: Dropout probability (for random init)
        num_action_samples: Number of actions to sample per state
    """
    self.config = config
    self.num_value_bins = config.num_value_bins
    self.value_weight = config.value_weight
    self.num_action_samples = num_action_samples
    self.params = {}  # For compatibility with existing code

    # Store model architecture params for saving
    self.model_num_layers = num_layers
    self.model_num_heads = num_heads
    self.model_dropout = dropout

    # Value bin centers for discretization
    self.value_bins = torch.linspace(
        config.no_legal_actions_value, 0, config.num_value_bins
    )

    # Determine initialization mode and build model
    if model_path is not None:
      self._load_from_checkpoint(model_path, config)
    elif pretrained_model_name is not None:
      self._load_pretrained(pretrained_model_name, config)
    else:
      self._build_random_model(
          hidden_dim, num_layers, num_heads, dropout, config
      )

    # Setup optimizer
    self.optimizer = torch.optim.Adam(
        self._get_trainable_parameters(), lr=config.lr
    )

  def _build_random_model(
      self,
      hidden_dim: int,
      num_layers: int,
      num_heads: int,
      dropout: float,
      config: Config,
  ):
    """Build model with random initialization."""
    self.is_pretrained = False
    self.pretrained_model_name = None

    # Create tokenizer
    self.tokenizer = SimpleTokenizer()

    # Create transformer
    self.transformer = TransformerEncoderDecoder(
        vocab_size=self.tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )

    self.hidden_dim = hidden_dim
    self.vocab_size = self.tokenizer.vocab_size

    # Create policy head (custom for random init)
    self.policy_head = self.transformer.output_projection  # Reuse output projection

    # Create value head
    self.value_head = ValueHead(hidden_dim, config.num_value_bins)

  def _load_pretrained(self, model_name: str, config: Config):
    """Load pretrained model from HuggingFace."""

    self.is_pretrained = True
    self.pretrained_model_name = model_name

    # Load pretrained model and tokenizer
    self.transformer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get hidden dimension from model config
    if hasattr(self.transformer.config, 'd_model'):
      self.hidden_dim = self.transformer.config.d_model
    elif hasattr(self.transformer.config, 'hidden_size'):
      self.hidden_dim = self.transformer.config.hidden_size
    else:
      raise ValueError(f"Cannot determine hidden_dim from {model_name} config")

    self.vocab_size = self.transformer.config.vocab_size

    # Policy head is the pretrained LM head (don't create new one)
    self.policy_head = None  # Will use transformer's built-in lm_head

    # Create value head (task-specific, always random init)
    self.value_head = ValueHead(self.hidden_dim, config.num_value_bins)

  def _load_from_checkpoint(self, path: str, config: Config):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model_config = checkpoint['model_config']
    self.is_pretrained = model_config.get('is_pretrained', False)

    # Restore model architecture params
    self.model_num_layers = model_config.get('num_layers', 6)
    self.model_num_heads = model_config.get('num_heads', 8)
    self.model_dropout = model_config.get('dropout', 0.1)

    if self.is_pretrained:
      # Reload pretrained model
      pretrained_name = model_config['pretrained_model_name']
      self._load_pretrained(pretrained_name, config)
      # Load transformer weights
      self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
    else:
      # Reconstruct custom transformer with SAME architecture
      self.tokenizer = SimpleTokenizer.from_config(checkpoint['tokenizer_config'])
      self.transformer = TransformerEncoderDecoder(
          vocab_size=model_config['vocab_size'],
          hidden_dim=model_config['hidden_dim'],
          num_layers=self.model_num_layers,
          num_heads=self.model_num_heads,
          dropout=self.model_dropout,
      )
      self.transformer.load_state_dict(checkpoint['transformer_state_dict'])

      self.hidden_dim = model_config['hidden_dim']
      self.vocab_size = model_config['vocab_size']
      self.policy_head = self.transformer.output_projection

    # Always load value head
    self.value_head = ValueHead(
        model_config['hidden_dim'], config.num_value_bins
    )
    self.value_head.load_state_dict(checkpoint['value_head_state_dict'])

  def _get_trainable_parameters(self):
    """Get all trainable parameters."""
    params = list(self.transformer.parameters())
    params.extend(self.value_head.parameters())
    return params

  def forward(
      self, params: Params, observation: torch.Tensor, action: torch.Tensor
  ) -> NetworkTrainingOutput:
    """Forward pass for training.

    Args:
        params: Placeholder for compatibility
        observation: Tokenized observation (batch_size, src_seq_len)
        action: Tokenized action (batch_size, tgt_seq_len)

    Returns:
        NetworkTrainingOutput with value_logits and policy_logits
    """
    if self.is_pretrained:
      # Use pretrained model's native forward
      outputs = self.transformer(
          input_ids=observation,
          labels=action,
          output_hidden_states=True,
      )
      policy_logits = outputs.logits[:, -1, :]  # Last token logits
      decoder_hidden = outputs.decoder_hidden_states[-1][:, -1, :]
    else:
      # Use custom transformer
      decoder_output, output_logits = self.transformer(observation, action)
      policy_logits = output_logits[:, -1, :]  # Last token logits
      decoder_hidden = decoder_output[:, -1, :]

    # Compute value from decoder hidden state
    value_logits = self.value_head(decoder_hidden)

    return NetworkTrainingOutput(
        value_logits=value_logits, policy_logits=policy_logits
    )

  def sample(self, observation: str) -> NetworkSamplingOutput:
    """Sample multiple actions from observation.

    Uses encoder caching optimization: encodes state once, then samples
    multiple actions from the cached encoder output.

    Args:
        observation: Observation string (tactic state)

    Returns:
        NetworkSamplingOutput with action_logprobs and value
    """
    # Tokenize observation
    obs_tokens = self.tokenizer.encode(
        observation, add_special_tokens=True, return_tensors="pt"
    )

    with torch.no_grad():
      if self.is_pretrained:
        # Encode once with pretrained model
        encoder = self.transformer.get_encoder()
        encoder_outputs = encoder(input_ids=obs_tokens)
        encoder_hidden = encoder_outputs.last_hidden_state

        # Compute value from encoder representation
        state_repr = encoder_hidden.mean(dim=1)  # Mean pooling
        value_logits = self.value_head(state_repr)
        value = self._value_from_logits(value_logits)

        # Sample multiple actions from cached encoder output
        action_logprobs = {}
        for _ in range(self.num_action_samples):
          # Generate with HuggingFace generate method
          generated = self.transformer.generate(
              encoder_outputs=encoder_outputs,
              max_length=100,
              do_sample=True,
              temperature=1.0,
              num_return_sequences=1,
          )

          # Decode and compute log probability
          action_str = self.tokenizer.decode(
              generated[0], skip_special_tokens=True
          )

          # Compute log probability
          logprob = self._compute_action_logprob_pretrained(
              generated[0], encoder_outputs
          )

          # Avoid duplicate actions
          if action_str not in action_logprobs:
            action_logprobs[action_str] = logprob

      else:
        # Encode once with custom model
        encoder_output = self.transformer.encode(obs_tokens)

        # Compute value from encoder representation
        state_repr = encoder_output.mean(dim=1)  # Mean pooling
        value_logits = self.value_head(state_repr)
        value = self._value_from_logits(value_logits)

        # Sample multiple actions from cached encoder output
        action_logprobs = {}
        for _ in range(self.num_action_samples):
          # Generate sequence
          generated = self.transformer.generate_sequence_cached(
              encoder_output,
              max_length=100,
              start_token_id=self.tokenizer.bos_token_id,
              end_token_id=self.tokenizer.eos_token_id,
              temperature=1.0,
              do_sample=True,
          )

          # Decode
          action_str = self.tokenizer.decode(
              generated[0], skip_special_tokens=True
          )

          # Compute log probability
          logprob = self.transformer.compute_sequence_logprob(
              generated, encoder_output
          ).item()

          # Avoid duplicates
          if action_str not in action_logprobs:
            action_logprobs[action_str] = logprob

    return NetworkSamplingOutput(action_logprobs=action_logprobs, value=value)

  def _compute_action_logprob_pretrained(
      self, sequence: torch.Tensor, encoder_outputs
  ) -> float:
    """Compute log probability for pretrained model."""
    # Use model to compute logits for sequence
    outputs = self.transformer(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=sequence[:-1].unsqueeze(0),
    )
    logits = outputs.logits[0]  # Remove batch dim
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather log probs of actual tokens
    target_tokens = sequence[1:]  # Skip BOS
    token_logprobs = log_probs[
        torch.arange(len(target_tokens)), target_tokens
    ]

    return token_logprobs.sum().item()

  def _value_from_logits(self, value_logits: torch.Tensor) -> float:
    """Convert value logits to scalar value.

    Args:
        value_logits: (batch_size, num_value_bins)

    Returns:
        Scalar value
    """
    # Compute expected value using bin centers
    probs = torch.softmax(value_logits, dim=-1)
    value = (probs * self.value_bins).sum(dim=-1)
    return value.item()

  def _compute_loss(self, batch):
    """Compute loss for a batch."""
    loss = torch.tensor(0.0, requires_grad=True)

    for observations, actions, value_targets in batch:
      network_output = self.forward(self.params, observations, actions)

      # Policy loss: cross-entropy on last token prediction
      # actions is (batch_size, seq_len), we want to predict the last token
      target_tokens = actions[:, -1]  # Last token
      policy_loss = F.cross_entropy(
          network_output.policy_logits, target_tokens
      )

      # Value loss: cross-entropy on value bins
      # Convert value_targets to bin indices
      value_bin_indices = self._value_to_bin_index(value_targets)
      v_loss = F.cross_entropy(
          network_output.value_logits,
          value_bin_indices.unsqueeze(0),  # Add batch dim
      )

      loss = loss + policy_loss + self.value_weight * v_loss

    return loss

  def _value_to_bin_index(self, value: float) -> torch.Tensor:
    """Convert scalar value to bin index."""
    # Find closest bin
    dists = torch.abs(self.value_bins - value)
    bin_idx = torch.argmin(dists)
    return bin_idx

  def update(self, batch: list[tuple[torch.Tensor, torch.Tensor, float]]):
    """Update network weights."""
    self.optimizer.zero_grad()
    loss = self._compute_loss(batch)
    loss.backward()
    self.optimizer.step()

  def save_checkpoint(self, path: str):
    """Save model checkpoint."""
    checkpoint = {
        'transformer_state_dict': self.transformer.state_dict(),
        'value_head_state_dict': self.value_head.state_dict(),
        'model_config': {
            'hidden_dim': self.hidden_dim,
            'vocab_size': self.vocab_size,
            'num_value_bins': self.num_value_bins,
            'is_pretrained': self.is_pretrained,
            'pretrained_model_name': self.pretrained_model_name if self.is_pretrained else None,
        },
        'optimizer_state_dict': self.optimizer.state_dict(),
    }

    # For custom models, save tokenizer and additional config
    if not self.is_pretrained:
      checkpoint['tokenizer_config'] = self.tokenizer.get_config()
      checkpoint['model_config'].update({
          'num_layers': self.model_num_layers,
          'num_heads': self.model_num_heads,
          'dropout': self.model_dropout,
      })

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


class ReplayBuffer:

  def __init__(self, config: Config, tokenizer=None):
    """Initialize replay buffer.

    Args:
        config: AlphaProof config
        tokenizer: Tokenizer for encoding observations and actions
    """
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.sequence_length = config.sequence_length
    self.buffer = []
    self.tokenizer = tokenizer

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
    """Tokenize input string.

    Args:
        input_string: String to tokenize

    Returns:
        Tokenized tensor (1, seq_len)
    """
    if self.tokenizer is None:
      # Fallback to dummy tokenization if no tokenizer provided
      return torch.zeros((1, self.sequence_length), dtype=torch.long)

    # Use actual tokenizer
    tokens = self.tokenizer.encode(
        input_string,
        add_special_tokens=True,
        max_length=self.sequence_length,
        return_tensors="pt",
    )

    # Pad if needed
    if tokens.size(1) < self.sequence_length:
      padding = torch.zeros(
          (1, self.sequence_length - tokens.size(1)), dtype=torch.long
      )
      if hasattr(self.tokenizer, 'pad_token_id'):
        padding.fill_(self.tokenizer.pad_token_id)
      tokens = torch.cat([tokens, padding], dim=1)

    return tokens


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
