import torch
import torch.nn.functional as F

import typing
from typing import Dict

from src.environment import Action
from src.environment import (
    Environment, Config, Node, Game, Theorem, Player, Observation, Action
)


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
    self.model = None

    self.num_value_bins = config.num_value_bins
    self.value_weight = config.value_weight
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

  def value_loss(self, value_logits: torch.Tensor, value_targets: float) -> torch.Tensor:
    """Calculate the categorical cross-entropy loss for value prediction.

    Args:
      value_logits: Tensor of shape (num_value_bins,) with logits for each bin
      value_targets: Scalar representing -T_steps (negative number of tactics needed)
                    e.g., -10.0 means 10 steps remaining

    Returns:
      Cross-entropy loss between predicted categorical distribution and target bin
    """
    # Define value range based on paper and config
    # With 64 bins, we represent values from -63 to 0
    # where 0 means proof is complete, -63 means 63+ steps remaining
    VALUE_MIN = -self.num_value_bins + 1  # -63 for 64 bins
    VALUE_MAX = 0

    # Convert continuous value target to bin index
    # Clamp value to valid range
    clamped_value = max(VALUE_MIN, min(VALUE_MAX, value_targets))

    # Map to bin index: bin 0 corresponds to VALUE_MIN (-63)
    #                   bin 63 corresponds to VALUE_MAX (0)
    bin_index = int(clamped_value - VALUE_MIN)

    # Create target tensor with the bin index
    target = torch.tensor(bin_index, dtype=torch.long)

    # Calculate cross-entropy loss
    # F.cross_entropy expects input of shape (N, C) and target of shape (N,)
    # where N is batch size and C is number of classes
    loss = F.cross_entropy(value_logits.unsqueeze(0), target.unsqueeze(0))

    return loss

  def _compute_loss(self, batch):
    loss = torch.tensor(0.0, requires_grad=True)
    for observations, actions, value_targets in batch:
      network_output = self.forward(observations, actions)
      # Policy loss
      policy_loss = F.cross_entropy(
          network_output.policy_logits, actions
      )
      # Value loss
      v_loss = self.value_loss(network_output.value_logits, value_targets)
      loss = loss + policy_loss + self.value_weight * v_loss

    return loss

  def forward(
      self, observation: torch.Tensor, action: torch.Tensor
  ) -> NetworkTrainingOutput:
    """Forward pass through the model for training.

    Args:
      observation: Tokenized observation tensor, shape (seq_len,)
      action: Tokenized action tensor, shape (seq_len,)

    Returns:
      NetworkTrainingOutput with value_logits and policy_logits
    """
    # Add batch dimension: (seq_len,) -> (1, seq_len)
    obs_batch = observation.unsqueeze(0)
    action_batch = action.unsqueeze(0)

    # Forward through model
    model_output = self.model(
        input_ids=obs_batch,
        decoder_input_ids=action_batch
    )

    # Extract and remove batch dimension
    value_logits = model_output['value_logits'].squeeze(0)  # (1, num_bins) -> (num_bins,)
    policy_logits = model_output['policy_logits'].squeeze(0)  # (1, seq_len, vocab) -> (seq_len, vocab)

    return NetworkTrainingOutput(
        value_logits=value_logits,
        policy_logits=policy_logits
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