import typing
from typing import Any, Dict

from alphaproof.config import Config
from alphaproof.environment import Action


Params = Any


class NetworkTrainingOutput(typing.NamedTuple):
    """Output of the network during training."""
    value_logits: jax.Array
    policy_logits: jax.Array


class NetworkSamplingOutput(typing.NamedTuple):
    """Output of the network when sampling actions."""
    action_logprobs: Dict[Action, float]
    value: float


class Network:
    """Placeholder neural network used by the training pseudocode."""

    def __init__(self, config: Config):
        """Initialize parameters, optimizer state, and loss closure."""
        self.params = {'weights': jnp.array([0])}

        self.num_value_bins = config.num_value_bins
        self.value_weight = config.value_weight
        self.optimizer = optax.adam(config.lr)
        self.opt_state = self.optimizer.init(self.params)

        def _loss_fn(params, batch):
            """Compute policy and value loss over a replay batch."""
            loss = 0
            for observations, actions, value_targets in batch:
                network_output = self.forward(params, observations, actions)
                # Policy loss
                loss += jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(
                        network_output.policy_logits, actions
                    )
                )
                # Value loss
                loss += self.value_weight * value_loss(
                    network_output.value_logits,
                    value_targets,
                )

            return loss

        self._loss_grad = jax.grad(_loss_fn)

    def forward(
        self, params: Params, observation: jax.Array, action: jax.Array
    ) -> NetworkTrainingOutput:
        """Run the placeholder network for supervised training."""
        # Predict value logits and policy logits from given observation and action.
        # observation and action are passed to the network.
        value_logits = jnp.zeros(self.num_value_bins)
        policy_logits = jnp.array([0])
        return NetworkTrainingOutput(
            value_logits=value_logits, policy_logits=policy_logits
        )

    def sample(self, observation: str) -> NetworkSamplingOutput:
        """Return sampled tactics and a value estimate for search."""
        # Predict value and sample actions from a given observation.
        # observation is tokenized and passed to the network to produce value
        # logits. The value is then calcualated from value logits and bin locations.
        value = 0.
        return NetworkSamplingOutput(
            action_logprobs={'placeholder_action': -2.},
            value=value,
        )

    def update(self, batch: list[tuple[jax.Array, jax.Array, float]]):
        """Apply one optimizer update from a replay batch."""
        # Update the network weights.
        grads = self._loss_grad(self.params, batch)
        updates, self.opt_state = self.optimizer.update(
            grads, self.opt_state, self.params
        )
        self.params = optax.apply_updates(self.params, updates)
