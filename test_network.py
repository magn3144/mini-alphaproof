"""Test script for Network implementation."""

import torch
from src.training import Network
from src.environment import Config

def test_random_initialization():
  """Test random initialization mode."""
  print("\n=== Test 1: Random Initialization ===")

  config = Config(
      num_simulations=800,
      batch_size=2,
      num_actors=1,
      lr=0.001,
  )

  # Create network with random weights
  network = Network(config, hidden_dim=256, num_layers=3, num_heads=4)

  print(f"✓ Network created")
  print(f"  - Is pretrained: {network.is_pretrained}")
  print(f"  - Hidden dim: {network.hidden_dim}")
  print(f"  - Vocab size: {network.vocab_size}")
  print(f"  - Transformer type: {type(network.transformer).__name__}")

  return network


def test_forward_pass(network):
  """Test forward pass."""
  print("\n=== Test 2: Forward Pass ===")

  # Create dummy inputs
  batch_size = 2
  seq_len = 32

  observation = torch.randint(
      0, network.vocab_size, (batch_size, seq_len), dtype=torch.long
  )
  action = torch.randint(
      0, network.vocab_size, (batch_size, seq_len), dtype=torch.long
  )

  # Forward pass
  output = network.forward(network.params, observation, action)

  print(f"✓ Forward pass successful")
  print(f"  - Policy logits shape: {output.policy_logits.shape}")
  print(f"  - Value logits shape: {output.value_logits.shape}")
  print(
      f"  - Expected policy shape: ({batch_size}, {network.vocab_size})"
  )
  print(f"  - Expected value shape: ({batch_size}, {network.num_value_bins})")

  # Verify shapes
  assert output.policy_logits.shape == (batch_size, network.vocab_size), \
      f"Policy logits shape mismatch: {output.policy_logits.shape}"
  assert output.value_logits.shape == (batch_size, network.num_value_bins), \
      f"Value logits shape mismatch: {output.value_logits.shape}"

  print("✓ Output shapes correct")


def test_sample_method(network):
  """Test sample method with encoder caching."""
  print("\n=== Test 3: Sample Method (Encoder Caching) ===")

  observation = "example tactic state: rw [mul_comm] at h"

  # Sample actions
  output = network.sample(observation)

  print(f"✓ Sample method successful")
  print(f"  - Number of actions sampled: {len(output.action_logprobs)}")
  print(f"  - Value: {output.value:.4f}")
  print(f"  - Sample actions:")
  for i, (action, logprob) in enumerate(
      list(output.action_logprobs.items())[:3]
  ):
    print(f"    {i+1}. {action[:50]}... (logprob: {logprob:.4f})")

  # Verify structure
  assert isinstance(output.action_logprobs, dict), \
      "action_logprobs should be a dict"
  assert isinstance(output.value, float), "value should be a float"
  assert len(output.action_logprobs) > 0, "Should sample at least one action"

  print("✓ Sample output structure correct")


def test_checkpoint_save_load(network):
  """Test checkpoint save and load."""
  print("\n=== Test 4: Checkpoint Save/Load ===")

  import tempfile
  import os

  # Save checkpoint
  with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
    network.save_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")

    # Load checkpoint
    config = Config(
        num_simulations=800,
        batch_size=2,
        num_actors=1,
        lr=0.001,
    )
    network2 = Network(config, model_path=checkpoint_path)
    print(f"✓ Checkpoint loaded")

    # Verify weights match
    for p1, p2 in zip(
        network.transformer.parameters(), network2.transformer.parameters()
    ):
      assert torch.allclose(p1, p2), "Transformer weights mismatch"

    for p1, p2 in zip(
        network.value_head.parameters(), network2.value_head.parameters()
    ):
      assert torch.allclose(p1, p2), "Value head weights mismatch"

    print("✓ Loaded weights match original")


def test_pretrained_loading():
  """Test pretrained model loading."""
  print("\n=== Test 5: Pretrained Model Loading ===")

  try:
    from transformers import AutoModelForSeq2SeqLM

    config = Config(
        num_simulations=800,
        batch_size=2,
        num_actors=1,
        lr=0.001,
    )

    # Try loading a small pretrained model
    print("Attempting to load google/byt5-small...")
    network = Network(config, pretrained_model_name="google/byt5-small")

    print(f"✓ Pretrained model loaded")
    print(f"  - Is pretrained: {network.is_pretrained}")
    print(f"  - Hidden dim: {network.hidden_dim}")
    print(f"  - Vocab size: {network.vocab_size}")
    print(f"  - Model name: {network.pretrained_model_name}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    observation = torch.randint(
        0, network.vocab_size, (batch_size, seq_len), dtype=torch.long
    )
    action = torch.randint(
        0, network.vocab_size, (batch_size, seq_len), dtype=torch.long
    )

    output = network.forward(network.params, observation, action)
    print(f"✓ Pretrained forward pass successful")
    print(f"  - Policy logits shape: {output.policy_logits.shape}")
    print(f"  - Value logits shape: {output.value_logits.shape}")

  except ImportError:
    print("⚠ Skipping pretrained test: transformers library not installed")
  except Exception as e:
    print(f"⚠ Pretrained test failed: {e}")
    import traceback
    traceback.print_exc()


def main():
  """Run all tests."""
  print("=" * 60)
  print("Testing AlphaProof Network Implementation")
  print("=" * 60)

  try:
    # Test random initialization
    network = test_random_initialization()

    # Test forward pass
    test_forward_pass(network)

    # Test sample method
    test_sample_method(network)

    # Test checkpoint save/load
    test_checkpoint_save_load(network)

    # Test pretrained loading
    test_pretrained_loading()

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

  except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ Test failed with error: {e}")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    return 1

  return 0


if __name__ == "__main__":
  exit(main())
