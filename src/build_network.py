"""Build two-headed encoder-decoder models with value heads for AlphaProof."""

import os
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM


class TwoHeadedEncoderDecoder(nn.Module):
    """Encoder-decoder model with additional value head for value prediction.

    This module wraps a HuggingFace encoder-decoder model and adds a value head
    that predicts categorical value distributions over bins.
    """

    def __init__(self, encoder_decoder, num_value_bins: int):
        """Initialize the two-headed model.

        Args:
            encoder_decoder: HuggingFace encoder-decoder model (e.g., T5, BART)
            num_value_bins: Number of bins for categorical value prediction
        """
        super().__init__()
        self.encoder_decoder = encoder_decoder
        self.num_value_bins = num_value_bins

        # Get hidden size from model config
        # Different models use different attribute names
        if hasattr(encoder_decoder.config, 'd_model'):
            hidden_size = encoder_decoder.config.d_model  # T5, mT5
        elif hasattr(encoder_decoder.config, 'hidden_size'):
            hidden_size = encoder_decoder.config.hidden_size  # BART, others
        else:
            raise ValueError(
                f"Could not determine hidden size from model config. "
                f"Available attributes: {dir(encoder_decoder.config)}"
            )

        # Value head: maps encoder hidden states to value logits
        self.value_head = nn.Linear(hidden_size, num_value_bins)

        # Initialize value head with Xavier uniform
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        """Forward pass through encoder-decoder and value head.

        Args:
            input_ids: Input token IDs for the encoder
            attention_mask: Attention mask for the encoder input
            decoder_input_ids: Input token IDs for the decoder

        Returns:
            Dictionary containing:
                - encoder_decoder_outputs: Full outputs from the encoder-decoder
                - value_logits: Value prediction logits [batch_size, num_value_bins]
        """
        # Encoder-decoder forward pass
        outputs = self.encoder_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        # Extract encoder hidden states for value prediction
        # Use the last layer's hidden states
        encoder_hidden = outputs.encoder_hidden_states[-1]  # [batch, seq_len, hidden]

        # Pool over sequence dimension (mean pooling)
        pooled = encoder_hidden.mean(dim=1)  # [batch_size, hidden_size]

        # Value head
        value_logits = self.value_head(pooled)  # [batch_size, num_value_bins]

        return {
            'encoder_decoder_outputs': outputs,
            'value_logits': value_logits,
            'policy_logits': outputs.logits  # Decoder logits for policy prediction
        }


def sanitize_model_name(model_name: str) -> str:
    """Convert model name to valid filename.

    Args:
        model_name: HuggingFace model identifier (e.g., "google/t5-v1_1-small")

    Returns:
        Sanitized filename-safe string (e.g., "google-t5-v1_1-small")
    """
    return model_name.replace('/', '-').replace('\\', '-')


def build_two_headed_model(
    model_name: str,
    num_value_bins: int,
    save_dir: str = "models"
) -> str:
    """Download encoder-decoder model from HuggingFace and add a value head.

    This function downloads a pre-trained encoder-decoder model, wraps it with
    a randomly initialized value head for categorical value prediction, and saves
    the complete model as a PyTorch state dict.

    Args:
        model_name: HuggingFace model identifier (e.g., "google/t5-v1_1-small",
                    "facebook/bart-base", "google/flan-t5-base")
        num_value_bins: Number of bins for value prediction output (must be > 0)
        save_dir: Directory to save the model (default: "models")

    Returns:
        Path to the saved model file

    Raises:
        ValueError: If num_value_bins <= 0 or model cannot be downloaded

    Example:
        >>> path = build_two_headed_model("google/t5-v1_1-small", 64)
        >>> print(f"Model saved to: {path}")
        Model saved to: models/google-t5-v1_1-small_vhead_64bins.pt
    """
    # Validation
    if num_value_bins <= 0:
        raise ValueError(f"num_value_bins must be positive, got {num_value_bins}")

    print(f"Downloading model: {model_name}")

    # Download base encoder-decoder model
    try:
        encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except OSError as e:
        raise ValueError(f"Failed to download model '{model_name}': {e}")
    except Exception as e:
        raise ValueError(f"Error loading model '{model_name}': {e}")

    # Get hidden size for logging
    if hasattr(encoder_decoder.config, 'd_model'):
        hidden_size = encoder_decoder.config.d_model
    elif hasattr(encoder_decoder.config, 'hidden_size'):
        hidden_size = encoder_decoder.config.hidden_size
    else:
        hidden_size = "unknown"

    print(f"Model downloaded. Hidden size: {hidden_size}")

    # Create two-headed model
    model = TwoHeadedEncoderDecoder(encoder_decoder, num_value_bins)

    # Prepare save path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    sanitized_name = sanitize_model_name(model_name)
    filename = f"{sanitized_name}_vhead_{num_value_bins}bins.pt"
    filepath = os.path.join(save_dir, filename)

    # Save model state dict
    print(f"Saving model to: {filepath}")
    torch.save(model.state_dict(), filepath)

    # Report success with file size
    file_size_mb = os.path.getsize(filepath) / 1e6
    print(f"Model saved successfully!")
    print(f"  - Base model: {model_name}")
    print(f"  - Value bins: {num_value_bins}")
    print(f"  - File size: {file_size_mb:.1f} MB")

    return filepath
