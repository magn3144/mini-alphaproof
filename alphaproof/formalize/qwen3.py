from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from alphaproof.core.config import MODELS_DIR


QWEN3_MODEL_NAME = 'Qwen/Qwen3-32B'
QWEN3_MODEL_DIR = MODELS_DIR / QWEN3_MODEL_NAME.replace('/', '--')
QWEN3_8B_MODEL_NAME = 'Qwen/Qwen3-8B'
QWEN3_8B_MODEL_DIR = MODELS_DIR / QWEN3_8B_MODEL_NAME.replace('/', '--')


class Qwen3:
    """Local wrapper for loading and sampling Qwen3 instruction models."""

    model_name = QWEN3_MODEL_NAME

    def __init__(
        self,
        model_dir: str | Path = QWEN3_MODEL_DIR,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | str = 'auto',
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
        quantization: str | None = None,
    ):
        """Store model settings without loading weights until load() is called."""
        self.model_dir = Path(model_dir)
        self.device = (
            torch.device(device)
            if device is not None
            else self._default_device()
        )
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.enable_thinking = enable_thinking
        self.quantization = quantization
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> 'Qwen3':
        """Load the tokenizer and model from the local models directory."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f'Expected Qwen3 model at {self.model_dir}. '
                f'Download {self.model_name} into models/ first.'
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            **self._model_load_kwargs(),
        )
        if self.quantization is None:
            self.model.to(self.device)
        self.model.eval()
        return self

    def sample(
        self,
        prompt: str,
        num_samples: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
    ) -> list[str]:
        """Sample one or more completions from the loaded model."""
        self._ensure_loaded()
        tokenizer = self.tokenizer
        model = self.model
        assert tokenizer is not None
        assert model is not None

        if num_samples < 1:
            raise ValueError('num_samples must be at least 1.')

        encoded = self._encode_prompt(prompt, tokenizer)
        encoded = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self._pad_token_id(tokenizer),
            )

        prompt_length = encoded['input_ids'].shape[-1]
        completion_ids = generated[:, prompt_length:]
        completions = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        return [self._clean_completion(completion) for completion in completions]

    def _encode_prompt(self, prompt: str, tokenizer: Any) -> dict[str, torch.Tensor]:
        """Encode a user prompt with Qwen3's chat template."""
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return tokenizer([text], return_tensors='pt')

    def _clean_completion(self, completion: str) -> str:
        """Remove Qwen3 thinking markup when present."""
        end_tag = '</think>'
        if end_tag in completion:
            completion = completion.split(end_tag, maxsplit=1)[1]
        return completion.strip()

    def _ensure_loaded(self) -> None:
        """Raise a clear error if load() has not been called."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError('Call load() before sampling.')

    def _model_load_kwargs(self) -> dict[str, Any]:
        """Build model loading options, including optional quantization."""
        kwargs: dict[str, Any] = {
            'torch_dtype': self.torch_dtype,
            'trust_remote_code': self.trust_remote_code,
        }
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            if self.device.type == 'mps':
                raise ValueError(
                    'bitsandbytes quantization is not supported on Apple MPS. '
                    'Use quantization=None and a smaller dtype such as float16.'
                )
            kwargs['device_map'] = 'auto'
            kwargs['quantization_config'] = quantization_config
        return kwargs

    def _quantization_config(self) -> BitsAndBytesConfig | None:
        """Return a bitsandbytes quantization config when requested."""
        if self.quantization is None:
            return None
        if self.quantization == '8bit':
            return BitsAndBytesConfig(load_in_8bit=True)
        if self.quantization == '4bit':
            return BitsAndBytesConfig(load_in_4bit=True)
        raise ValueError("quantization must be None, '8bit', or '4bit'.")

    def _pad_token_id(self, tokenizer: Any) -> int | None:
        """Return a usable pad token id for generation."""
        if tokenizer.pad_token_id is not None:
            return tokenizer.pad_token_id
        return tokenizer.eos_token_id

    def _default_device(self) -> torch.device:
        """Pick the best available local torch device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')


class Qwen3_8B(Qwen3):
    """Local wrapper for loading and sampling Qwen3-8B."""

    model_name = QWEN3_8B_MODEL_NAME

    def __init__(
        self,
        model_dir: str | Path = QWEN3_8B_MODEL_DIR,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | str = 'auto',
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
        quantization: str | None = None,
    ):
        """Store Qwen3-8B settings without loading weights until load()."""
        super().__init__(
            model_dir=model_dir,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            quantization=quantization,
        )
