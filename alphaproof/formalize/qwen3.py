from importlib.metadata import PackageNotFoundError, version
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers.tokenization_utils_base as transformers_tokenization_utils_base

# lm-format-enforcer 0.11.3 still imports the removed Transformers 4 path.
sys.modules.setdefault(
    'transformers.tokenization_utils',
    transformers_tokenization_utils_base,
)

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from alphaproof.core.config import MODELS_DIR


QWEN3_MODEL_NAME = 'Qwen/Qwen3.6-27B'
QWEN3_MODEL_DIR = MODELS_DIR / QWEN3_MODEL_NAME.replace('/', '--')
QWEN3_9B_MODEL_NAME = 'Qwen/Qwen3.5-9B'
QWEN3_9B_MODEL_DIR = MODELS_DIR / QWEN3_9B_MODEL_NAME.replace('/', '--')
PARALLELISM_MODES = {'none', 'balanced', 'tensor', 'data'}


class TensorParallelError(RuntimeError):
    """Raised when tensor-parallel ranks can no longer safely continue."""


@dataclass
class ModelCall:
    """Metrics for one actual model.generate call."""

    stage: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return this call as JSON-compatible data."""
        return asdict(self)


class Qwen3:
    """Local wrapper for loading and sampling Qwen3 instruction models."""

    model_name = QWEN3_MODEL_NAME

    def __init__(
        self,
        model_dir: str | Path = QWEN3_MODEL_DIR,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | str = 'auto',
        trust_remote_code: bool = True,
        quantization: str | None = None,
        parallelism: str = 'none',
        seed: int = 0,
        max_batch_size: int | None = None,
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
        self.quantization = quantization
        if parallelism not in PARALLELISM_MODES:
            choices = ', '.join(sorted(PARALLELISM_MODES))
            raise ValueError(f'parallelism must be one of: {choices}.')
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError('max_batch_size must be at least 1.')
        if quantization is not None and parallelism != 'none':
            raise ValueError('quantization is only supported with parallelism=none.')
        self.parallelism = parallelism
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.metric_stage = 'sample'
        self.model_calls: list[ModelCall] = []
        self._sample_index = 0

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
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                **self._model_load_kwargs(),
            )
        except ValueError as error:
            self._raise_model_load_error(error)
        assert self.model is not None
        if self.quantization is None and self.parallelism in {'none', 'data'}:
            self.model.to(self.device)
        self.model.eval()
        return self

    def sample(
        self,
        prompts: list[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        json_schema: dict[str, Any] | None = None,
    ) -> list[str]:
        """Sample one completion per prompt, chunking oversized batches."""
        self._ensure_loaded()
        if not prompts:
            return []

        batch_size = self.max_batch_size or len(prompts)
        completions = []
        for start in range(0, len(prompts), batch_size):
            completions.extend(
                self._sample_batch(
                    prompts[start:start + batch_size],
                    max_new_tokens,
                    temperature,
                    top_p,
                    json_schema,
                )
            )
        return completions

    def _sample_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        json_schema: dict[str, Any] | None,
    ) -> list[str]:
        """Sample one completion per prompt in one model.generate call."""
        tokenizer = self.tokenizer
        model = self.model
        assert tokenizer is not None
        assert model is not None

        encoded = self._encode_prompts(prompts, tokenizer)
        encoded = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }
        generate_kwargs = self._generate_kwargs(tokenizer, json_schema)

        self._set_call_seed()
        start = perf_counter()
        try:
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self._pad_token_id(tokenizer),
                    **generate_kwargs,
                )
        except Exception as error:
            if self.parallelism == 'tensor':
                raise TensorParallelError(
                    f'Tensor-parallel generation failed: {error}'
                ) from error
            raise
        seconds = perf_counter() - start

        prompt_length = encoded['input_ids'].shape[-1]
        completion_ids = generated[:, prompt_length:]
        completions = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        completions = [
            self._clean_completion(completion)
            for completion in completions
        ]
        self.model_calls.append(
            ModelCall(
                stage=self.metric_stage,
                batch_size=len(prompts),
                prompt_tokens=int(encoded['attention_mask'].sum().item()),
                generated_tokens=self._count_generated_tokens(
                    completion_ids,
                    tokenizer.eos_token_id,
                ),
                seconds=seconds,
            )
        )
        self._verify_tensor_completions(completions)
        return completions

    def reset_metrics(self) -> None:
        """Reset measured model-call metrics after warmup."""
        self.model_calls.clear()

    def metrics(self) -> dict[str, Any]:
        """Return aggregate and per-stage model-call metrics."""
        prompt_tokens = sum(call.prompt_tokens for call in self.model_calls)
        generated_tokens = sum(call.generated_tokens for call in self.model_calls)
        generation_seconds = sum(call.seconds for call in self.model_calls)
        stages: dict[str, dict[str, float | int]] = {}
        for call in self.model_calls:
            stage = stages.setdefault(call.stage, {'calls': 0, 'seconds': 0.0})
            stage['calls'] += 1
            stage['seconds'] += call.seconds
        return {
            'calls': len(self.model_calls),
            'prompt_tokens': prompt_tokens,
            'generated_tokens': generated_tokens,
            'generation_seconds': generation_seconds,
            'generated_tokens_per_second': (
                generated_tokens / generation_seconds if generation_seconds else 0.0
            ),
            'total_tokens_per_second': (
                (prompt_tokens + generated_tokens) / generation_seconds
                if generation_seconds else 0.0
            ),
            'max_model_call_batch_size': max(
                (call.batch_size for call in self.model_calls),
                default=0,
            ),
            'stages': stages,
            'model_calls': [call.to_dict() for call in self.model_calls],
        }

    def _set_call_seed(self) -> None:
        """Set the deterministic seed stream for this actual model call."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        rank_offset = rank * 1_000_000 if self.parallelism == 'data' else 0
        call_seed = self.seed + rank_offset + self._sample_index
        self._sample_index += 1
        torch.manual_seed(call_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(call_seed)

    def _verify_tensor_completions(self, completions: list[str]) -> None:
        """Require every tensor-parallel rank to observe identical output."""
        if self.parallelism != 'tensor' or not dist.is_initialized():
            return
        gathered: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, completions)
        if any(rank_completions != completions for rank_completions in gathered):
            raise TensorParallelError(
                'Tensor-parallel ranks produced different completions.'
            )

    def _count_generated_tokens(
        self,
        completion_ids: torch.Tensor,
        eos_token_id: int | None,
    ) -> int:
        """Count generated tokens through the first EOS in each completion."""
        if eos_token_id is None:
            return int(completion_ids.numel())
        count = 0
        for token_ids in completion_ids.tolist():
            count += len(token_ids)
            if eos_token_id in token_ids:
                count -= len(token_ids) - token_ids.index(eos_token_id) - 1
        return count

    def _encode_prompts(
        self,
        prompts: list[str],
        tokenizer: Any,
    ) -> dict[str, torch.Tensor]:
        """Encode user prompts with Qwen3's chat template."""
        texts = []
        for prompt in prompts:
            messages = [{'role': 'user', 'content': prompt}]
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer(texts, return_tensors='pt', padding=True)

    def _generate_kwargs(
        self,
        tokenizer: Any,
        json_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return extra generate() kwargs for optional structured output."""
        if json_schema is None:
            return {}

        parser = JsonSchemaParser(json_schema)
        return {
            'prefix_allowed_tokens_fn': build_transformers_prefix_allowed_tokens_fn(
                tokenizer,
                parser,
            )
        }

    def _clean_completion(self, completion: str) -> str:
        """Clean model output text."""
        return completion.strip()

    def _ensure_loaded(self) -> None:
        """Raise a clear error if load() has not been called."""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError('Call load() before sampling.')

    def _raise_model_load_error(self, error: ValueError) -> None:
        """Raise an actionable error for unsupported local model configs."""
        message = str(error)
        if 'Transformers does not recognize this architecture' not in message:
            raise error

        try:
            transformers_version = version('transformers')
        except PackageNotFoundError:
            transformers_version = 'unknown'

        raise RuntimeError(
            f'{self.model_name} at {self.model_dir} uses a model architecture '
            'that this Transformers install does not support. '
            f'Installed Transformers version: {transformers_version}. '
            'Install Transformers from source, then rerun the data cleaning '
            'command: uv pip install git+https://github.com/huggingface/'
            'transformers.git'
        ) from error

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
        elif self.parallelism == 'balanced':
            kwargs['device_map'] = 'balanced'
        elif self.parallelism == 'tensor':
            kwargs['tp_plan'] = 'auto'
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


class Qwen3_9B(Qwen3):
    """Local wrapper for loading and sampling Qwen3.5-9B."""

    model_name = QWEN3_9B_MODEL_NAME

    def __init__(
        self,
        model_dir: str | Path = QWEN3_9B_MODEL_DIR,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype | str = 'auto',
        trust_remote_code: bool = True,
        quantization: str | None = None,
        parallelism: str = 'none',
        seed: int = 0,
        max_batch_size: int | None = None,
    ):
        """Store Qwen3.5-9B settings without loading weights until load()."""
        super().__init__(
            model_dir=model_dir,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            quantization=quantization,
            parallelism=parallelism,
            seed=seed,
            max_batch_size=max_batch_size,
        )
