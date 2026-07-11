import torch

from alphaproof.formalize.qwen3 import Qwen3, Qwen3_9B


CLEANING_MODEL_ALIASES = {
    'qwen3.6-27b': Qwen3,
    'qwen/qwen3.6-27b': Qwen3,
    'qwen3.5-9b': Qwen3_9B,
    'qwen/qwen3.5-9b': Qwen3_9B,
}
DEFAULT_CLEANING_MODEL = 'qwen3.6-27b'


def is_out_of_memory_error(error: Exception) -> bool:
    """Return whether an exception looks like a torch accelerator OOM."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    message = str(error).lower()
    return (
            'out of memory' in message
            or 'failed to allocate' in message
    )


def clear_accelerator_cache() -> None:
    """Release cached accelerator memory after a failed generation."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def load_cleaning_model(
        model_name: str,
        device: str | None,
        torch_dtype: str,
        quantization: str | None,
) -> Qwen3:
    """Load the model used for data cleaning."""
    if device is not None:
        model_device = torch.device(device)
    elif torch.cuda.is_available():
        model_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        model_device = torch.device('mps')
    else:
        model_device = torch.device('cpu')

    if torch_dtype == 'auto' and model_device.type == 'mps':
        torch_dtype = 'float16'

    model_class = CLEANING_MODEL_ALIASES.get(model_name.lower())
    if model_class is None:
        choices = ', '.join(sorted(CLEANING_MODEL_ALIASES))
        raise ValueError(f'model must be one of: {choices}.')

    qwen = model_class(
            device=model_device,
            torch_dtype=torch_dtype,
            quantization=quantization,
    )
    qwen.load()
    print(f'Loaded {qwen.model_name} from {qwen.model_dir} on {qwen.device}.')
    return qwen
