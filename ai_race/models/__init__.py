"""Model backend adapters reused by the AI Race runners."""

from .factory import (
    free_offline_backend,
    get_send_batch,
    init_offline_backend,
    vllm_init_kwargs,
)

__all__ = [
    "free_offline_backend",
    "get_send_batch",
    "init_offline_backend",
    "vllm_init_kwargs",
]

