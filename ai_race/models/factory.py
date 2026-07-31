"""Bridge the AI Race engine to the vendored FAIRGAME model connectors."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

OFFLINE_MODELS = {
    "LocalQwen",
    "LocalLlama",
    "LocalGemma",
    "LocalMistral",
    "LocalModel",
}


def is_offline_model(name: str) -> bool:
    return name in OFFLINE_MODELS


def merged_engine_settings(offline_settings: dict, model_preset: dict) -> dict:
    settings = dict(offline_settings.get("engine", {}) or {})
    settings.update(model_preset.get("engine", {}) or {})
    return settings


def vllm_init_kwargs(offline_settings: dict, model_preset: dict) -> dict[str, Any]:
    sampling = offline_settings.get("sampling", {}) or {}
    engine = merged_engine_settings(offline_settings, model_preset)
    kwargs: dict[str, Any] = {
        "max_model_len": int(engine.get("maxModelLen", 4096)),
        "temperature": float(sampling.get("temperature", 0.7)),
        "max_tokens": int(sampling.get("maxTokens", 256)),
        "logprobs": int(sampling.get("logprobs", 0)),
        "gpu_memory_utilization": float(engine.get("gpuMemoryUtilization", 0.90)),
        "tensor_parallel_size": int(engine.get("tensorParallelSize", 1)),
        "enforce_eager": bool(engine.get("enforceEager", True)),
        "trust_remote_code": bool(engine.get("trustRemoteCode", False)),
    }
    optional = {
        "quantization": engine.get("quantization"),
        "dtype": engine.get("dtype"),
        "kv_cache_dtype": engine.get("kvCacheDtype"),
        "max_num_seqs": engine.get("maxNumSeqs"),
        "cpu_offload_gb": engine.get("cpuOffloadGb"),
    }
    if optional["quantization"]:
        kwargs["quantization"] = str(optional["quantization"])
    if optional["dtype"] and str(optional["dtype"]).lower() != "auto":
        kwargs["dtype"] = str(optional["dtype"])
    if optional["kv_cache_dtype"] and str(optional["kv_cache_dtype"]).lower() != "auto":
        kwargs["kv_cache_dtype"] = str(optional["kv_cache_dtype"])
    if optional["max_num_seqs"]:
        kwargs["max_num_seqs"] = int(optional["max_num_seqs"])
    if optional["cpu_offload_gb"]:
        kwargs["cpu_offload_gb"] = float(optional["cpu_offload_gb"])
    return kwargs


def transformers_init_kwargs(offline_settings: dict, model_preset: dict) -> dict[str, Any]:
    sampling = offline_settings.get("sampling", {}) or {}
    engine = merged_engine_settings(offline_settings, model_preset)
    if int(sampling.get("logprobs", 0)) > 0:
        raise ValueError(
            "sampling.logprobs is supported only by the vLLM backend"
        )
    kwargs: dict[str, Any] = {
        "temperature": float(sampling.get("temperature", 0.7)),
        "max_tokens": int(sampling.get("maxTokens", 256)),
        "trust_remote_code": bool(engine.get("trustRemoteCode", False)),
    }
    if engine.get("quantization"):
        kwargs["quantization"] = str(engine["quantization"])
    return kwargs


def init_offline_backend(
    offline_settings: dict,
    model_preset: dict,
    *,
    force: bool = False,
) -> None:
    """Load one offline model. Kaggle may call with ``force=True`` between models."""
    from FAIRGAME.src.llm_connectors.local_vllm_connector import init_local_llm

    backend = str(offline_settings.get("backend", "vllm"))
    model_path = str(model_preset["modelPath"])
    if backend == "vllm":
        init_local_llm(
            model_path,
            engine="vllm",
            force=force,
            **vllm_init_kwargs(offline_settings, model_preset),
        )
        return
    if backend == "transformers":
        init_local_llm(
            model_path,
            engine="transformers",
            force=force,
            **transformers_init_kwargs(offline_settings, model_preset),
        )
        return
    raise ValueError(f"Unsupported offline backend: {backend!r}")


def free_offline_backend() -> None:
    from FAIRGAME.src.llm_connectors.local_vllm_connector import free_local_llm

    free_local_llm()


def get_send_batch(
    model_name: str,
    *,
    offline: bool = True,
    batch_size: int = 0,
    backend: str = "auto",
    proxy_options: dict[str, Any] | None = None,
) -> Callable[..., list[str | dict[str, Any]]]:
    """Return ``send_batch(prompts, seeds=None)`` for the selected backend.

    ``backend`` is ``"offline"`` (vLLM/transformers), ``"proxy"`` (Kaggle model
    proxy), ``"api"`` (FAIRGAME provider SDK connectors), or ``"auto"``, which
    resolves from the legacy ``offline`` flag.
    """
    backend = str(backend or "auto").lower()
    if backend == "auto":
        backend = "offline" if offline else "api"
    if backend not in {"offline", "proxy", "api"}:
        raise ValueError(f"Unsupported model backend: {backend!r}")

    if backend == "proxy":
        from ai_race.models.proxy import make_proxy_send_batch

        return make_proxy_send_batch(model_name, **(proxy_options or {}))

    if backend == "offline":
        from FAIRGAME.src.llm_connectors.local_vllm_connector import send_prompts_global

        def send(
            prompts: list[str],
            seeds: list[int] | None = None,
        ) -> list[str | dict[str, Any]]:
            return send_prompts_global(prompts, batch_size, seeds=seeds)

        return send

    from FAIRGAME.src.llm_connectors.llm_factory_connector import ChatModelFactory

    model = ChatModelFactory.get_model(model_name)

    def send_api(
        prompts: list[str],
        seeds: list[int] | None = None,
    ) -> list[str]:
        del seeds
        return [model.send_prompt(prompt) for prompt in prompts]

    return send_api
