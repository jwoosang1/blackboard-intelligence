from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .model import LLaDAModel


@dataclass
class LLaDAModelBundle:
    model: LLaDAModel
    tokenizer: PreTrainedTokenizerBase
    hidden_size: int
    mask_token_id: int


def _resolve_dtype(dtype_name: str) -> torch.dtype | str:
    name = dtype_name.lower()
    if name == "auto":
        return "auto"
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _infer_hidden_size(backbone: PreTrainedModel) -> int:
    for attr in ("hidden_size", "d_model", "dim", "n_embd", "model_dim"):
        value = getattr(backbone.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not infer hidden size from backbone config.")


def _apply_lora(
    backbone: PreTrainedModel,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Tuple[str, ...],
) -> PreTrainedModel:
    if not use_lora:
        return backbone

    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(lora_target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(backbone, lora_cfg)


def _is_llada_tied_weights_compat_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "all_tied_weights_keys" in text
        or "tie_weights() got an unexpected keyword argument" in text
        or "tie_weights() got an unexpected" in text
    )


def _ensure_llada_config_compat(backbone: PreTrainedModel) -> None:
    cfg = getattr(backbone, "config", None)
    if cfg is None:
        return
    if not hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    if not hasattr(cfg, "use_return_dict"):
        cfg.use_return_dict = True


def _patch_llada_remote_model_for_new_transformers(
    model_name: str,
    trust_remote_code: bool,
) -> bool:
    if not trust_remote_code or "llada" not in model_name.lower():
        return False

    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except Exception:
        return False

    cfg_module = getattr(cfg.__class__, "__module__", "")
    if not cfg_module:
        return False

    if cfg_module.endswith(".configuration_llada"):
        modeling_module = cfg_module[: -len(".configuration_llada")] + ".modeling_llada"
    else:
        parts = cfg_module.rsplit(".", 1)
        if len(parts) != 2:
            return False
        modeling_module = f"{parts[0]}.modeling_llada"

    try:
        mod = importlib.import_module(modeling_module)
        cls = getattr(mod, "LLaDAModelLM", None)
    except Exception:
        return False

    if cls is None:
        return False
    if getattr(cls, "_llada_hf5_compat_patched", False):
        return True

    orig_init = cls.__init__

    def patched_init(self, config, model=None, init_params=False):
        orig_init(self, config, model=model, init_params=init_params)
        # transformers>=5 expects this to exist during from_pretrained finalize.
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    cls.__init__ = patched_init

    try:
        tie_sig = inspect.signature(cls.tie_weights)
        wants_new_signature = (
            "missing_keys" in tie_sig.parameters
            and "recompute_mapping" in tie_sig.parameters
        )
    except Exception:
        wants_new_signature = False

    if not wants_new_signature:
        def patched_tie_weights(self, missing_keys=None, recompute_mapping=True):
            if getattr(self.config, "weight_tying", False):
                self.model.transformer.ff_out = self.model.transformer.wte
            return self

        cls.tie_weights = patched_tie_weights

    cls._llada_hf5_compat_patched = True
    return True


def _load_backbone_with_fallback(
    model_name: str,
    trust_remote_code: bool,
    torch_dtype: torch.dtype | str,
    return_dict: bool = True,
    rank: int = 0,
) -> PreTrainedModel:
    kwargs = dict(
        pretrained_model_name_or_path=model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        return_dict=return_dict,
    )
    try:
        model = AutoModel.from_pretrained(**kwargs)
        _ensure_llada_config_compat(model)
        return model
    except (AttributeError, TypeError) as exc:
        if not _is_llada_tied_weights_compat_error(exc):
            raise
        patched = _patch_llada_remote_model_for_new_transformers(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
        )
        if not patched:
            raise
        if rank == 0:
            print("[model] transformers/LLaDA compatibility fallback enabled; retrying load.")
        model = AutoModel.from_pretrained(**kwargs)
        _ensure_llada_config_compat(model)
        return model



def build_llada_bundle(
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    torch_dtype: str = "bf16",
    trust_remote_code: bool = True,
    mask_token_id: int = 126336,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> LLaDAModelBundle:
    dtype = _resolve_dtype(torch_dtype)

    backbone = _load_backbone_with_fallback(
        model_name=model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        return_dict=True,
        rank=0,
    )
    backbone.config.output_hidden_states = True
    backbone.config.return_dict = True

    backbone = _apply_lora(
        backbone=backbone,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = _infer_hidden_size(backbone)
    model = LLaDAModel(backbone=backbone, hidden_size=hidden_size)

    return LLaDAModelBundle(
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        mask_token_id=int(mask_token_id),
    )