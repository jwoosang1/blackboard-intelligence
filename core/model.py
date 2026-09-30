from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class LLaDAModel(nn.Module):
    """Headless LLaDA wrapper used by the paper's final experiments.

    The final ZebraLogic, Nurse Rostering, and JSSP configurations use
    All reported solvers read backbone logits directly.
    Keeping the historical confidence head here would add unused parameters and
    imply that release checkpoints require an unused artifact.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        for key in ("x0", "t", "masked_indices"):
            kwargs.pop(key, None)
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        if not hasattr(out, "logits") or out.logits is None:
            raise RuntimeError("Backbone output has no logits.")
        return {"logits": out.logits}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
