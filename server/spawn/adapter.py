"""Bridges a sampler that exposes predict_vector to spawn.FlowModel."""
from dataclasses import dataclass

import torch

from .base import Conditioning


@dataclass
class SamplerModel:
    sampler: object

    def velocity(self, x: torch.Tensor, t: float, cond: Conditioning,
                 t_next: float, guidance: float) -> torch.Tensor:
        return self.sampler.predict_vector(
            x, t,
            prompt_embeds=cond.prompt_embeds[:1],
            pooled_prompt_embeds=cond.pooled_prompt_embeds[:1],
            guidance_scale=guidance,
            t_next=t_next,
        )
