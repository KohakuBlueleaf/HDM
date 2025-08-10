import json
import torch

from xut.xut import XUDiT
from .base import *


class XUDiTConditionModel(BasicUNet):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = XUDiT(*args, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any] | str) -> "XUDiTConditionModel":
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        return cls(**config)

    def enable_gradient_checkpointing(self):
        return self.model.set_grad_ckpt(True)

    def disable_gradient_checkpointing(self):
        return self.model.set_grad_ckpt(False)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        pos_map: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        result = self.model(
            sample, timestep, encoder_hidden_states, pos_map, **added_cond_kwargs
        )
        if return_dict:
            return UNet2DConditionOutput(sample=result)
        else:
            return (sample,)
