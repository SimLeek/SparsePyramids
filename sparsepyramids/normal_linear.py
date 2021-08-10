import torch
import torch.nn.grad
from torch.nn.common_types import _size_2_t
from torch import Tensor
from typing import Optional, List


class NormLinear(torch.nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ):
        super(NormLinear, self).__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype
        )

        self.restrict_parameters()

    def restrict_parameters(self):
        eps = 1e-6
        disteps = 1e-2

        with torch.no_grad():
            # mimic physical weights. You can't be more than 100% connected, or nan% connected
            max = torch.max(torch.abs(self.weight)) + eps
            if max > 1.0 + disteps:
                self.weight = torch.nn.Parameter(self.weight / max)

            if self.bias is not None:
                max = torch.max(torch.abs(self.bias)) + eps
                if max > 1.0 + disteps:
                    self.bias = torch.nn.Parameter(self.bias / max)

    def forward(self, input: Tensor) -> Tensor:
        eps = 1e-6
        disteps = 1e-2

        if self.training:
            self.restrict_parameters()

        f = super(NormLinear, self).forward(input)
        f = f / (torch.max(torch.abs(f)) + eps)

        return f