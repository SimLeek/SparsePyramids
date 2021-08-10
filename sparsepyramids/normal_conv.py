import torch
import torch.nn.grad
from torch.nn.common_types import _size_2_t
from torch import Tensor
from typing import Optional, List


class NormConv2d(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ):
        super(NormConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        eps = 1e-6
        disteps = 1e-2

        if self.training:
            with torch.no_grad():
                # mimic physical weights. You can't be more than 100% connected, and having nan values pop up is stupid.
                max = torch.max(torch.abs(self.weight)) + eps
                if max > 1.0 + disteps:
                    self.weight = torch.nn.Parameter(self.weight / max)

                if self.bias is not None:
                    max = torch.max(torch.abs(self.bias)) + eps
                    if max > 1.0 + disteps:
                        self.bias = torch.nn.Parameter(self.bias / max)

                # Could set percentage of norm, but weights aren't 0-1, but 0-0.1 or something, but the norm is 1,
                # which we don't actually care about
                # self.weight.div_(torch.norm(self.weight, dim=(2,3), keepdim=True))

        f = self._conv_forward(input, self.weight, self.bias)

        f = f / (torch.max(torch.abs(f)) + eps)

        # f = f/torch.norm(f, dim=(2,3), keepdim=True)

        return f


class NormConvTranspose2d(torch.nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ):
        super(NormConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:

        if self.training:
            with torch.no_grad():
                # mimic physical weights. You can't be more than 100% connected, and having nan values pop up is stupid.
                max = torch.max(torch.abs(self.weight))
                self.weight.div_(max)

                if self.bias is not None:
                    max = torch.max(torch.abs(self.bias))
                    self.bias.div_(max)

                # Could set percentage of norm, but weights aren't 0-1, but 0-0.1 or something, but the norm is 1,
                # which we don't actually care about
                # self.weight.div_(torch.norm(self.weight, dim=(2,3), keepdim=True))

        f = super(NormConvTranspose2d, self).forward(input, output_size)

        f = f / torch.max(torch.abs(f))

        # f = f/torch.norm(f, dim=(2,3), keepdim=True)

        return f
