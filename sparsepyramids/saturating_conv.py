import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn.grad
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch import Tensor
from typing import Optional, Union
from torch.types import _int, _size
from functools import reduce
from .fixed_saturate_tensor import saturate_tensor, saturate_duplicates_tensor


class SaturatingConv2DFunc(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        saturate_weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size] = 1,
        dilation: Union[_int, _size] = 1,
        groups: _int = 1,
        saturation_lr=1e-4,
    ):
        out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        out2 = F.conv2d(out, saturate_weight, None, stride, 0, dilation, groups)

        max2 = torch.max(torch.abs(out2))
        out2 = out2 / max2

        ctx.save_for_backward(input, out, out2, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.saturation_lr = saturation_lr

        return out2

    @staticmethod
    def backward(ctx, grad_output):
        x, y, y2, w = ctx.saved_tensors
        x_grad = w_grad = None

        sat_diff = (y2 - y) * ctx.saturation_lr

        grad_z = grad_output + sat_diff

        x_grad = torch.nn.grad.conv2d_input(
            x.shape, w, grad_z, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        w_grad = torch.nn.grad.conv2d_weight(
            x, w.shape, grad_z, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        )
        return x_grad, w_grad, None, None, None, None, None, None


class SaturatingConv2D(torch.nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        saturation_lr=1e-4,
        device=None,
        dtype=None,
    ):
        super(SaturatingConv2D, self).__init__(
            in_channels,
            in_channels,
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
        # assert padding == 1

        self.conv = SaturatingConv2DFunc()
        self.saturation_lr = saturation_lr
        self.sat_weights = saturate_tensor(ndim=2, in_channels=in_channels)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        self.sat_weights = self.sat_weights.to(input.device)
        if self.padding_mode != "zeros":
            return self.conv.apply(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                self.sat_weights,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return self.conv.apply(
            input,
            weight,
            self.sat_weights,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
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

        f = self._conv_forward(input, self.weight, self.bias)

        return f
