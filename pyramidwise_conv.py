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


class SparsifyingConv2DFunc(Function):
    @staticmethod
    def forward(ctx,
                input: Tensor,
                weight: Tensor,
                bias: Optional[Tensor] = None,
                stride: Union[_int, _size] = 1,
                padding: Union[_int, _size] = 0,
                dilation: Union[_int, _size] = 1,
                groups: _int = 1,
                variational_lr=1e-4,
                sparsity_lr=1e-5,
                order_lr=1e-3,
                ):
        out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input, out, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.variational_lr = variational_lr
        ctx.sparsity_lr = sparsity_lr
        ctx.order_lr = order_lr

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, y, w = ctx.saved_tensors
        x_grad = w_grad = None
        xy_sparsity = (
                torch.sign(y) * (torch.max(abs(y)) - abs(y)) *
                torch.log1p(torch.sum(abs(y), dim=(-1, -2))).detach()[..., None, None] *
                ctx.sparsity_lr
        )
        c_sparsity = (
                torch.sign(y) * (torch.max(abs(y)) - abs(y)) *
                torch.log1p(torch.sum(abs(y), dim=(-3))).detach()[:, None, ...] *
                ctx.variational_lr
        )

        pad = torch.zeros([y.shape[0], 1, *y.shape[2:]]).to(y.device)

        c_order = (
                (abs(y) < abs(torch.cat((y[:, 1:, ], pad), 1))) * torch.sign(y) * abs(y) * ctx.order_lr
        )

        grad_z = grad_output + xy_sparsity + c_sparsity + c_order

        x_grad = torch.nn.grad.conv2d_input(x.shape, w, grad_z,
                                            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        w_grad = torch.nn.grad.conv2d_weight(x, w.shape, grad_z,
                                             ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        return x_grad, w_grad, None, None, None, None, None


class SparsifyingConv2D(torch.nn.Conv2d):

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
            padding_mode: str = 'zeros',  # TODO: refine this type
            xy_sparsity_lr=1e-5,
            c_sparsity_lr=1e-4,
            device=None,
            dtype=None
    ):
        super(SparsifyingConv2D, self).__init__(
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
            dtype
        )
        self.conv = SparsifyingConv2DFunc()
        self.xy_sparsity_lr = xy_sparsity_lr
        self.c_sparsity = c_sparsity_lr
        with torch.no_grad():
            self.mean = torch.tensor(0)
            self.max = torch.tensor(2.0 / (in_channels * reduce(lambda x, y: x * y, self.kernel_size)))

        self.ncn = None
        self.ncp = None
        self.cc1 = None
        self.diffn = None
        self.diffp = None
        self.diff1 = None

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return self.conv.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                   weight, bias, self.stride,
                                   _pair(0), self.dilation, self.groups)
        return self.conv.apply(input, weight, bias, self.stride,
                               self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            max = torch.max(torch.abs(self.weight))
            self.weight.div_(max / self.max)

        f = self._conv_forward(input, self.weight, self.bias)

        return f
