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
                sparsity_lr=1e-5
                ):
        out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input, out, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.variational_lr = variational_lr
        ctx.sparsity_lr = sparsity_lr

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

        grad_z = grad_output + xy_sparsity + c_sparsity

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
            c_organization_lr=1.0
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
            padding_mode
        )
        self.conv = SparsifyingConv2DFunc()
        self.xy_sparsity_lr = xy_sparsity_lr
        self.c_sparsity = c_sparsity_lr
        with torch.no_grad():
            self.out_channel_correlation_indices = torch.zeros((out_channels, 2, 2))
            self.out_channel_correlation_values = torch.zeros((out_channels, 2, 2))
            # todo: make this match the device self.kernel/self.weight is on
            self.out_channel_correlation_values = self.out_channel_correlation_values.cuda()
            self.out_channel_correlation_indices = self.out_channel_correlation_indices.cuda()

            self.near_channels = torch.empty((self.out_channels, 2), dtype=torch.long)
            self.near_channels[:, 0] = torch.arange(-1, self.out_channels - 1)
            self.near_channels[0, 0] = self.out_channels - 1
            self.near_channels[:, 1] = torch.arange(1, self.out_channels + 1)
            self.near_channels[self.out_channels - 1, 1] = 0

            self.compare_channels = torch.randint(self.out_channels - 3, (self.out_channels, 1), dtype=torch.long)

            self.mean = torch.tensor(0)
            self.max = torch.tensor(2.0 / (in_channels * reduce(lambda x, y: x * y, self.kernel_size)))

        self.ncn = None
        self.ncp = None
        self.cc1 = None
        self.diffn = None
        self.diffp = None
        self.diff1 = None
        self.new_weights = None

        self.c_organization_lr = c_organization_lr

    def sort_channel_pass_pre(self):
        with torch.no_grad():
            del self.new_weights
            self.new_weights = self.weight.detach().clone()
            self.new_weights.requires_grad = True
            # max = torch.max(torch.abs(self.new_weights))
            # self.new_weights.div_(max/self.max)

            for o in range(self.out_channels):
                if self.out_channel_correlation_values[o, 0, 1] > self.out_channel_correlation_values[o, 0, 0]:
                    tv = self.out_channel_correlation_values[o, 0, 1].item()
                    self.out_channel_correlation_values[o, 0, 1] = self.out_channel_correlation_values[o, 0, 0]
                    self.out_channel_correlation_values[o, 0, 0] = tv

                    ti1 = int(self.out_channel_correlation_indices[o, 0, 1].item())
                    ti0 = int(self.out_channel_correlation_indices[o, 0, 0].item())
                    self.out_channel_correlation_indices[o, 0, 1] = self.out_channel_correlation_indices[o, 0, 0]
                    self.out_channel_correlation_indices[o, 0, 0] = ti1

                    # swap kernel _and_ output so backprop works
                    # inplace operation here
                    temp = self.new_weights[ti0, ...].clone()
                    self.new_weights[ti0, ...] = self.new_weights[ti1, ...]
                    self.new_weights[ti1, ...] = temp

                if self.out_channel_correlation_values[o, 1, 1] > self.out_channel_correlation_values[o, 1, 0]:
                    tv = self.out_channel_correlation_values[o, 1, 1].item()
                    self.out_channel_correlation_values[o, 1, 1] = self.out_channel_correlation_values[o, 1, 0]
                    self.out_channel_correlation_values[o, 1, 0] = tv

                    ti1 = int(self.out_channel_correlation_indices[o, 1, 1].item())
                    ti0 = int(self.out_channel_correlation_indices[o, 1, 0].item())
                    self.out_channel_correlation_indices[o, 1, 1] = self.out_channel_correlation_indices[o, 1, 0]
                    self.out_channel_correlation_indices[o, 1, 0] = ti1

                    # swap kernel _and_ output so backprop works
                    # inplace operation here
                    temp = self.new_weights[ti0, ...].clone()
                    self.new_weights[ti0, ...] = self.new_weights[ti1, ...]
                    self.new_weights[ti1, ...] = temp

            # del self.weight
            self.weight[...] = self.new_weights

            self.compare_channels = None
            self.compare_channels = torch.randint(self.out_channels - 3, (self.out_channels, 1))
            self.compare_channels[:, 0][self.compare_channels[:, 0] >= self.near_channels[:, 0]] += 3

    def sort_channel_pass_post(self):
        with torch.no_grad():
            for o in range(self.out_channels):

                if self.diff1[o].item() > 0:
                    if self.out_channel_correlation_indices[o, 1, 1] == self.cc1[o]:
                        self.out_channel_correlation_values[o, 1, 1] += 1.0 / torch.abs(self.diff1[o])
                    elif 1.0 / torch.abs(self.diff1[o]) > self.out_channel_correlation_values[o, 1, 1]:
                        self.out_channel_correlation_indices[o, 1, 1] = self.cc1[o]
                        self.out_channel_correlation_values[o, 1, 1] = 1.0 / torch.abs(self.diff1[o])
                else:
                    if self.out_channel_correlation_indices[o, 0, 1] == self.cc1[o]:
                        self.out_channel_correlation_values[o, 0, 1] += 1.0 / torch.abs(self.diff1[o])
                    elif 1.0 / torch.abs(self.diff1[o]) > self.out_channel_correlation_values[o, 0, 1]:
                        self.out_channel_correlation_indices[o, 0, 1] = self.cc1[o]
                        self.out_channel_correlation_values[o, 0, 1] = 1.0 / torch.abs(self.diff1[o])

                self.out_channel_correlation_values[o, 0, 0] += (1.0 / torch.abs(
                    self.diffn[o]) * self.c_organization_lr) / (self.out_channels - 3)
                self.out_channel_correlation_values[o, 1, 0] += (1.0 / torch.abs(
                    self.diffp[o]) * self.c_organization_lr) / (self.out_channels - 3)

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

        self.ncn = self.near_channels[:, 0]
        self.ncp = self.near_channels[:, 1]
        self.cc1 = self.compare_channels[:, 0]
        self.diffn = torch.sum(torch.abs(f[...]) - torch.abs(f[:, self.ncn, ...]), dim=(0, 2, 3))
        self.diffp = torch.sum(torch.abs(f[...]) - torch.abs(f[:, self.ncp, ...]), dim=(0, 2, 3))
        self.diff1 = torch.sum(torch.abs(f[...]) - torch.abs(f[:, self.cc1, ...]), dim=(0, 2, 3))

        return f
