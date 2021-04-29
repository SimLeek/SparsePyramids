import logging

import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn.grad
from torch import nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch import Tensor
from typing import Optional, Union
from torch.types import _int, _size
from torch.autograd import Variable
import numpy as np
from visual_debug import make_dispalyable_image_tensor
from displayarray import display
# from tests.videos import test_video_2
from tests.pics import neuron_pic_small

displayer = display(neuron_pic_small)

import logging

log_stats = False
intermediate_layer_visualizer = None
_logger_dict = {}


class SpBpConv2D(Function):
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

        if log_stats:
            outp1 = torch.numel(out[torch.abs(out) > 1])
            outp01 = torch.numel(out[torch.abs(out) > 0.1])
            outp001 = torch.numel(out[torch.abs(out) > 0.01])
            outp0001 = torch.numel(out[torch.abs(out) > 0.001])
            outpmax = torch.numel(out)

            if hash(ctx) not in _logger_dict.keys():
                logger = logging.getLogger(f'sparse_variational_convolution_layer_logger{hash(ctx)}')
                logger.setLevel(logging.INFO)
                handler = logging.FileHandler(f'sparse_variational_convolution_layer_log{hash(ctx)}.csv')
                logger.addHandler(handler)
                logger.info('Num Elements, Abs(Elements)>1, Abs(Elements)>.1, Abs(Elements)>.01, Abs(Elements)>.001,\n')
                _logger_dict[hash(ctx)] = logger
            else:
                logger = _logger_dict[hash(ctx)]
            logger.info(f'{outpmax}, {outp1}, {outp01}, {outp001}, {outp0001},\n')

        ctx.save_for_backward(input, out, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.variational_lr = variational_lr
        ctx.sparsity_lr = sparsity_lr

        # note: On most images/videos, you should eventually see a grey background, rgb lines depending on orientation,
        # and textures seperated into rgb. This depends on which output channel you map to rgb for your visualizer.
        if displayer is not None:
            displayer.update(make_dispalyable_image_tensor(out)[0].detach().cpu().numpy(),
                             f'SpBpConv2D {hash(ctx)} forward output')

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, y, w = ctx.saved_tensors
        x_grad = w_grad = None
        xy_sparsity = (
                torch.sign(y) * (torch.max(abs(y)) - abs(y)) *
                torch.log1p(torch.sum(abs(y), dim=(-1, -2))).detach()[..., None, None] *
                1e-5
        )
        c_sparsity = (
                torch.sign(y) * (torch.max(abs(y)) - abs(y)) *
                torch.log1p(torch.sum(abs(y), dim=(-3))).detach()[:, None, ...] *
                1e-4
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
            padding_mode: str = 'zeros'  # TODO: refine this type
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
        self.conv = SpBpConv2D()

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return self.conv.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                   weight, bias, self.stride,
                                   _pair(0), self.dilation, self.groups)
        return self.conv.apply(input, weight, bias, self.stride,
                               self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


if __name__ == '__main__':
    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = SparsifyingConv2D(3, 64, (3, 3), 1, 0, 1, 1, True, 'zeros')
            self.decoder = nn.ConvTranspose2d(64, 3, (3, 3), 1, 0, 0, 1, True, 1, 'zeros')

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    learning_rate = 1e-4
    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-3)

    while displayer:
        displayer.update()
        grab = torch.from_numpy(
            next(iter(displayer.FRAME_DICT.values()))[np.newaxis, ...].astype(np.float32) / 255.0
        )
        grab = torch.swapaxes(grab, 1, 3)
        grab = torch.swapaxes(grab, 2, 3)
        img = Variable(grab).cuda()
        output = model(img)

        vis_output = output.detach()
        vis_output = torch.swapaxes(vis_output, 2, 3)
        vis_output = torch.swapaxes(vis_output, 1, 3)
        displayer.update(
            (vis_output.cpu().numpy()[0] * 255.0).astype(np.uint8), "autoencoder output"
        )
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
