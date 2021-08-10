from displayarray import display

# from tests.videos import test_video_2
from sparsepyramids.sparsifying_conv import SparsifyingConv2DFunc, SparsifyingConv2D
from sparsepyramids.tests.pics import smol
from torch import Tensor
from torch.nn.common_types import _size_2_t
from typing import Optional, Union
from torch.types import _int, _size
import torch
from sparsepyramids.visual_debug import make_dispalyable_image_tensor
from torch import nn
from torch.autograd import Variable
import numpy as np

import logging

log_stats = False

displayer = display(smol)
intermediate_layer_visualizer = displayer

_logger_dict = {}


class DbgSpBpConv2D(SparsifyingConv2DFunc):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Union[_int, _size] = 1,
        padding: Union[_int, _size] = 0,
        dilation: Union[_int, _size] = 1,
        groups: _int = 1,
        variational_lr=1e-4,
        sparsity_lr=1e-5,
    ):
        out = SparsifyingConv2DFunc.forward(
            ctx,
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            variational_lr,
            sparsity_lr,
        )

        if log_stats:
            outp1 = torch.numel(out[torch.abs(out) > 1])
            outp01 = torch.numel(out[torch.abs(out) > 0.1])
            outp001 = torch.numel(out[torch.abs(out) > 0.01])
            outp0001 = torch.numel(out[torch.abs(out) > 0.001])
            outpmax = torch.numel(out)

            if hash(ctx) not in _logger_dict.keys():
                logger = logging.getLogger(
                    f"sparse_variational_convolution_layer_logger{hash(ctx)}"
                )
                logger.setLevel(logging.INFO)
                handler = logging.FileHandler(
                    f"sparse_variational_convolution_layer_log{hash(ctx)}.csv"
                )
                logger.addHandler(handler)
                logger.info(
                    "Num Elements, Abs(Elements)>1, Abs(Elements)>.1, Abs(Elements)>.01, Abs(Elements)>.001,\n"
                )
                _logger_dict[hash(ctx)] = logger
            else:
                logger = _logger_dict[hash(ctx)]
            logger.info(f"{outpmax}, {outp1}, {outp01}, {outp001}, {outp0001},\n")

        # note: On most images/videos, you should eventually see a grey background, rgb lines depending on orientation,
        # and textures seperated into rgb. This depends on which output channel you map to rgb for your visualizer.
        if intermediate_layer_visualizer is not None:
            intermediate_layer_visualizer.update(
                make_dispalyable_image_tensor(out)[0].detach().cpu().numpy(),
                f"SpBpConv2D 0 forward output",
            )

        return out


class DbgSparsifyingConv2D(SparsifyingConv2D):
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
        xy_sparsity_lr=1e-5,
        c_sparsity_lr=1e-4,
    ):
        super(DbgSparsifyingConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            xy_sparsity_lr,
            c_sparsity_lr,
        )
        self.conv = DbgSpBpConv2D()


if __name__ == "__main__":

    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = SparsifyingConv2D(3, 32, (3, 3), 1, 0, 1, 1, True, "zeros")
            self.decoder = nn.ConvTranspose2d(
                32, 3, (3, 3), 1, 0, 0, 1, True, 1, "zeros"
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    learning_rate = 1e-3
    model = autoencoder().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-3
    )

    i = 0
    skip_bool = False
    while displayer:
        displayer.update()
        if not skip_bool:
            print(i)
            grab = torch.from_numpy(
                next(iter(displayer.FRAME_DICT.values()))[np.newaxis, ...].astype(
                    np.float32
                )
                / 255.0
            )
            grab = torch.swapaxes(grab, 1, 3)
            grab = torch.swapaxes(grab, 2, 3)
            img = Variable(grab).cuda()

            output = model(img)

            vis_output = output.detach()
            vis_output = torch.swapaxes(vis_output, 2, 3)
            vis_output = torch.swapaxes(vis_output, 1, 3)
            displayer.update(
                (vis_output.cpu().numpy()[0] * 255.0).astype(np.uint8),
                "autoencoder output",
            )
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            i += 1
