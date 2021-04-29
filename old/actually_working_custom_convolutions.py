import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn.grad
from torch import nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair
from torch import Tensor
from typing import Tuple, Optional, Union
from torch.types import _int, _size
from displayarray.window.subscriber_windows import display, SubscriberWindows
from torch.autograd import Variable
import numpy as np


def make_dispalyable_image_tensor(tens, normalize=True, min_target=0.0, max_target=1.0, swaps=((2,3),(1,3))):
    t = tens.detach()
    if normalize:
        max_vis = torch.max(t)
        min_vis = torch.min(t)
        t = t - min_vis + min_target
        t = t * max_target / (max_vis - min_vis)
    t = torch.swapaxes(t, *swaps[0])
    t = torch.swapaxes(t, *swaps[1])
    return t




class VisualDebugModule(torch.nn.Module):
    visual_debug = (False, False, False, False)

    def __init__(self, module: nn.Module, display: SubscriberWindows):
        self.module = module
        self.display = display
        super().__init__()

    def hook_up(self, lr = 1e-3):
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
            for e, inpt in enumerate(input):
                self.display.update(make_dispalyable_image_tensor(inpt),
                                f'{self.module._get_name() + str(self.module.__hash__())} -- forward hook input {e}')
            for e, outpt in enumerate(input):
                self.display.update(make_dispalyable_image_tensor(outpt),
                                    f'{self.module._get_name() + str(self.module.__hash__())} -- forward hook output {e}')

        def backward_hook(module, grad_input, grad_output: Tuple[torch.Tensor]):
            for e, inpt in enumerate(grad_input):
                self.display.update(make_dispalyable_image_tensor(inpt),
                                    f'{self.module._get_name() + str(self.module.__hash__())} -- backward hook input {e}')
            for e, outpt in enumerate(grad_output):
                self.display.update(make_dispalyable_image_tensor(outpt),
                                    f'{self.module._get_name() + str(self.module.__hash__())} -- backward hook output {e}')

        self.module.register_full_backward_hook(backward_hook)
        self.module.register_forward_hook(forward_hook)

    def forward(self, x):
        # Note: to use the forward and backward hooks, you have to call conv2d, not conv2d.forward
        x = self.module(x)

        self.display.update(make_dispalyable_image_tensor(x),
                            f'{self.module._get_name()+str(self.module.__hash__())} -- forward')

        return x


class SpBpConv2D(Function):
    @staticmethod
    def forward(ctx,
                input: Tensor,
                weight: Tensor,
                bias: Optional[Tensor]=None,
                stride: Union[_int, _size]=1,
                padding: Union[_int, _size]=0,
                dilation: Union[_int, _size]=1,
                groups: _int=1
                ):
        out = F.conv2d(input, weight, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(input, out, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, y, w  = ctx.saved_variables
        x_grad = w_grad = None
        xy_sparsity = torch.log1p(torch.sum(abs(y), dim=(-1, -2))).detach()[..., None, None] * 1e-7
        c_sparsity =  torch.log1p(torch.sum(abs(y), dim=(-3))).detach()[:, None, ...] * 1e-7

        '''print(x.shape)
        print(y.shape)
        print(xy_sparsity.shape)
        print(c_sparsity.shape)
        print(grad_output.shape)'''

        grad_z = grad_output+ xy_sparsity + c_sparsity

        #print(grad_z.shape)

        x_grad = torch.nn.grad.conv2d_input(x.shape, w, grad_z,
                                            ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        print(f'grad_output[0] min:{torch.min(grad_output[0]).item()}, max:{torch.max(grad_output[0]).item()}')
        print(f'xy_sparsity min:{torch.min(xy_sparsity).item()}, max:{torch.max(xy_sparsity).item()}')
        print(f'c_sparsity min:{torch.min(c_sparsity).item()}, max:{torch.max(c_sparsity).item()}')

        w_grad = torch.nn.grad.conv2d_weight(x, w.shape, grad_z,
                                             ctx.stride, ctx.padding, ctx.dilation, ctx.groups)
        #grad_input[0][:] = x_grad
        '''print(grad_input[0].shape)
        print(grad_input[1].shape)
        print(grad_output[0].shape)
        print(x_grad.shape)
        print(w_grad.shape)'''
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
        #self.hook_up()
        self.conv = SpBpConv2D()

    '''def hook_up(self, lr = 1e-3):
        def backward_hook(module, grad_input, grad_output: Tuple[torch.Tensor]):
            for e, grout in enumerate(grad_output):
                print('out', e, grout.shape)
            for e, grin in enumerate(grad_input):
                print('in', e, grin.shape)
            return self.conv.backward(self.conv, grad_input, grad_output)

        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
            return self._conv_forward(input[0], self.weight, self.bias)

        self.register_backward_hook(backward_hook)
        self.register_forward_hook(forward_hook)'''


    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':

            return self.conv.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return self.conv.apply(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    #def forward(self, input: Tensor) -> Tensor:
    #    return self._conv_forward(input, self.weight, self.bias)

if __name__ == '__main__':
    from displayarray import display
    #from tests.videos import test_video_2
    from tests.pics import neuron_pic_small

    displayer = display(neuron_pic_small)

    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = SparsifyingConv2D(3, 24, (3,3), 1, 0, 1, 1, True, 'zeros')
            self.decoder = nn.ConvTranspose2d(24, 3, (3,3), 1, 0, 0, 1, True, 1, 'zeros')

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