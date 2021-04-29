from typing import Tuple

import torch
import torch.nn
from displayarray.window import SubscriberWindows
from torch import nn


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

    def __init__(self, disp_op: nn.Module, display: SubscriberWindows):
        super().__init__()
        self.disp_op = disp_op
        self.display = display
        self.disp_op.eval()
        self.disp_op.requires_grad_()
        self.eval()
        self.requires_grad_()
        self.hook_up()

    def hook_up(self):
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):

            for e, inpt in enumerate(input):
                self.display.update(make_dispalyable_image_tensor(inpt)[0].detach().cpu().numpy(),
                                f'{self.disp_op._get_name() + str(self.disp_op.__hash__())} -- forward hook input {e}')
            for e, outpt in enumerate(input):
                self.display.update(make_dispalyable_image_tensor(outpt)[0].detach().cpu().numpy(),
                                    f'{self.disp_op._get_name() + str(self.disp_op.__hash__())} -- forward hook output {e}')

        def backward_hook(module, grad_input, grad_output: Tuple[torch.Tensor]):
            for e, inpt in enumerate(grad_input):
                self.display.update(make_dispalyable_image_tensor(inpt)[0].detach().cpu().numpy(),
                                    f'{self.disp_op._get_name() + str(self.disp_op.__hash__())} -- backward hook input {e}')
            for e, outpt in enumerate(grad_output):
                self.display.update(make_dispalyable_image_tensor(outpt)[0].detach().cpu().numpy(),
                                    f'{self.disp_op._get_name() + str(self.disp_op.__hash__())} -- backward hook output {e}')

        self.disp_op.register_full_backward_hook(backward_hook)
        self.disp_op.register_forward_hook(forward_hook)

    def forward(self, x):
        # Note: to use the forward and backward hooks, you have to call conv2d, not conv2d.forward
        x = self.disp_op(x)

        #self.display.update(make_dispalyable_image_tensor(x)[0].detach().cpu().numpy(),
        #                    f'{self.disp_op._get_name()+str(self.disp_op.__hash__())} -- forward')

        return x