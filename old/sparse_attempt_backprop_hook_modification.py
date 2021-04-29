import torch
from torch import nn
from typing import Tuple
import math as m
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.nn.functional as F
visual_debug = False


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


# todo:
#  * Only delete a neuron if it's new and hasn't been firing a lot consecutively. That way you don't forget important
#    things in new environments.
#  * Delete new neurons if they haven't fired after 100 inputs or so
#  * Add new neurons that take input from unique areas outside of most of the dendrites of the other nearby neurons
class SparsifyingConv2D(nn.Module):
    def __init__(self,
                 conv2d: nn.Conv2d,
                 target_sparsity: float = 0.02,
                 enforcement: str = 'global',
                 local_size: Tuple[float, float] = (64, 64),
                 enforce_channel_sparsity=True):
        """
        A Conv2D that automatically enforces that output should be a specific sparsity.

        :param target_sparsity: target portion of neurons that should be on. 1=100%.
        :param enforcement: Can be 'grid', 'local square', 'local cartesian', or 'global'
            'grid' divides the output up into a grid with rectangles of size 'local_size'
                and enforces the desired sparsity within each of those rectangles separately.
            'local square' (slow) enforces the desired sparsity on a rectangle of 'local_size' around
                each 'enforcement_dims' output.
            'local cartesian' (slowest) enforces the desired sparsity on an oval of radii 'local_size'
                around each 'enforcement_dims' output.
            'global' enforces sparsity in the entire output
        :param local_size: size of the area to enforce sparcity on, if not using global enforcement.
        """
        super(SparsifyingConv2D, self).__init__()
        self.conv2d = conv2d
        self.target_sparsity = target_sparsity
        self.enforcement = enforcement
        self.local_size = local_size
        self.enforce_channel_sparsity = enforce_channel_sparsity
        self.sparsity = 0.0

        self.eval()
        self.requires_grad_()
        self.hook_up()

    def hook_up(self, lr = 1e-3):
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
            if self.enforcement == 'global':
                if self.enforce_channel_sparsity:
                    with torch.no_grad():
                        min_out = output[0].clone().detach()
                        #min_out[0][abs(min_out[0]) < 0.01] = 0
                        #print(torch.count_nonzero(min_out[0]).item())
                        #total_min_out = abs(min_out)
                        s = torch.log1p(torch.sum(abs(min_out))).detach()
                        #s2 = torch.log1p(torch.sum(s)).detach()
                        #s2 = torch.sum(s)*(max(s)/s.shape[0]).detach()
                        #print(s)
                        #print(s2)
                else:
                    min_out = output[0].clone().detach()
                    s = torch.log1p(torch.sum(abs(min_out), dim=(1,2))).detach()
                max_vis = torch.max(min_out)
                min_vis = torch.min(min_out)
                t = min_out - min_vis
                t = t * 1.0 / (max_vis - min_vis)
                t = 1.0 - t
                self.s = s
                self.sparsified_forward = s * t

        def backward_hook(module, grad_input, grad_output: Tuple[torch.Tensor]):
            #print('backward_hook called')
            #if isinstance(self.sparsity, float):
            #    log_sp = m.log1p(self.sparsity)*lr
            #elif isinstance(self.sparsity, torch.Tensor):
            #    log_sp = torch.log1p(self.sparsity)*lr
            #else:
            #    raise NotImplementedError(f'sparsity of type {type(self.sparsity)} is not supported')
            if self.enforce_channel_sparsity:
                #print(f'sparsified forward: {torch.sum(self.sparsitied_forward).item()}')
                #print(f'grad input: {torch.sum(abs(grad_input[0])).item()}')
                modified_img_in = grad_input[0] + self.sparsified_forward * lr
                #modified_img_in = grad_input[0] + log_sp[:, None, None]*torch.sign(grad_input[0])
                #modified_conv_in = grad_input[1] + log_sp[:, None, None]*torch.sign(grad_input[1])
            else:
                pass
                #modified_img_in = grad_input[0] + log_sp.item()*torch.sign(grad_input[0])
                #modified_conv_in = grad_input[1] + log_sp.item()*torch.sign(grad_input[1])

            return (modified_img_in, grad_input[1])

        self.conv2d.register_backward_hook(backward_hook)
        self.conv2d.register_forward_hook(forward_hook)

    def forward(self, x):
        # Note: to use the forward and backward hooks, you have to call conv2d, not conv2d.forward
        x = self.conv2d(x)

        self.vis_output = make_dispalyable_image_tensor(x)

        return x


class SparsifyingConvTranspose2d(nn.Module):
    def __init__(self,
                 convtranspose2d: nn.ConvTranspose2d,
                 target_sparsity: float = 0.02,
                 enforcement: str = 'global',
                 local_size: Tuple[float, float] = (64, 64),
                 enforce_channel_sparsity=True):
        """
        A Conv2D that automatically enforces that output should be a specific sparsity.

        :param target_sparsity: target portion of neurons that should be on. 1=100%.
        :param enforcement: Can be 'grid', 'local square', 'local cartesian', or 'global'
            'grid' divides the output up into a grid with rectangles of size 'local_size'
                and enforces the desired sparsity within each of those rectangles separately.
            'local square' (slow) enforces the desired sparsity on a rectangle of 'local_size' around
                each 'enforcement_dims' output.
            'local cartesian' (slowest) enforces the desired sparsity on an oval of radii 'local_size'
                around each 'enforcement_dims' output.
            'global' enforces sparsity in the entire output
        :param local_size: size of the area to enforce sparcity on, if not using global enforcement.
        """
        super(SparsifyingConvTranspose2d, self).__init__()
        self.convtranspose2d = convtranspose2d
        self.target_sparsity = target_sparsity
        self.enforcement = enforcement
        self.local_size = local_size
        self.enforce_channel_sparsity = enforce_channel_sparsity
        self.sparsity = 0.0

        self.eval()
        self.requires_grad_()
        self.hook_up()

    def hook_up(self, lr = 1e-8):
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor], output: Tuple[torch.Tensor]):
            if self.enforcement == 'global':
                if self.enforce_channel_sparsity:
                    with torch.no_grad():
                        min_out = input[0].clone().detach()
                        s = torch.log1p(torch.sum(abs(min_out))).detach()
                else:
                    min_out = input[0].clone().detach()
                    s = torch.log1p(torch.sum(abs(min_out), dim=(2,3))).detach()
                sc = torch.log1p(torch.sum(abs(min_out), dim=1)).detach()

                max_vis = torch.max(min_out)
                min_vis = torch.min(min_out)
                t = min_out - min_vis
                t = t * 1.0 / (max_vis - min_vis)
                t = 1.0 - t
                self.sparsified_forward = s * t + sc * t * 10
                print(f'sparsified forward min:{torch.min(self.sparsified_forward).item()}, max:{torch.max(self.sparsified_forward).item()}')
                print(f'sc * t forward min:{torch.min(sc * t).item()}, max:{torch.max(sc * t).item()}')

                self.forward_vis_output = make_dispalyable_image_tensor(self.sparsified_forward[0],
                                                                        swaps=((1,2),(0,2)))

        def backward_hook(module, grad_input, grad_output: Tuple[torch.Tensor]):
            print(grad_input[0].shape)
            print(grad_output[0].shape)

            if self.enforce_channel_sparsity:
                c_diff = (self.sparsified_forward.shape[1] - grad_input[0].shape[1])//2
                x_diff = (grad_input[0].shape[2] - self.sparsified_forward.shape[2])//2
                y_diff = (grad_input[0].shape[3] - self.sparsified_forward.shape[3])//2
                print(c_diff, x_diff, y_diff)

                modified_forward = F.pad((self.sparsified_forward * lr),
                                         (0,0,11,10, x_diff, x_diff, y_diff, y_diff))
                print(modified_forward.shape)
                modified_img_in = torch.zeros_like(grad_input[0]) + modified_forward
                print(f'sparsify: {torch.max(self.sparsified_forward).item()*lr}')
                print(f'grad input: {torch.max(grad_input[0])}')
            else:
                pass
            self.back_vis_output = make_dispalyable_image_tensor(modified_img_in)

            return (modified_img_in, grad_input[1])

        self.convtranspose2d.register_backward_hook(backward_hook)
        self.convtranspose2d.register_forward_hook(forward_hook)

    def forward(self, x):
        # Note: to use the forward and backward hooks, you have to call conv2d, not conv2d.forward
        x = self.convtranspose2d(x)

        self.vis_output = make_dispalyable_image_tensor(x)

        return x


class PyramidifyingConv2D(nn.Module):
    pass

if __name__ == '__main__':
    from displayarray import display
    #from tests.videos import test_video_2
    from tests.pics import neuron_pic_small

    displayer = display(neuron_pic_small)

    class autoencoder(nn.Module):
        def __init__(self):
            super(autoencoder, self).__init__()
            self.encoder = SparsifyingConv2D(nn.Conv2d(3, 24, (3,3), 1, 0, 1, 1, True, 'zeros'))
            self.decoder = SparsifyingConvTranspose2d(nn.ConvTranspose2d(24, 3, (3,3), 1, 0, 0, 1, True, 1, 'zeros'))

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x


    learning_rate = 1e-3
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
        dispo = model.encoder.vis_output.cpu().numpy()[0]
        dispo[dispo>1]=1
        dispo[dispo < 0] = 0
        displayer.update(
            (dispo * 255.0).astype(np.uint8), "conv2d output"
        )

        vis_output = output.detach()
        vis_output = torch.swapaxes(vis_output, 2, 3)
        vis_output = torch.swapaxes(vis_output, 1, 3)
        displayer.update(
            (vis_output.cpu().numpy()[0] * 255.0).astype(np.uint8), "autoencoder output"
        )

        vis_output3 = model.decoder.forward_vis_output.cpu().numpy()*255
        displayer.update(
            (vis_output3).astype(np.uint8), "convtranspose forward"
        )

        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()

        vis_output2 = model.decoder.back_vis_output.cpu().numpy()[0]*255
        displayer.update(
            (vis_output2).astype(np.uint8), "convtranspose back"
        )

        optimizer.step()

