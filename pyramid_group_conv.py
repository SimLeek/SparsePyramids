import torch
from typing import Optional, Dict, Any, List, Union
import math as m
from torchvision import transforms

class PyramidGroup2D():
    def __init__(self,
                 tensors:List[torch.Tensor],
                 conv_args: Optional[Dict[str, Any]] = None,
                 start_size = (360,640),  # h,w
                 halfing_val=2
                 ):
        self.tensors = tensors
        self.conv_args = conv_args
        self.halfing_val = halfing_val
        self.start_size = start_size
        self.sizes = [start_size]
        self.transformed_inputs = [[]]
        self.transforms = []
        x,y = start_size
        while x>1 or y>1:
            x = max(int(m.floor(self.sizes[-1][0]/m.sqrt(halfing_val))),1)
            y = max(int(m.floor(self.sizes[-1][1]/m.sqrt(halfing_val))),1)
            self.sizes.append((x,y))
            self.transformed_inputs.append([])
        for s in self.sizes:
            self.transforms.append(transforms.Resize(s))

    def apply_resize(self):
        for i in range(len(self.sizes)):
            for t in self.tensors:
                if t.shape[-2]>self.sizes[i][0] and t.shape[-1] > self.sizes[i][1]:
                    self.transformed_inputs[i].append(((t.shape[-2], t.shape[-1]),self.transforms[i](t)))
                else:
                    continue

    def apply_convolution(self):


    def append(self, t:torch.Tensor):
        self.tensors.append(t)

    @staticmethod
    def apply(x: Union['PyramidGroup2D', torch.Tensor], conv: torch.nn.Conv2d):
        y = PyramidGroup2D([], x.conv_args, x.halfing_val)
        initial_conv_size = torch.Size([1+(k-1)*d for k, d in zip(x.conv_args['kernel_size'], x.conv_args['dilation'])])
        for t in x.tensors:
            if t.shape[-2]==torch.Size([1, 1]):
                # Images that were already converted to vectors pass through
                y.append(t)
            elif t.shape[-2]==initial_conv_size:
                y.append(conv.apply())


def test_init():
    p = PyramidGroup2D([])
    print(p.sizes)

def test_apply_resize():
    from tests.pics import larg
    import PIL.Image
    import numpy as np
    from displayarray import breakpoint_display
    t = torch.Tensor(np.asarray(PIL.Image.open(larg)))
    t = torch.swapaxes(t, 0, -1)
    t = torch.swapaxes(t, 1, 2)
    p = PyramidGroup2D([t])
    p.apply_resize()
    for p in p.transformed_inputs:
        for i in p:
            b = torch.swapaxes(i[1], 0, -1)
            b = torch.swapaxes(b, 0, 1)
            breakpoint_display(b.numpy()/255)
