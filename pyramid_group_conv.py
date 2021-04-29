import torch
from typing import Optional, Dict, Any, List
import math as m

class PyramidGroup2D():
    def __init__(self,
                 tensors:List[torch.Tensor],
                 conv_args: Optional[Dict[str, Any]] = None,
                 halfing_val=m.sqrt(2)
                 ):
        self.tensors = tensors
        self.conv_args = conv_args
        self.halfing_val = halfing_val

    def append(self, t:torch.Tensor):
        self.tensors.append(t)

    @staticmethod
    def apply(x: 'PyramidGroup2D', conv: torch.nn.Conv2d):
        y = PyramidGroup2D([], x.conv_args, x.halfing_val)
        initial_conv_size = torch.Size([1+(k-1)*d for k, d in zip(x.conv_args['kernel_size'], x.conv_args['dilation'])])
        for t in x.tensors:
            if t.shape[-2]==torch.Size([1, 1]):
                # Images that were already converted to vectors pass through
                y.append(t)
            elif t.shape[-2]==initial_conv_size:
                y.append(conv.apply())


