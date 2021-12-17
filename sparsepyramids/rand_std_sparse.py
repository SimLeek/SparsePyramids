from typing import Any
from displayarray import display
from torch.autograd import Function
from torch import Tensor
import torch
import numpy as np


def _eliminate_zeros(x: Tensor):
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse_coo_tensor(ni, nv, x.shape)


from scipy.special import erfi
import math as m


def percentile_to_std(percentile):
    """Percentile to std, assuming a half normal distribution

    :param percentile: float between 0 and 1.
    """
    x = m.sqrt(2) * erfi(percentile)
    return x


fifty_percent_std = 0.8696735925295496849668947505414


class RandStdSparseFunc(Function):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        avg_percent_activation: float = 0.05,
        make_sparse_tensor=True,
        mean=None,
        rand_portion=0.5,
    ):
        """

        Note: closeness to actual sparsity depends on rand_portion and luck. As rand_portion approaches 1, sparsity
        will match avg_percent_activation more and more.

        :param ctx:
        :param input:
        :param avg_percent_activation:
        :return:
        """
        out = input.clone()
        if mean is None:
            std, mean = torch.std_mean(out)
        else:
            std = torch.std(out)
        goal_std = percentile_to_std(1.0 - avg_percent_activation)
        normalized = torch.abs(out / std - mean) / fifty_percent_std
        renorm = torch.normal(mean=0.0, std=normalized)
        final_norm = rand_portion * renorm + (1.0 - rand_portion) * normalized
        out[final_norm < goal_std] = 0

        ctx.was_sparse = input.is_sparse
        if input.is_sparse and make_sparse_tensor:
            out = _eliminate_zeros(out)
        elif input.is_sparse and not make_sparse_tensor:
            out = out.to_dense()
        elif not input.is_sparse and make_sparse_tensor:
            i = torch.nonzero(out)
            v = out[i]
            s = torch.sparse_coo_tensor(i, v, out.shape)
            out = s
        else:  # not input.is_sparse and not make_sparse_tensor:
            pass

        return out

    @staticmethod
    def backward(ctx: Any, grad_output) -> Any:
        """if ctx.is_sparse and not ctx.was_sparse:
            return grad_output.to_dense(), None
        elif not ctx.is_sparse and ctx.was_sparse:
            return _handle_sparse_conversion(grad_output)"""
        return grad_output


class RandStdSparse(torch.nn.Module):
    def __init__(
        self,
        avg_percent_activation: float = 0.05,
        make_sparse_tensor=True,
        mean=None,
        rand_portion=0.5,
    ):
        super(RandStdSparse, self).__init__()
        self.std_func = RandStdSparseFunc()
        self.avg_percent_activation = avg_percent_activation
        self.make_sparse_tensor = make_sparse_tensor
        self.mean = mean
        self.rand_portion = rand_portion

    def forward(self, input: Tensor) -> Tensor:
        out = self.std_func.apply(
            input,
            self.avg_percent_activation,
            self.make_sparse_tensor,
            self.mean,
            self.rand_portion,
        )

        return out


if __name__ == "__main__":

    class sparsifier(torch.nn.Module):
        def __init__(self):
            super(sparsifier, self).__init__()
            self.sp = RandStdSparse(
                avg_percent_activation=0.02, make_sparse_tensor=False, mean=0.0
            )

        def forward(self, x):
            x = self.sp(x)
            return x

    displayer = display(dark_neuron)
    model = sparsifier().cuda()

    while displayer:
        displayer.update()
        grab = torch.from_numpy(
            next(iter(displayer.FRAME_DICT.values()))[np.newaxis, ...].astype(
                np.float32
            )
            / 255.0
        )
        grab = torch.swapaxes(grab, 1, 3)
        grab = torch.swapaxes(grab, 2, 3)
        img = torch.autograd.Variable(grab).cuda()

        output = model(img)

        vis_output = output.detach()
        vis_output = torch.swapaxes(vis_output, 2, 3)
        vis_output = torch.swapaxes(vis_output, 1, 3)
        displayer.update(
            (vis_output.cpu().numpy()[0] * 255.0).astype(np.uint8), "sparsifier output"
        )
