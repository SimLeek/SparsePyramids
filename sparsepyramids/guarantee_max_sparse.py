from torch.autograd import Function
from torch import Tensor
import torch
from typing import Any
import math as m


def _eliminate_zeros(x: Tensor):
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse_coo_tensor(ni, nv, x.shape)


class RandMaxSparseFunc(Function):
    @staticmethod
    def forward(
        ctx, input: Tensor, out_sparsity: float = 0.05, make_sparse_tensor=True,
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
        nz = torch.nonzero(out).split(1, dim=1)
        count_nz = nz[0].shape[0]
        goal_nz = m.floor(out_sparsity * input.numel())
        if goal_nz < count_nz:
            perm = torch.randperm(count_nz)
            sel_perm = perm[0 : count_nz - goal_nz]
            sel_nz = tuple(nz[x][sel_perm] for x in range(len(nz)))
            out[sel_nz] = 0

            if input.is_sparse and make_sparse_tensor:
                out = _eliminate_zeros(out)

        ctx.was_sparse = input.is_sparse
        if input.is_sparse and not make_sparse_tensor:
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


class RandMaxSparse(torch.nn.Module):
    def __init__(
        self, out_percent_nonzero: float = 0.05, make_sparse_tensor=True,
    ):
        super(RandMaxSparse, self).__init__()
        self.max_func = RandMaxSparseFunc()
        self.out_sparsity = out_percent_nonzero
        self.make_sparse_tensor = make_sparse_tensor

    def forward(self, input: Tensor) -> Tensor:
        out = self.max_func.apply(input, self.out_sparsity, self.make_sparse_tensor)

        return out


if __name__ == "__main__":
    from rand_std_sparse import RandStdSparse

    from displayarray import display
    from sparsepyramids.tests import dark_neuron
    import numpy as np

    class sparsifier(torch.nn.Module):
        def __init__(self):
            super(sparsifier, self).__init__()
            self.sp = RandStdSparse(
                avg_percent_activation=0.02, make_sparse_tensor=False, mean=0.0
            )
            self.sp2 = RandMaxSparse(out_percent_nonzero=0.02, make_sparse_tensor=False)

        def forward(self, x):
            x = self.sp(x)
            x = self.sp2(x)
            # test out_percent_nonzero: torch.count_nonzero(x)/x.numel()
            # should always be less than out_percent_nonzero
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
