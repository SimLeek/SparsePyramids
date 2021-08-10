from typing import Any

from torch.autograd import Function
from torch import Tensor
import torch


def _handle_sparse_conversion(inp: Tensor):
    """Only convert to a sparse tensor if it will actually save memory/calculation."""
    i = torch.nonzero(inp)
    if i.shape[0] < inp.numel() / (inp.shape[1] + 1):
        v = inp[i]
        s = torch.sparse_coo_tensor(i, v, inp.shape)
        return s, True
    else:
        return inp, False


def _handle_dense_conversion(inp: Tensor):
    """Only convert to a dense tensor if it will actually save memory/calculation."""
    i = torch.nonzero(inp)
    if i.shape[0] > inp.numel() / (inp.shape[1] + 1):
        s = inp.to_dense()
        return s, True
    else:
        inp = _eliminate_zeros(inp)
        return inp, False


def _eliminate_zeros(x: Tensor):
    mask = x._values().nonzero()
    nv = x._values().index_select(0, mask.view(-1))
    ni = x._indices().index_select(1, mask.view(-1))
    return torch.sparse_coo_tensor(ni, nv, x.shape)


class PercentMax(Function):
    @staticmethod
    def forward(
        ctx, input: Tensor, percent: float = 0.5,
    ):
        """

        :param ctx:
        :param input:
        :param percent:
        :param make_sparse_tensor: Should be true if it will typically save memory/calculation.
                                    <~20% for typical image tensors.
                                    (NCHW index + value = 5 numbers to store 1 non-zero value)
        :return:
        """
        max_in = torch.max(input)
        out = input.clone()
        out[torch.abs(out) < max_in * percent] = 0

        if input.is_sparse:
            out, is_dense = _handle_dense_conversion(out)
            ctx.is_sparse = not is_dense
        else:
            out, is_sparse = _handle_sparse_conversion(out)
            ctx.is_sparse = is_sparse

        ctx.was_sparse = input.is_sparse

        return out

    @staticmethod
    def backward(ctx: Any, grad_output) -> Any:
        if ctx.is_sparse and not ctx.was_sparse:
            return grad_output.to_dense(), None
        elif not ctx.is_sparse and ctx.was_sparse:
            return _handle_sparse_conversion(grad_output)


# todo: create toy example where input starts 0, adds rnd nums by increasing index until full, then removes until 0 again, and repeats
#   print if sparse or not
