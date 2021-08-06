import math

import cv2
import torch


def map_cos(x):
    """Maps 0 -> 1 to cos(0) -> cos(pi) (1 -> -1)"""
    x *= math.pi
    o = math.cos(x)
    return o


def saturate_tensor(ndim,  # type: int
                    in_channels,  # type: int
                    dist_func=map_cos
                    ):
    mat = torch.zeros((in_channels, in_channels))
    for i in range(in_channels):
        for o in range(in_channels):
            dist1 = float(abs(i - o)) / (in_channels / 2)
            i2 = (i + in_channels / 2) % in_channels
            o2 = (o + in_channels / 2) % in_channels
            dist2 = float(abs(i2 - o2)) / (in_channels / 2)
            dist = min(dist2, dist1)
            dist = dist_func(dist)
            mat[i, o] = dist
    mat = mat[(...,) + (None,) * ndim]
    # mat = mat/torch.norm(mat)
    return mat


def test_saturate_tensor():
    from displayarray import breakpoint_display
    from tests.pics import smol
    from PIL import Image, ImageOps
    import numpy as np
    import torch.nn.functional as F

    s = saturate_tensor(2, 3)
    with Image.open(smol) as l:
        l = np.array(l)
        l = cv2.cvtColor(l, cv2.COLOR_RGB2BGR)
        l = torch.as_tensor(l)
        l = l[None, ...]
        l = torch.swapaxes(l, 1, 3)
        l = torch.swapaxes(l, 2, 3)
        l = l.float()
        lo = F.conv2d(l, s)

        # lo2 = torch.clone(lo)
        # lo2[lo2<0] = 0
        lo = lo / torch.max(lo)
        l = l / torch.max(l)

        lo2 = l * .5 + lo * .5
        lo2 = lo2 / torch.max(lo2)

        l = torch.squeeze(l)
        l = torch.swapaxes(l, 0, 2)
        l = torch.swapaxes(l, 0, 1)

        lo = torch.squeeze(lo)
        lo = torch.swapaxes(lo, 0, 2)
        lo = torch.swapaxes(lo, 0, 1)

        lo2 = torch.squeeze(lo2)
        lo2 = torch.swapaxes(lo2, 0, 2)
        lo2 = torch.swapaxes(lo2, 0, 1)

        breakpoint_display(l.cpu().numpy(), lo.cpu().numpy(), lo2.cpu().numpy())


def saturate_duplicates_tensor(ndim,  # type: int
                               in_channels,  # type: int
                               sub_channels,  # type: int
                               dist_func=map_cos):
    mat = saturate_tensor(ndim, in_channels, dist_func)
    dup_mat = torch.repeat_interleave(mat, sub_channels, dim=0)
    dup_mat = torch.repeat_interleave(dup_mat, sub_channels, dim=1)
    return dup_mat
