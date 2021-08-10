from displayarray import display

# from tests.videos import test_video_2
from typing import Optional
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import itertools
import math as m
import torch.nn.functional as F
from torchvision.transforms import functional as FV

log_stats = False

_logger_dict = {}


def center_surround_tensor(
    ndim,  # type: int
    center_in,  # type: List[int]
    center_out,  # type: List[int]
    surround_in,  # type: List[int]
    surround_out,  # type: List[int]
):
    """Generates a multi-channel center center surround matrix. Useful for isolating or enhancing edges.
    Note: center-surround tensors with 11 or more dimensions may take a while to generate. Make sure you cache those.
    :param ndim: number of dimensions
    :param center_in: input tensor of ints representing colors to look for in the center
    :param center_out: input tensor representing colors to output when more center is detected
    :param surround_in: tensor representing colors to look for outside of center
    :param surround_out: tensor representing colors to output when more surround color is detected
    """
    assert ndim >= 1

    center_surround = np.ndarray(
        shape=[3 for _ in range(ndim)] + [len(center_in), len(center_out)]
    )

    total = 0
    for tup in itertools.product(*[range(3) for _ in range(ndim)]):
        inv_manhattan_dist = sum([abs(t - 1) for t in tup])
        if inv_manhattan_dist == 0:
            center_surround[tup] = [[0 for _ in center_out] for _ in center_in]
        else:
            euclidian_dist = 1.0 / m.sqrt(inv_manhattan_dist)
            center_surround[tup] = [
                [o * i * euclidian_dist for o in surround_out] for i in surround_in
            ]
            total += euclidian_dist
    center_index = tuple([1 for _ in range(ndim)])
    center_surround[center_index] = [
        [o * i * total for o in center_out] for i in center_in
    ]
    return center_surround


def normalize_tensor_positive_negative(
    tensor,  # type: np.ndarray
    positive_value=1.0,
    negative_value=1.0,
    epsilon=1e-12,
):
    """ Normalizes a tensor so values above zero all add up to positive_value, and values below zero add up to
    -negative_value.
    :param tensor: Input tensor to normalize.
    :param positive_value: Positive parts of the tensor will sum up to this value.
    :param negative_value: Negative parts of the tensor will sum up to this value.
    :return: Normalized tensor.
    """
    sum_pos = max(sum([abs(x) if x > 0 else 0 for x in np.nditer(tensor)]), epsilon)
    sum_neg = max(sum([abs(x) if x < 0 else 0 for x in np.nditer(tensor)]), epsilon)
    for tup in itertools.product(*[range(x) for x in tensor.shape]):
        if tensor[tup] > 0:
            tensor[tup] *= positive_value / sum_pos
        if tensor[tup] < 0:
            tensor[tup] *= negative_value / sum_neg
    return tensor


def midget_rgc(
    n,  # type: int
):  # type: (...)->np.ndarray
    """Returns a tensor that can convolve a color image for better edge_orientation_detector detection.
    Based off of human retinal ganglian cells.
    :param n: number of dimensions
    :return: tensor used for convolution
    """
    d = 1.0
    out = (
        center_surround_tensor(
            n,
            center_in=[d, 0, 0],
            center_out=[d, 0, 0],
            surround_in=[d, 0, 0],
            surround_out=[-d, 0, 0],
        )
        + center_surround_tensor(
            n,
            center_in=[0, d, 0],
            center_out=[0, d, 0],
            surround_in=[0, d, 0],
            surround_out=[0, -d, 0],
        )
        + center_surround_tensor(
            n,
            center_in=[0, 0, d],
            center_out=[0, 0, d],
            surround_in=[0, 0, d],
            surround_out=[0, 0, -d],
        )
    )

    return normalize_tensor_positive_negative(out, 1.0, 1.0)


def edge_detector(
    n,  # type: int
):  # type: (...)->np.ndarray
    """Returns a tensor that can convolve a color image for better edge detection.
    Based off of retinal ganglian cells.
    :param n: number of dimensions
    :return: tensor used for convolution
    """
    d = 1.0
    out = center_surround_tensor(
        n, center_in=[d], center_out=[d], surround_in=[d], surround_out=[-d]
    )

    return normalize_tensor_positive_negative(out, 1.0, 1.0)


# RGC, Retinal Ganglion Cells, Simple Cells, all really just edge-detectors, are necessary when working with VAEs.
# VAEs remove edges due to the bottleneck they create by trying to organize all the
rgc = torch.FloatTensor(midget_rgc(2)) / 2.0
rgc = torch.swapaxes(rgc, 0, 3)
rgc = torch.swapaxes(rgc, 1, 2)

edge = torch.FloatTensor(edge_detector(2)) / 2.0
edge = torch.swapaxes(edge, 0, 3)
edge = torch.swapaxes(edge, 1, 2)

import time


def image_to_edge_pyramid_cv(img, scale_val=m.sqrt(2)):
    t0 = time.time()
    dst = cv2.filter2D(img, -1, rgc)
    t1 = time.time()
    print(f"cv2 filter time: {(t1 - t0) * 1000.0}ms")

    return dst


def image_to_rgc_pyramid(img, scale_val=m.sqrt(2)):
    global rgc
    rgc = rgc.to(img.device)

    in_shape = list(img.shape[-2:])

    imgs = [img]
    while True:
        in_shape[0] = m.floor(in_shape[0] / scale_val)
        in_shape[1] = m.floor(in_shape[1] / scale_val)
        if in_shape[0] <= 3 or in_shape[1] <= 3:
            break
        imgs.append(FV.resize(img, in_shape, FV.InterpolationMode.BILINEAR))

    out_imgs = []
    for i in imgs:
        out_imgs.append(F.conv2d(i, rgc))

    return out_imgs


def image_to_edge_pyramid(img, scale_val=m.sqrt(2)):
    global edge
    edge = edge.to(img.device)

    in_shape = list(img.shape[-2:])

    imgs = [img]
    while True:
        in_shape[0] = m.floor(in_shape[0] / scale_val)
        in_shape[1] = m.floor(in_shape[1] / scale_val)
        if in_shape[0] <= 3 or in_shape[1] <= 3:
            break
        imgs.append(FV.resize(img, in_shape, FV.InterpolationMode.BILINEAR))

    out_imgs = []
    for i in imgs:
        out_imgs.append(F.conv2d(i, edge))

    return out_imgs


def sum_any_pyramid(img_list, target_shape):
    """Sum up a list of images into a single image the size of targer_shape.

    Twice as slow as sum_pyramid, but works for lists with any image shapes.

    :param img_list: List of images. First image should be the largest.
    :return:
    """
    out_img = torch.zeros(list(img_list[0].shape[0:2]) + target_shape).to(
        img_list[0].device
    )
    for i, img in enumerate(img_list):
        out_img += FV.resize(img, target_shape, FV.InterpolationMode.BILINEAR) * (
            np.prod(list(img.shape[-2:])) / np.prod(target_shape)
        )
    out_img /= 2.0

    return out_img


def sum_pyramid(img_list, scale_val=m.sqrt(2)):
    """Sum up a list of images into a single image

    :param img_list: List of images. First image should be the largest.
    :return:
    """
    to_shape = list(img_list[0].shape[-2:])

    out_img: Optional[torch.Tensor] = None

    for i, img in enumerate(img_list):
        if i == 0:
            out_img = img
        else:
            out_img += FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR) * (
                np.prod(list(img.shape[-2:])) / np.prod(to_shape)
            )
    out_img /= 2.0

    return out_img


if __name__ == "__main__":
    from sparsepyramids.tests import smol

    displayer = display(smol)
    intermediate_layer_visualizer = displayer

    while displayer:
        displayer.update()
        img = next(iter(displayer.FRAME_DICT.values()))
        t0 = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grab = torch.from_numpy(
            img[np.newaxis, np.newaxis, ...].astype(np.float32) / 255.0
        )
        img = Variable(grab).cuda()

        output = image_to_edge_pyramid(img)
        output = sum_pyramid(output)
        t1 = time.time()
        print(f"pyramid edge filter time: {(t1 - t0) * 1000.0}ms")

        vis_output = output.detach()
        vis_output = torch.squeeze(vis_output)
        # vis_output = torch.swapaxes(vis_output, 1, 3)

        vis_output = (vis_output - vis_output.min()) / (
            vis_output.max() - vis_output.min()
        )

        displayer.update(
            (vis_output.cpu().numpy() * 255.0).astype(np.uint8),
            f"summed pyramid output",
        )

        # for i, o in enumerate(vis_output):
        #    displayer.update(
        #        (o.cpu().numpy()[0] * 255.0).astype(np.uint8), f"pyramid output {i}"
        #    )
