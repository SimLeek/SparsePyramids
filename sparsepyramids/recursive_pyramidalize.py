import torch
import torch.nn.grad
import torchvision.transforms.functional
from torch.nn.common_types import _size_2_t
from torch import Tensor
from typing import Union, List, Sequence
import math as m
from torchvision.transforms import functional as FV
import warnings
import numpy as np
import sys
from torch.nn import functional as F
from sparsepyramids.normal_conv import NormConv2d, NormConvTranspose2d
from sparsepyramids.fixed_saturate_tensor import saturate_tensor, saturate_duplicates_tensor


class RecursivePyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            scale_val: float = m.sqrt(2),
            interpolation=FV.InterpolationMode.BILINEAR,
            min_size=3,
            max_size=None,
            antialias=None,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()
        if not isinstance(scale_val, float):
            raise TypeError(
                "scale_val should be a real number. Got {}".format(type(scale_val))
            )
        assert scale_val > 1.0, "scale_val needs to be greater than one."
        if not isinstance(min_size, (int, Sequence)) and not min_size is None:
            raise TypeError(
                "min_size should be int, sequence, or None. Got {}".format(
                    type(scale_val)
                )
            )
        if isinstance(min_size, (float, int)):
            min_size = [min_size, min_size]
        if isinstance(min_size, Sequence):
            if len(min_size) not in (1, 2):
                raise ValueError(
                    "If min_size is a sequence, it should have 1 or 2 values"
                )
            elif len(min_size) == 1:
                min_size = list(min_size) * 2
        self.min_size = min_size

        self.scale_val = scale_val
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__

    def pyramidalize(self, input: Tensor):
        in_shape = list(input.shape[-2:])

        out = [input]

        while True:
            in_shape[0] = max(m.floor(in_shape[0] / self.scale_val), 1)
            in_shape[1] = max(m.floor(in_shape[1] / self.scale_val), 1)
            if self.min_size is None:
                if in_shape[0] <= 1 and in_shape[1] <= 1:
                    break
            else:
                if in_shape[0] <= self.min_size[0] or in_shape[1] <= self.min_size[1]:
                    break
            out.append(FV.resize(input, in_shape, self.interpolation))

        return out

    def pyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out = []
            for i in input:
                out.append(self.pyramidalize_list(i))
            return out
        elif isinstance(input, Tensor):
            return self.pyramidalize(input)

    def forward(self, input: Tensor):
        out = self.pyramidalize_list(input)

        return out


class RecursiveSumDepyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            scale_pow: float = 1,
            interpolation=FV.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=None,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()
        if not isinstance(scale_pow, (float, int)):
            raise TypeError(
                "scale_val should be a real number. Got {}".format(type(scale_pow))
            )

        self.scale_pow = scale_pow
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__

    def depyramidalize(self, input: List[Tensor]):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img += (
                        FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR)
                        * (np.prod(list(img.shape[-2:])) / np.prod(to_shape))
                        ** self.scale_pow
                )
        out_img /= 2.0

        return out_img

    def depyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self, input: Tensor):
        out = self.depyramidalize_list(input)

        return out


class RecursiveChanDepyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            scale_pow: float = 1,
            interpolation=FV.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=None,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()
        if not isinstance(scale_pow, (float, int)):
            raise TypeError(
                "scale_pow should be a real number. Got {}".format(type(scale_pow))
            )

        self.scale_pow = scale_pow
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return (
                self.__class__.__name__
                + "(size={0}, interpolation={1}, max_size={2}, antialias={3})".format(
            self.size, interpolate_str, self.max_size, self.antialias
        )
        )

    def depyramidalize(self, input: List[Tensor]):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img = torch.cat(
                    [out_img, FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR)],
                    dim=1,
                )

        return out_img

    def depyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self, input: Tensor):
        out = self.depyramidalize_list(input)

        return out


class RecursiveConvDepyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            shape=(1, 1),
            interpolation=FV.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=None,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()

        self.in_channels = in_channels
        self.conv = NormConv2d(in_channels, out_channels, shape)
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__

    def depyramidalize(self, input: List[Tensor]):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img = torch.cat(
                    [out_img, FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR)],
                    dim=1,
                )

        # necessary for level 2 depyramidalizations and up
        # todo: use 3d and 4d convolutions for level 2 depyramidalizations and up
        if out_img.shape[1] < self.in_channels:
            # images sets should always contain smallest images,
            # but not largest ones, so place at end, or pad front with zeros:
            out_img = F.pad(
                out_img, (0, 0, 0, 0, self.in_channels - out_img.shape[1], 0)
            )
        elif out_img.shape[1] > self.in_channels:
            out_img = out_img[:, 0: self.in_channels, ...]

        out_img = self.conv.forward(out_img)

        return out_img

    def depyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self, input: Tensor):
        out = self.depyramidalize_list(input)

        return out


class RecursiveSaturateDepyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            levels,
            in_channels=1,
            interpolation=FV.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=None,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()

        self.in_channels = levels
        self.max_size = max_size

        if in_channels == 1:
            self.weights = saturate_tensor(ndim=2, in_channels=levels)
        else:
            self.weights = saturate_duplicates_tensor(
                ndim=2, in_channels=levels, sub_channels=in_channels
            )

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__

    def depyramidalize(self, input: List[Tensor]):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img = torch.cat(
                    [out_img, FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR)],
                    dim=1,
                )

        # necessary for level 2 depyramidalizations and up
        # todo: use 3d and 4d convolutions for level 2 depyramidalizations and up
        if out_img.shape[1] < self.in_channels:
            # images sets should always contain smallest images,
            # but not largest ones, so place at end, or pad front with zeros:
            out_img = F.pad(
                out_img, (0, 0, 0, 0, self.in_channels - out_img.shape[1], 0)
            )
        elif out_img.shape[1] > self.in_channels:
            out_img = out_img[:, 0: self.in_channels, ...]

        out_img = F.conv2d(out_img, self.weights)

        return out_img

    def depyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self, input: Tensor):
        out = self.depyramidalize_list(input)

        return out


from sparsepyramids.saturating_conv import SaturatingConv2D


class RecursiveDepthDepyramidalize2D(torch.nn.Module):
    def __init__(
            self,
            levels,
            in_channels,
            out_channels,
            shape=(1, 1),
            interpolation=FV.InterpolationMode.BILINEAR,
            max_size=None,
            antialias=None,
            self_train_percent=0.25,
    ):
        """Recursively create image pyramids, even if the input is already an image pyramid.

        :param scale_val (float): Desired downscaling size.
            Input tensor width and height will be sequentially downscaled by this value.
        :param interpolation (InterpolationMode): Desired interpolation enum
        :param min_size: Minimum width and height for any output tensor.
            Useful for allowing convolutions on all tensors. If the size isn't limited and the convolution is larger
            than some tensors, then those tensors will need to be teased out and either padded or used for other
            operations. Teasing out must be done with another module, which should return a list of lists/tensors of
            conv-able tensors and a list of lists/tensors of non-conv-able tensors.
        :param max_size: The maximum allowed for the longer edge of the resized image
        :param antialias (bool, optional): antialias flag. Only works for BILINEAR interpolation with tensors.

        """
        super().__init__()

        self.in_channels = in_channels
        self.levels = levels
        self.max_size = max_size
        self.self_train_percent = self_train_percent

        if in_channels != 1:
            raise NotImplementedError("Currently this only supports c ~= distance")
        self.conv = SaturatingConv2D(levels, shape)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = FV._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__

    def depyramidalize(self, input: List[Tensor]):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img = torch.cat(
                    [out_img, FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR)],
                    dim=1,
                )

        # necessary for level 2 depyramidalizations and up
        # todo: use 3d and 4d convolutions for level 2 depyramidalizations and up
        if out_img.shape[1] < self.levels:
            # images sets should always contain smallest images,
            # but not largest ones, so place at end, or pad front with zeros:
            out_img = F.pad(out_img, (0, 0, 0, 0, self.levels - out_img.shape[1], 0))
        elif out_img.shape[1] > self.levels:
            out_img = out_img[:, 0: self.levels, ...]

        out_img = self.conv.forward(out_img)

        return out_img

    def depyramidalize_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self, input: Tensor):
        out = self.depyramidalize_list(input)

        return out


def get_module_for_applying_module_to_nested_tensors(op, *args, **argv):
    class ApplyToNestedTensors(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.op = op(*args, **argv)

        def __repr__(self):
            return self.__class__.__name__ + f"(args={args})"

        def apply_list(self, input: Union[Tensor, List[Tensor]]):
            if isinstance(input, list):
                out = []
                for i in input:
                    out.append(self.apply_list(i))
                return out
            elif isinstance(input, Tensor):
                return self.op.forward(input)

        def forward(self, input: Tensor):
            out = self.apply_list(input)

            return out

    return ApplyToNestedTensors()


def apply_func_to_nested_tensors(input, op, *args, **argv):
    def apply_list(input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out = []
            for i in input:
                out.append(apply_list(i))
            return out
        elif isinstance(input, Tensor):
            return op(input, *args, **argv)

    return apply_list(input)


def apply_2func_to_nested_tensors(input, input2, op, *args, **argv):
    def apply_list(
            input: Union[Tensor, List[Tensor]], input2: Union[Tensor, List[Tensor]]
    ):
        if isinstance(input, list):
            out = []
            if isinstance(input2, list):
                for i, i2 in zip(input, input2):
                    out.append(apply_list(i, i2))
            else:
                for i in input:
                    out.append(apply_list(i, input2))
            return out
        elif isinstance(input, Tensor):
            if isinstance(input2, Tensor) and len(input2.shape) < len(input.shape):
                return op(
                    input,
                    input2[
                        [...]
                        + [None for _ in range(len(input.shape) - len(input2.shape))]
                        ],
                    *args,
                    **argv,
                )
            else:
                return op(input, input2, *args, **argv)

    return apply_list(input, input2)


def ravel_nested(input):
    partial_ravel = apply_func_to_nested_tensors(input, torch.ravel)

    def cat_nested(input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out = None
            for i in input:
                if out is None:
                    out = cat_nested(i)
                else:
                    out = torch.cat([out, cat_nested(i)])
            return out
        elif isinstance(input, Tensor):
            return input

    return cat_nested(partial_ravel)


def nested_convs(
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
):
    return get_module_for_applying_module_to_nested_tensors(
        NormConv2d,
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype,
    )


def log2_channel_distance_to_float_distance(inp: Tensor, scale=2):
    eps = 1e-6
    inp_c = inp.shape[1]
    t = torch.arange(inp_c)
    t = t.to(inp.device)
    t = t[None, ..., None, None]
    inp2 = torch.clone(inp)
    inp2 = (inp2 - torch.min(inp2)) / (torch.max(inp2) - torch.min(inp2)) * 2 - 1
    inp2[inp < 0] = 0
    findex_max = torch.sum(t * inp2, dim=1, keepdim=True) / (
            torch.sum(inp2, dim=1, keepdim=True) + eps
    )
    probability = torch.sum(abs(inp2), dim=1, keepdim=True)
    pixel_disparity = scale ** findex_max
    # for distance, first determine the approximate distance where pixel_disparity=1, let's call it d1_dist
    # then, the distance of any object is now d1_dist/pixel_disparity
    return pixel_disparity, probability


from torch.testing import assert_allclose


def test_log2_channel_distance_to_float_distance():
    inp_tensor = torch.as_tensor(
        [
            [[0, 0], [0, 0.1]],
            [[0, 0], [0, 0.7]],
            [[0.1, 0], [0, 0.2]],
            [[0.5, 0.1], [0, 0.01]],
            [[0.1, 0.6], [0.2, 0]],
            [[0, 0.2], [0.9, 0]],
        ]
    )

    inp_tensor = inp_tensor[None, ...]

    assert inp_tensor.shape == torch.Size([1, 6, 2, 2])

    out_disparity, out_probability = log2_channel_distance_to_float_distance(inp_tensor)

    approximate_index = torch.log2(out_disparity)
    approximate_index_assert = torch.as_tensor([[[3.0000, 4.1111], [4.8182, 1.1188]]])

    assert_allclose(approximate_index, approximate_index_assert)

    pixel_disparity_assert = torch.as_tensor([[[8.0000, 17.2810], [28.2109, 2.1717]]])

    assert_allclose(out_disparity, pixel_disparity_assert)

    out_probability_assert = torch.as_tensor([[[0.7000, 0.9000], [1.1000, 1.0100]]])
    # Note: the probabilities > 1 might be something that should be trained/punished
    assert_allclose(out_probability, out_probability_assert)


"""
import sparsifying_conv


def get_nested_sparsifying_convs_module(
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        xy_sparsity_lr=1e-5,
        c_sparsity_lr=1e-4,
):
    return get_module_for_applying_module_to_nested_tensors(sparsifying_conv.SparsifyingConv2D,
                                                            in_channels,
                                                            out_channels,
                                                            kernel_size,
                                                            stride=stride,
                                                            padding=padding,
                                                            dilation=dilation,
                                                            groups=groups,
                                                            bias=bias,
                                                            padding_mode=padding_mode,
                                                            device=device,
                                                            dtype=dtype,
                                                            xy_sparsity_lr=xy_sparsity_lr,
                                                            c_sparsity_lr=c_sparsity_lr,
                                                            )
"""


def get_nested_conv_transposes_module(
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
):
    return get_module_for_applying_module_to_nested_tensors(
        NormConvTranspose2d,
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding_mode=padding_mode,
        device=device,
        dtype=dtype,
    )


def cat_nested_tensors(t, dim=0):
    def get_nested(nest, idx):
        r = nest
        for i in idx:
            r = r[i]
        return r

    def cat_list(idx: List[int]):
        t_val = get_nested(t[0], idx)
        if isinstance(t_val, list):
            out = []
            for i in range(len(t_val)):
                out.append(cat_list(idx + [i]))
            return out
        elif isinstance(t_val, Tensor):
            try:
                l = len(t)
                catted = torch.cat([get_nested(t, [x] + idx) for x in range(l)], dim)
            except Exception as e:
                print(
                    "Every pyramid in outermost list must have similar nesting and tensor sizes.",
                    file=sys.stderr,
                )
                raise e
            return catted

    return cat_list([])


def test_cat_nested_tensors():
    from edge_pyramid import image_to_edge_pyramid
    from displayarray import breakpoint_display
    from sparsepyramids.tests import smol
    from PIL import Image, ImageOps

    with Image.open(smol) as l:
        l = ImageOps.grayscale(l)
        l = torch.as_tensor((np.array(l)))
        l = l[None, None, ...]
        l = torch.swapaxes(l, 2, 3)
        l = l.float()
        r = torchvision.transforms.functional.rotate(l, 20)
        r2 = torchvision.transforms.functional.rotate(l, 40)

        lpyr = image_to_edge_pyramid(l)
        rpyr = image_to_edge_pyramid(r)
        r2pyr = image_to_edge_pyramid(r2)

        edge = cat_nested_tensors([lpyr, rpyr, r2pyr], 1)
        for i, e in enumerate(edge):
            e = torch.squeeze(e)
            e = torch.swapaxes(e, 0, 2)

            breakpoint_display(e.cpu().numpy())


class SplitImagesFromPyramidBySize2D(torch.nn.Module):
    def __init__(self, split_size=3):
        super().__init__()

        if not isinstance(split_size, (int, Sequence)) and not split_size is None:
            raise TypeError(
                "min_size should be int, sequence, or None. Got {}".format(
                    type(split_size)
                )
            )
        if isinstance(split_size, float):
            split_size = [split_size, split_size]
        if isinstance(split_size, Sequence):
            if len(split_size) not in (1, 2):
                raise ValueError(
                    "If min_size is a sequence, it should have 1 or 2 values"
                )
            elif len(split_size) == 1:
                split_size = list(split_size) * 2
        self.split_size = split_size

    def split_list(self, input: Union[Tensor, List[Tensor]]):
        if isinstance(input, list):
            out_large = []
            out_small = []
            for i in input:
                large, small = self.split_list(i)
                if out_large is not None:
                    out_large.append(large)
                if out_small is not None:
                    out_small.append(small)
            return out_large, out_small
        elif isinstance(input, Tensor):
            size = input.shape[-2:]
            if size[0] < self.min_size[0] or size[1] < self.min_size[1]:
                return None, input
            else:
                return input, None

    def forward(self, t):
        return self.split_list(t)
