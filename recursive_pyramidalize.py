import torch
import torch.nn.grad
from torch.nn.common_types import _size_2_t
from torch import Tensor
from typing import Union, List, Sequence
import math as m
from torchvision.transforms import functional as FV
import warnings
import numpy as np
import sys


class RecursivePyramidalize2D(torch.nn.Module):

    def __init__(self,
                 scale_val: float = m.sqrt(2),
                 interpolation=FV.InterpolationMode.BILINEAR,
                 min_size=3,
                 max_size=None,
                 antialias=None
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
            raise TypeError("scale_val should be a real number. Got {}".format(type(scale_val)))
        assert scale_val > 1.0, "scale_val needs to be greater than one."
        if not isinstance(min_size, (int, Sequence)) and not min_size is None:
            raise TypeError("min_size should be int, sequence, or None. Got {}".format(type(scale_val)))
        if isinstance(min_size, float):
            min_size = [min_size, min_size]
        if isinstance(min_size, Sequence):
            if len(min_size) not in (1, 2):
                raise ValueError("If min_size is a sequence, it should have 1 or 2 values")
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
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)

    def pyramidalize(self,
                     input: Tensor
                     ):
        in_shape = list(input.shape[-2:])

        out = [input]

        while True:
            in_shape[0] = m.floor(in_shape[0] / self.scale_val)
            in_shape[1] = m.floor(in_shape[1] / self.scale_val)
            if in_shape[0] <= self.min_size[0] or in_shape[1] <= self.min_size[1]:
                break
            out.append(FV.resize(input, in_shape, self.interpolation))

        return out

    def pyramidalize_list(self,
                          input: Union[Tensor, List[Tensor]]
                          ):
        if isinstance(input, list):
            out = []
            for i in input:
                out.append(self.pyramidalize_list(i))
            return out
        elif isinstance(input, Tensor):
            return self.pyramidalize(input)

    def forward(self,
                input: Tensor
                ):
        out = self.pyramidalize_list(input)

        return out


class RecursiveSumDepyramidalize2D(torch.nn.Module):

    def __init__(self,
                 scale_val: float = m.sqrt(2),
                 interpolation=FV.InterpolationMode.BILINEAR,
                 max_size=None,
                 antialias=None
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
            raise TypeError("scale_val should be a real number. Got {}".format(type(scale_val)))
        assert scale_val > 1.0, "scale_val needs to be greater than one."

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
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)

    def depyramidalize(self,
                       input: List[Tensor]
                       ):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        sum = 1
        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img += FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR) * \
                           (np.prod(list(img.shape[-2:])) / np.prod(to_shape))
                sum += (np.prod(list(img.shape[-2:])) / np.prod(to_shape))
        out_img /= sum

        return out_img

    def depyramidalize_list(self,
                            input: Union[Tensor, List[Tensor]]
                            ):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self,
                input: Tensor
                ):
        out = self.depyramidalize_list(input)

        return out


class RecursiveChanDepyramidalize2D(torch.nn.Module):

    def __init__(self,
                 scale_val: float = m.sqrt(2),
                 interpolation=FV.InterpolationMode.BILINEAR,
                 max_size=None,
                 antialias=None
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
            raise TypeError("scale_val should be a real number. Got {}".format(type(scale_val)))
        assert scale_val > 1.0, "scale_val needs to be greater than one."

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
        return self.__class__.__name__ + '(size={0}, interpolation={1}, max_size={2}, antialias={3})'.format(
            self.size, interpolate_str, self.max_size, self.antialias)

    def depyramidalize(self,
                       input: List[Tensor]
                       ):
        to_shape = list(input[0].shape[-2:])

        out_img = None

        for i, img in enumerate(input):
            if i == 0:
                out_img = img
            else:
                out_img = torch.cat([out_img, FV.resize(img, to_shape, FV.InterpolationMode.BILINEAR)], dim=1)

        return out_img

    def depyramidalize_list(self,
                            input: Union[Tensor, List[Tensor]]
                            ):
        if isinstance(input[0], list):
            out = []
            for i in input:
                out.append(self.depyramidalize_list(i))
            return out
        elif isinstance(input[0], Tensor):
            return self.depyramidalize(input)

    def forward(self,
                input: Tensor
                ):
        out = self.depyramidalize_list(input)

        return out


def get_module_for_applying_module_to_nested_tensors(op, *args, **argv):
    class ApplyToNestedTensors(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.op = op(*args, **argv)

        def __repr__(self):
            return self.__class__.__name__ + f'(args={args})'

        def apply_list(self, input: Union[Tensor, List[Tensor]]):
            if isinstance(input, list):
                out = []
                for i in input:
                    out.append(self.apply_list(i))
                return out
            elif isinstance(input, Tensor):
                return op.apply(input)

        def forward(self,
                    input: Tensor
                    ):
            out = self.apply_list(input)

            return out

    return ApplyToNestedTensors


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


def get_nested_convs_module(
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
):
    return get_module_for_applying_module_to_nested_tensors(torch.nn.Conv2d,
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
                                                            dtype=dtype)


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
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
):
    return get_module_for_applying_module_to_nested_tensors(torch.nn.ConvTranspose2d,
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
                                                            dtype=dtype)


def cat_nested_tensors(t, dim=0):
    def cat_list(t_: List[int]):
        if isinstance(t[0, *t_], list):
            out = []
            for i in t:
                out.append(cat_list(i))
            return out
        elif isinstance(t[0, *t_], Tensor):
            try:
                catted = torch.cat(t[slice(None), *t_], dim)
            except Exception as e:
                print("Every pyramid in outermost list must have similar nesting and tensor sizes.", file=sys.stderr)
                raise e
            return catted

    return cat_list([])


class SplitImagesFromPyramidBySize2D(torch.nn.Module):
    def __init__(self,
                 split_size=3):
        super().__init__()

        if not isinstance(split_size, (int, Sequence)) and not split_size is None:
            raise TypeError("min_size should be int, sequence, or None. Got {}".format(type(split_size)))
        if isinstance(split_size, float):
            split_size = [split_size, split_size]
        if isinstance(split_size, Sequence):
            if len(split_size) not in (1, 2):
                raise ValueError("If min_size is a sequence, it should have 1 or 2 values")
            elif len(split_size) == 1:
                split_size = list(split_size) * 2
        self.split_size = split_size

    def split_list(self,
                   input: Union[Tensor, List[Tensor]]
                   ):
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
