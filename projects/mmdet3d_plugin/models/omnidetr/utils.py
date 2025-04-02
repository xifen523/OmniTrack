# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Module utils."""

import copy
import math
import platform
import re
from importlib import metadata
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import uniform_

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
TORCHVISION_VERSION = importlib.metadata.version("torchvision")

__all__ = "multi_scale_deformable_attn_pytorch", "inverse_sigmoid"



def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        print(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0
    
def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string

def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.
        msg (str, optional): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
        ```
    """
    if not current:  # if current is '' or None
        print(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in {">=", ""} and not (c >= v):  # if no constraint passed assume '>=required'
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))  # assert version requirements met
        if verbose:
            print(warning)
    return result


TORCH_1_9 = check_version(torch.__version__, "1.9.0")
TORCH_1_13 = check_version(torch.__version__, "1.13.0")
TORCH_2_0 = check_version(torch.__version__, "2.0.0")
TORCHVISION_0_10 = check_version(TORCHVISION_VERSION, "0.10.0")
TORCHVISION_0_11 = check_version(TORCHVISION_VERSION, "0.11.0")
TORCHVISION_0_13 = check_version(TORCHVISION_VERSION, "0.13.0")
TORCHVISION_0_18 = check_version(TORCHVISION_VERSION, "0.18.0")


def _get_clones(module, n):
    """Create a list of cloned modules from the given module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module."""
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Multiscale deformable attention.

    https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()

def autocast(enabled: bool, device: str = "cuda"):
    """
    Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast. Defaults to 'cuda'.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Note:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.autocast`.

    Example:
        ```python
        with autocast(amp=True):
            # Your mixed precision operations here
            pass
        ```
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)