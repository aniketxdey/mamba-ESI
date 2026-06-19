"""Pure-PyTorch fallbacks used when CUDA/Triton kernels are unavailable.

These are functional replacements for the kernels in
``mamba_ssm.ops.triton.layernorm`` so the model can be imported and run on
CPU / MPS backends (e.g. macOS) without compiling the original CUDA / Triton
sources. They are slower than the fused kernels but produce numerically
equivalent outputs for inference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """A simple, pure-PyTorch RMSNorm implementation.

    The signature mirrors ``mamba_ssm.ops.triton.layernorm.RMSNorm`` closely
    enough to be drop-in compatible for the parts of the codebase that use it.
    The ``bias`` attribute is exposed (and kept ``None``) because some call
    sites still reference ``norm.bias``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm_fn(x, self.weight, None, eps=self.eps)


def _add_if_needed(x: torch.Tensor, residual: Optional[torch.Tensor]) -> torch.Tensor:
    if residual is None:
        return x
    return (x + residual).to(x.dtype)


def layer_norm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
):
    """Pure-PyTorch substitute for the fused LayerNorm kernel."""
    dtype = x.dtype
    if residual is not None:
        x_with_res = x.to(torch.float32) + residual.to(torch.float32) if residual_in_fp32 else x + residual
        x_for_norm = x_with_res.to(weight.dtype)
        residual_out = x_with_res
    else:
        x_for_norm = x.to(weight.dtype)
        residual_out = x
    out = F.layer_norm(x_for_norm, x_for_norm.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype)
    return out if not prenorm else (out, residual_out)


def rms_norm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
):
    """Pure-PyTorch substitute for the fused RMSNorm kernel."""
    dtype = x.dtype
    if residual is not None:
        x_with_res = x.to(torch.float32) + residual.to(torch.float32) if residual_in_fp32 else x + residual
        x_for_norm = x_with_res
        residual_out = x_with_res
    else:
        x_for_norm = x
        residual_out = x

    x_for_norm = x_for_norm.to(torch.float32)
    rstd = torch.rsqrt(x_for_norm.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_for_norm * rstd
    out = out.to(weight.dtype) * weight
    if bias is not None:
        out = out + bias
    out = out.to(dtype)
    return out if not prenorm else (out, residual_out)
