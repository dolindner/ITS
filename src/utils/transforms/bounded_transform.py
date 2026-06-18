import math
from typing import Optional

import torch

from src.utils.transforms.base import Transform


class BoundedTransform(Transform):
    """Base class for transforms_old that require parameter bounds checking and projection."""

    def __init__(self, log: bool = False):
        """
        Initialize the BoundedTransform.

        Args:
            log: If True, use logarithmic scaling for parameters; otherwise, use linear scaling.
        """
        super().__init__()
        self.log = log

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate lower and upper bounds for parameters based on the specified domain.
        
        Args:
            domain: Can be a scalar, a tuple (min, max), or a tensor with bound information
            dtype: dtype for the output tensors
            device: device for the output tensors

        Returns:
            A tuple of (lower_bounds, upper_bounds) as tensors
        """
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = self.param_size()

        if dom.ndim == 0:
            lower = -torch.ones(param_size, dtype=dtype, device=device) * dom
            upper = torch.ones(param_size, dtype=dtype, device=device) * dom
        elif dom.ndim == 1 and len(dom) == 2:
            lower = torch.ones(param_size, dtype=dtype, device=device) * dom[0]
            upper = torch.ones(param_size, dtype=dtype, device=device) * dom[1]
        elif dom.ndim == 2:
            if dom.shape[0] < param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain tensor shape: {dom.shape}")

        min_vals = torch.min(lower, upper)
        max_vals = torch.max(lower, upper)
        return min_vals, max_vals

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        :return: The sampled parameter.
        """
        low, up = self.calc_bounds(domain, dtype=dtype, device=device)
        return torch.rand(batch_size, self.param_size(), device=device, dtype=dtype) * (up - low) + low

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        """
        Project parameters to stay within the specified domain.

        Args:
            param: Parameter tensor to project
            domain: Domain specification (scalar, tuple, or tensor)
            reflect: If True, reflect parameters at the boundaries; if False, clamp them

        Returns:
            Projected parameter tensor
        """
        if self.log:
            raise NotImplementedError("Logarithmic projection not implemented yet.")
        dtype = param.dtype
        device = param.device
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)

        if not reflect:
            return torch.clamp(param, min=lower, max=upper)

        span = upper - lower
        x = param - lower
        period = 2 * span
        mod = torch.remainder(x, period)
        return torch.where(mod <= span, mod, period - mod) + lower

    def supports_sobol(self) -> bool:
        """
        Indicates whether the transform supports Sobol sampling.

        Returns:
            True if Sobol sampling is supported, False otherwise.
        """
        return True

    def supports_orbit(self) -> bool:
        return True

    def support_calc_bounds(self) -> bool:
        return True

    # methods that have to be potentially overridden by subclasses
    def sobol_to_param(self, sparam: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Convert Sobol sample-space parameters (in [0,1]) to actual parameters using bounds.
        Args:
            sparam: (..., d) tensor with entries in [0,1]
            domain: domain spec (scalar, (min,max), or tensor Nx2). Required unless already in target space.
        Returns:
            Tensor of same shape as sparam in parameter space.
        """
        lower, upper = self.calc_bounds(domain, dtype=sparam.dtype, device=sparam.device)
        # ensure shape matches
        if lower.numel() != sparam.shape[-1]:
            raise ValueError(
                f"Domain parameter size ({lower.numel()}) does not match sparam last dim ({sparam.shape[-1]}).")
        span = upper - lower
        if self.log:
            # uniform in log-space of (param + 1)
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + sparam * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + sparam * span

    def orbit(self,
              n_samples: int,
              domain,
              dim=0,
              extend: int = 0,
              shift: int = 0) -> Optional[torch.Tensor]:

        # get per-dimension bounds and determine parameter dimension
        low_all, high_all = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        param_dim = low_all.numel()
        low_p, high_p = low_all[dim].item(), high_all[dim].item()

        # total samples including padding
        total = n_samples + 2 * extend

        if self.log:
            # logarithmic spacing in [low_p, high_p]
            s_min, s_max = low_p + 1.0, high_p + 1.0
            log_min, log_max = math.log(s_min), math.log(s_max)
            spacing = (log_max - log_min) / (n_samples - 1) if n_samples > 1 else 0
            start = log_min - extend * spacing
            logs = torch.linspace(start,
                                  log_max + extend * spacing,
                                  total) + shift * spacing
            values = torch.exp(logs) - 1.0
        else:
            # linear spacing in [low_p, high_p]
            rng = high_p - low_p
            spacing = rng / (n_samples - 1) if n_samples > 1 else 0
            start = low_p - extend * spacing
            values = torch.linspace(start,
                                    high_p + extend * spacing,
                                    total) + shift * spacing

        # embed into full parameter vectors
        params = torch.zeros((total, param_dim), dtype=torch.float32)
        params[:, dim] = values

        return params

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        For bounded 2-vector representation: use per-parameter span (upper - lower).
        Fallback to a modest constant if bounds cannot be computed.
        """
        if domain is not None:
            lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
            return torch.clamp((upper - lower).to(dtype=dtype, device=device), min=1e-8)
        else:
            return torch.full((self.param_size(),), 1.0, dtype=dtype, device=device)
