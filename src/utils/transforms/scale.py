import numpy as np
import matplotlib.pyplot as plt
from src.utils.transforms.apply import grid_resample, transform_3d_point_cloud
import math
from typing import Optional

import torch

from src.utils.helper import identity
from src.utils.transforms.base import Transform  # updated path
from src.utils.transforms.bounded_transform import BoundedTransform  # updated path


class Scale(Transform):
    """Scaling in D dimensions. Log specifies that parameter is in log space. Bounds are still given in original space.
    But sampling becomes non uniform in original space if log is true."""

    def __init__(self, dims: int, log: bool = True):
        self.dims = dims
        self.log = log

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        dim = param.shape[-1]
        batch_size = param.shape[:-1]
        # Create a scaling matrix
        scaling_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        # Fill the scaling matrix with param + 1 to ensure identity for zero input
        scaling_matrix[..., :-1, :-1] = torch.diag_embed(param + 1.0)
        return scaling_matrix

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = self.dims

        if dom.ndim == 0:
            # single value → same interval for each dim
            dom = dom.expand(param_size)
            lower = 1.0 / (1.0 + dom) - 1.0
            upper = dom
        elif dom.ndim == 1:
            # vector → interval per dim
            lower = dom[0].unsqueeze(0).expand(param_size)
            upper = dom[1].unsqueeze(0).expand(param_size)
        elif dom.ndim == 2:
            # each row [low_i, high_i]; expand if needed
            if dom.shape[0] != param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain tensor shape: {dom.shape}")

        # ensure lower ≤ upper
        min_vals = torch.min(lower, upper)
        max_vals = torch.max(lower, upper)
        return min_vals, max_vals

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, lower, upper)
        # choose log or linear per self.log
        if self.log:
            eps = torch.finfo(param.dtype).eps * 10
            s = torch.clamp(param + 1.0, min=eps)
            lm = torch.log(torch.clamp(lower + 1.0, min=eps))
            um = torch.log(torch.clamp(upper + 1.0, min=eps))
            span = torch.clamp(um - lm, min=eps)
            x = torch.log(s) - lm
            period = 2 * span
            mod = torch.remainder(x, period)
            refl = torch.where(mod <= span, mod, period - mod) + lm
            result = torch.exp(refl) - 1.0
            # result = torch.nan_to_num(result, nan=0.0, posinf=upper.item(), neginf=lower.item())
            return torch.clamp(result, lower, upper)
        else:
            # linear reflection
            span = upper - lower
            x = param - lower
            period = 2 * span
            mod = torch.remainder(x, period)
            return torch.where(mod <= span, mod, period - mod) + lower

    def param_size(self) -> int:
        return self.dims

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0) -> None:
        # multi-param transform: no single-parameter orbit
        return None

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        r = torch.rand(batch_size, self.param_size(), device=device, dtype=dtype)
        if self.log:
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + r * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + r * (upper - lower)

    def supports_sobol(self) -> bool:
        return True

    def supports_orbit(self) -> bool:
        return False  # multi-parameter

    def support_calc_bounds(self) -> bool:
        return True

    def sobol_to_param(self, sparam: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Convert Sobol sample-space parameters (in [0,1]) to actual parameters using bounds.
        """
        lower, upper = self.calc_bounds(domain, dtype=sparam.dtype, device=sparam.device)
        if lower.numel() != sparam.shape[-1]:
            raise ValueError(
                f"Domain parameter size ({lower.numel()}) does not match sparam last dim ({sparam.shape[-1]}).")
        if self.log:
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + sparam * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + sparam * (upper - lower)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood for scale parameters: per-parameter span from calc_bounds.
        If domain is None, assume a modest scalar domain of 1.0.
        """
        dom = 1.0 if domain is None else domain
        lower, upper = self.calc_bounds(dom, dtype=dtype, device=device)
        return (upper - lower).to(dtype=dtype, device=device)


class ScaleAllSame(Transform):
    """Uniform scaling in all D dimensions."""

    def __init__(self, dims: int, log: bool = True):
        self.dims = dims
        self.log = log

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        scales = param.expand(param.shape[:-1] + (self.dims,))
        batch_size = scales.shape[:-1]
        matrix = identity(batch_size, self.dims + 1, dtype=param.dtype, device=param.device)
        # Add 1.0 to ensure identity when param is zero
        matrix[..., :-1, :-1] = torch.diag_embed(scales + 1.0)
        return matrix

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = 1
        if dom.ndim == 0:
            dom = dom.expand(param_size)
            lower = 1.0 / (1.0 + dom) - 1.0
            upper = dom
        elif dom.ndim == 1:
            lower = dom[0].unsqueeze(0).expand(param_size)
            upper = dom[1].unsqueeze(0).expand(param_size)
        elif dom.ndim == 2:
            if dom.shape[0] != param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return lower, upper

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        # (shape fix: remove extra unsqueeze that produced (B,1,1))
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, lower, upper)
        if self.log:
            eps = torch.finfo(param.dtype).eps * 10
            s = torch.clamp(param + 1.0, min=eps)
            lm = torch.log(torch.clamp(lower + 1.0, min=eps))
            um = torch.log(torch.clamp(upper + 1.0, min=eps))
            span = torch.clamp(um - lm, min=eps)
            x = torch.log(s) - lm
            period = 2 * span
            mod = torch.remainder(x, period)
            refl = torch.where(mod <= span, mod, period - mod) + lm
            result = torch.exp(refl) - 1.0
            # result = torch.nan_to_num(result, nan=0.0, posinf=upper.item(), neginf=lower.item())
            return torch.clamp(result, lower, upper)
        else:
            span = upper - lower
            x = param - lower
            period = 2 * span
            mod = torch.remainder(x, period)
            return torch.where(mod <= span, mod, period - mod) + lower

    def param_size(self) -> int:
        return 1

    def orbit(self, n_samples: int, domain, dim=0, extend: int = 0, shift: int = 0) -> torch.Tensor:
        # Reuse calc_bounds to properly parse domain
        low_p, high_p = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_p.item(), high_p.item()

        if self.log:
            # shift param‐domain by +1 to get scale factors >0
            s_min, s_max = low_p + 1.0, high_p + 1.0
            log_min, log_max = math.log(s_min), math.log(s_max)
            total = n_samples + 2 * extend
            spacing = (log_max - log_min) / (n_samples - 1) if n_samples > 1 else 0
            start = log_min - extend * spacing
            end = log_max + extend * spacing
            logs = torch.linspace(start, end, total) + shift * spacing
            scales = torch.exp(logs)
            params = scales - 1.0
            return params[..., None].expand(params.shape + (self.param_size(),))
        else:
            # linear sampling over positive param‐domain
            total = n_samples + 2 * extend
            rng = high_p - low_p
            spacing = rng / (n_samples - 1) if n_samples > 1 else 0
            start = low_p - extend * spacing
            end = high_p + extend * spacing
            lin = torch.linspace(start, end, total) + shift * spacing
            return lin[..., None].expand(lin.shape + (self.param_size(),))

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        r = torch.rand(batch_size, 1, device=device, dtype=dtype)
        if self.log:
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + r * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + r * (upper - lower)

    def supports_sobol(self) -> bool:
        return True

    def supports_orbit(self) -> bool:
        return True  # single parameter

    def support_calc_bounds(self) -> bool:
        return True

    def sobol_to_param(self, sparam: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Convert Sobol sample-space parameters (in [0,1]) to actual parameters using bounds.
        """
        lower, upper = self.calc_bounds(domain, dtype=sparam.dtype, device=sparam.device)
        if lower.numel() != sparam.shape[-1]:
            raise ValueError(
                f"Domain parameter size ({lower.numel()}) does not match sparam last dim ({sparam.shape[-1]}).")
        if self.log:
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + sparam * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + sparam * (upper - lower)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood: span from calc_bounds for the single scale parameter.
        If domain is None, assume a modest scalar domain of 1.0.
        """
        dom = 1.0 if domain is None else domain
        lower, upper = self.calc_bounds(dom, dtype=dtype, device=device)
        return (upper - lower).to(dtype=dtype, device=device)


class DirectedScale(Transform):
    """Scaling along a single axis in D dimensions."""

    def __init__(self, dims: int, axis: int, log: bool = True):
        self.dims = dims
        self.axis = axis
        self.log = log

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        res = torch.zeros(*param.shape[:-1], self.dims, dtype=param.dtype, device=param.device)
        res[..., self.axis] = param.squeeze(-1)

        batch_size = res.shape[:-1]
        dim = res.shape[-1]
        # Create a scaling matrix
        scaling_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        # Fill the scaling matrix with param + 1 to ensure identity for zero input
        scaling_matrix[..., :-1, :-1] = torch.diag_embed(res + 1.0)

        return scaling_matrix

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        param_size = 1
        if dom.ndim == 0:
            dom = dom.expand(param_size)
            lower = 1.0 / (1.0 + dom) - 1.0
            upper = dom
        elif dom.ndim == 1:
            lower = dom[0].unsqueeze(0).expand(param_size)
            upper = dom[1].unsqueeze(0).expand(param_size)
        elif dom.ndim == 2:
            if dom.shape[0] != param_size:
                dom = dom.expand(param_size, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return lower, upper

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=param.dtype, device=param.device)
        if not reflect:
            return torch.clamp(param, lower, upper)
        # log‐space or linear
        if self.log:
            eps = torch.finfo(param.dtype).eps * 10
            s = torch.clamp(param + 1.0, min=eps)
            lm = torch.log(torch.clamp(lower + 1.0, min=eps))
            um = torch.log(torch.clamp(upper + 1.0, min=eps))
            span = torch.clamp(um - lm, min=eps)
            x = torch.log(s) - lm
            period = 2 * span
            mod = torch.remainder(x, period)
            refl = torch.where(mod <= span, mod, period - mod) + lm
            result = torch.exp(refl) - 1.0
            # result = torch.nan_to_num(result, nan=0.0, posinf=upper.item(), neginf=lower.item())
            return torch.clamp(result, lower, upper)
        else:
            span = upper - lower
            x = param - lower
            period = 2 * span
            mod = torch.remainder(x, period)
            return (torch.where(mod <= span, mod, period - mod) + lower)

    def param_size(self) -> int:
        return 1

    def orbit(self, n_samples: int, domain, extend: int = 0, shift: int = 0, dim=0) -> torch.Tensor:
        # Reuse calc_bounds to properly parse domain
        low_p, high_p = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_p.item(), high_p.item()

        if self.log:
            s_min, s_max = low_p + 1.0, high_p + 1.0
            log_min, log_max = math.log(s_min), math.log(s_max)
            total = n_samples + 2 * extend
            spacing = (log_max - log_min) / (n_samples - 1) if n_samples > 1 else 0
            start = log_min - extend * spacing
            end = log_max + extend * spacing
            logs = torch.linspace(start, end, total) + shift * spacing
            if total == 1:
                logs = torch.tensor([(start + end) / 2], dtype=torch.float32, device="cpu")
            scales = torch.exp(logs)
            params = scales - 1.0
            return params[..., None]
        else:
            total = n_samples + 2 * extend
            rng = high_p - low_p
            spacing = rng / (n_samples - 1) if n_samples > 1 else 0
            start = low_p - extend * spacing
            end = high_p + extend * spacing
            lin = torch.linspace(start, end, total) + shift * spacing
            return lin[..., None]

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        r = torch.rand(batch_size, 1, device=device, dtype=dtype)
        if self.log:
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + r * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + r * (upper - lower)

    def supports_sobol(self) -> bool:
        return True

    def supports_orbit(self) -> bool:
        return True  # single parameter

    def support_calc_bounds(self) -> bool:
        return True

    @torch.no_grad()
    def sobol_to_param(self, sparam: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Convert Sobol sample-space parameters (in [0,1]) to actual parameters using bounds.
        """
        lower, upper = self.calc_bounds(domain, dtype=sparam.dtype, device=sparam.device)
        if lower.numel() != sparam.shape[-1]:
            raise ValueError(
                f"Domain parameter size ({lower.numel()}) does not match sparam last dim ({sparam.shape[-1]}).")
        if self.log:
            l_log = torch.log(lower + 1.0)
            u_log = torch.log(upper + 1.0)
            logs = l_log + sparam * (u_log - l_log)
            return torch.exp(logs) - 1.0
        return lower + sparam * (upper - lower)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Default neighbourhood: span from calc_bounds for the single directed scale parameter.
        If domain is None, assume a modest scalar domain of 1.0.
        """
        dom = 1.0 if domain is None else domain
        lower, upper = self.calc_bounds(dom, dtype=dtype, device=device)
        return (upper - lower).to(dtype=dtype, device=device)


class Reflection(BoundedTransform):
    """Reflection across a hyperplane in D dimensions."""

    def __init__(self, dims: int, axis: int):
        super().__init__()
        self.dims = dims
        self.axis = axis
        # Small magnitude so that a previously zero parameter is nudged to ±eps.
        self.eps = 1e-3
        self.warned = False
        self.warned_2 = False

    def matrix(self, param: torch.Tensor) -> torch.Tensor:
        batch_size = param.shape[:-1]
        dim = self.dims
        reflection_matrix = identity(batch_size, dim + 1, dtype=param.dtype, device=param.device)
        p = torch.sign(param)
        reflection_matrix[..., self.axis, self.axis] = 1.0 * p.squeeze(-1)
        return reflection_matrix

    def param_size(self) -> int:
        return 1

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample a parameter for the transformation.
        :return: The sampled parameter.
        """
        low, up = self.calc_bounds(domain, dtype=dtype, device=device)
        res = torch.rand(batch_size, self.param_size(), device=device, dtype=dtype) * (up - low) + low
        return self.normalize_parameters(res)

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
            res = torch.exp(logs) - 1.0
            return self.normalize_parameters(res, domain)

        res = lower + sparam * span
        return self.normalize_parameters(res)

    def project_parameters(self, param: torch.Tensor, domain=None, reflect: bool = False) -> torch.Tensor:
        # Force to exactly ±1 by sign; 0 stays 0 here, handled by normalize_parameters to avoid degeneracy.
        return self.normalize_parameters(param)

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        """
        Maps parameter to ±eps so as a hack so that uniform neighborhood sampling approximatly uniformly samples ±1.
        """
        eps = torch.as_tensor(self.eps, dtype=param.dtype, device=param.device)
        return torch.sign(param) * eps

    def orbit(self, n_samples: int, domain=None, extend: int = 0, shift: int = 0, dim=0) -> torch.Tensor:
        # Return the two possible reflections
        vals = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        if n_samples == 1:
            # Just return the identity (no reflection)
            vals = torch.tensor([1.0], dtype=torch.float32)
            if not self.warned:
                print("Warning: Orbit with n_samples=1 returns only the identity (no reflection).")
                self.warned = True
        elif n_samples > 2:
            vals = torch.linspace(-1.0, 1.0, n_samples)
            if not self.warned_2:
                print(
                    f"Warning: Orbit with n_samples={n_samples} returns {n_samples} evenly spaced values between -1 and 1, which may not be meaningful for a reflection transform. The search would ideally not pass this value.")
                self.warned_2 = True
        return vals[:, None]

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        Reflection uses a sign parameter normalized to ±eps. Use 1.0 per default neighbourhood size so that
        sampling can cross zero and flip the sign. Normaliztaion should then map to ±eps. so that the flips are bout
        uniformly sampled.
        """
        return torch.full((self.param_size(),), 1.0, dtype=dtype, device=device)

    def num_discrete_values(self) -> Optional[int]:
        # Reflection is a discrete transform: ±1
        return 2

    def identity_param(self, batch_size: Optional[int] = 1, dtype=torch.float32, device="cpu") -> torch.Tensor:
        # Identity reflection parameter is 1.0 (no reflection)
        return torch.ones(batch_size, self.param_size(), dtype=dtype, device=device)


# Instantiate common transforms_old:
Scale2D = Scale(2)
Scale3D = Scale(3)
UniformScale2D = ScaleAllSame(2)
UniformScale3D = ScaleAllSame(3)
ScaleX2D = DirectedScale(2, 0)
ScaleY2D = DirectedScale(2, 1)
ScaleX3D = DirectedScale(3, 0)
ScaleY3D = DirectedScale(3, 1)
ScaleZ3D = DirectedScale(3, 2)


def compare_and_plot(dims=2, domain=0.5, N=20000, seed=42, log=True):
    import numpy as np
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    transform = Scale(dims, log=log)

    # Sobol-like uniform samples in [0,1]
    sparam = torch.rand(N, dims, dtype=torch.float64)

    # Map via sobol_to_param
    sobol_mapped = transform.sobol_to_param(sparam, domain=domain).cpu().numpy()

    # Direct sampling (unpaired, iid)
    direct = transform.sample_param(N, domain, device="cpu", dtype=torch.float64).cpu().numpy()

    # Quick numeric sanity
    diff_max = np.max(np.abs(sobol_mapped - direct[:sobol_mapped.shape[0]]))
    print(f"log={log}  max absolute diff (unpaired sample arrays): {diff_max:.6g}")

    # Per-dimension histograms (overlay)
    for d in range(dims):
        plt.figure(figsize=(6, 3))
        plt.hist(direct[:, d], bins=100, alpha=0.5, density=True, label='direct', color='C0')
        plt.hist(sobol_mapped[:, d], bins=100, alpha=0.5, density=True, label='sobol_to_param', color='C1')
        plt.xlabel(f'param_dim_{d}')
        plt.ylabel('density')
        plt.title(f'Histogram dim {d}  (log={log})')
        plt.legend()
        plt.tight_layout()

    # 2D joint scatter (both overlaid)
    if dims >= 2:
        plt.figure(figsize=(5, 5))
        plt.scatter(sobol_mapped[:, 0], sobol_mapped[:, 1], s=2, alpha=0.4, label='sobol_to_param', color='C1')
        plt.scatter(direct[:, 0], direct[:, 1], s=2, alpha=0.4, label='direct', color='C0')
        plt.xlabel('dim 0')
        plt.ylabel('dim 1')
        plt.title(f'Joint samples (log={log})')
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    import torch
    from src.utils.transforms.apply import grid_resample, transform_3d_point_cloud

    print("=== Class-based Scale Tests ===")

    # 1. Zero-parameter identity checks
    zero2d = torch.zeros(1, 2)
    id3 = torch.eye(3).unsqueeze(0)
    assert torch.allclose(Scale2D.matrix(zero2d), id3), "Scale2D zero-param ≠ identity"

    zero1d = torch.zeros(1, 1)
    assert torch.allclose(UniformScale2D.matrix(zero1d), id3), "UniformScale2D zero-param ≠ identity"
    assert torch.allclose(ScaleX2D.matrix(zero1d), id3), "ScaleX2D zero-param ≠ identity"
    assert torch.allclose(ScaleY2D.matrix(zero1d), id3), "ScaleY2D zero-param ≠ identity"
    print("✓ Zero-parameter identity tests passed")

    # 2. Gradient check with 2D image
    param_img = torch.randn(1, 2, requires_grad=True)
    mat_img = Scale2D.matrix(param_img)
    x_img = torch.randn(1, 1, 28, 28)
    out_img = grid_resample(x_img, mat_img)
    out_img.sum().backward()
    assert param_img.grad is not None and param_img.grad.abs().sum() > 0, "Scale2D image grad failed"
    print("✓ Scale2D image gradient test passed")

    # 3. Numeric gradcheck for Scale2D
    param_img_d = torch.randn(1, 2, dtype=torch.double, requires_grad=True)
    x_img_d = x_img.to(torch.double)
    fn_img = lambda p: grid_resample(x_img_d, Scale2D.matrix(p))
    assert torch.autograd.gradcheck(fn_img, (param_img_d,), eps=1e-6, atol=1e-4), "Scale2D gradcheck failed"
    print("✓ Scale2D gradcheck passed")

    # 4. Gradient check with 3D point cloud
    param_pc = torch.randn(1, 3, requires_grad=True)
    mat_pc = Scale3D.matrix(param_pc)
    x_pc = torch.randn(1, 1024, 3)
    out_pc = transform_3d_point_cloud(x_pc, mat_pc)
    out_pc.sum().backward()
    assert param_pc.grad is not None and param_pc.grad.abs().sum() > 0, "Scale3D point-cloud grad failed"
    print("✓ Scale3D point-cloud gradient test passed")

    # 5. Numeric gradcheck for Scale3D
    param_pc_d = torch.randn(1, 3, dtype=torch.double, requires_grad=True)
    x_pc_d = x_pc.to(torch.double)
    fn_pc = lambda p: transform_3d_point_cloud(x_pc_d, Scale3D.matrix(p))
    assert torch.autograd.gradcheck(fn_pc, (param_pc_d,), eps=1e-6, atol=1e-4), "Scale3D gradcheck failed"
    print("✓ Scale3D gradcheck passed")

    # 6. Uniform scaling 2D and 3D gradient checks
    p2 = torch.randn(1, 1, requires_grad=True)
    m2 = UniformScale2D.matrix(p2)
    out2 = grid_resample(x_img, m2)
    out2.sum().backward()
    assert p2.grad.abs().sum() > 0, "UniformScale2D grad failed"
    print("✓ UniformScale2D gradient test passed")

    p3 = torch.randn(1, 1, requires_grad=True)
    m3 = UniformScale3D.matrix(p3)
    out3 = transform_3d_point_cloud(x_pc, m3)
    out3.sum().backward()
    assert p3.grad.abs().sum() > 0, "UniformScale3D grad failed"
    print("✓ UniformScale3D gradient test passed")

    print("\nTesting project_parameters reflect behavior:")
    # choose a domain and an offset beyond the domain
    domain = 0.5
    delta = -1
    p = torch.tensor([[domain + delta]])
    # UniformScale2D
    low_u, up_u = UniformScale2D.calc_bounds(domain, dtype=p.dtype, device=p.device)
    print(low_u, up_u)
    refl_u = UniformScale2D.project_parameters(p, domain, reflect=True)
    clip_u = UniformScale2D.project_parameters(p, domain, reflect=False)
    print(f"UniformScale2D: param={p.item()} → reflect={refl_u.item()}, clip={clip_u.item()}")
    # DirectedScale example: ScaleX2D
    low_d, up_d = ScaleX2D.calc_bounds(domain, dtype=p.dtype, device=p.device)
    refl_d = ScaleX2D.project_parameters(p, domain, reflect=True)
    clip_d = ScaleX2D.project_parameters(p, domain, reflect=False)
    print(f"ScaleX2D: param={p.item()} → reflect={refl_d.item()}, clip={clip_d.item()}")
    print("Reflect tests completed.")
