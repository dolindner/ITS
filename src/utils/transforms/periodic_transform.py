import math

import torch

from src.utils.transforms.base import Transform


def _generate_orbit_samples(low: torch.Tensor, high: torch.Tensor, n_samples: int,
                            extend: int = 0, shift: int = 0, is_full_circle: bool = False) -> torch.Tensor:
    """
    Helper function to generate orbit samples between low and high bounds.

    Args:
        low: Lower bound
        high: Upper bound
        n_samples: Number of samples to generate
        extend: Number of additional samples to extend on each side
        shift: Amount to shift the samples
        is_full_circle: If True, generate one extra sample and discard the last one to avoid duplicating start/end

    Returns:
        Tensor of shape (n_samples + 2*extend, 1) containing orbit samples
    """
    total_samples = n_samples + 2 * extend

    if is_full_circle:
        # For full circle, generate one extra point and discard the last one
        return _generate_orbit_samples(low, high, n_samples + 1, extend, shift, False)[:-1]
    else:
        # For partial arcs
        spacing = (high - low) / (n_samples - 1) if n_samples > 1 else 0
        start = low - extend * spacing
        end = high + extend * spacing
        orbit = torch.linspace(start, end, total_samples)
        if shift != 0:
            orbit = orbit + shift * (2 * (high - low) / n_samples)
        return orbit


class PeriodicTransform(Transform):
    """Base class for transforms_old with periodic parameters (e.g. angles).
       Subclasses must implement:
         - matrix(param)
         - param_size()
    """

    def supports_sobol(self) -> bool:
        return True

    def supports_orbit(self) -> bool:
        return True

    def interval(self):
        """Default fundamental interval (one period) for all parameters."""
        return -math.pi, math.pi  # can be overridden

    def calc_bounds(self, domain, dtype=torch.float32, device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpret domain similarly to BoundedTransform but allow per-dimension wrap intervals.
        domain:
          - scalar R  -> [-R, R] for each param
          - (a,b)     -> [a,b] for each param
          - tensor (p,2) -> per-parameter [low, high]
        """
        if domain is None:
            domain = self.interval()
        dom = torch.as_tensor(domain, dtype=dtype, device=device)
        p = self.param_size()
        if dom.ndim == 0:
            lower = -torch.ones(p, dtype=dtype, device=device) * dom
            upper = torch.ones(p, dtype=dtype, device=device) * dom
        elif dom.ndim == 1 and dom.numel() == 2:
            lower = torch.full((p,), dom[0].item(), dtype=dtype, device=device)
            upper = torch.full((p,), dom[1].item(), dtype=dtype, device=device)
        elif dom.ndim == 2 and dom.shape[1] == 2:
            if dom.shape[0] < p:
                dom = dom.expand(p, -1)
            lower = dom[:, 0]
            upper = dom[:, 1]
        else:
            raise ValueError(f"Unsupported domain shape: {dom.shape}")
        return lower, upper

    def normalize_parameters(self, param: torch.Tensor) -> torch.Tensor:
        """
        Wrap parameters into the fundamental interval returned by interval().
        """
        default_lower, default_upper = self.interval()
        # ensure scalars
        if torch.is_tensor(default_lower):
            default_lower = default_lower.item() if default_lower.ndim == 0 else default_lower.view(-1)[0].item()
        if torch.is_tensor(default_upper):
            default_upper = default_upper.item() if default_upper.ndim == 0 else default_upper.view(-1)[0].item()
        span_interval = default_upper - default_lower
        return torch.remainder(param - default_lower, span_interval) + default_lower

    def reproject_to_interval(self, param: torch.Tensor) -> torch.Tensor:
        # Delegate to normalize_parameters for backward compatibility
        return self.normalize_parameters(param)

    def project_parameters(self, param: torch.Tensor, domain, reflect: bool = True) -> torch.Tensor:
        dtype = param.dtype
        device = param.device
        lower_bounds, upper_bounds = self.calc_bounds(domain, dtype=dtype, device=device)
        wrapped = self.reproject_to_interval(param)
        default_lower, default_upper = self.interval()

        # Wrap the parameters into the interval
        span_interval = default_upper - default_lower

        if reflect:
            # detect if [ℓ, u] is “inverted” (i.e. ℓ > u means it wraps around the period boundary)
            invert = lower_bounds > upper_bounds

            # compute a “continuous” upper bound u_mod = (u + span_interval) when inverted, else u
            u_mod = torch.where(invert, upper_bounds + span_interval, upper_bounds)

            w_mod = torch.where(
                invert & (wrapped < lower_bounds),
                wrapped + span_interval,
                wrapped
            )

            span_domain = u_mod - lower_bounds
            period_domain = 2 * span_domain

            # reflect w_mod around [ℓ, u_mod]
            y = w_mod - lower_bounds
            m = torch.remainder(y, period_domain)
            folded = torch.where(m <= span_domain, m, period_domain - m) + lower_bounds

            # map anything ≥ β back into [α, β)
            reflected = torch.where(
                folded >= default_upper,
                folded - span_interval,
                folded
            )
            return reflected

        else:
            inside = (wrapped >= lower_bounds) & (wrapped <= upper_bounds)

            # ivert where lower_bounds is larger than upper_bounds
            invert = lower_bounds > upper_bounds
            inside2 = (wrapped >= lower_bounds) | (wrapped <= upper_bounds)
            inside = torch.where(invert, inside2, inside)

            diff_l = torch.abs(wrapped - lower_bounds)
            diff_u = torch.abs(wrapped - upper_bounds)

            dist_l = torch.minimum(diff_l, span_interval - diff_l)
            dist_u = torch.minimum(diff_u, span_interval - diff_u)

            clamp_to_lower = dist_l < dist_u

            proj = torch.where(
                inside,
                wrapped,
                torch.where(clamp_to_lower, lower_bounds, upper_bounds)
            )
            return proj

    def orbit(self, n_samples: int, domain=2 * math.pi, dim: int = 0, extend: int = 0, shift: int = 0) -> torch.Tensor:
        """Generate an orbit of parameters, supporting wrapped domains where lower > upper."""
        low_vec, high_vec = self.calc_bounds(domain, dtype=torch.float32, device="cpu")
        low_p, high_p = low_vec[dim].item(), high_vec[dim].item()
        interval_low, interval_high = self.interval()
        if torch.is_tensor(interval_low):
            interval_low = interval_low.item() if interval_low.ndim == 0 else interval_low.view(-1)[0].item()
        if torch.is_tensor(interval_high):
            interval_high = interval_high.item() if interval_high.ndim == 0 else interval_high.view(-1)[0].item()
        period = interval_high - interval_low
        epsilon = 1e-4

        wrap = low_p > high_p  # wrapped arc crossing boundary
        if wrap:
            high_mod = high_p + period  # make continuous ascending segment
            arc_len = high_mod - low_p
            is_full_circle = abs(arc_len - period) < epsilon
            if is_full_circle:
                samples = _generate_orbit_samples(low_p, low_p + period, n_samples, extend, shift, True)
            else:
                samples = _generate_orbit_samples(low_p, high_mod, n_samples, extend, shift, False)
        else:
            arc_len = high_p - low_p
            is_full_circle = abs(abs(arc_len) - period) < epsilon
            if is_full_circle:
                samples = _generate_orbit_samples(low_p, low_p + period, n_samples, extend, shift, True)
            else:
                samples = _generate_orbit_samples(low_p, high_p, n_samples, extend, shift, False)

        params = torch.zeros((n_samples + 2 * extend, self.param_size()), dtype=torch.float32)
        # Wrap samples back into fundamental interval
        span_interval = period
        wrapped = torch.remainder(samples - interval_low, span_interval) + interval_low
        params[:, dim] = wrapped
        # due to rounding this may have set some samples slightly outside the domain bounds; reproject
        params = self.project_parameters(params, domain, reflect=False)
        return params

    def sample_param(self, batch_size, domain, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """
        Sample a batch of parameters from the periodic transform's domain.
        This method generates random parameters within the specified domain.

        Args:
            batch_size: Number of parameters to sample
            domain: Domain over which to sample the parameters
            device: Device to place the sampled parameters on
            dtype: Data type of the sampled parameters

        Returns:
            A tensor of shape (batch_size, param_size) containing sampled parameters
        """
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        default_lower, default_upper = self.interval()
        span_interval = default_upper - default_lower
        p = self.param_size()

        # clamp arc length into one period
        span_raw = torch.remainder(upper - lower, span_interval)
        span = torch.where(span_raw > 0, span_raw, span_interval)

        u = torch.rand(batch_size, p, dtype=dtype, device=device)
        raw = lower.unsqueeze(0) + u * span.unsqueeze(0)
        return self.normalize_parameters(raw)

    def sobol_to_param(self, sparam: torch.Tensor, domain=None) -> torch.Tensor:
        """
        Convert Sobol samples (in [0,1]) to periodic parameters.
        If domain is given: linear map via calc_bounds, then wrap.
        Else: use interval() (must return (low, high)); fall back to passthrough if interval() is None.
        """
        if sparam.shape[-1] != self.param_size():
            raise ValueError(f"Sobol sample last dim {sparam.shape[-1]} != param_size {self.param_size()}")
        if domain is not None:
            lower, upper = self.calc_bounds(domain, dtype=sparam.dtype, device=sparam.device)
        else:
            interval_vals = self.interval()
            if interval_vals is None:
                return sparam  # no mapping possible; assume already parameterized
            lower, upper = interval_vals
            # ensure tensors
            lower = torch.as_tensor(lower, dtype=sparam.dtype, device=sparam.device).expand(self.param_size())
            upper = torch.as_tensor(upper, dtype=sparam.dtype, device=sparam.device).expand(self.param_size())
        span = upper - lower
        mapped = lower + sparam * span
        return self.normalize_parameters(mapped)

    def default_neighbourhood_size(self, domain=None, dtype=torch.float32, device="cpu") -> torch.Tensor:
        """
        For periodic transforms, default neighbourhood size is the arc length (accounting for wrap).
        Returns a tensor of length param_size() with per-parameter spans (>= eps).
        """
        interval_vals = self.interval()
        low_i, high_i = interval_vals
        if torch.is_tensor(low_i):
            low_i = float(low_i.item())
        if torch.is_tensor(high_i):
            high_i = float(high_i.item())
        period = float(high_i - low_i)
        if domain is None:
            return torch.full((self.param_size(),), period, dtype=dtype, device=device)
        lower, upper = self.calc_bounds(domain, dtype=dtype, device=device)
        lower_t = lower.to(dtype=dtype, device=device)
        upper_t = upper.to(dtype=dtype, device=device)
        wrap_mask = lower_t > upper_t
        span = torch.where(wrap_mask, (upper_t + period) - lower_t, upper_t - lower_t)
        return torch.clamp(span, min=0.0)


if __name__ == '__main__':
    import math
    import torch


    class DummyPeriodic(PeriodicTransform):
        def matrix(self, param: torch.Tensor) -> torch.Tensor:
            # Dummy implementation for testing
            return param

        def param_size(self):
            return 1

        def interval(self):
            # CHANGED: return plain floats to avoid tensor shape issues
            return -1.0, 1.0


    tp = DummyPeriodic()

    # now we have a domain over the interval from 0.9 to -0.9
    values = [-0.125, 0.125, -1.125, 1.05]
    domain = [0.9, -0.9]
    for reflect in (False, True):
        print(f"\nTesting with reflect={reflect}")
        for v in values:
            x = torch.tensor([v], dtype=torch.float32)
            y = tp.project_parameters(x, domain=domain, reflect=reflect)
            print(f" value={v: .2f} → projected = {y.item(): .2f}")

    values = [-0.125, 0.125, -1.125, 1.125]
    domain = [-0.1, 0.1]
    expected = [-0.075, 0.075, 0.075, -0.075]

    for reflect in (False, True):
        print(f"\nTesting with reflect={reflect}")
        for i, v in enumerate(values):
            x = torch.tensor([v], dtype=torch.float32)
            y = tp.project_parameters(x, domain=domain, reflect=reflect)
            if reflect:
                assert torch.allclose(y, torch.ones_like(y) * expected[i],
                                      atol=1e-6), f"Reflection failed: {y} != {expected[i]}"

    values = [-0.125, 0.125, -1.125, 1.125]
    domain = [-2, 2]
    for reflect in (False, True):
        print(f"\nTesting with reflect={reflect}")
        for v in values:
            x = torch.tensor([v], dtype=torch.float32)
            y = tp.project_parameters(x, domain=domain, reflect=reflect)
            print(f" value={v: .2f} → projected={y.item(): .2f}")

    wrap_domain = [0.9, -0.9]
    orb = tp.orbit(n_samples=9, domain=wrap_domain, dim=0)
    inside = (orb[:, 0] >= wrap_domain[0]) | (orb[:, 0] <= wrap_domain[1])
    assert inside.all(), f"Orbit produced out-of-domain samples for wrapped domain: {orb[:, 0]}"
    # Ensure coverage of arc endpoints
    assert (orb[:, 0] >= 0.9).any() and (orb[:, 0] <= -0.9).any(), "Wrapped arc endpoints missing in orbit."

    print("Wrapped domain orbit test passed.")
