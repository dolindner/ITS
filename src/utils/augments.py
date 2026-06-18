import torch
import torch.nn.functional as F
import kornia

from src.utils.transforms.apply import grid_resample
import torch
import torch.nn.functional as F
import kornia as K
from typing import Tuple, Callable, List, Optional


def small_affine_augment_2d(images, max_shift=1.0, max_angle=3.0, max_scale=0.04,resample_function=grid_resample):
    """
    Small augments to simulate some artficats from bilinear interpolation and small misalignments
    Args:
        images: Batched image tensor.
        max_shift: Max shift to apply
        max_angle: Max angle for angle jitter
        max_scale: Max scale for scale jitter

    Returns:
        augmented image tensor

    """
    B, C, H, W = images.shape
    device = images.device

    # random translations in pixels
    translations = torch.empty(B, 2, device=device).uniform_(-max_shift, max_shift)

    # random angles in radians
    angles = torch.empty(B, device=device).uniform_(-max_angle, max_angle)

    # random scales
    scales = 1.0 + torch.empty(B, device=device).uniform_(-max_scale, max_scale)
    scales = scales.unsqueeze(1).repeat(1, 2)

    # center of image
    center = torch.tensor([[W / 2, H / 2]], device=device).expand(B, -1)

    # get 3x3 affine matrices in pixel coordinates
    M_3x3 = kornia.geometry.transform.get_affine_matrix2d(translations, center, scales, angles)

    # normalize to PyTorch [-1,1] coordinates
    norm_mat = torch.tensor([
        [2.0/(W-1), 0, -1],
        [0, 2.0/(H-1), -1],
        [0, 0, 1]
    ], device=device).unsqueeze(0).repeat(B,1,1)

    norm_mat_inv = torch.tensor([
        [(W-1)/2.0, 0, (W-1)/2.0],
        [0, (H-1)/2.0, (H-1)/2.0],
        [0, 0, 1]
    ], device=device).unsqueeze(0).repeat(B,1,1)

    # properly transform 3x3 matrices
    M_norm_3x3 = torch.bmm(norm_mat, torch.bmm(M_3x3, norm_mat_inv))

    res = resample_function(images, M_norm_3x3)
    return res


Tensor = torch.Tensor

def _rand_params(shape, device, low, high):
    return torch.empty(shape, device=device).uniform_(low, high)

@torch.no_grad()
def random_gaussian_blur(x: Tensor,
                         p: float = 0.5,
                         ksize_choices: Tuple[int, ...] = (3, 5),
                         sigma_range: Tuple[float, float] = (0.1, 1.5)) -> Tensor:
    """
    Vectorized per-sample Gaussian blur.
    """
    if p <= 0:
        return x
    B, C, H, W = x.shape
    device = x.device
    apply_mask = torch.rand(B, device=device) < p
    if not apply_mask.any():
        return x
    # Choose kernel size per sample
    k_choices = torch.tensor(ksize_choices, device=device)
    k_idx = torch.randint(0, len(k_choices), (B,), device=device)
    k_sizes = k_choices[k_idx]

    # Sample sigmas per sample (Kornia expects (B,2))
    sigmas = _rand_params((B, 2), device, *sigma_range)
    # Build blurred batch
    out = x.clone()
    # Group by kernel size to avoid looping over samples
    for k in k_choices.unique():
        mask_k = (k_sizes == k) & apply_mask
        if not mask_k.any():
            continue
        x_sub = x[mask_k]
        sigma_sub = sigmas[mask_k]
        blur = K.filters.GaussianBlur2d((int(k.item()), int(k.item())), sigma_sub)(x_sub)
        out[mask_k] = blur
    return out

@torch.no_grad()
def random_unsharp_mask(x: Tensor,
                        p: float = 0.5,
                        ksize: int = 5,
                        sigma_range: Tuple[float, float] = (0.5, 1.5),
                        amount_range: Tuple[float, float] = (0.5, 1.5),
                        clamp: bool = True) -> Tensor:
    """
    Sharpen / deblur via unsharp masking: y = x + a * (x - blur(x)).
    """
    if p <= 0:
        return x
    B = x.shape[0]
    device = x.device
    apply_mask = torch.rand(B, device=device) < p
    if not apply_mask.any():
        return x
    sigmas = _rand_params((B, 2), device, *sigma_range)
    amounts = _rand_params((B, 1, 1, 1), device, *amount_range)
    blur_all = K.filters.GaussianBlur2d((ksize, ksize), sigmas)(x)
    sharpened = x + amounts * (x - blur_all)
    if clamp:
        sharpened = sharpened.clamp(0.0, 1.0)
    return torch.where(apply_mask.view(B, 1, 1, 1), sharpened, x)

@torch.no_grad()
def random_gaussian_noise(x: Tensor,
                          p: float = 0.5,
                          sigma_range: Tuple[float, float] = (0.001, 0.05),
                          clamp: bool = True) -> Tensor:
    if p <= 0:
        return x
    B = x.shape[0]
    device = x.device
    mask = torch.rand(B, device=device) < p
    if not mask.any():
        return x
    sigmas = _rand_params((B, 1, 1, 1), device, *sigma_range)
    noise = torch.randn_like(x) * sigmas
    out = x + noise
    if clamp:
        out = out.clamp(0.0, 1.0)
    return torch.where(mask.view(B,1,1,1), out, x)

@torch.no_grad()
def random_contrast(x: Tensor,
                    p: float = 0.5,
                    factor_range: Tuple[float, float] = (0.75, 1.25),
                    eps: float = 1e-8) -> Tensor:
    if p <= 0:
        return x
    B = x.shape[0]
    device = x.device
    mask = torch.rand(B, device=device) < p
    if not mask.any():
        return x
    factors = _rand_params((B,1,1,1), device, *factor_range)
    mean = x.mean(dim=(2,3), keepdim=True)
    out = (x - mean) * factors + mean
    out = torch.where(mask.view(B,1,1,1), out, x)
    return out.clamp(0.0, 1.0)

@torch.no_grad()
def random_gamma(x: Tensor,
                 p: float = 0.5,
                 gamma_range: Tuple[float, float] = (0.7, 1.5),
                 eps: float = 1e-8) -> Tensor:
    if p <= 0:
        return x
    B = x.shape[0]
    device = x.device
    mask = torch.rand(B, device=device) < p
    if not mask.any():
        return x
    gammas = _rand_params((B,1,1,1), device, *gamma_range)
    out = (x.clamp(min=eps)) ** gammas
    return torch.where(mask.view(B,1,1,1), out, x)

@torch.no_grad()
def random_cutout(x: Tensor,
                  p: float = 0.5,
                  scale_range: Tuple[float, float] = (0.1, 0.25),
                  fill: float = 0.0) -> Tensor:
    """
    Rectangular mask with random size per sample.
    """
    if p <= 0:
        return x
    B, C, H, W = x.shape
    device = x.device
    mask_apply = torch.rand(B, device=device) < p
    if not mask_apply.any():
        return x
    out = x.clone()
    for_mask = torch.nonzero(mask_apply).flatten()
    # Vectorizable approach: build per-pixel mask
    yy = torch.arange(H, device=device).view(1, H, 1)
    xx = torch.arange(W, device=device).view(1, 1, W)
    for b in for_mask:
        scale = _rand_params((1,), device, *scale_range).item()
        cut_h = max(1, int(H * scale))
        cut_w = max(1, int(W * scale))
        cy = torch.randint(0, H, (1,), device=device).item()
        cx = torch.randint(0, W, (1,), device=device).item()
        y1 = max(0, cy - cut_h // 2); y2 = min(H, y1 + cut_h)
        x1 = max(0, cx - cut_w // 2); x2 = min(W, x1 + cut_w)
        out[b,:,y1:y2,x1:x2] = fill
    return out

@torch.no_grad()
def random_blur_or_sharpen(
    x: torch.Tensor,
    p: float = 0.4,
    prob_blur: float = 0.5,           # fraction of affected samples that get blur; rest sharpen
    blur_ks_choices=(3,5),
    blur_sigma_range=(0.2,1.2),
    usm_ksize=5,
    usm_sigma_range=(0.5,1.5),
    usm_amount_range=(0.5,1.3),
    clamp: bool = True
) -> torch.Tensor:
    """
    Per sample choose: no-op, Gaussian blur, or unsharp mask (sharpen).
    """
    if p <= 0:
        return x
    B, C, H, W = x.shape
    device = x.device

    apply_mask = torch.rand(B, device=device) < p
    if not apply_mask.any():
        return x

    # Decide which of the applying samples get blur vs sharpen
    apply_indices = torch.nonzero(apply_mask).flatten()
    n_apply = apply_indices.numel()
    n_blur = int(round(n_apply * prob_blur))
    # Shuffle
    perm = apply_indices[torch.randperm(n_apply, device=device)]
    blur_indices = perm[:n_blur]
    sharp_indices = perm[n_blur:]

    out = x.clone()

    # --- Blur branch ---
    if blur_indices.numel() > 0:
        k_choices = torch.tensor(blur_ks_choices, device=device)
        # kernel per sample
        k_idx = torch.randint(0, len(k_choices), (blur_indices.numel(),), device=device)
        k_sizes = k_choices[k_idx]
        sigmas = torch.empty(blur_indices.numel(), 2, device=device).uniform_(*blur_sigma_range)
        for ks in k_choices.unique():
            sel = (k_sizes == ks)
            if not sel.any():
                continue
            idx_sel = blur_indices[sel]
            x_sub = x[idx_sel]
            sigma_sub = sigmas[sel]
            blur_op = K.filters.GaussianBlur2d((int(ks.item()), int(ks.item())), sigma_sub)
            out[idx_sel] = blur_op(x_sub).to(dtype=x.dtype)

    # --- Sharpen (unsharp mask) branch ---
    if sharp_indices.numel() > 0:
        sigmas = torch.empty(sharp_indices.numel(), 2, device=device).uniform_(*usm_sigma_range)
        amounts = torch.empty(sharp_indices.numel(), 1, 1, 1, device=device).uniform_(*usm_amount_range)
        blur_op = K.filters.GaussianBlur2d((usm_ksize, usm_ksize), sigmas)
        base = x[sharp_indices]
        blurred = blur_op(base).to(dtype=x.dtype)
        sharpened = base + amounts * (base - blurred)
        if clamp:
            sharpened = sharpened.clamp(0,1)
        out[sharp_indices] = sharpened

    return out

class ComposeAugmentations:
    def __init__(self, ops: List[Callable[[Tensor], Tensor]]):
        self.ops = ops
    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        for op in self.ops:
            x = op(x)
        return x

    def as_single_sample(self):
        """
        Returns a callable that accepts (C,H,W) or (B,C,H,W).
        """
        parent = self
        @torch.no_grad()
        def _single(x: Tensor) -> Tensor:
            if x.dim() == 3:
                return parent(x.unsqueeze(0)).squeeze(0)
            return parent(x)
        return _single

def build_default_augmentations() -> ComposeAugmentations:
    return ComposeAugmentations([
        lambda x: random_blur_or_sharpen(x, p=0.8, prob_blur=0.5,
                                         blur_ks_choices=(3, 5), blur_sigma_range=(0.2, 1.8),
                                         usm_ksize=5, usm_sigma_range=(0.5, 1.5),
                                         usm_amount_range=(0.5, 1.3), clamp=True),
        lambda x: random_gaussian_noise(x, p=0.3),
        lambda x: random_contrast(x, p=0.3),
        lambda x: random_gamma(x, p=0.3),
    ])



#Single-sample wrappers for affine augments
@torch.no_grad()
def small_affine_augment_2d_single(x: torch.Tensor, **kwargs):
    return small_affine_augment_2d(x.unsqueeze(0), **kwargs).squeeze(0)
