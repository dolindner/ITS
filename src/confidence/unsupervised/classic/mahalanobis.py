from typing import Optional, List, Union
import torch
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from pytorch_ood.detector.mahalanobis import Mahalanobis
from torch import device, Tensor
from confidence.unsupervised.classic.nn_pytorch import DistanceConfidence, _remap_labels_to_indices, _compute_per_class_means_inv_covs, _compute_global_mean_inv_cov
import torch.nn.functional as F

#TODO remove unused cases namely low rank and cov mode.
class PrototypeMahalanobisConfidence(DistanceConfidence):
    """
    Mahalanobis distance using DistanceConfidence that is more general and supports features like
    distance per class. For work only standard mahalanobis distance is used.(Shared Covariance, Full)
    Supports per-class or shared covariance, with optional diagonal or low-rank approximations.

    Args:
        eps: small regularizer added to covariance diagonal.
        shared_covariance: Whether to use shared covariance across classes.
        use_raw_scatter: Use raw scatter (mathces pytorch ood)
        cov_mode: "full" | "diag" | "lowrank"
        low_rank_r: Rank for lowrank mode.
        **kwargs: forwarded to base class.

    Returns:
        Instance configured for Mahalanobis computations.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 shared_covariance: bool = True,
                 use_raw_scatter: bool = False, #matches pytorch OOD behavior
                 cov_mode: str = "full",     # "full" | "diag" | "lowrank"
                 low_rank_r: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.mahalanobis_eps = eps
        self.shared_covariance = shared_covariance
        self.use_raw_scatter = use_raw_scatter
        self.cov_mode = cov_mode
        self.low_rank_r = low_rank_r

    def _fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit class means and covariances.

        Args:
            x: Features tensor (N x D).
            y: Labels tensor (N,).

        Returns:
            self after registration of buffers.
        """
        with torch.no_grad():
            labels_unique, y_idx = _remap_labels_to_indices(y)
            self.register_buffer('y_idx', y_idx)
            self.register_buffer('labels_unique', labels_unique)

            original_dtype = x.dtype
            x_f32 = x.to(torch.float32)
            C = int(y_idx.max().item()) + 1
            one_hot = F.one_hot(y_idx, num_classes=C).to(x_f32.dtype)
            counts = one_hot.sum(dim=0)
            class_sums = one_hot.T @ x_f32
            class_means_f32 = class_sums / counts.unsqueeze(1)
            self.register_buffer('class_means', class_means_f32.to(original_dtype))

            N, D = x_f32.shape

            if self.shared_covariance:
                # Compute residuals from class means
                residuals = x_f32 - class_means_f32[y_idx]
                
                if self.cov_mode == "diag":
                    # Direct diagonal computation: variance per feature
                    denom = N if self.use_raw_scatter else max(1, N - C)
                    var = (residuals ** 2).sum(0) / denom
                    var = var + self.mahalanobis_eps
                    inv_diag = 1.0 / var
                    self.register_buffer('shared_inv_diag', inv_diag.to(original_dtype))

                elif self.cov_mode == "lowrank":
                    # Compute covariance and apply SVD
                    denom = N if self.use_raw_scatter else max(1, N - C)
                    cov = (residuals.T @ residuals) / denom
                    cov = cov + torch.eye(D, device=x.device, dtype=torch.float32) * self.mahalanobis_eps
                    r = min(self.low_rank_r, D)
                    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
                    # For low-rank: inverse is approximated by U @ diag(1/S) @ U.T
                    # Store U and 1/S for efficient computation
                    self.register_buffer("shared_U", U[:, :r].to(original_dtype))
                    self.register_buffer("shared_inv_S", (1.0 / S[:r]).to(original_dtype))

                else:  # full
                    _, inv_cov_sample = _compute_global_mean_inv_cov(residuals, self.mahalanobis_eps)
                    inv_cov_sample = inv_cov_sample.to(torch.float32)
                    if self.use_raw_scatter:
                        inv_cov_sample *= float(max(1, N - C))
                    self.register_buffer('shared_inv_cov', inv_cov_sample.to(original_dtype))

            else:
                # Per-class covariances
                if self.cov_mode == "diag":
                    # Direct diagonal computation per class
                    residuals = x_f32 - class_means_f32[y_idx]
                    var_num = torch.einsum('nd,nk,nd->kd', residuals, one_hot, residuals)
                    denom = counts.unsqueeze(1) if self.use_raw_scatter else (counts).clamp_min(1).unsqueeze(1)
                    var = var_num / denom.to(var_num.dtype)
                    var = var + self.mahalanobis_eps
                    inv_diag = 1.0 / var
                    self.register_buffer('class_inv_diag', inv_diag.to(original_dtype))

                elif self.cov_mode == "lowrank":
                    # Per-class low-rank: compute scatter per class and apply SVD
                    residuals = x_f32 - class_means_f32[y_idx]
                    # Compute per-class scatter matrices
                    cov_num = torch.einsum('nd,nk,nm->kdm', residuals, one_hot, residuals)
                    denom = counts.view(C, 1, 1) if self.use_raw_scatter else (counts).clamp_min(1).view(C, 1, 1)
                    covs = cov_num / denom.to(cov_num.dtype)
                    eye = torch.eye(D, device=x.device, dtype=torch.float32).unsqueeze(0)
                    covs = covs + self.mahalanobis_eps * eye

                    r = min(self.low_rank_r, D)
                    U_list, inv_S_list = [], []
                    for c in range(C):
                        U, S, _ = torch.linalg.svd(covs[c], full_matrices=False)
                        U_list.append(U[:, :r])
                        inv_S_list.append(1.0 / S[:r])

                    self.register_buffer('class_U', torch.stack(U_list).to(original_dtype))
                    self.register_buffer('class_inv_S', torch.stack(inv_S_list).to(original_dtype))

                else:  # full
                    _, invcovs_sample = _compute_per_class_means_inv_covs(x_f32, y_idx, self.mahalanobis_eps)
                    invcovs_sample = invcovs_sample.to(torch.float32)
                    if self.use_raw_scatter:
                        denom = counts.clamp_min(1).view(C, 1, 1).to(invcovs_sample.dtype)
                        invcovs_sample *= denom
                    self.register_buffer('class_inv_covs', invcovs_sample.to(original_dtype))
            return self

    def _compute_distance(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute Mahalanobis distances to class means.

        Args:
            x: Query features (N x D).
            y: Ignored (kept for API compatibility).

        Returns:
            Tensor of per-sample distances (N,).
        """
        if self.shared_covariance:
            diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)
            if self.cov_mode == "diag":
                inv_diag = self.shared_inv_diag
                sq = (diff ** 2 * inv_diag.unsqueeze(0).unsqueeze(0)).sum(-1)
            elif self.cov_mode == "lowrank":
                U, inv_S = self.shared_U, self.shared_inv_S
                # Mahalanobis with low-rank inverse: d^2 = x^T (U @ diag(1/S) @ U.T) x
                proj = torch.matmul(diff, U)  # [N, C, r]
                sq = (proj ** 2 * inv_S.unsqueeze(0).unsqueeze(0)).sum(-1)
            else:
                inv = self.shared_inv_cov
                temp = torch.matmul(diff.unsqueeze(2), inv)
                sq = (temp @ diff.unsqueeze(3)).squeeze(-1).squeeze(-1)
        else:
            diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)
            if self.cov_mode == "diag":
                inv_diag = self.class_inv_diag
                sq = (diff ** 2 * inv_diag.unsqueeze(0)).sum(-1)
            elif self.cov_mode == "lowrank":
                U, inv_S = self.class_U, self.class_inv_S
                # [N, C, D] @ [C, D, r] -> [N, C, r]
                proj = torch.einsum('ncd,cdr->ncr', diff, U)
                sq = (proj ** 2 * inv_S.unsqueeze(0)).sum(-1)
            else:
                inv_covs = self.class_inv_covs
                temp = torch.einsum('ncd,cde->nce', diff, inv_covs)
                sq = (temp * diff).sum(-1)


        dist = torch.sqrt(sq.clamp(min=0)).min(dim=1).values
        return dist


#reference implementation
class MahalanobisConfidence(ClassicConfidenceBase):
    """
    Wraps pytorch_ood.detector.Mahalanobis into a ClassicConfidenceBase.
    Expects X to be features and y to be integer labels.

    Args:
        eps: regularization added to covariance.
        norm_std: optional normalization std list.
        input_transform: optional InputTransform.
        map_function: optional mapping applied to detector score.

    Returns:
        A MahalanobisConfidence instance.
    """

    def __init__(
        self,
        eps: float = 0,
        norm_std: Optional[List[float]] = None,
        input_transform: Optional[InputTransform] = None,
        map_function = None,
    ):
        super().__init__(input_transform=input_transform)
        # model=None since we operate directly on features
        self.detector = Mahalanobis(model=None, eps=eps, norm_std=norm_std)
        self.fitted = False
        self.map_function = map_function or (lambda x: -x)

    def _fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> "MahalanobisConfidence":
        """
        Fit the wrapped Mahalanobis detector.

        Args:
            X: Feature tensor.
            y: Labels tensor (required).

        Returns:
            self after fitting.
        """
        if y is None:
            raise ValueError("Mahalanobis requires labels to fit.")
        X, y = X, y
        self.detector.fit_features(X, y,device=X.device)
        self.fitted = True
        return self

    def _forward(
        self,
        x: torch.Tensor,
        y=None
    ) -> torch.Tensor:
        """
        Predict mapped confidence for given features.

        Args:
            x: Feature tensor to score.
            y: Optional labels (unused).

        Returns:
            Mapped detector scores as confidence tensor.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")
        self.to(x.device)
        score = self.detector.predict_features(x)
        return self.map_function(score)

    def to(self, device: Optional[Union[int, device]] = None) -> "MahalanobisConfidence":
        """
        Move detector buffers to target device.

        Args:
            device: Target device.

        Returns:
            self after move.
        """
        super().to(device)
        self.detector.mu = self.detector.mu.to(device)
        self.detector.cov = self.detector.cov.to(device)
        self.detector.precision = self.detector.precision.to(device)
        return self

    def cuda(self, device: Optional[Union[int, device]] = None) -> "MahalanobisConfidence":
        """
        Move to CUDA device.

        Args:
            device: CUDA device.

        Returns:
            self on CUDA.
        """
        self.to("cuda" if device is None else device)
        return self

    def cpu(self) -> "MahalanobisConfidence":
        """
        Move to CPU.

        Returns:
            self on CPU.
        """
        self.to("cpu")
        return self


if __name__ == "__main__":
    torch.manual_seed(0)

    # toy data
    N, D, C = 200, 2, 3
    z = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    # old mahalanobis (pytorch_ood wrapper) -- keep its cov as raw scatter (do NOT divide/normalize)
    old_mah = MahalanobisConfidence(eps=1e-6)
    old_mah._fit(z, y)

    new_mah_sample = PrototypeMahalanobisConfidence(eps=1e-6, shared_covariance=True, use_raw_scatter=False)
    new_mah_sample._fit(z, y)

    new_mah_raw = PrototypeMahalanobisConfidence(eps=1e-6, shared_covariance=True, use_raw_scatter=True, confidence_function=lambda x: -x)
    new_mah_raw._fit(z, y)

    # prototype with mahalanobis (kept for inspection only; NOT used to compare old)
    from confidence.unsupervised.classic.prototype import ClassPrototypeConfidence
    proto_mah = ClassPrototypeConfidence(metric="mahalanobis", mahalanobis_eps=1e-6, shared_covariance=True)
    proto_mah._fit(z, y)

    print("\n=== Learned parameters comparison ===")
    print("Empirical global mean (z.mean):", z.mean(0))

    # Old / pytorch_ood (raw scatter)
    print("\n-- Old (pytorch_ood.Mahalanobis) --")
    if hasattr(old_mah, "detector") and old_mah.detector is not None:
        det = old_mah.detector
        print("Class means (detector.mu):", det.mu)
        if hasattr(det, "cov") and det.cov is not None:
            print("Detector cov (raw scatter):", det.cov)
        if hasattr(det, "precision") and det.precision is not None:
            print("Detector precision (inverse):", det.precision)

    print("reproduced class means (manual)")
    if hasattr(new_mah_raw, "class_means"):
        print("Class means:", new_mah_raw.class_means)
    if hasattr(new_mah_raw, "shared_inv_cov"):
        print("Shared inverse covariance (raw-scatter inv):", new_mah_raw.shared_inv_cov)
        try:
            print("Shared covariance (inv of shared_inv_cov):", torch.linalg.inv(new_mah_raw.shared_inv_cov))
        except Exception:
            pass

    print("\n-- New (StandardMahalanobisConfidence) sample-cov --")
    print("Class means:", new_mah_sample.class_means)
    if hasattr(new_mah_sample, "shared_inv_cov"):
        print("Shared inverse covariance (sample-cov inv):", new_mah_sample.shared_inv_cov)
        try:
            print("Shared covariance (inv of shared_inv_cov):", torch.linalg.inv(new_mah_sample.shared_inv_cov))
        except Exception:
            pass

    # Prototype (for inspection only)
    print("\n-- Prototype (ClassPrototypeConfidence metric='mahalanobis') --")
    print("Class means:", proto_mah.class_means)
    if hasattr(proto_mah, "shared_inv_cov"):
        print("Prototype shared inverse covariance (shared_inv_cov):", proto_mah.shared_inv_cov)
        try:
            print("Prototype shared covariance (inv of shared_inv_cov):", torch.linalg.inv(proto_mah.shared_inv_cov))
        except Exception:
            pass

    # --- COMPARE SCORES ---
    test_z = torch.randn(10, D)
    raw_old = old_mah._forward(test_z)          # old raw scores (pytorch_ood)
    new_raw_like = new_mah_raw._forward(test_z)  # new raw-like
    new_sample_like = new_mah_sample._forward(test_z)  # new sample-like (for inspection only)

    print("\n=== Score comparison on test data ===")



    print("\nOld Mahalanobis (pytorch_ood) raw scores:", raw_old)
    print("New Mahalanobis (our raw-scatter) raw-like scores:", new_raw_like)
    print("New Mahalanobis (our sample-cov) sample-like scores (for inspection only):", new_sample_like)
    print("Prototype Mahalanobis scores (for inspection only):", proto_mah._forward(test_z, None))

    # --- ASSERT: old raw == new_raw_like (compare raw->raw) ---
    assert torch.allclose(raw_old, new_raw_like, atol=1e-5), "Old (pytorch_ood raw) should match new (raw-scatter) raw-like scores"

    print("Comparison passed: old (pytorch_ood raw) == new (raw-scatter) raw-like.")