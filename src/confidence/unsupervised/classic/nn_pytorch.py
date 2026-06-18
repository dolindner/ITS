from abc import abstractmethod
from typing import Optional, Callable, Tuple, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F

from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase


def default_confidence_function(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + x)


# Add distance helper functions
def _compute_cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance = 1 - cosine_similarity
    Output is a matrix of shape (N, M) where N is the number of samples in x and M is the number of samples in y.

    Args:
        x: Tensor of shape (N, D).
        y: Tensor of shape (M, D).

    Returns:
        Cosine distance matrix of shape (N, M).
    """
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    sim = x_norm @ y_norm.t()
    return 1.0 - sim


def mixed_distance_fast(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 0.5,  # weight for Euclidean part
        eps: float = 1e-8,
        squared: bool = False,  # if True return squared-Euclidean (avoids sqrt)
        normalize_euclid: bool = True,
) -> torch.Tensor:
    """
    Mixed distance between unnormalized x (N,d) and y (M,d).
    Reuses cross = x @ y.T for both cosine and euclidean.
    Returns (N, M) matrix: alpha * euclid + (1-alpha) * cos_dist

    Args:
        x: (N, D) tensor.
        y: (M, D) tensor.
        alpha: weight for euclidean component.
        eps: numerical epsilon.
        squared: whether to return squared Euclidean.
        normalize_euclid: whether to normalize Euclidean by sqrt(D).

    Returns:
        Mixed distance matrix (N, M).
    """
    # Precompute norms and cross-product
    x_norm_sq = (x * x).sum(dim=1)  # (N,)
    y_norm_sq = (y * y).sum(dim=1)  # (M,)
    cross = x @ y.t()  # (N, M)

    # Cosine distance: 1 - (x·y) / (||x|| * ||y||)
    x_norm = torch.sqrt(x_norm_sq.clamp(min=eps))  # (N,)
    y_norm = torch.sqrt(y_norm_sq.clamp(min=eps))  # (M,)
    denom = x_norm[:, None] * y_norm[None, :]  # (N, M)
    cos_sim = cross / (denom + eps)
    # numeric safety: clamp cosine to [-1,1] if you rely on it
    cos_sim = cos_sim.clamp(-1.0, 1.0)
    cos_dist = 1.0 - cos_sim

    # Euclidean distance (or squared)
    euclid_sq = x_norm_sq[:, None] + y_norm_sq[None, :] - 2.0 * cross
    euclid_sq = euclid_sq.clamp(min=0.0)  # avoid negatives from precision
    if squared:
        euclid = euclid_sq
    else:
        euclid = torch.sqrt(euclid_sq + eps)

    if normalize_euclid:
        # Normalize by sqrt(dimension) instead of batch statistics
        d = x.shape[1]
        scale = torch.sqrt(torch.tensor(d, device=x.device, dtype=x.dtype))
        euclid = euclid / (scale + eps)

    return alpha * euclid + (1.0 - alpha) * cos_dist


def _compute_mahalanobis_distance(x: torch.Tensor, y: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    """
    Compute Mahalanobis distance between points in x and y using inverse covariance matrix

    Args:
        x: (N, D)
        y: (M, D)
        inv_cov: (D, D) inverse covariance or (M, D, D) depending on usage.

    Returns:
        Mahalanobis distances (N, M).
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # [N, M, D]
    temp = torch.matmul(diff.unsqueeze(2), inv_cov.unsqueeze(0).unsqueeze(0))
    sq = torch.sum(temp * diff.unsqueeze(2), dim=-1).squeeze(2).clamp(min=0)
    return torch.sqrt(sq)


def _compute_distance_metric(x: torch.Tensor, y: torch.Tensor, metric, inv_cov=None, **kwargs) -> torch.Tensor:
    """
    Unified distance computation function that handles different metrics

    Args:
        x: (N, D) queries.
        y: (M, D) database.
        metric: "euclidean", "cosine", "mixed", "mahalanobis", or callable.
        inv_cov: optional inverse covariance for Mahalanobis.
        **kwargs: metric-specific parameters.

    Returns:
        Distance matrix (N, M).
    """
    if callable(metric):
        return metric(x, y)
    elif metric == "euclidean":
        return torch.cdist(x, y, p=2)
    elif metric == "cosine":
        return _compute_cosine_distance(x, y)
    elif metric == "mixed":
        return mixed_distance_fast(x, y, **kwargs)
    elif metric == "mahalanobis" and inv_cov is not None:
        return _compute_mahalanobis_distance(x, y, inv_cov)
    else:
        raise ValueError(f"Unknown metric: {metric} or missing inverse covariance matrix")


# Add top-k recomputation helper
def _topk_recompute_euclidean(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute top-k Euclidean distances with recomputation for efficient backward.
    Forward: compute all distances to find top-k indices (no grad)
    Backward: recompute only top-k distances with grad enabled

    Args:
        x: (N, D)
        y: (M, D)
        k: top-k

    Returns:
        Top-k distances per query (N, k).
    """
    with torch.no_grad():
        D_all = torch.cdist(x, y, p=2)
        vals, idx = D_all.topk(k, dim=1, largest=False)

    # Recompute only top-k distances for backward
    y_topk = y[idx]  # [N_query, k, D]
    x_expand = x[:, None, :]  # [N_query, 1, D]
    D_topk = (x_expand - y_topk).pow(2).sum(dim=2).sqrt()

    return D_topk


def _topk_recompute_cosine(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    """
    Compute top-k cosine distances with recomputation for efficient backward.

    Args:
        x: (N, D)
        y: (M, D)
        k: top-k

    Returns:
        Top-k cosine distances (N, k).
    """
    with torch.no_grad():
        D_all = _compute_cosine_distance(x, y)
        vals, idx = D_all.topk(k, dim=1, largest=False)

    # Recompute only top-k cosine distances for backward
    y_topk = y[idx]  # [N_query, k, D]
    x_expand = x[:, None, :]  # [N_query, 1, D]

    x_norm = F.normalize(x_expand, p=2, dim=2)
    y_norm = F.normalize(y_topk, p=2, dim=2)
    sim = (x_norm * y_norm).sum(dim=2)
    D_topk = 1.0 - sim

    return D_topk


def _topk_recompute_mixed(x: torch.Tensor, y: torch.Tensor, k: int, alpha: float = 0.5,
                          eps: float = 1e-8, squared: bool = False,
                          normalize_euclid: bool = True) -> torch.Tensor:
    """
    Compute top-k mixed distances with recomputation for efficient backward.

    Args:
        x, y, k: see mixed_distance_fast.
        alpha, eps, squared, normalize_euclid: same semantics as mixed_distance_fast.

    Returns:
        Top-k mixed distances (N, k).
    """
    with torch.no_grad():
        # Find top-k using full distance computation
        D_all = mixed_distance_fast(x, y, alpha=alpha, eps=eps, squared=squared,
                                    normalize_euclid=normalize_euclid)
        vals, idx = D_all.topk(k, dim=1, largest=False)

    # Recompute only top-k mixed distances for backward
    # Must use the SAME computation as mixed_distance_fast to get correct gradients
    y_topk = y[idx]  # [N_query, k, D]

    # Recompute exactly as in mixed_distance_fast but only for top-k
    x_norm_sq = (x * x).sum(dim=1)  # (N,)
    y_topk_norm_sq = (y_topk * y_topk).sum(dim=2)  # (N, k)

    # Cross product for top-k
    cross = torch.bmm(x.unsqueeze(1), y_topk.transpose(1, 2)).squeeze(1)  # (N, k)

    # Cosine distance
    x_norm = torch.sqrt(x_norm_sq.clamp(min=eps))  # (N,)
    y_topk_norm = torch.sqrt(y_topk_norm_sq.clamp(min=eps))  # (N, k)
    denom = x_norm[:, None] * y_topk_norm  # (N, k)
    cos_sim = (cross / (denom + eps)).clamp(-1.0, 1.0)
    cos_dist = 1.0 - cos_sim

    # Euclidean distance
    euclid_sq = x_norm_sq[:, None] + y_topk_norm_sq - 2.0 * cross
    euclid_sq = euclid_sq.clamp(min=0.0)
    if squared:
        euclid = euclid_sq
    else:
        euclid = torch.sqrt(euclid_sq + eps)

    if normalize_euclid:
        # Normalize by sqrt(dimension) instead of batch statistics
        d = x.shape[1]
        scale = torch.sqrt(torch.tensor(d, device=x.device, dtype=x.dtype))
        euclid = euclid / (scale + eps)

    D_topk = alpha * euclid + (1.0 - alpha) * cos_dist

    return D_topk


def _topk_recompute_mahalanobis(x: torch.Tensor, y: torch.Tensor, k: int,
                                inv_cov: torch.Tensor) -> torch.Tensor:
    """
    Compute top-k Mahalanobis distances with recomputation for efficient backward.

    Args:
        x: (N, D)
        y: (M, D)
        k: top-k
        inv_cov: inverse covariance

    Returns:
        Top-k Mahalanobis distances (N, k).
    """
    with torch.no_grad():
        D_all = _compute_mahalanobis_distance(x, y, inv_cov)
        vals, idx = D_all.topk(k, dim=1, largest=False)

    # Recompute only top-k Mahalanobis distances for backward
    y_topk = y[idx]  # [N_query, k, D]
    diff = x[:, None, :] - y_topk  # [N_query, k, D]
    temp = torch.matmul(diff.unsqueeze(2), inv_cov.unsqueeze(0).unsqueeze(0))
    sq = torch.sum(temp * diff.unsqueeze(2), dim=-1).squeeze(2).clamp(min=0)
    D_topk = torch.sqrt(sq)

    return D_topk


class DistanceConfidence(ClassicConfidenceBase):
    """Abstract base for distance-based confidence modules."""

    def __init__(
            self,
            confidence_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            input_transform: Optional[InputTransform] = None,
            dtype: torch.dtype = None
    ):
        super().__init__(input_transform=input_transform)
        self.confidence_function = confidence_function or default_confidence_function

    @property
    def dtype(self) -> torch.dtype:
        """The data type used for storing training data and for computation."""
        return torch.float32

    def _forward(self, x: torch.Tensor, y=None) -> torch.Tensor:
        """
        Compute confidence from distances for a batch.

        Args:
            x: Query features.
            y: Optional labels (passed to distance computation).

        Returns:
            Confidence tensor in the original input dtype.
        """
        original_dtype = x.dtype
        x_compute = x.to(self.dtype)
        dist = self._compute_distance(x_compute, y)
        conf = self.confidence_function(dist)
        return conf.to(original_dtype)

    @abstractmethod
    def _compute_distance(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute distance metric for queries.

        Args:
            x: Query features (converted to self.dtype).
            y: Optional labels or extra info.

        Returns:
            Per-sample distance tensor.
        """
        pass


# uses biased estimator by diving by n like in paper https://proceedings.neurips.cc/paper_files/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf
def _compute_global_mean_inv_cov(x: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    mu = x_f32.mean(0)
    diff = x_f32 - mu
    # Use population denominator N (per provided GDA formula) instead of N-1
    cov = diff.T @ diff / float(x_f32.size(0))
    cov = cov + torch.eye(x_f32.size(1), dtype=torch.float32, device=x.device) * eps
    inv_cov = torch.linalg.pinv(cov)
    return mu.to(original_dtype), inv_cov.to(original_dtype)


def _compute_per_class_means_inv_covs(x: torch.Tensor, y_idx: torch.Tensor, eps: float):
    """
    Compute class means and per-class inverse covariances.

    Args:
        x: (N, D) features.
        y_idx: (N,) integer class indices.
        eps: diagonal regularizer.

    Returns:
        (class_means, invcovs) both in original dtype.
    """
    original_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    C = int(y_idx.max()) + 1  # removed .item()
    one_hot = F.one_hot(y_idx, num_classes=C).to(x_f32.dtype)
    counts = one_hot.sum(dim=0)
    class_sums = one_hot.T @ x_f32
    class_means = class_sums / counts.unsqueeze(1)
    diff = x_f32 - class_means[y_idx]
    cov_num = torch.einsum('nd,nk,nm->kdm', diff, one_hot, diff)
    denom = counts.view(C, 1, 1)

    # avoid division by zero for empty classes
    denom_safe = denom.clone()
    denom_safe[denom_safe <= 0] = 1.0
    covs = cov_num / denom_safe
    eye = torch.eye(x_f32.shape[1], device=x.device).unsqueeze(0).repeat(C, 1, 1)
    covs = covs + eps * eye
    # replace covs for zero-count classes by eps*I explicitly
    low_count_mask = (counts <= 0)
    if low_count_mask.any():
        covs[low_count_mask, :, :] = eps * eye[0]
    invcovs = torch.linalg.pinv(covs)
    return class_means.to(original_dtype), invcovs.to(original_dtype)


def _remap_labels_to_indices(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels_unique = torch.unique(y, sorted=True)
    y_idx = torch.searchsorted(labels_unique, y)
    return labels_unique, y_idx


def _mahalanobis_distance_per_class_vectorized(
        x_query, class_means, class_inv_covs, y_idx_query
) -> torch.Tensor:
    mu = class_means[y_idx_query]
    invcov = class_inv_covs[y_idx_query]
    diff = x_query - mu
    left = (diff.unsqueeze(1) @ invcov)  # N×1×D
    sq = (left @ diff.unsqueeze(2)).squeeze()
    return torch.sqrt(torch.clamp(sq, min=0.0))


def _knn_distance_per_class_vectorized(
        x_query, x_train, y_idx_query, y_idx_train, k: int, metric="euclidean", inv_cov=None, **kwargs
) -> torch.Tensor:
    dists = _compute_distance_metric(x_query, x_train, metric, inv_cov, **kwargs)
    mask = y_idx_query.unsqueeze(1) == y_idx_train.unsqueeze(0)
    INF = torch.finfo(dists.dtype).max
    # avoid inplace modification which can break autograd
    dists = dists.masked_fill(~mask, INF)
    vals = torch.topk(dists, k=k, dim=1, largest=False).values
    return vals.mean(1)


def _mahalanobis_knn_per_class(
        x_query, x_train, y_idx_q, y_idx_t, class_inv_covs, k: int
) -> torch.Tensor:
    inv = class_inv_covs[y_idx_q]
    diff = x_query.unsqueeze(1) - x_train.unsqueeze(0)
    temp = (diff.unsqueeze(2) @ inv.unsqueeze(1))  # N×T×1×D
    sq = (temp * diff.unsqueeze(2)).sum(-1).clamp(min=0)
    d = torch.sqrt(sq)
    mask = y_idx_q.unsqueeze(1) == y_idx_t.unsqueeze(0)
    INF = torch.finfo(d.dtype).max
    # avoid inplace modification which can break autograd
    d = d.masked_fill(~mask, INF)
    vals = torch.topk(d, k=k, dim=1, largest=False).values
    return vals.mean(1)


def _knn_distance_per_class_vectorized_topk_recompute(
        x_query, x_train, y_idx_query, y_idx_train, k: int, metric="euclidean", inv_cov=None, **kwargs
) -> torch.Tensor:
    """
    Vectorized per-class KNN with top-k recomputation optimization.
    """
    # First pass: find top-k indices without gradients
    with torch.no_grad():
        dists_nograd = _compute_distance_metric(x_query, x_train, metric, inv_cov, **kwargs)
        mask = y_idx_query.unsqueeze(1) == y_idx_train.unsqueeze(0)
        INF = torch.finfo(dists_nograd.dtype).max
        dists_nograd = dists_nograd.masked_fill(~mask, INF)
        topk_result = torch.topk(dists_nograd, k=k, dim=1, largest=False)
        idx = topk_result.indices  # [N_query, k]

    # Second pass: recompute distances only for top-k with gradients
    x_train_topk = x_train[idx]  # [N_query, k, D]

    if metric == "euclidean":
        x_expand = x_query[:, None, :]
        D_topk = (x_expand - x_train_topk).pow(2).sum(dim=2).sqrt()
    elif metric == "cosine":
        x_expand = x_query[:, None, :]
        x_norm = F.normalize(x_expand, p=2, dim=2)
        y_norm = F.normalize(x_train_topk, p=2, dim=2)
        sim = (x_norm * y_norm).sum(dim=2)
        D_topk = 1.0 - sim
    elif metric == "mixed":
        # Recompute mixed distance exactly as in mixed_distance_fast
        alpha = kwargs.get("alpha", 0.5)
        eps = kwargs.get("eps", 1e-8)
        squared = kwargs.get("squared", False)
        normalize_euclid = kwargs.get("normalize_euclid", True)

        # Precompute norms and cross-product for top-k
        x_norm_sq = (x_query * x_query).sum(dim=1)  # (N,)
        y_topk_norm_sq = (x_train_topk * x_train_topk).sum(dim=2)  # (N, k)

        # Cross product for top-k
        cross = torch.bmm(x_query.unsqueeze(1), x_train_topk.transpose(1, 2)).squeeze(1)  # (N, k)

        # Cosine distance
        x_norm = torch.sqrt(x_norm_sq.clamp(min=eps))  # (N,)
        y_topk_norm = torch.sqrt(y_topk_norm_sq.clamp(min=eps))  # (N, k)
        denom = x_norm[:, None] * y_topk_norm  # (N, k)
        cos_sim = (cross / (denom + eps)).clamp(-1.0, 1.0)
        cos_dist = 1.0 - cos_sim

        # Euclidean distance
        euclid_sq = x_norm_sq[:, None] + y_topk_norm_sq - 2.0 * cross
        euclid_sq = euclid_sq.clamp(min=0.0)
        if squared:
            euclid = euclid_sq
        else:
            euclid = torch.sqrt(euclid_sq + eps)

        if normalize_euclid:
            # Normalize by sqrt(dimension) instead of batch statistics
            d = x_query.shape[1]
            scale = torch.sqrt(torch.tensor(d, device=x_query.device, dtype=x_query.dtype))
            euclid = euclid / (scale + eps)

        D_topk = alpha * euclid + (1.0 - alpha) * cos_dist
    elif metric == "mixed_faiss":
        alpha = kwargs.get("alpha", 0.5)
        eps = kwargs.get("eps", 1e-8)

        x_norm = x_query / (x_query.norm(dim=1, keepdim=True) + eps)
        y_topk_norm = x_train_topk / (x_train_topk.norm(dim=2, keepdim=True) + eps)

        sqrt_alpha = torch.sqrt(torch.tensor(alpha, dtype=x_query.dtype, device=x_query.device))
        sqrt_1ma = torch.sqrt(torch.tensor(1.0 - alpha, dtype=x_query.dtype, device=x_query.device))

        x_mix = torch.cat([sqrt_alpha * x_query, sqrt_1ma * x_norm], dim=1)
        y_mix = torch.cat([sqrt_alpha * x_train_topk, sqrt_1ma * y_topk_norm], dim=2)

        x_expand = x_mix[:, None, :]
        dist_sq = ((x_expand - y_mix).pow(2).sum(dim=2)).clamp(min=0.0)
        D_topk = torch.sqrt(dist_sq + eps)
    elif metric == "mahalanobis":
        diff = x_query[:, None, :] - x_train_topk
        temp = torch.matmul(diff.unsqueeze(2), inv_cov.unsqueeze(0).unsqueeze(0))
        sq = torch.sum(temp * diff.unsqueeze(2), dim=-1).squeeze(2).clamp(min=0)
        D_topk = torch.sqrt(sq)
    else:
        raise ValueError(f"Unknown metric for top-k recompute: {metric}")

    return D_topk.mean(1)


def _knn_distance_per_class_loop(
        x_query: torch.Tensor,
        x_train: torch.Tensor,
        y_idx_query: torch.Tensor,
        y_idx_train: torch.Tensor,
        k: int,
        metric="euclidean",
        inv_cov=None,
        **kwargs
) -> torch.Tensor:
    """
    Loop over classes; compute distances only within the same class.
    More memory efficient than masked full-matrix approach.

    Args:
        x_query, x_train: feature tensors.
        y_idx_query, y_idx_train: integer label indices.
        k: top-k.
        metric: distance metric.

    Returns:
        Per-sample averaged top-k distance.
    """
    classes = torch.unique(y_idx_query)
    out = torch.empty(x_query.size(0), dtype=x_query.dtype, device=x_query.device)
    for c in classes:
        q_mask = (y_idx_query == c)
        t_mask = (y_idx_train == c)
        x_q_c = x_query[q_mask]
        x_t_c = x_train[t_mask]
        if x_t_c.numel() == 0:
            out[q_mask] = 1e38
            continue
        d = _compute_distance_metric(x_q_c, x_t_c, metric, inv_cov, **kwargs)
        out[q_mask] = torch.topk(d, k=k, dim=1, largest=False).values.mean(1)
    return out


def _knn_distance_per_class_loop_topk_recompute(
        x_query: torch.Tensor,
        x_train: torch.Tensor,
        y_idx_query: torch.Tensor,
        y_idx_train: torch.Tensor,
        k: int,
        metric="euclidean",
        inv_cov=None,
        **kwargs
) -> torch.Tensor:
    """
    Loop over classes with top-k recomputation optimization.

    Args:
        x_query, x_train: feature tensors.
        y_idx_query, y_idx_train: integer label indices.
        k: top-k.
        metric: distance metric.

    Returns:
        Per-sample averaged top-k distance.
    """
    classes = torch.unique(y_idx_query)
    out = torch.empty(x_query.size(0), dtype=x_query.dtype, device=x_query.device)

    for c in classes:
        q_mask = (y_idx_query == c)
        t_mask = (y_idx_train == c)
        x_q_c = x_query[q_mask]
        x_t_c = x_train[t_mask]

        if x_t_c.numel() == 0:
            out[q_mask] = 1e38
            continue

        # Find top-k without gradients
        with torch.no_grad():
            d_nograd = _compute_distance_metric(x_q_c, x_t_c, metric, inv_cov, **kwargs)
            idx = torch.topk(d_nograd, k=k, dim=1, largest=False).indices

        # Recompute top-k with gradients
        x_t_topk = x_t_c[idx]  # [N_q_c, k, D]
        y_t_topk = x_t_topk  # for consistency in naming

        if metric == "euclidean":
            x_expand = x_q_c[:, None, :]
            d_topk = (x_expand - x_t_topk).pow(2).sum(dim=2).sqrt()
        elif metric == "cosine":
            x_expand = x_q_c[:, None, :]
            x_norm = F.normalize(x_expand, p=2, dim=2)
            y_norm = F.normalize(x_t_topk, p=2, dim=2)
            sim = (x_norm * y_norm).sum(dim=2)
            d_topk = 1.0 - sim
        elif metric == "mixed":
            # Recompute mixed distance exactly as in mixed_distance_fast
            alpha = kwargs.get("alpha", 0.5)
            eps = kwargs.get("eps", 1e-8)
            squared = kwargs.get("squared", False)
            normalize_euclid = kwargs.get("normalize_euclid", True)

            # Precompute norms and cross-product for top-k
            x_norm_sq = (x_q_c * x_q_c).sum(dim=1)  # (N,)
            y_topk_norm_sq = (y_t_topk * y_t_topk).sum(dim=2)  # (N, k)

            # Cross product for top-k
            cross = torch.bmm(x_q_c.unsqueeze(1), y_t_topk.transpose(1, 2)).squeeze(1)  # (N, k)

            # Cosine distance
            x_norm = torch.sqrt(x_norm_sq.clamp(min=eps))  # (N,)
            y_topk_norm = torch.sqrt(y_topk_norm_sq.clamp(min=eps))  # (N, k)
            denom = x_norm[:, None] * y_topk_norm  # (N, k)
            cos_sim = (cross / (denom + eps)).clamp(-1.0, 1.0)
            cos_dist = 1.0 - cos_sim

            # Euclidean distance
            euclid_sq = x_norm_sq[:, None] + y_topk_norm_sq - 2.0 * cross
            euclid_sq = euclid_sq.clamp(min=0.0)
            if squared:
                euclid = euclid_sq
            else:
                euclid = torch.sqrt(euclid_sq + eps)

            if normalize_euclid:
                d = x_q_c.shape[1]
                scale = torch.sqrt(torch.tensor(d, device=x_q_c.device, dtype=x_q_c.dtype))
                euclid = euclid / (scale + eps)

            d_topk = alpha * euclid + (1.0 - alpha) * cos_dist
        elif metric == "mixed_faiss":
            alpha = kwargs.get("alpha", 0.5)
            eps = kwargs.get("eps", 1e-8)

            x_norm = x_q_c / (x_q_c.norm(dim=1, keepdim=True) + eps)
            y_topk_norm = y_t_topk / (y_t_topk.norm(dim=2, keepdim=True) + eps)

            sqrt_alpha = torch.sqrt(torch.tensor(alpha, dtype=x_q_c.dtype, device=x_q_c.device))
            sqrt_1ma = torch.sqrt(torch.tensor(1.0 - alpha, dtype=x_q_c.dtype, device=x_q_c.device))

            x_mix = torch.cat([sqrt_alpha * x_q_c, sqrt_1ma * x_norm], dim=1)
            y_mix = torch.cat([sqrt_alpha * y_t_topk, sqrt_1ma * y_topk_norm], dim=2)

            x_expand = x_mix[:, None, :]
            dist_sq = ((x_expand - y_mix).pow(2).sum(dim=2)).clamp(min=0.0)
            d_topk = torch.sqrt(dist_sq + eps)
        elif metric == "mahalanobis":
            diff = x_q_c[:, None, :] - x_t_topk
            temp = torch.matmul(diff.unsqueeze(2), inv_cov.unsqueeze(0).unsqueeze(0))
            sq = torch.sum(temp * diff.unsqueeze(2), dim=-1).squeeze(2).clamp(min=0)
            d_topk = torch.sqrt(sq)
        else:
            raise ValueError(f"Unknown metric for top-k recompute: {metric}")

        out[q_mask] = d_topk.mean(1)

    return out


def _mahalanobis_knn_per_class_loop(
        x_query: torch.Tensor,
        x_train: torch.Tensor,
        y_idx_q: torch.Tensor,
        y_idx_t: torch.Tensor,
        class_inv_covs: torch.Tensor,
        k: int,
        shared: bool = False
) -> torch.Tensor:
    """
    Per-class loop variant for Mahalanobis KNN.
    Handles shared or per-class covariance.

    Args:
        x_query, x_train: features.
        y_idx_q, y_idx_t: label indices.
        class_inv_covs: inverse covariances.
        k: top-k.
        shared: whether class_inv_covs is shared.

    Returns:
        Per-sample averaged top-k Mahalanobis distance.
    """
    classes = torch.unique(y_idx_q)
    out = torch.empty(x_query.size(0), dtype=x_query.dtype, device=x_query.device)
    for c in classes:
        q_mask = (y_idx_q == c)
        t_mask = (y_idx_t == c)
        x_q_c = x_query[q_mask]
        x_t_c = x_train[t_mask]
        if x_t_c.numel() == 0:
            out[q_mask] = torch.finfo(out.dtype).max
            continue
        inv = class_inv_covs if shared else class_inv_covs[c]
        diff = x_q_c.unsqueeze(1) - x_t_c.unsqueeze(0)
        temp = diff @ inv
        sq = (temp * diff).sum(-1).clamp(min=0)
        d = torch.sqrt(sq)
        out[q_mask] = torch.topk(d, k=k, dim=1, largest=False).values.mean(1)
    return out


class KNNConfidence(DistanceConfidence):
    """
    K-NN confidence with selectable metric: "euclidean", "cosine", "mixed",
    or a custom callable function.
    """

    def __init__(
            self,
            k: int = 3,
            metric: Union[Literal["euclidean", "cosine", "mahalanobis", "mixed", "mixed_faiss"], Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
            dtype: torch.dtype = torch.float16,
            mixed_alpha: float = 0.5,
            mixed_squared: bool = False,
            mixed_normalize_euclid: bool = True,
            mahalanobis_eps: float = 1e-6,
            use_topk_recompute: bool = True,  # parameter enabling top-k recompute optimization
            **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric
        self._dtype = dtype
        self.mixed_alpha = mixed_alpha
        self.mixed_squared = mixed_squared
        self.mixed_normalize_euclid = mixed_normalize_euclid
        self.mahalanobis_eps = mahalanobis_eps
        self.use_topk_recompute = use_topk_recompute

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Fit KNN backing data and optional Mahalanobis inverse covariance.

        Args:
            x: Training features (stored as self.train_data).
            y: Optional labels (unused here).

        Returns:
            self after fit.
        """
        x_fit = x.to(self.dtype)
        self.register_buffer("train_data", x_fit)
        if self.metric == "mahalanobis":
            # For Mahalanobis, compute global mean and inverse covariance
            mu, inv_cov = _compute_global_mean_inv_cov(x_fit, eps=self.mahalanobis_eps)
            self.register_buffer("inv_cov", inv_cov)
            self.register_buffer("mu", mu)
        return self

    def _compute_distance(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Compute per-sample KNN distance (averaged top-k) using selected metric.

        Args:
            x: Query features (converted to self.dtype before call).
            y: Optional labels (unused here).

        Returns:
            Per-sample averaged top-k distance (N,).
        """
        if not self.use_topk_recompute:
            # Original implementation
            if self.metric == "mahalanobis":
                inv_cov = getattr(self, "inv_cov", None)
                if inv_cov is None:
                    raise ValueError("Mahalanobis metric requires fitting first to compute inverse covariance.")
                d = _compute_distance_metric(x, self.train_data, self.metric, inv_cov)
            else:
                kwargs = {}
                if self.metric == "mixed":
                    kwargs = {
                        "alpha": self.mixed_alpha,
                        "squared": self.mixed_squared,
                        "normalize_euclid": self.mixed_normalize_euclid
                    }
                elif self.metric == "mixed_faiss":
                    kwargs = {
                        "alpha": self.mixed_alpha,
                    }
                d = _compute_distance_metric(x, self.train_data, self.metric, **kwargs)
            vals = torch.topk(d, k=self.k, dim=1, largest=False).values
            return vals.mean(1)

        # Optimized top-k recompute version
        if self.metric == "euclidean":
            D_topk = _topk_recompute_euclidean(x, self.train_data, self.k)
        elif self.metric == "cosine":
            D_topk = _topk_recompute_cosine(x, self.train_data, self.k)
        elif self.metric == "mixed":
            D_topk = _topk_recompute_mixed(
                x, self.train_data, self.k,
                alpha=self.mixed_alpha,
                squared=self.mixed_squared,
                normalize_euclid=self.mixed_normalize_euclid
            )
        elif self.metric == "mahalanobis":
            inv_cov = getattr(self, "inv_cov", None)
            if inv_cov is None:
                raise ValueError("Mahalanobis metric requires fitting first to compute inverse covariance.")
            D_topk = _topk_recompute_mahalanobis(x, self.train_data, self.k, inv_cov)
        elif callable(self.metric):
            # For custom metrics, fall back to original implementation
            d = self.metric(x, self.train_data)
            D_topk = torch.topk(d, k=self.k, dim=1, largest=False).values
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return D_topk.mean(1)


def _knn_distance_per_class_jit(
        x_query: torch.Tensor,
        x_train: torch.Tensor,
        y_idx_query: torch.Tensor,
        y_idx_train: torch.Tensor,
        k: int,
        metric: str = "euclidean",
        inv_cov: Optional[torch.Tensor] = None,
        **kwargs
) -> torch.Tensor:
    """
    Compute per-class KNN distances using a loop over classes in JIT.
    Supports all metrics from _compute_distance.

    Args:
        x_query, x_train: features.
        y_idx_query, y_idx_train: label indices.
        k: top-k.
        metric: distance metric.

    Returns:
        Per-sample averaged top-k distance.
    """
    INF: float = float('inf')
    N_query = x_query.size(0)
    result = torch.empty(N_query, dtype=x_query.dtype, device=x_query.device)

    # Get unique classes
    classes = torch.unique(y_idx_query)

    for i in range(classes.size(0)):
        c = classes[i]
        mask_q = y_idx_query == c
        mask_t = y_idx_train == c

        x_q_c = x_query[mask_q]
        x_t_c = x_train[mask_t]

        if x_t_c.size(0) == 0:
            result[mask_q] = INF
            continue

        # Compute distances
        if metric == "mahalanobis" and inv_cov is not None:
            inv = inv_cov if inv_cov.ndim == 2 else inv_cov[c]
            diff = x_q_c.unsqueeze(1) - x_t_c.unsqueeze(0)
            temp = diff.unsqueeze(2) @ inv.unsqueeze(0).unsqueeze(0)
            dists = torch.sqrt(torch.clamp((temp * diff.unsqueeze(2)).sum(-1), min=0.0))
        else:
            dists = _compute_distance_metric(x_q_c, x_t_c, metric, inv_cov, **kwargs)

        kk = k  # removed .item()
        result[mask_q] = torch.topk(dists, k=kk, largest=False).values.mean(dim=1)

    return result


class PerClassKNNConfidence(DistanceConfidence):
    """
    Per-class K-nearest neighbor confidence using various distance metrics.
    computation_mode:
      - "masked": (default) compute full distance matrix and mask non-class entries (faster when classes few)
      - "per_class": iterate per class; lower memory when many classes / large dataset
    """

    def __init__(
            self,
            k: int = 3,
            metric: Union[Literal["euclidean", "cosine", "mixed", "mahalanobis", "mixed_faiss"], Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
            mahalanobis_eps: float = 1e-6,
            shared_covariance: bool = False,
            computation_mode: Literal["masked", "per_class"] = "masked",
            dtype: torch.dtype = torch.float16,
            mixed_alpha: float = 0.0,
            mixed_squared: bool = False,
            mixed_normalize_euclid: bool = True,
            use_topk_recompute: bool = True,  # parameter enabling top-k recompute optimization
            **kwargs
    ):
        super().__init__(**kwargs)
        self.k = k
        self.metric = metric
        self.mahalanobis_eps = mahalanobis_eps
        self.shared_covariance = shared_covariance
        self.computation_mode = computation_mode
        self._dtype = dtype
        self.mixed_alpha = mixed_alpha
        self.mixed_squared = mixed_squared
        self.mixed_normalize_euclid = mixed_normalize_euclid
        self.use_topk_recompute = use_topk_recompute

        if computation_mode not in ("masked", "per_class"):
            raise ValueError("computation_mode must be 'masked', 'per_class',")

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Fit per-class training data and optionally compute Mahalanobis statistics.

        Args:
            x: Training features.
            y: Labels.

        Returns:
            self after fit.
        """
        x_fit = x.to(self.dtype)
        self.register_buffer("train_data", x_fit)
        labels_unique, y_idx = _remap_labels_to_indices(y)
        # keep aliases for downstream methods
        _labels_unique = labels_unique
        _train_labels = y_idx
        self.register_buffer("_labels_unique", _labels_unique)
        self.register_buffer("_train_labels", _train_labels)

        # For Mahalanobis, compute class means and covariances
        if self.metric == "mahalanobis":
            # Compute class means
            class_means, inv_covs = _compute_per_class_means_inv_covs(x_fit, self._train_labels, self.mahalanobis_eps)
            self.register_buffer("class_means", class_means)
            if self.shared_covariance:
                # Use a single shared inverse covariance
                _, shared_inv_cov = _compute_global_mean_inv_cov(x_fit - self.class_means[self._train_labels],
                                                                 self.mahalanobis_eps)
                self.register_buffer("shared_inv_cov", shared_inv_cov)
            else:
                self.register_buffer("class_inv_covs", inv_covs)
        return self

    def _compute_distance(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute per-sample per-class KNN distances using selected computation mode.

        Args:
            x: Query features.
            y: Labels for query samples (required to select class-specific neighbors).

        Returns:
            Per-sample averaged top-k distance (N,).
        """
        if y is None:
            raise NotImplementedError("Per-class KNN requires labels y to compute distances.")

        y_idx = torch.searchsorted(self._labels_unique, y)
        # check that none are not found
        if (y_idx >= len(self._labels_unique)).any():
            raise ValueError("Some labels in y are not found in the training labels.")

        if self.metric == "mahalanobis":
            inv_covs = self.shared_inv_cov if self.shared_covariance else self.class_inv_covs
            if self.computation_mode == "masked" and not self.shared_covariance:
                # original vectorized masked version
                dist = _mahalanobis_knn_per_class(
                    x, self.train_data, y_idx, self._train_labels,
                    inv_covs, self.k)
            else:
                # loop version supports both shared / per-class
                dist = _mahalanobis_knn_per_class_loop(
                    x, self.train_data, y_idx, self._train_labels,
                    inv_covs, self.k, shared=self.shared_covariance)
        else:
            kwargs = {}
            if self.metric == "mixed":
                kwargs = {
                    "alpha": self.mixed_alpha,
                    "squared": self.mixed_squared,
                    "normalize_euclid": self.mixed_normalize_euclid
                }
            elif self.metric == "mixed_faiss":
                kwargs = {
                    "alpha": self.mixed_alpha,
                }

            if self.use_topk_recompute:
                # Top-k recompute versions
                if self.computation_mode == "masked":
                    dist = _knn_distance_per_class_vectorized_topk_recompute(
                        x, self.train_data, y_idx, self._train_labels, self.k,
                        metric=self.metric, **kwargs)
                else:  # per_class
                    dist = _knn_distance_per_class_loop_topk_recompute(
                        x, self.train_data, y_idx, self._train_labels, self.k,
                        metric=self.metric, **kwargs)
            else:
                if self.computation_mode == "masked":
                    dist = _knn_distance_per_class_vectorized(
                        x, self.train_data, y_idx, self._train_labels, self.k,
                        metric=self.metric, **kwargs)
                else:  # per_class
                    dist = _knn_distance_per_class_loop(
                        x, self.train_data, y_idx, self._train_labels, self.k,
                        metric=self.metric, **kwargs)

        return dist


if __name__ == "__main__":
    skip_test = True
    if not skip_test:
        torch.manual_seed(0)
        np.random.seed(0)
        N_per = 50;
        D = 2
        mu0 = np.array([0., 0.]);
        cov0 = np.eye(2) * 0.5
        mu1 = np.array([5., 5.]);
        cov1 = np.eye(2) * 0.5
        data0 = np.random.multivariate_normal(mu0, cov0, N_per)
        data1 = np.random.multivariate_normal(mu1, cov1, N_per)
        X = np.vstack([data0, data1]);
        y = np.hstack([np.zeros(N_per), np.ones(N_per)])
        X_t = torch.from_numpy(X).float();
        y_t = torch.from_numpy(y).long()
        X_test = torch.tensor([[0.1, -0.2], [4.8, 5.2], [2.5, 2.5]])
        y_test = torch.tensor([0, 1, 0])


        # Custom distance function example - Manhattan distance
        def manhattan_distance(x, y):
            return torch.cdist(x, y, p=1)


        # Test the distance-based confidence classes
        for name, cls in [
            ("KNN Euclidean", KNNConfidence(k=3, metric="euclidean")),
            ("KNN Cosine", KNNConfidence(k=3, metric="cosine")),
            ("KNN Mixed", KNNConfidence(k=3, metric="mixed")),
            ("KNN Custom", KNNConfidence(k=3, metric=manhattan_distance)),
            ("Per-Class KNN Euclidean", PerClassKNNConfidence(k=3, metric="euclidean")),
            ("Per-Class KNN Cosine", PerClassKNNConfidence(k=3, metric="cosine")),
            ("Per-Class KNN Mahalanobis", PerClassKNNConfidence(k=3, metric="mahalanobis")),
        ]:
            print(f"\n==== {name} ====")
            mdl = cls
            mdl.fit(X_t, y_t)
            conf = mdl.forward(X_test, y_test)
            print(f"{name} confidences:", conf.tolist())

        # Test the new Triton-accelerated variants
        triton_tests = [
            ("Per-Class KNN Triton Euclidean",
             PerClassKNNConfidence(k=3, metric="euclidean", computation_mode="triton")),
            ("Per-Class KNN Triton Cosine", PerClassKNNConfidence(k=3, metric="cosine", computation_mode="triton")),
        ]

        for name, cls in triton_tests:
            print(f"\n==== {name} ====")
            mdl = cls
            mdl.fit(X_t, y_t)
            conf = mdl.forward(X_test, y_test)
            print(f"{name} confidences:", conf.tolist())

        # Test half precision
        if torch.cuda.is_available():
            print("\n==== Testing Half Precision (FP16) ====")
            X_t_cuda = X_t.cuda()
            y_t_cuda = y_t.cuda()
            X_test_cuda = X_test.cuda()
            y_test_cuda = y_test.cuda()

            half_tests = [
                ("KNN Euclidean FP16", KNNConfidence(k=3, metric="euclidean", dtype=torch.half)),
                ("Per-Class KNN Euclidean FP16", PerClassKNNConfidence(k=3, metric="euclidean", dtype=torch.half)),
                ("Per-Class KNN Triton FP16",
                 PerClassKNNConfidence(k=3, metric="euclidean", computation_mode="triton", dtype=torch.half)),
                ("Per-Class KNN Mahalanobis FP16", PerClassKNNConfidence(k=3, metric="mahalanobis", dtype=torch.half)),
            ]
            for name, cls in half_tests:
                print(f"\n==== {name} ====")
                mdl = cls
                mdl.fit(X_t_cuda, y_t_cuda)
                conf = mdl.forward(X_test_cuda, y_test_cuda)
                print(f"{name} confidences:", conf.tolist())
                print(f"Train data dtype: {mdl.train_data.dtype}")

    # CUDA speed test with large database and batches
    if torch.cuda.is_available():
        import time

        print("\n==== CUDA Speed Test: 50000x512 DB, 128x512 batches ====")
        N_db = 50000
        D = 512
        N_batch = 128
        n_batches = 100

        # Generate random database and batch samples efficiently
        X_db = torch.randn(N_db, D, device="cuda")
        y_db = torch.randint(0, 50, (N_db,), device="cuda")
        X_batches = torch.randn(n_batches, N_batch, D, device="cuda")
        y_batches = torch.randint(0, 50, (n_batches, N_batch), device="cuda")

        mdl = PerClassKNNConfidence(
            k=3, metric="cosine", dtype=torch.float32, computation_mode="masked"
        ).to("cuda")
        mdl.fit(X_db, y_db)

        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_batches):
            conf = mdl.forward(X_batches[i], y_batches[i])
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"Processed {n_batches} batches of {N_batch}x{D} against {N_db}x{D} DB in {t1 - t0:.3f} seconds")
        print(f"Avg time per batch: {(t1 - t0) / n_batches:.3f} seconds")
        print("Sample confidences:", conf[:5].cpu())

        # speed test for top-k recompute vs original with backward to input
        if torch.cuda.is_available():
            print("\n==== CUDA Speed Test: Top-K Recompute vs Original with Backward ====")
            N_db = 20000
            D = 256
            N_batch = 256
            k_val = 10

            X_db = torch.randn(N_db, D, device="cuda", dtype=torch.float32)
            y_db = torch.randint(0, 20, (N_db,), device="cuda")
            X_batch = torch.randn(N_batch, D, device="cuda", dtype=torch.float32)
            y_batch = torch.randint(0, 20, (N_batch,), device="cuda")

            # Original
            mdl_orig = KNNConfidence(
                k=k_val, metric="euclidean", dtype=torch.float32,
                use_topk_recompute=False
            ).to("cuda")
            mdl_orig.fit(X_db, y_db)

            X_batch_orig = X_batch.clone().detach().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(100):
                conf_orig = mdl_orig.forward(X_batch_orig, y_batch)
                loss_orig = conf_orig.sum()
                loss_orig.backward()
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"Original time (forward + backward): {t1 - t0:.3f} seconds")

            # Top-K Recompute
            mdl_opt = KNNConfidence(
                k=k_val, metric="euclidean", dtype=torch.float32,
                use_topk_recompute=True
            ).to("cuda")
            mdl_opt.fit(X_db, y_db)

            X_batch_opt = X_batch.clone().detach().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(100):
                conf_opt = mdl_opt.forward(X_batch_opt, y_batch)
                loss_opt = conf_opt.sum()
                loss_opt.backward()
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"Top-K Recompute time (forward + backward): {t1 - t0:.3f} seconds")

    # Add gradient verification tests
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("=== TOP-K RECOMPUTE GRADIENT VERIFICATION ===")
        print("=" * 80)

        torch.manual_seed(42)
        N_db = 1000
        D = 128
        N_batch = 32
        k_test = 5

        X_db = torch.randn(N_db, D, device="cuda", dtype=torch.float32)
        y_db = torch.randint(0, 10, (N_db,), device="cuda")
        X_test = torch.randn(N_batch, D, device="cuda", dtype=torch.float32)
        y_test = torch.randint(0, 10, (N_batch,), device="cuda")

        metrics_to_test = ["euclidean", "cosine", "mixed", "mixed_faiss"]

        for metric in metrics_to_test:
            print(f"\n{'=' * 60}")
            print(f"Testing metric: {metric}")
            print(f"{'=' * 60}")

            # Test KNNConfidence
            print(f"\n--- KNNConfidence ({metric}) ---")

            # Original version
            X_test_orig = X_test.clone().detach().requires_grad_(True)
            knn_orig = KNNConfidence(k=k_test, metric=metric, dtype=torch.float32,
                                     use_topk_recompute=False, mixed_alpha=0.4).to("cuda")
            knn_orig.fit(X_db, y_db)
            conf_orig = knn_orig.forward(X_test_orig, y_test)
            loss_orig = conf_orig.sum()
            loss_orig.backward()
            grad_orig = X_test_orig.grad.clone()

            # Optimized version
            X_test_opt = X_test.clone().detach().requires_grad_(True)
            knn_opt = KNNConfidence(k=k_test, metric=metric, dtype=torch.float32,
                                    use_topk_recompute=True, mixed_alpha=0.4).to("cuda")
            knn_opt.fit(X_db, y_db)
            conf_opt = knn_opt.forward(X_test_opt, y_test)
            loss_opt = conf_opt.sum()
            loss_opt.backward()
            grad_opt = X_test_opt.grad.clone()

            # Compare
            conf_diff = (conf_orig - conf_opt).abs().max().item()
            grad_diff = (grad_orig - grad_opt).abs().max().item()
            grad_rel_diff = (grad_diff / (grad_orig.abs().mean().item() + 1e-8))

            print(f"  Confidence diff: {conf_diff:.2e}")
            print(f"  Gradient abs diff: {grad_diff:.2e}")
            print(f"  Gradient rel diff: {grad_rel_diff:.2e}")
            print(f"  Gradient norm (orig): {grad_orig.norm().item():.6f}")
            print(f"  Gradient norm (opt):  {grad_opt.norm().item():.6f}")

            assert conf_diff < 1e-4, f"Confidence mismatch for {metric}: {conf_diff}"
            assert grad_diff < 1e-3, f"Gradient mismatch for {metric}: {grad_diff}"
            print(f"  ✓ Gradients match!")

            # Test PerClassKNNConfidence with different computation modes
            for comp_mode in ["masked", "per_class"]:
                print(f"\n--- PerClassKNNConfidence ({metric}, {comp_mode}) ---")

                # Original version
                X_test_orig = X_test.clone().detach().requires_grad_(True)
                pc_knn_orig = PerClassKNNConfidence(
                    k=k_test, metric=metric, dtype=torch.float32,
                    computation_mode=comp_mode, use_topk_recompute=False,
                    mixed_alpha=0.4
                ).to("cuda")
                pc_knn_orig.fit(X_db, y_db)
                conf_orig = pc_knn_orig.forward(X_test_orig, y_test)
                loss_orig = conf_orig.sum()
                loss_orig.backward()
                grad_orig = X_test_orig.grad.clone()

                # Optimized version
                X_test_opt = X_test.clone().detach().requires_grad_(True)
                pc_knn_opt = PerClassKNNConfidence(
                    k=k_test, metric=metric, dtype=torch.float32,
                    computation_mode=comp_mode, use_topk_recompute=True,
                    mixed_alpha=0.4
                ).to("cuda")
                pc_knn_opt.fit(X_db, y_db)
                conf_opt = pc_knn_opt.forward(X_test_opt, y_test)
                loss_opt = conf_opt.sum()
                loss_opt.backward()
                grad_opt = X_test_opt.grad.clone()

                # Compare
                conf_diff = (conf_orig - conf_opt).abs().max().item()
                grad_diff = (grad_orig - grad_opt).abs().max().item()
                grad_rel_diff = (grad_diff / (grad_orig.abs().mean().item() + 1e-8))

                print(f"  Confidence diff: {conf_diff:.2e}")
                print(f"  Gradient abs diff: {grad_diff:.2e}")
                print(f"  Gradient rel diff: {grad_rel_diff:.2e}")
                print(f"  Gradient norm (orig): {grad_orig.norm().item():.6f}")
                print(f"  Gradient norm (opt):  {grad_opt.norm().item():.6f}")

                assert conf_diff < 1e-4, f"Confidence mismatch for {metric} {comp_mode}: {conf_diff}"
                assert grad_diff < 1e-3, f"Gradient mismatch for {metric} {comp_mode}: {grad_diff}"
                print(f"Gradients match!")

        print("\n" + "=" * 80)
        print("=== ALL GRADIENT TESTS PASSED ===")
        print("=" * 80)
