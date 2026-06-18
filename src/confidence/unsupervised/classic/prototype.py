from typing import Union, Literal, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

from confidence.unsupervised.classic.nn_pytorch import DistanceConfidence, _compute_global_mean_inv_cov, \
    _compute_distance_metric, _remap_labels_to_indices, _compute_per_class_means_inv_covs, \
    _mahalanobis_distance_per_class_vectorized


class GlobalPrototypeConfidence(DistanceConfidence):
    """
    Global prototype confidence using various distance metrics. Distance is calculated to global mean.
    """

    def __init__(
            self,
            metric: Union[Literal["euclidean", "cosine", "mixed", "mahalanobis"], Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
            mahalanobis_eps: float = 1e-6,
            mixed_alpha: float = 0.5,
            mixed_squared: bool = False,
            mixed_normalize_euclid: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.metric = metric
        self.mahalanobis_eps = mahalanobis_eps
        self.mixed_alpha = mixed_alpha
        self.mixed_squared = mixed_squared
        self.mixed_normalize_euclid = mixed_normalize_euclid

    def _fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Compute and store the global mean and optional inverse covariance.

        Args:
            x: Input features tensor of shape [N, D].
            y: Optional labels tensor (unused for global mean).

        Returns:
            self
        """
        # Store global mean
        self.register_buffer("mean", x.mean(0))

        # For Mahalanobis, compute inverse covariance
        if self.metric == "mahalanobis":
            _, inv_cov = _compute_global_mean_inv_cov(x, self.mahalanobis_eps)
            self.register_buffer("inv_cov", inv_cov)
        return self

    def _compute_distance(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        """
        Compute distance from each sample to the global mean using the configured metric.

        Args:
            x: Input features tensor [N, D].
            y: Optional labels (not used).

        Returns:
            Tensor: distances per input (shape [N]).
        """
        mean_batch = self.mean.unsqueeze(0)

        kwargs = {}
        if self.metric == "mixed":
            kwargs["alpha"] = self.mixed_alpha
            kwargs["squared"] = self.mixed_squared
            kwargs["normalize_euclid"] = self.mixed_normalize_euclid
        if self.metric == "mahalanobis":
            return _compute_distance_metric(x, mean_batch, self.metric, self.inv_cov).squeeze(1)
        else:
            return _compute_distance_metric(x, mean_batch, self.metric, **kwargs).squeeze(1)


class ClassPrototypeConfidence(DistanceConfidence):
    """
    Class prototype confidence using various distance metrics. Distance is calculated to all classes if
    not class is passed otherwise distance is calculated to the class mean of the corresponding class.
    """

    def __init__(
            self,
            metric: Union[Literal["euclidean", "cosine", "mixed", "mahalanobis"], Callable[
                [torch.Tensor, torch.Tensor], torch.Tensor]] = "euclidean",
            mahalanobis_eps: float = 1e-6,
            shared_covariance: bool = False,
            mixed_alpha: float = 0.5,  # use a mixture between cosine and euclidean distance.
            mixed_squared: bool = False,
            mixed_normalize_euclid: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.metric = metric
        self.mahalanobis_eps = mahalanobis_eps
        self.shared_covariance = shared_covariance
        self.mixed_alpha = mixed_alpha
        self.mixed_squared = mixed_squared
        self.mixed_normalize_euclid = mixed_normalize_euclid

    def _fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute per-class means and (optionally) per-class inverse covariances.

        Args:
            x: Input features tensor [N, D].
            y: Label tensor [N].

        Returns:
            self
        """
        labels_unique, y_idx = _remap_labels_to_indices(y)
        self.register_buffer("_labels_unique", labels_unique)

        # Compute class means
        C = int(y_idx.max().item()) + 1
        one_hot = F.one_hot(y_idx, num_classes=C).float()
        counts = one_hot.sum(dim=0)
        class_sums = one_hot.T @ x
        class_means = class_sums / counts.unsqueeze(1)
        self.register_buffer("class_means", class_means)

        # For Mahalanobis, compute covariances
        if self.metric == "mahalanobis":
            _, inv_covs = _compute_per_class_means_inv_covs(x, y_idx, self.mahalanobis_eps)
            if self.shared_covariance:
                # Shared inverse covariance across classes
                _, shared_inv_cov = _compute_global_mean_inv_cov(x - self.class_means[y_idx], self.mahalanobis_eps)
                self.register_buffer("shared_inv_cov", shared_inv_cov)
            else:
                self.register_buffer("class_inv_covs", inv_covs)
        return self

    def _compute_distance(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute distances either to the class mean corresponding to y, or the closest class mean.

        Very important: when y is provided this returns the per-sample distance to the corresponding class mean.

        Args:
            x: Input features tensor [N, D].
            y: Optional labels tensor [N].

        Returns:
            Tensor: distance(s) depending on y (shape [N] when y provided, else [N]).
        """
        kwargs = {}
        if self.metric == "mixed":
            kwargs["alpha"] = self.mixed_alpha
            kwargs["squared"] = self.mixed_squared
            kwargs["normalize_euclid"] = self.mixed_normalize_euclid

        if y is not None:
            y_idx = torch.searchsorted(self._labels_unique, y)

            if self.metric == "mahalanobis":
                # Mahalanobis distance to class mean
                inv_covs = self.shared_inv_cov if self.shared_covariance else self.class_inv_covs
                dist = _mahalanobis_distance_per_class_vectorized(
                    x, self.class_means, inv_covs, y_idx)
            else:
                # Use unified distance function for other metrics
                # Compute full distance matrix between samples and class means
                all_dists = _compute_distance_metric(x, self.class_means, self.metric, **kwargs)  # (N, C)

                # Create a mask to select the correct distance for each sample
                mask = F.one_hot(y_idx, num_classes=self.class_means.shape[0]).bool()

                # Select the distances using the mask
                dist = all_dists[mask]
        else:
            # Compute distances to all class prototypes and take minimum
            if self.metric == "mahalanobis":
                # Mahalanobis distance to each class mean
                inv_covs = self.shared_inv_cov if self.shared_covariance else self.class_inv_covs
                C = inv_covs.shape[0] if not self.shared_covariance else inv_covs.dim() == 2
                diff = x.unsqueeze(1) - self.class_means.unsqueeze(0)  # [N, C, D]

                if self.shared_covariance:
                    inv = inv_covs
                    temp = torch.matmul(diff.unsqueeze(2), inv)
                    sq = (temp @ diff.unsqueeze(3)).squeeze(-1).squeeze(-1).clamp(min=0)
                else:
                    inv_exp = inv_covs.unsqueeze(0).expand(x.size(0), -1, -1, -1)
                    temp = torch.matmul(diff.unsqueeze(2), inv_exp)
                    sq = (temp @ diff.unsqueeze(3)).squeeze(-1).squeeze(-1).clamp(min=0)
                dist = torch.sqrt(sq).min(dim=1).values
            else:
                # For other metrics, calculate distance to each class mean and find min
                dist = _compute_distance_metric(
                    x, self.class_means, self.metric, **kwargs).min(dim=1).values
        # Apply confidence function

        return dist


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    N_per = 50
    D = 2
    mu0 = np.array([0., 0.])
    cov0 = np.eye(2) * 0.5
    mu1 = np.array([5., 5.])
    cov1 = np.eye(2) * 0.5
    data0 = np.random.multivariate_normal(mu0, cov0, N_per)
    data1 = np.random.multivariate_normal(mu1, cov1, N_per)
    X = np.vstack([data0, data1])
    y = np.hstack([np.zeros(N_per), np.ones(N_per)])
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    X_test = torch.tensor([[0.1, -0.2], [4.8, 5.2], [2.5, 2.5]])
    y_test = torch.tensor([0, 1, 0])


    # Custom distance function example - Manhattan distance
    def manhattan_distance(x, y):
        return torch.cdist(x, y, p=1)


    # Test the distance-based confidence classes
    for name, cls in [
        ("Global Prototype Euclidean", GlobalPrototypeConfidence(metric="euclidean")),
        ("Global Prototype Cosine", GlobalPrototypeConfidence(metric="cosine")),
        ("Class Prototype Euclidean", ClassPrototypeConfidence(metric="euclidean")),
        ("Class Prototype Cosine", ClassPrototypeConfidence(metric="cosine")),
        ("Global Prototype Mahalanobis", GlobalPrototypeConfidence(metric="mahalanobis")),
        ("Class Prototype Mahalanobis", ClassPrototypeConfidence(metric="mahalanobis")),
    ]:
        print(f"\n==== {name} ====")
        mdl = cls
        mdl.fit(X_t, y_t)
        conf = mdl.forward(X_test, y_test)
        print(f"{name} confidences:", conf.tolist())
