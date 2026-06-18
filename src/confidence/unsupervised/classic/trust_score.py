import torch
from typing import Optional, Union
from confidence.base_confidence import ConfidenceModule
from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
#todo maybe include alpha preprocessing as in the original paper

class TrustScoreTorchConfidence(ClassicConfidenceBase):
    """
    Vectorized PyTorch Trust Score using k-nearest average distances. In the original definition this requires predicted class from the classifier to be passed.
    In the orginal dalpha ased density preprocessing was also done. #TODO
    Requires the predicted class to be passed. If this is not done. We predict using the nearest class.
    Supports 'euclidean' or 'cosine' distance.
    """
    def __init__(
        self,
        k_neighbors: int = 5,
        k_distance: Optional[int] = None,
        eps: float = 1e-10,
        input_transform: Optional[InputTransform] = None,
        alpha: float = 0.0,
        distance: str = "euclidean"
    ):
        super().__init__(input_transform=input_transform)
        self.k = k_neighbors
        self.k_dist = k_distance or k_neighbors
        self.eps = eps
        self.X_train = None
        self.y_train = None
        self.warned = False
        self.alpha = alpha  # density thresholding parameter
        self.distance = distance

    def _compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distance matrix between a and b.

        Args:
            a: Tensor of shape [M, D].
            b: Tensor of shape [N, D].

        Returns:
            Tensor: pairwise distances of shape [M, N].
        """
        if self.distance == "euclidean":
            return torch.cdist(a, b)
        elif self.distance == "cosine":
            norm_a = a / a.norm(dim=-1, keepdim=True)
            norm_b = b / b.norm(dim=-1, keepdim=True)
            sim = norm_a @ norm_b.T
            return 1 - sim
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Fit the module and optionally perform density-based preprocessing.

        Args:
            X: Training feature tensor [N, D].
            y: Training labels tensor [N].

        Returns:
            None
        """
        # Move to device and transform if needed

        # TODO check again if this matches offically implementation.
        if self.alpha > 0:
            X_hd_list = []
            y_hd_list = []
            for cls in torch.unique(y):
                mask = (y == cls)
                Xc = X[mask]
                N = Xc.size(0)
                if N == 0:
                    continue
                k_effective = min(self.k, N - 1)
                # compute pairwise distances within class
                dists = self._compute_distance(Xc, Xc)
                # ignore self-distances
                diag_idx = torch.arange(dists.size(0), device=dists.device)
                dists[diag_idx, diag_idx] = float('inf')
                # get k-th nearest neighbor distance
                knn_dists, _ = torch.topk(dists, k_effective + 1, largest=False)
                radii = knn_dists[:, -1]
                # threshold at percentile
                thresh = torch.quantile(radii, 1 - self.alpha)
                high_density_mask = radii <= thresh
                Xc_hd = Xc[high_density_mask]
                yc_hd = torch.full((Xc_hd.size(0),), cls, dtype=y.dtype, device=y.device)
                X_hd_list.append(Xc_hd)
                y_hd_list.append(yc_hd)
        else:
            # No density filtering, use all points
            X_hd_list = [X]
            y_hd_list = [y]

        # concatenate high-density sets
        self.X_train = torch.cat(X_hd_list, dim=0)
        self.y_train = torch.cat(y_hd_list, dim=0)

    def _forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute trust-based confidence scores for inputs.

        Args:
            x: Input features tensor [B, D].
            y: Optional predicted class labels tensor [B].

        Returns:
            Tensor: confidence per sample (shape [B]).
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Call fit() before forward().")

        x = x
        if y is None and self.warned is False:
            print("Warning: TrustScoreTorchConfidence requires predicted labels y for correct computation. "
                  "Using k-NN to predict labels instead.")
            self.warned = True

        # pairwise distances (B, N)
        d_xt = self._compute_distance(x, self.X_train)

        # predict class via k-NN if y is None
        d_knn, idx_knn = torch.topk(d_xt, self.k, largest=False)
        if y is None:
            neigh_labels = self.y_train[idx_knn]               # (B, k)
            y_pred = torch.mode(neigh_labels, dim=1).values    # (B,)
        else:
            y_pred = y.long()

        # create masks (B, N)
        eq_mask = self.y_train.unsqueeze(0) == y_pred.unsqueeze(1)
        inf_mat = torch.full_like(d_xt, float('inf'))

        # average of k nearest same-class distances
        d_same_masked = torch.where(eq_mask, d_xt, inf_mat)
        d_same_k, _ = torch.topk(d_same_masked, self.k_dist, largest=False)
        d_same = d_same_k.mean(dim=1)

        # average of k nearest other-class distances
        d_other_masked = torch.where(~eq_mask, d_xt, inf_mat)
        d_other_k, _ = torch.topk(d_other_masked, self.k_dist, largest=False)
        d_other = d_other_k.mean(dim=1)

        # compute trust and convert to confidence
        trust = d_other / (d_same + self.eps)
        conf = 1.0 - 1.0 / (1.0 + trust)

        return conf