import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import numpy as np
import torch
from pytorch_ood.detector import OpenMax
from scipy.stats import exponweib
from torch import Tensor

from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase


class OpenMaxConfidence(ClassicConfidenceBase):
    """
    OpenMax with support for Euclidean, Cosine, or blended Euclidean-Cosine distances via euclid_weight.
    """

    def __init__(
            self,
            tail_size: float = 25,
            alpha: int = 10,
            euclid_weight: float = 0.5,
            input_transform: Optional[InputTransform] = None,
            input_is_logits: bool = True
    ):
        super().__init__(input_transform=input_transform)
        self.tail_size = tail_size
        self.alpha = alpha
        self.euclid_weight = euclid_weight
        self.cos_weight = 1.0 - euclid_weight
        self.translation = 10000.0
        self.register_buffer('class_means', None)
        self.register_buffer('shapes', None)
        self.register_buffer('scales', None)
        self.register_buffer('min_vals', None)
        self.n_classes = None

    def _get_dists(self, x: Tensor) -> Tensor:
        # x: [B, D], class_means: [C, D] -> output [B, C]
        if self.euclid_weight == 1.0:
            return torch.cdist(x, self.class_means, p=2)
        elif self.euclid_weight == 0.0:
            x_norm = F.normalize(x, dim=1)
            m_norm = F.normalize(self.class_means, dim=1)
            cos_sim = torch.matmul(x_norm, m_norm.t())
            return 1 - cos_sim
        else:
            # Euclidean
            euclid = torch.cdist(x, self.class_means, p=2)
            # Cosine
            x_norm = F.normalize(x, dim=1)
            m_norm = F.normalize(self.class_means, dim=1)
            cos_sim = torch.matmul(x_norm, m_norm.t())
            cos_dist = 1 - cos_sim
            # Weighted sum
            return self.euclid_weight * euclid + self.cos_weight * cos_dist

    def _fit(self, X: Tensor, y: Tensor) -> "OpenMaxConfidence":
        """
        Fit per-class Weibull tails and class means for OpenMax.

        Args:
            X: Feature tensor of shape [N, D].
            y: Label tensor of shape [N].

        Returns:
            OpenMaxConfidence: self
        """
        classes = torch.unique(y)
        means, shapes, scales, min_vals = [], [], [], []
        for c in classes.tolist():
            feats = X[y == c]
            mu = feats.mean(dim=0)
            dists = torch.norm(feats - mu, dim=1).cpu().numpy()
            k = int(self.tail_size) if self.tail_size >= 1 else max(1, int(len(dists) * self.tail_size))
            tail = np.sort(dists)[-k:]
            min_val = tail.min()
            tail_trans = tail + self.translation - min_val
            _, c_shape, _, scale = exponweib.fit(tail_trans, f0=1, floc=0)
            means.append(mu)
            shapes.append(c_shape)
            scales.append(scale)
            min_vals.append(min_val)
        device = X.device
        self.class_means = torch.stack(means, dim=0).to(device)
        self.shapes = torch.tensor(shapes, device=device, dtype=self.class_means.dtype)
        self.scales = torch.tensor(scales, device=device, dtype=self.class_means.dtype)
        self.min_vals = torch.tensor(min_vals, device=device, dtype=self.class_means.dtype)
        self.n_classes = len(means)
        return self

    def _forward_logits(self, x: Tensor, y=None) -> Tensor:
        """
        Convert logits to unknown-class probability using OpenMax-style re-calibration.


        Args:
            x: Logits/features tensor of shape [B, C].
            y: Optional labels (not used here).

        Returns:
            Tensor: unknown-class probability per sample (shape [B]).
        """
        assert self.n_classes is not None, "Model not fitted"
        assert x.dim() == 2, "Input must be 2D tensor [B, D]"
        batch = x.size(0)
        # get weighted distances
        dists = self._get_dists(x)  # [B, C]
        # translate/clamp
        tail = dists + self.translation - self.min_vals.unsqueeze(0)
        tail = torch.clamp(tail, min=0)
        # Weibull CDF
        c = self.shapes.unsqueeze(0)
        s = self.scales.unsqueeze(0)
        cdf = 1 - torch.exp(- (tail / s) ** c)
        raw_w = cdf
        effective_alpha = min(self.alpha, self.n_classes)
        topk_vals, topk_idx = torch.topk(x, effective_alpha, dim=1)
        idx = torch.arange(1, effective_alpha + 1, device=x.device, dtype=torch.float32)
        alpha_w = ((effective_alpha + 1) - idx) / float(effective_alpha)
        w = torch.zeros_like(x)
        for i in range(effective_alpha):
            cls = topk_idx[:, i]
            w[torch.arange(batch), cls] = raw_w[torch.arange(batch), cls] * alpha_w[i]
        rev = x * (1 - w)
        outlier = (x * w).sum(dim=1, keepdim=True)
        logits = torch.cat([outlier, rev], dim=1)
        return F.softmax(logits, dim=1)[:, 0]

    def _forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Return confidence as 1 - unknown-class probability.

        Args:
            x: Logits/features tensor [B, C].
            y: Optional labels.

        Returns:
            Tensor: confidence per sample (shape [B]).
        """
        return 1 - self._forward_logits(x, y)


# reference implementaion via pytorch_ood.
class OpenMaxConfidenceNumpy(ClassicConfidenceBase):
    """
    Simple wrapper around PyTorch-OOD's OpenMax.
    Fits on model logits and returns unknown-class probability.
    """

    def __init__(
            self,
            model: Optional[torch.nn.Module] = None,
            tailsize: int = 25,
            alpha: int = 10,
            input_transform: Optional[InputTransform] = None,
            euclid_weight=0.5
    ):
        super().__init__(input_transform=input_transform)
        self.model = model
        self._openmax = OpenMax(model=model, tailsize=tailsize, alpha=alpha, euclid_weight=euclid_weight)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def _fit(self, x, y: Tensor) -> "OpenMaxConfidence":
        """
        Fit OpenMax on features via the wrapped OpenMax instance.

        Args:
            x: Feature tensor used for fitting.
            y: Label tensor corresponding to x.

        Returns:
            OpenMaxConfidence: self
        """
        self._openmax.fit_features(x, y)
        return self

    @torch.no_grad()
    def _forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Compute unknown-class probability using the wrapped OpenMax predictor.

        Args:
            x: Feature tensor to score.
            y: Optional labels (unused).

        Returns:
            Tensor: unknown-class probability per sample (moved to input device/dtype).
        """
        res = 1 - self._openmax.predict_features(x)
        return res.to(x.device, x.dtype)

    @property
    def centers(self) -> Tensor:
        """
        Class centers (means) from the fitted OpenMax model.

        Returns:
            Tensor: class centers tensor.
        """
        return self._openmax._openmax.centers


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.datasets import make_blobs

    # Test equivlance on blob data.

    # Toy blob data
    X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)


    # linear classifier
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)


    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # 3. Train classifier
    for epoch in range(100):
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_logits = model(X)

    openmax = OpenMaxConfidence(tail_size=10, alpha=1, euclid_weight=1.0)
    openmax.fit(train_logits, y)

    openmax_numpy = OpenMaxConfidenceNumpy(model=None, tailsize=10, alpha=1, euclid_weight=1.0)
    openmax_numpy.fit(train_logits, y)

    test_points = torch.tensor([[0, 0], [2, 2], [10, 10], [-10, -10]], dtype=torch.float32)
    test_logits = model(test_points)

    conf_openmax = openmax(test_logits)
    conf_numpy = openmax_numpy(test_logits)

    print("OpenMaxConfidence unknown-class probabilities:", conf_openmax.detach().numpy())
    print("OpenMaxConfidenceNumpy unknown-class probabilities:", conf_numpy.detach().numpy())
    print("Difference:", (conf_openmax - conf_numpy).detach().numpy())
