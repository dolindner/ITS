from typing import Optional, TypeVar
import torch
from torch import nn, Tensor

from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase

Self = TypeVar("Self")

class SHETorchConfidence(ClassicConfidenceBase):
    """
    Simplified Hopfield Energy (feature-only). Adapted from pytorch_ood.detector.SHE.
    Fit per‐class mean patterns, then score via -⟨z, pattern_y⟩.
    """

    def __init__(self, input_transform: Optional[nn.Module] = None,map_function: Optional[callable] = None):

        super().__init__(input_transform=input_transform)
        self.patterns_: Optional[Tensor] = None
        self.fitted = False
        self.map_function = map_function if map_function is not None else lambda x: -x
        #print("For correct functionality one must prefilter embeddings to only keep correctly classified ones.")
        self.cosine_debug = False  # debug flag for cosine similarity

    def _fit(self, z: Tensor, y: Tensor) -> Self:
        """
        Compute per-class mean patterns and mark the module as fitted.

        Args:
            z: Feature tensor of shape [N, D].
            y: Label tensor of shape [N].

        Returns:
            Self: self
        """
        # classes must be 0..C-1
        classes = torch.unique(y)
        assert len(classes) == classes.max().item() + 1, "labels must cover 0..C-1"
        # compute per-class mean feature
        patterns = [z[y == c].mean(dim=0) for c in classes]
        self.patterns_ = torch.stack(patterns, dim=0)
        self.fitted = True
        return self

    def _forward(self, z: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Score inputs using stored per-class patterns.

        Requires predicted class labels y at inference; raises if y is None.
        The returned score is mapped via map_function for final confidence-like output.

        Args:
            z: Feature tensor [B, D].
            y: Label tensor [B] indicating predicted class for each sample.

        Returns:
            Tensor: mapped scores per sample (shape [B]).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")
        if y is None:
            raise ValueError("Predicted Class labels 'y' required at inference")

        # compute only true‐class scores
        patterns_y = self.patterns_[y]  # (batch_size, feature_dim)
        if not self.cosine_debug:
            true_scores = (z * patterns_y).sum(dim=1)  # (batch_size,)
        else:
            z_norm = z / z.norm(dim=1, keepdim=True)
            patterns_y_norm = patterns_y / patterns_y.norm(dim=1, keepdim=True)
            cosine_distance = 1-(z_norm * patterns_y_norm).sum(dim=1)
            true_scores = -cosine_distance  # (batch_size,)

        return self.map_function(-true_scores)



if __name__ == "__main__":
    import torch
    from torch import nn, Tensor
    from torch.utils.data import DataLoader, TensorDataset

    from pytorch_ood.detector import SHE

    # --- Dummy data ---
    num_classes = 3
    feature_dim = 5
    num_samples = 100
    torch.manual_seed(42)

    X = torch.randn(num_samples, feature_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    # --- Dummy backbone and head ---
    class DummyBackbone(nn.Module):
        def forward(self, x):
            return x

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(feature_dim, num_classes)
        def forward(self, z):
            return self.linear(z)

    backbone = DummyBackbone()
    head = DummyHead()

    #Fit SHETorchConfidence
    # Use a dummy classifier to generate predicted labels
    logits = head(X)
    y_hat = logits.argmax(dim=1)

    she_torch = SHETorchConfidence(map_function=lambda x: x)
    she_torch._fit(X, y_hat)  # fit patterns using predicted labels
    scores_torch = she_torch._forward(X, y_hat)  # forward with predicted labels

    #Fit original SHE
    she_orig = SHE(backbone=backbone, head=head)
    # For demonstration, we compute patterns using predicted classes
    y_hat_orig = head(X).argmax(dim=1)
    she_orig.patterns = torch.stack([X[y_hat_orig == c].mean(dim=0) for c in range(num_classes)])
    scores_orig = she_orig.predict_features(X)

    #Compare
    print("SHETorchConfidence scores (predicted classes):", scores_torch)
    print("SHE original scores:", scores_orig)

    diff = scores_torch - scores_orig
    print("Difference:", diff)
    print("Correlation:", torch.corrcoef(torch.stack([scores_torch, scores_orig]))[0, 1])