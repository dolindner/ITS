import torch
import warnings
import torch
import torch_geometric
from torch import nn, Tensor
from typing import Optional
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from confidence.input_transform import InputTransform
from pytorch_ood.detector import ViM


from torch import nn



def find_last_linear_layer(model):
    """
    Find the last linear layer in a model (supports torch_geometric.nn.Linear).

    Args:
        model: nn.Module to inspect.

    Returns:
        The last nn.Linear or torch_geometric.nn.Linear instance found, or None.
    """
    last_linear = None
    for module in model.modules():
        if isinstance(module,(nn.Linear, torch_geometric.nn.Linear)):
            last_linear = module
    return last_linear


import torch
from torch import nn, Tensor
from typing import Optional
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from confidence.input_transform import InputTransform
#based on pytorch_ood
class ViMTorchConfidence(ClassicConfidenceBase):
    def __init__(
        self,
        model: nn.Module,
        n_dim: int,
        input_transform: Optional[InputTransform] = None,
        use_energy: bool = True,
    ):
        if input_transform is not None:
            warnings.warn(
                "Input transforms_old are not compatible with ViMTorchConfidence and have been disabled"
            )
        super().__init__(input_transform=None)
        linear = find_last_linear_layer(model)
        self.w = linear.weight.detach()
        self.b = linear.bias.detach()
        self.n_dim = n_dim
        self.use_energy = use_energy
        self.u = -torch.pinverse(self.w) @ self.b
        self.principal_subspace: Optional[Tensor] = None
        self.alpha: Optional[float] = None
        self.fitted = False

    def _fit(self, X: Tensor, y: Tensor) -> "ViMTorchConfidence":
        """
        Fit ViM internals: compute principal subspace and alpha scaling.

        Very important: computes principal subspace and alpha used in scoring.

        Args:
            X: Feature tensor [N, D].
            y: Label tensor [N].

        Returns:
            ViMTorchConfidence: self
        """
        N, _ = X.shape
        logits = X @ self.w.T + self.b
        Xc = X - self.u.unsqueeze(0)
        cov = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        idx = torch.argsort(eigvals, descending=True)[self.n_dim:]
        self.principal_subspace = eigvecs[:, idx]
        resid = (Xc @ self.principal_subspace).norm(dim=1)
        max_logit = logits.max(dim=1).values
        self.alpha = (max_logit.mean() / resid.mean()).item()
        self.fitted = True
        return self

    def _forward(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Score inputs using ViM scoring rule.

        Args:
            X: Feature tensor [B, D].
            y: Optional labels (unused).

        Returns:
            Tensor: score per input (shape [B]).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")

        logits = X @ self.w.T + self.b
        Xc = X - self.u.unsqueeze(0)
        resid = (Xc @ self.principal_subspace).norm(dim=1)
        vlogit = resid * self.alpha

        if self.use_energy:
            energy = torch.logsumexp(logits, dim=1)
            score = -vlogit + energy
        else:
            score = -vlogit

        return score


#Comparision Module from pytorch_ood.
class ViMConfidenceNumpy(ClassicConfidenceBase):
    def __init__(
        self,
        model: nn.Module,
        n_dim: int,
        input_transform: Optional[InputTransform] = None,
    ):
        if input_transform is not None:
            warnings.warn(
                "Input transforms_old are not compatible with ViMConfidenceNumpy and have been disabled"
            )
        super().__init__(input_transform=None)
        linear = find_last_linear_layer(model)
        w_np = linear.weight.detach()
        b_np = linear.bias.detach()
        self.vim = ViM(None, n_dim, w_np, b_np)

    def _fit(self, X: Tensor, y: Tensor) -> "ViMConfidenceNumpy":
        """
        Fit the wrapped ViM Pyotorch OOD implementation on provided features.

        Args:
            X: Feature tensor [N, D].
            y: Label tensor [N].

        Returns:
            ViMConfidenceNumpy: self
        """
        feats = X.detach()
        labs = y.detach()
        self.vim.fit_features(feats, labs)
        return self

    def _forward(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Predict ViM scores via the wrapped  Pyotorch OOD implementation.

        Args:
            X: Feature tensor [B, D].
            y: Optional labels (unused).

        Returns:
            Tensor: negative ViM scores moved to input device/dtype.
        """
        feats = X
        scores = self.vim.predict_features(feats)
        return -scores.to(X.device, X.dtype)


import warnings


if __name__ == '__main__':
    # Synthetic data
    N, D, C = 100, 16, 4
    X = torch.randn(N, D)
    y = torch.randint(0, C, (N,))

    # Simple linear model
    model = nn.Sequential(nn.Linear(D, C))

    # NumPy-wrapper ViM
    vim_np = ViMConfidenceNumpy(model=model, n_dim=8)
    vim_np.fit(X, y)
    scores_np = vim_np(X)

    # Pure-PyTorch ViM
    vim_t = ViMTorchConfidence(model=model, n_dim=8)
    vim_t.fit(X, y)
    scores_t = vim_t(X)
    #print alpha
    print("Alpha (ViM):", vim_t.alpha)
    #print principal subspace
    print("Principal subspace (ViM):", vim_t.principal_subspace)

    # Compare outputs
    print("ViM scores match:", torch.allclose(scores_np, scores_t, atol=1e-6))