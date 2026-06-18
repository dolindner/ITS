import torch

from confidence.base_confidence import ConfidenceModule
from confidence.direct.logit_based import EnergyConfidence
from confidence.model.base_model import ModelBasedConfidence


class ASHConfidence(ModelBasedConfidence):
    """
    Wraps a backbone+head and applies ASH transform.

    Args:
        backbone: Feature extractor module.
        head: Final classification head module.
        variant: ASH variant string (e.g. "ash-s").
        percentile: Fraction for ASH trimming.
        index_feat: Optional index selecting feature to transform.
        index_logits: Optional index selecting logits for confidence.
        confidence: ConfidenceModule used after head.
        use_feature_confidence: If True compute confidence from features instead of logits.

    Returns:
        Tuple (confidences, selected_logits)
    """

    def __init__(
            self,
            backbone: torch.nn.Module,
            head: torch.nn.Module,
            variant: str = "ash-s",
            percentile: float = 0.90,
            index_feat: int = None,
            index_logits: int = None,
            confidence: ConfidenceModule = EnergyConfidence(),
            use_feature_confidence: bool = False,
            # calculates distance on ash features before passing them through the head
    ):
        # confidence and index for final logits
        super().__init__(head, confidence, index_logits)
        self.backbone = backbone
        self.head = head
        # ASH transform only; do not set ASH's detector
        self.ash = ASH(backbone, head, variant, percentile)
        self.index_feat = index_feat
        self.use_feature_confidence = use_feature_confidence

    def forward(self, x: torch.Tensor, y=None):
        """
        Forward pass applying ASH on selected features and computing confidence.

        Args:
            x: Input tensor (batch, ...).
            y: Optional labels passed to the confidence module.

        Returns:
            Tuple (confidences, selected_logits)
        """
        # 1) backbone
        all_feat = self.backbone(x)
        # 2) select which feat to use for ASH
        if self.index_feat is not None:
            feats = all_feat[self.index_feat]
        else:
            feats = all_feat

        added_dims = False
        if feats.dim() != 4:
            feats_for_ash = feats.unsqueeze(-1).unsqueeze(-1)
            added_dims = True
        else:
            feats_for_ash = feats
        # ------------------------------------------------------------------------------
        feats_for_ash = feats_for_ash.clone()

        # 3) ASH transform
        ash_feat_trans = self.ash.ash(feats_for_ash, self.ash.percentile)

        # If extra spatial dims were inserted, flatten back to (b, dim) to feed head/confidence expecting vectors.
        if added_dims:
            ash_feat = ash_feat_trans.view(ash_feat_trans.shape[0], -1)
        else:
            ash_feat = ash_feat_trans

        if self.index_feat is not None:
            all_feat = list(all_feat)  # ensure mutable
            all_feat[self.index_feat] = ash_feat
        else:
            all_feat = ash_feat

        # 4) confidences
        if self.use_feature_confidence:  # not tested
            confidences, logits = self.confidence(ash_feat, y)
        else:
            # TODO: fix potential assumptions about head input shape
            logits = self.head(ash_feat)
            confidences = self.confidence(logits, y)

        # select logits if index_logits set
        output = logits if self.index is None else all_feat[self.index]
        return confidences, output


import torch
from math import floor, ceil
from confidence.direct.logit_based import EnergyConfidence
from confidence.model.base_model import ModelBasedConfidence
from confidence.base_confidence import ConfidenceModule


def torch_quantile(  # noqa: PLR0913 (too many arguments)
        tensor: torch.Tensor,
        q: float | torch.Tensor,
        dim: int | None = None,
        *,
        keepdim: bool = False,
        interpolation: str = "linear",
        out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Improved ``torch.quantile`` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.

    References
    ----------
    Taken from the PyTorch issue discussion:
    "torch.quantile is slow and has a 2^24 element limit"
    (https://github.com/pytorch/pytorch/issues/157431#issuecomment-3026856373).
    BSD 3-Clause from pytorch.
    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)


class ReActConfidence(ModelBasedConfidence):
    """
    Wraps a backbone+head with ReAct activation clipping.
    #https://arxiv.org/abs/2111.12797

    Returns (confidences, selected_logits).
    """

    def __init__(
            self,
            backbone: torch.nn.Module,
            head: torch.nn.Module,
            percentile: float = 0.9,
            index_feat: int = None,
            index_logits: int = None,
            confidence: ConfidenceModule = EnergyConfidence(),
            use_feature_confidence: bool = False,
            threshold: float = None,
    ):
        super().__init__(head, confidence, index_logits)
        self.backbone = backbone
        self.percentile = percentile
        self.index_feat = index_feat
        self.head = head
        self.threshold = threshold  # to be computed in fit() or passed directly
        self.use_feature_confidence = use_feature_confidence

    def fit(self, features: torch.Tensor):
        """
        Calculates the clipping threshold based on the given percentile of activations.
        """
        # note that quantile is heavy operation; consider using a subset of features
        if self.threshold is not None:
            return  # Do not re-compute if already set

        self.threshold = torch_quantile(features, self.percentile)

    def forward(self, x: torch.Tensor, y=None):
        if self.threshold is None:
            raise RuntimeError("ReActConfidence has not been fitted. Call .fit(features) first.")

        # 1) backbone
        all_feat = self.backbone(x)

        # Ensure all_feat is a mutable list if it's a tuple
        is_tuple = isinstance(all_feat, tuple)
        if is_tuple:
            all_feat = list(all_feat)

        # 2) select which feat to clip
        if self.index_feat is not None:
            feats = all_feat[self.index_feat]
        else:
            feats = all_feat

        # 3) clip activations
        self.threshold = self.threshold.to(feats.device)
        react_feat = feats.clamp(max=self.threshold)

        # 4) logits
        if self.index_feat is not None:
            # Pass modified feature list to head
            all_feat[self.index_feat] = react_feat
            head_input = tuple(all_feat) if is_tuple else all_feat
        else:
            # Pass clipped features directly to head
            head_input = react_feat

        # 5) confidences
        if self.use_feature_confidence:
            confidences, logits = self.confidence(react_feat, y)
        else:
            logits = self.head(react_feat)
            confidences = self.confidence(logits, y)

        # select logits if index_logits set
        index_logit_overide = self.index

        output = logits if index_logit_overide is None else head_input[
            index_logit_overide]  # either take the rectifeid output or assume that the module also outputs its logits
        return confidences, output


# compare against ASH from pytorch OOD. #TODO: move react to separate file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from confidence.direct.logit_based import EnergyConfidence

from pytorch_ood.detector import ASH  # for reference/comparison


# -------------------------------
# Dummy dataset for testing
# -------------------------------
def create_dummy_data(num_samples=1000, num_features=10, num_classes=5):
    X = torch.randn(num_samples, num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


# -------------------------------
# Dummy model backbone and head
# -------------------------------
class SimpleBackbone(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return x  # features


class SimpleHead(nn.Module):
    def __init__(self, feature_dim=20, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)  # logits


from pytorch_ood.detector import ReAct


# -------------------------------
# Main comparison
# -------------------------------
def main():
    # Data
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Models
    backbone = SimpleBackbone()
    head = SimpleHead()

    # Confidence modules
    energy_conf = EnergyConfidence()

    react_conf = ReActConfidence(backbone, head, confidence=energy_conf, percentile=0.92)

    # Fit ReAct (needed for threshold)
    all_features = backbone(X)
    react_conf.fit(all_features)

    react_ood = ReAct(backbone, head, threshold=react_conf.threshold.item(), detector=energy_conf)

    # Evaluation loop
    for batch_X, batch_y in dataloader:
        # ASH
        # ReAct
        react_confidences, react_logits = react_conf(batch_X, batch_y)
        react_ood_conf = react_ood.predict(batch_X)

        print("ReAct confidences:", react_confidences[:5])
        print("ReAct logits shape:", react_logits.shape)
        print("ReAct OOD scores:", react_ood_conf[:5])
        print("---")
        mean_dif = torch.abs(react_confidences - react_ood_conf).mean().item()
        print(f"Mean absolute difference between ReActConfidence and ReAct OOD scores: {mean_dif:.6f}")
        break  # just first batch for demo


if __name__ == "__main__":
    main()
