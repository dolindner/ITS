import torch


from torch import Tensor
from pytorch_ood.utils import extract_features, is_known
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from confidence.input_transform import InputTransform
from pytorch_ood.detector import DICE
from confidence.base_confidence import ConfidenceModule
from confidence.direct.logit_based import EnergyConfidence
from model.basic_networks import find_last_linear_layer

#TODO Note DICE uses a newer version as was used during the paper. See if things changed. Maybe remove or rerun.
class DICEConfidence(ClassicConfidenceBase):
    """
    Feature-based confidence for DICE with passable score-to-confidence mapping.

    Args:
        model: The full model (used to find last linear layer weights).
        percentile: Percentile parameter for DICE detector.
        input_transform: Optional input transform wrapper.
        map_function: Optional mapping from raw score to confidence.
        confidence: Detector/confidence module to apply on DICE outputs.

    Returns:
        A configured DICEConfidence instance when constructed.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            percentile: float,
            input_transform: InputTransform = None,
            map_function=None,
            confidence: ConfidenceModule = EnergyConfidence()
    ):
        super().__init__(input_transform)
        linear = find_last_linear_layer(model)
        w = linear.weight.detach().cpu()
        b = linear.bias.detach().cpu()
        self.dice = DICE(encoder=None, w=w, b=b, p=percentile, detector=None)
        self.dice.detector = confidence  # Set detector to the confidence module
        self.map_fn = map_function or (lambda score: score)
        self.confidence = confidence
        self.linear = linear  # Store the linear layer for feature extraction

    def _fit(self, X: Tensor, y: Tensor) -> "DICEConfidence":
        """
        Fit DICE on features.

        Args:
            X: Feature tensor (N x D).
            y: Labels tensor (N,).

        Returns:
            self after fitting.
        """
        feats = X.detach().cpu()
        labs = y.detach().cpu()
        keep = is_known(labs)
        if not keep.any():
            raise ValueError("No in-distribution data for DICEConfidence")
        self.dice.fit_features(feats[keep], labs[keep])
        return self

    def _predict_features_with_grad(self, x: Tensor) -> Tensor:
        """
        Predict raw DICE detector scores using features and preserved gradients.

        Args:
            x: Feature tensor (N x D).

        Returns:
            Raw detector scores tensor.
        """
        vote = x.unsqueeze(1) * self.dice.masked_w.to(x.device)
        output = vote.sum(2) + self.dice.bias.to(x.device)

        score = self.dice.detector(output)
        return score

    def _forward(self, X: Tensor, y: Tensor = None) -> Tensor:
        """
        Forward mapping from features to confidence.

        Args:
            X: Feature tensor.
            y: Optional labels (unused).

        Returns:
            Confidence values tensor matching X device and dtype.
        """
        feats = X

        scores = self._predict_features_with_grad(feats)

        conf = self.map_fn(scores)
        return conf.to(X.device, X.dtype)