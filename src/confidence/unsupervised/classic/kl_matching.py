import torch
from typing import Optional, Union

from torch import device
from torch.nn import Module
from torch.nn.modules.module import T

from confidence.input_transform import InputTransform
from confidence.unsupervised.unsupervised_base import ClassicConfidenceBase
from pytorch_ood.detector.klmatching import KLMatching

class KLMatchingConfidence(ClassicConfidenceBase):
    """
    Wraps pytorch_ood.detector.KLMatching into a ClassicConfidenceBase.
    Expects x to be model logits

    Args:
        model: Optional model used by KLMatching.
        input_transform: Optional InputTransform.
        map_function: Optional mapping applied to detector scores.

    Returns:
        A KLMatchingConfidence instance.
    """

    def __init__(
        self,
        model: Module=None,
        input_transform: Optional[InputTransform] = None,
        map_function: Optional[callable] = None
    ):
        super().__init__(input_transform=input_transform)
        self.model = model
        self.detector = KLMatching(model)
        self.fitted = False
        self.map_function = map_function or (lambda x: -x)

    def _fit(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> "KLMatchingConfidence":
        """
        Fit detector using logits and labels.

        Args:
            X: Logits tensor.
            y: Labels tensor (required).

        Returns:
            self after fitting.
        """
        if y is None:
            raise ValueError("KLMatching requires labels to fit.")
        # X should be logits, fit_features will convert to probabilities internally
        self.detector.fit_features(X, y, device=X.device)
        self.fitted = True
        return self

    def _forward(
        self,
        x: torch.Tensor,
        y=None
    ) -> torch.Tensor:
        """
        Predict mapped confidence from logits.

        Args:
            x: Logits tensor.
            y: Optional labels (unused).

        Returns:
            Mapped detector scores as confidence tensor.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before forward()")
        # x should be logits, convert to probabilities for predict_features
        x = torch.nn.functional.softmax(x, dim=-1)
        scores = self.detector.predict_features(x)
        return self.map_function(scores)


    def to(self, *args, **kwargs) -> "KLMatchingConfidence":
        """
        Move module to device and ensure detector tensors moved.

        Args:
            device: target device as first positional arg or kwargs.

        Returns:
            self after move.
        """
        super().to(*args, **kwargs)
        # device is the first positional argument
        device = args[0] if args else kwargs.get("device")
        for key in self.detector.dists:
            if isinstance(self.detector.dists[key], torch.Tensor):
                self.detector.dists[key] = self.detector.dists[key].to(device)

        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> "KLMatchingConfidence":
        """
        Move module to CUDA device.

        Args:
            device: CUDA device.

        Returns:
            self on CUDA.
        """
        super().cuda(device)
        for key in self.detector.dists:
            if isinstance(self.detector.dists[key], torch.Tensor):
                self.detector.dists[key] = self.detector.dists[key].cuda(device)

        return self

    def cpu(self) -> "KLMatchingConfidence":
        """
        Move module to CPU.

        Returns:
            self on CPU.
        """
        super().cpu()
        for key in self.detector.dists:
            if isinstance(self.detector.dists[key], torch.Tensor):
                self.detector.dists[key] = self.detector.dists[key].cpu()

        return self