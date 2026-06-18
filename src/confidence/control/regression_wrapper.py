import torch
import torch.nn as nn

from confidence.control.regression import RegressionConfidence


class RegressionWrapper(nn.Module):
    """
    A wrapper that combines a multi-output model wrapper with RegressionConfidence.
    It ensures the model is called only once to extract all features, which are then
    passed to RegressionConfidence for aggregation.
    """

    def __init__(self, model_wrapper: nn.Module, regression_confidence: RegressionConfidence):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.regression_confidence = regression_confidence

    def forward(self, x, y=None):
        # The model_wrapper extracts all features and the final output in one pass.
        features = self.model_wrapper(x)
        # Pass the pre-computed features to RegressionConfidence
        return self.regression_confidence(features, y), None

    @torch.no_grad()
    def fit(self, in_loader, out_loader):
        """Fits the underlying RegressionConfidence module."""
        # The fit method of RegressionConfidence needs the model_wrapper to collect features.
        self.regression_confidence.fit(in_loader, out_loader, model_wrapper=self.model_wrapper)
        return self

    @torch.no_grad()
    def evaluate(self, in_loader, out_loader):
        return self.regression_confidence.evaluate(in_loader, out_loader, model_wrapper=self.model_wrapper)
