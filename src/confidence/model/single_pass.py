from typing import Optional, Union

import torch

from confidence.model.base_model import ModelBasedConfidence


class SinglePassConfidence(ModelBasedConfidence):
    """
    This class calls a model a single time and passes the output to a confidence module without doing any modifications.
    The index parameter is used to select a specific output from the model's output. If index is None,
    the whole output is used for confidence computation.
    """

    def __init__(self, model, confidence, index=None, input_dim=None):
        """ Initializes the SinglePassConfidence class. This class computes
        confidence scores using a single forward pass of the model by calling
        the model and passing the output to a confidence module.

        Args:
            model: The model whose output is used to compute confidence scores.
            confidence: A torch.nn.Module that computes the confidence scores.
            index: Index of the output to use for confidence computation.
                If None, use the whole output.
        """
        super(SinglePassConfidence, self).__init__(model, confidence, index)
        self.input_dim = input_dim

    def forward(self, x, y=None):
        """
        Calculates a single forward pass of a model and computes confidence scores using the provided confidence module.
        Args:
            x:  Input tensor for the model. Shape: (*batch_dims, *data_dims)
            y: Optional labels for modules that use them.

        Returns:
            A tuple (confidence_scores, model_output).
            confidence_scores: Confidence scores computed by the confidence module. Shape: (*batch_dims)
            model_output: Output of the model. Shape: (*batch_dims, ...) if index is None, else same as output[self.index]
        """

        output = self.model(x)

        confidence_scores = self.confidence(output, y)
        if self.index is None:
            return confidence_scores, output
        else:
            return confidence_scores, output[self.index]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self.confidence, 'to'):
            self.confidence.to(*args, **kwargs)
        return self

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        if hasattr(self.confidence, 'cuda'):
            self.confidence.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        if hasattr(self.confidence, 'cpu'):
            self.confidence.cpu()
        return self
