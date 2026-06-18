import abc

import torch


class ModelBasedConfidence(torch.nn.Module, abc.ABC):
    """Base class for model-based confidence estimation.
    These call a model and pass the output to a confidence module.

    This class is an abstract base class. Subclasses should implement the forward method.
    """

    def __init__(self, model: torch.nn.Module, confidence: torch.nn.Module, index=None):
        """Initializes the ModelBasedConfidence class.

        Args:
            model: The model whose output is used to compute confidence scores.
            confidence: A torch.nn.Module that computes the confidence scores.
            index: Index of the output to use for confidence computation.
                If None, use the whole output.
        """
        super(ModelBasedConfidence, self).__init__()
        self.model = model #this is a classifier model that outputs logits
        self.confidence = confidence
        self.index = index

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> tuple:
        """Abstract method to be implemented by subclasses.

        Args:
            x: Input tensor for the model. Shape: (*batch_dims, *data_dims)
            y: Optional labels for modules that use them.

        Returns:
            A tuple (confidence_scores, model_output).
            confidence_scores: Confidence scores computed by the confidence module. Shape: (*batch_dims)
            model_output: Output of the model. Shape: (*batch_dims, ...)
        """
        pass