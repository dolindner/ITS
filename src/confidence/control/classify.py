import torch

from confidence.base_confidence import ConfidenceModule


class ClassifyingConfidence(ConfidenceModule):
    """
    Calculates predicted class and passes both the x and the predicted class to another confidence module.
    """

    def __init__(self,confidence: torch.nn.Module = None, index=None,index_confidence=None):
        """
        Initializes the ClassifyingConfidence.

        Args:
            confidence: Confidence module used to calculate confidence scores.
            index: Index of the input x that specifies the logits for class prediction.
                   If None, the entire input x is used for class prediction.
            index_confidence: Index of the input x that specifies the input for the confidence module.
                              If None, the entire input x is used for the confidence module.

        Returns:
            None
        """
        super(ClassifyingConfidence, self).__init__()
        self.index = index
        self.confidence = confidence
        self.index_confidence = index_confidence

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Computes confidence scores by calling a confidence that requires a class assignment.
        The input or an element of the input is interpreted as class logits to calculate a predicted class to guide
        the confidence calculation.

        Args:
            x: Input tensor for the model. Shape: (*batch_dims, *data_dims)
            y: Optional labels for modules that use them.

        Returns:
            confidence_scores: Confidence scores computed by the classification model. Shape: (*batch_dims)
        """
        log = x[self.index] if self.index is not None else x
        clas = torch.argmax(log, dim=-1)
        inp2 = x[self.index_confidence] if self.index_confidence is not None else x
        confidence_scores = self.confidence(inp2, clas)
        return confidence_scores