import torch

from confidence.base_confidence import ConfidenceModule


class CrossEntropyConfidence(ConfidenceModule):
    """
    Computes confidence as the negative cross‐entropy loss between logits and labels.
    Requires y to be provided.
    Basically cheating but can be used for finding points of high confidence in the correct class.
    Mainly for debug and comparison purposes.
    """

    def __init__(self):
        super(CrossEntropyConfidence, self).__init__()

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (*batch_dims, num_classes)
            y: (*batch_dims,) integer class labels

        Returns:
            confidence: (*batch_dims,) = -cross_entropy(logits, y)
        """
        if y is None:
            raise ValueError("CrossEntropyConfidence requires target labels y.")
        loss = torch.nn.functional.cross_entropy(logits, y, reduction='none')
        return -loss