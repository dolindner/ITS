#TODO adjust to make similar to other confidence modules

import torch
import torch.nn.functional as F

from confidence.base_confidence import ConfidenceModule

class MutualInformationCriterion(ConfidenceModule):
    """
    Computes the mutual information between the mean and individual predictions.
    Higher mutual information indicates more confidence in the predictions.
    Returns 1-MI so that the score is an inlier score (high for inliers).
    """
    def __init__(self, input_logits=False):
        super().__init__()
        self.input_logits = input_logits

    def forward(self, outputs, y=None):
        """
        Args:
            outputs: Tensor of shape [batch, samples, classes]
            y: Optional labels for modules that use them.

        Returns:
            Normalized mutual information scores.
        """
        if self.input_logits:
            outputs = F.softmax(outputs, dim=-1)
        p = outputs
        p_mean = p.mean(dim=-2)
        h_mean = -torch.sum(p_mean * torch.log(p_mean + 1e-12), dim=-1)
        h_per = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
        h_ind = h_per.mean(dim=-1)
        mi = h_mean - h_ind
        max_mi = torch.log(torch.tensor(outputs.size(-1), device=mi.device))
        mi_norm = mi / (max_mi + 1e-12)
        return 1.0 - mi_norm