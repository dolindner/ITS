import torch

from confidence.base_confidence import ConfidenceModule


class EnergyConfidence(ConfidenceModule):
    """
    This class computes confidence scores as the energy of the logits.

    The energy score for a single example is:

        energy_scores = logsumexp(logits) = log(∑_{i} e^{logits_i})

    From paper Energy-based Out-of-distribution Detection:
            `NeurIPS <https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf>`
    """

    def __init__(self, mean=False, t=1.0):
        super(EnergyConfidence, self).__init__()
        self.mean = mean  # If True, return mean energy score across batch
        self.t = t  # Temperature parameter for scaling logits, default is 1.0

    def forward(self, logits, y=None):
        """
        Computes confidence scores based on the energy of the logits.

        Formula:
            energy_scores = log(∑_{i} e^{logits_i})

        Args:
            logits: Logits output from a model. Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            energy_scores: Confidence scores based on the energy of the logits. Shape: (*batch_dims)
        """
        energy_scores = self.t * torch.logsumexp(logits / self.t, dim=-1)
        if self.mean:
            # If mean is True, return the mean energy score across the batch
            return energy_scores.mean(dim=-1)
        return energy_scores


class MaxLogitConfidence(ConfidenceModule):
    """
    This class computes confidence scores as the maximum logit value.

    The maximum logit score for a single example is:

        max_logit_scores = max(logits)
    """

    def __init__(self):
        super(MaxLogitConfidence, self).__init__()

    def forward(self, logits, y=None):
        """
        Computes confidence scores based on the maximum logit value.

        Args:
            logits: Logits output from a model. Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            max_logit_scores: Confidence scores based on the maximum logit value. Shape: (*batch_dims)
        """
        max_logit_scores = torch.max(logits, dim=-1).values
        return max_logit_scores


class CombinedEnergyMultiSampleConfidence(ConfidenceModule):
    """
    Combine an energy-based confidence (computed on mean logits) with a multi-sample
    confidence criterion (computed from per-sample probabilities). For use with bayesian
    or multi sample methods.

    Behavior:
      - Passes the scores directly to a confidence module taking multiple samples.
      - Computes mean logits across samples -> passes mean logits to EnergyConfidence.
      - Returns: alpha * energy_conf + (1-alpha) * multi_sample_conf
    """

    def __init__(self, multi_sample_confidence: ConfidenceModule, alpha: float = 0.5, energy_t: float = 1.0):
        super().__init__()
        self.multi_sample_confidence = multi_sample_confidence
        self.alpha = float(alpha)
        self.energy_confidence = EnergyConfidence(mean=False, t=energy_t)

    def forward(self, mc_logits, y=None):
        """
        Args:
            mc_logits: Logits tensor of shape [batch, samples, classes]
            y: Optional labels for modules that use them.

        Returns:
            Combined confidence scores.
        """
        if mc_logits.dim() < 3:
            raise ValueError(
                "Expected MC logits with shape [batch, samples, classes], got shape: " + str(mc_logits.shape))

        # compute per-sample probabilities for the multi-sample criterion
        multi_sample_conf = self.multi_sample_confidence(mc_logits, y)

        # compute energy on mean logits across samples
        mean_logits = mc_logits.mean(dim=1)  # shape [batch, classes]
        energy_conf = self.energy_confidence(mean_logits, y)

        # combine (alpha weights energy term)
        return self.alpha * energy_conf + (1.0 - self.alpha) * multi_sample_conf
