import torch

from confidence.base_confidence import ConfidenceModule


class EntropyConfidence(ConfidenceModule):
    """
    This class computes confidence scores as the negative entropy of the softmax probabilities.
    """

    def __init__(self, input_logits=False):
        super(EntropyConfidence, self).__init__()
        self.input_logits = input_logits  # If True, input is logits, otherwise probabilities

    def forward(self, logits, y=None):
        """
        Computes the negative entropy of the softmax probabilities as confidence scores.
        The formula is: -sum(p * log(p+1e-10))

        Args:
            logits: Logits output from a model. Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            entropy: The negative entropy of the softmax probabilities.
                Shape: (*batch_dims)
        """
        # Compute softmax probabilities
        if self.input_logits:
            probs = torch.nn.functional.softmax(logits, dim=-1)
        else:
            probs = logits
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return -entropy


class MaximumSoftmaxConfidence(ConfidenceModule):
    """
    This class takes the maximum softmax probability over the classes as the confidence score.
    """

    def __init__(self, input_logits=False):
        super(MaximumSoftmaxConfidence, self).__init__()
        self.input_logits = input_logits  # If True, input is logits, otherwise probabilities

    def forward(self, logits, y=None):
        """
        Takes the highest class probability as the confidence score.

        Args:
            logits: Logits output from a model. Shape: (*batch_dims, num_classes)
            y: Unused

        Returns:
            max_softmax_confidence: The maximum softmax probability over the classes.
                Shape: (*batch_dims)
        """
        # Compute softmax probabilities only if inputs are logits
        if self.input_logits:
            probs = torch.nn.functional.softmax(logits, dim=-1)
        else:
            probs = logits
        # Compute maximum softmax confidence
        max_softmax_confidence = torch.max(probs, dim=-1)[0]
        return max_softmax_confidence


class DifferentiableMaximumSoftmaxConfidence(ConfidenceModule):
    """
    This class computes confidence scores using a differentiable version of the maximum softmax confidence.

    Formula:
        confidence = exp((1/τ)·log(∑_{i} e^{τ·logits_i}) − log(∑_{j} e^{logits_j}))
    """

    def __init__(self, tau=1.0, input_logits=False):
        """
        Initializes the DifferentiableMaximumSoftmaxConfidence class.

        Args:
            tau: Temperature parameter for the softmax function. Default is 1.0.
            input_logits: If True, input is logits, otherwise probabilities.

        Formula:
            confidence = exp((1/τ)·log(∑_{i} e^{τ·logits_i}) − log(∑_{j} e^{logits_j}))
        """
        super(DifferentiableMaximumSoftmaxConfidence, self).__init__()
        self.tau = tau
        self.input_logits = input_logits  # If True, input is logits, otherwise probabilities

    def forward(self, logits, y=None):
        """
        Computes confidence scores using a differentiable maximum-softmax formula.

        Steps:
            1. log_sum_exp_tau = log(∑_{i} e^{τ·logits_i})
            2. log_sum_exp_normal = log(∑_{j} e^{logits_j})
            3. confidence = exp((log_sum_exp_tau / τ) − log_sum_exp_normal)

        Args:
            logits: Logits output from a model. Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            confidence: Confidence scores based on the differentiable max-softmax.
                Shape: (*batch_dims)
        """
        if not self.input_logits:
            # compute weights ∝ p^τ
            probs = logits
            weights = probs.pow(self.tau)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
            # smooth max‐probability
            return (weights * probs).sum(dim=-1)
        log_sum_exp_tau = torch.logsumexp(self.tau * logits, dim=-1)
        log_sum_exp_normal = torch.logsumexp(logits, dim=-1)
        confidence = torch.exp((log_sum_exp_tau / self.tau) - log_sum_exp_normal)
        return confidence


class GeneralizedEntropyConfidence(ConfidenceModule):
    """
    Implements the generalized (adjusted) entropy:

        E(x) = - sum_{i=1}^m p_i^λ (1 - p_i)^λ

    where p_i are the top-m predicted probabilities (sorted descending).
    By convention in this codebase, confidence values increase when uncertainty decreases,
    so this module returns -E(x) (i.e., negative of the above quantity).
    #https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GEN_Pushing_the_Limits_of_Softmax-Based_Out-of-Distribution_Detection_CVPR_2023_paper.pdf
    Args:
        lmbda: λ hyperparameter (float). Default 1.0.
        m: number of top classes to consider. If None, use all classes.
        input_logits: if True, inputs are logits and will be softmaxed.
    """

    def __init__(self, lmbda: float = 1.0, m: int = None, input_logits: bool = False):
        super(GeneralizedEntropyConfidence, self).__init__()
        self.lmbda = float(lmbda)
        self.m = None if m is None else int(m)
        self.input_logits = input_logits

    def forward(self, logits, y=None):
        """
        Args:
            logits: Logits or probabilities. Shape: (*batch_dims, num_classes)
            y: Optional labels for modules that use them.

        Returns:
            Generalized entropy scores.
        """
        # logits can be logits or probabilities depending on input_logits flag
        if self.input_logits:
            probs = torch.nn.functional.softmax(logits, dim=-1)
        else:
            probs = logits

        # determine m (number of top classes to use)
        num_classes = probs.shape[-1]
        m = num_classes if (self.m is None or self.m <= 0) else min(self.m, num_classes)

        # select top-m probabilities
        if m == num_classes:
            topk = probs
        else:
            topk = probs.topk(k=m, dim=-1).values  # shape (..., m)

        # compute sum_{i=1}^m p_i^λ (1 - p_i)^λ
        # clamp to ensure numerical stability
        eps = 1e-12
        p = topk.clamp(min=eps, max=1.0 - eps)
        term = (p.pow(self.lmbda) * (1.0 - p).pow(self.lmbda)).sum(dim=-1)

        # generalized entropy E(x) = -log(term)
        gen_entropy = -term

        return gen_entropy


class CombinedEntropyMultiSampleConfidence(ConfidenceModule):
    """
    Combines Entropy confidence on the mean prediction with a multi-sample confidence criterion.
    The input is expected to be a tensor of shape [batch, samples, classes] of probabilities.
    """

    def __init__(self, multi_sample_confidence: ConfidenceModule, alpha: float = 0.5, input_logits=False):
        """
        Args:
            multi_sample_confidence: A confidence module that operates on multiple samples (e.g., MutualInformationCriterion).
            alpha: The weight for the entropy-based confidence. The multi-sample confidence will be weighted by (1 - alpha).
            input_logits: If True, input is logits, otherwise probabilities.
        """
        super().__init__()
        if input_logits:
            raise NotImplementedError(
                "input_logits=True is not supported for CombinedEntropyMultiSampleConfidence yet.")
        self.multi_sample_confidence = multi_sample_confidence
        self.entropy_confidence = EntropyConfidence(input_logits=False)
        self.alpha = alpha

    def forward(self, outputs, y=None):
        """
        Computes the combined confidence score.

        Args:
            outputs: Tensor of shape [batch, samples, classes] representing probabilities from multiple samples.
            y: Optional labels.

        Returns:
            A combined confidence score.
        """
        # Confidence from the multi-sample criterion
        multi_sample_conf = self.multi_sample_confidence(outputs, y)

        # Mean prediction across samples
        mean_probs = outputs.mean(dim=-2)

        # Entropy-based confidence on the mean prediction
        entropy_conf = self.entropy_confidence(mean_probs, y)

        # Combined confidence
        return self.alpha * entropy_conf + (1.0 - self.alpha) * multi_sample_conf
