import torch

from confidence.base_confidence import ConfidenceModule


class SplitConfidence(ConfidenceModule):
    """
    This class computes confidence scores by combining the outputs of two confidence modules.
    This considers the output of the model as a tuple of (intermediate_output, logits) and forwards
    them to the respective confidence modules. The final confidence score is computed by either multiplying
    or adding the outputs of the two confidence modules, depending on the mult parameter.

    """

    def __init__(self, confidence_inter, confidence_final, mult=True, a=1, b=1, scale_inter=None, scale_final=None,
                 avg_inter=None):
        """
        Initializes the SplitConfidence class.
        Args:
            confidence_inter: The confidence module for the intermediate output.
            confidence_final: The confidence module for the final output.
            mult: If True, the outputs of the two confidence modules are multiplied.
            a: Multiplier for the intermediate confidence module. Ignored if mult is True.
            b: Multiplier for the final confidence module. Ignored if mult is True.
        """
        super(SplitConfidence, self).__init__()
        self.confidence_inter = confidence_inter
        self.confidence_final = confidence_final
        self.mult = mult
        self.a = a
        self.b = b

    def forward(self, x, y=None):
        """
        Computes confidence scores by combining the outputs of two confidence modules.

        Args:
            x: Input tensor for the model. This should be a tuple of (intermediate_output, logits).
            y: Not used for this.

        Returns:
            confidence: Confidence scores. Shape: (*batch_dims)
        """
        intermediate_output, logits = x

        # calculate trust confidence using intermediate representation without class information
        confidence_inter = self.confidence_inter(intermediate_output, None)

        # calculate final confidence using logits ithout class information
        confidence_final = self.confidence_final(logits, None)

        # combine confidences
        if self.mult:
            confidence = confidence_inter * confidence_final
            return confidence
        else:
            confidence = self.a * confidence_inter + self.b * confidence_final
            return confidence


class PredictedSplitConfidence(ConfidenceModule):
    """
    This class computes confidence scores by combining the outputs of two confidence modules.
    This considers the output of the model as a tuple of (intermediate_output, logits) and forwards
    them to the respective confidence modules. In additon the logits are inspected and the predicted class
    is forwarded to the modules(if not disabled by the predict_inter and predict_final flags).
    The final confidence score is computed by either multiplying
    or adding the outputs of the two confidence modules, depending on the mult parameter.
    """

    def __init__(self, confidence_trust, confidence_final, mult=True, a=1, b=1, predict_inter=True, predict_final=True):
        """
        Initializes the TrustSplitConfidence class.

        Args:
            confidence_trust: The confidence module for trust scores that takes
                              both intermediate output and predicted class.
            confidence_final: The confidence module for the final output.
            mult: If True, the outputs of the two confidence modules are multiplied.
            a: Multiplier for the trust confidence module. Ignored if mult is True.
            b: Multiplier for the final confidence module. Ignored if mult is True.
        """
        super(PredictedSplitConfidence, self).__init__()
        self.confidence_trust = confidence_trust
        self.confidence_final = confidence_final
        self.mult = mult
        self.a = a
        self.b = b
        self.predict_inter = predict_inter
        self.predict_final = predict_final

    def forward(self, x, y=None):
        """
        Computes confidence scores by combining the outputs of two confidence modules.

        Args:
            x: Input tensor for the model. This should be a tuple of (intermediate_output, logits).
            y: Unused by this. It uses labels derived from the logits instead.

        Returns:
            confidence: Confidence scores. Shape: (*batch_dims)
        """
        intermediate_output, logits = x

        # get predicted classes from logits
        predicted_classes = torch.argmax(logits, dim=-1)

        # combine confidences
        if self.mult:
            # calculate trust confidence using intermediate representation and predicted classes
            confidence_trust = self.confidence_trust(intermediate_output,
                                                     predicted_classes if self.predict_inter else y)

            # calculate final confidence using logits
            confidence_final = self.confidence_final(logits, predicted_classes if self.predict_final else y)

            confidence = confidence_trust * confidence_final
            return confidence
        else:
            if self.a == 0:
                return self.confidence_final(logits, predicted_classes if self.predict_final else y) * self.b
            if self.b == 0:
                return self.confidence_trust(intermediate_output,
                                             predicted_classes if self.predict_inter else y) * self.a

            # calculate trust confidence using intermediate representation and predicted classes
            confidence_trust = self.confidence_trust(intermediate_output,
                                                     predicted_classes if self.predict_inter else y)

            # calculate final confidence using logits
            confidence_final = self.confidence_final(logits, predicted_classes if self.predict_final else y)

            confidence = self.a * confidence_trust + self.b * confidence_final
            return confidence


class TrueLabelSplitConfidence(ConfidenceModule):
    """
    This class computes confidence scores by combining the outputs of two confidence modules.
    This considers the output of the model as a tuple of (intermediate_output, logits) and forwards
    them to the respective confidence modules.
    In addition a true label y is forwarded to both modules which has to be given as an input.
        is forwarded to the modules(if not disabled by the predict_inter and predict_final flags).
    The final confidence score is computed by either multiplying
        or adding the outputs of the two confidence modules, depending on the mult parameter.
    """

    def __init__(self, confidence_trust, confidence_final, mult=True, a=1, b=1):
        """
        Initializes the TrustSplitConfidence class.

        Args:
            confidence_trust: The confidence module for trust scores that takes
                              both intermediate output and true class.
            confidence_final: The confidence module for the final output.
            mult: If True, the outputs of the two confidence modules are multiplied.
            a: Multiplier for the trust confidence module. Ignored if mult is True.
            b: Multiplier for the final confidence module. Ignored if mult is True.
        """
        super(TrueLabelSplitConfidence, self).__init__()
        self.confidence_trust = confidence_trust
        self.confidence_final = confidence_final
        self.mult = mult
        self.a = a
        self.b = b

    def forward(self, x, y=None):
        """
        Computes confidence scores by combining the outputs of two confidence modules.

        Args:
            x: Input tensor for the model. This should be a tuple of (intermediate_output, logits).
            y: Optional labels for modules that use them.

        Returns:
            confidence: Confidence scores. Shape: (*batch_dims)
        """
        intermediate_output, logits = x

        # combine confidences
        if self.mult:
            # calculate trust confidence using intermediate representation and true classes
            confidence_trust = self.confidence_trust(intermediate_output, y)

            # calculate final confidence using logits
            confidence_final = self.confidence_final(logits, y)

            confidence = confidence_trust * confidence_final
            return confidence
        else:
            if self.a == 0:
                return self.confidence_final(logits, y) * self.b
            if self.b == 0:
                return self.confidence_trust(intermediate_output, y) * self.a

            # calculate trust confidence using intermediate representation and true classes
            confidence_trust = self.confidence_trust(intermediate_output, y)

            # calculate final confidence using logits
            confidence_final = self.confidence_final(logits, y)

            confidence = self.a * confidence_trust + self.b * confidence_final
            return confidence
