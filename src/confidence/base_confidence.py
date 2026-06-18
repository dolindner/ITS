"""
This file contains abstract base classes for confidence estimation modules and some simple modules that calculate confidence from model outputs.
More complex modules can be found in other files in the confidence directory.
"""
import abc
import inspect
from enum import IntEnum

import torch
import torch.nn



class ConfidenceModule(torch.nn.Module):
    """Base class for confidence modules. These modules compute
    confidence scores based on the output of a model.

    This class is an abstract base class and should not be instantiated
    directly. Subclasses should implement the forward method.
    """

    def __init__(self):
        super(ConfidenceModule, self).__init__()

    @abc.abstractmethod
    def forward(self, x, y=None):
        """Abstract method to be implemented by subclasses.

        Args:
            x: Input tensor for the confidence module.
            y: Optional labels for modules that use them.

        Returns:
            Confidence scores. Shape: (*batch_dims)
        """
        pass

