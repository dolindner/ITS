import inspect

import torch
import torch.utils._pytree
from torch.func import vmap

from confidence.model.base_model import ModelBasedConfidence


class MonteCarloDropoutConfidence(ModelBasedConfidence):
    """
    This class applies Monte Carlo Dropout during inference to compute confidence scores.
    This is done by setting the dropout layers to training mode when this is set to eval.
    If a model has a forward method that takes a kwarg mc_dropout, this is passed calling the model. The underlying
    model should implement dropout themself then.


    The model is called multiple times (samples) and the outputs are averaged to compute the final confidence scores.
    """

    def __init__(self, model, confidence, samples=4, data_dims=None, index=None, parallel=False, average=True,
                 softmax=False):
        """
        Initializes the MonteCarloDropoutConfidence class. This class computes confidence scores over multiple forward passes
        of the model by applying dropout during inference. The outputs of the model are averaged to compute the final confidence scores.

        Args:
            model: The model whose output is used to compute confidence scores.
                if the model has a forward method that takes a kwarg mc_dropout, this is passed calling the model when called.
            confidence: A ConfidenceModule that computes the confidence scores.
            samples: Number of forward passes to perform. Default is 4.
            data_dims: Number of non-batch dimensions of the input data (e.g. channels+spatial dims).
                Default is None which means assume a single leading batch dimension (i.e. batch_dim = 1).
            parallel: Whether to use parallel MC-dropout via vmap. Default is False.
            average: Whether to return averaged outputs or full MC sample tensor. Default is True.
            softmax_before_average: Whether to apply softmax to logits before averaging MC samples. Default is False.
        """
        super(MonteCarloDropoutConfidence, self).__init__(model, confidence, index)
        self.samples = samples
        self.parallel = parallel
        self.average = average  # whether to average outputs across MC samples
        self.softmax = softmax  # renamed flag: whether to apply softmax to samples before passing to confidence

        self.takes_mc_samples = 'mc_dropout' in inspect.signature(model.forward).parameters

        self.data_dims = data_dims

        self.eval()

    def eval(self):
        super(MonteCarloDropoutConfidence, self).eval()
        self.model.eval()

        if self.samples > 1:
            for layer in self.model.modules():
                if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout2d)):
                    layer.train()
        return self

    def train(self, mode=True):
        super(MonteCarloDropoutConfidence, self).train(mode)
        return self

    @staticmethod
    def _aggregate_outputs(outputs, dim=0):
        """
        Recursively aggregates model outputs across MC samples.

        args:
            outputs: List of outputs to aggregate
            dim: Dimension along which to aggregate

        returns:
            Aggregated output (stacked and averaged) maintaining structure
        """
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            stacked = torch.stack(outputs, dim=dim)
            return stacked.mean(dim=dim)
        elif isinstance(first, tuple):
            return tuple(
                MonteCarloDropoutConfidence._aggregate_outputs([o[i] for o in outputs], dim)
                for i in range(len(first))
            )
        elif isinstance(first, list):
            return [
                MonteCarloDropoutConfidence._aggregate_outputs([o[i] for o in outputs], dim)
                for i in range(len(first))
            ]
        elif isinstance(first, dict):
            return {
                k: MonteCarloDropoutConfidence._aggregate_outputs([o[k] for o in outputs], dim)
                for k in first
            }
        else:
            raise TypeError(f"Unsupported output type: {type(first)}")

    def forward(self, x, y=None):
        """
        Computes confidence scores using Monte Carlo Dropout averaging.

        Args:
            x: Input tensor
            y: Optional labels

        Returns:
            confidence: Confidence scores
            output: Averaged or stacked logits depending on configuration
        """
        kwargs = {}
        if self.takes_mc_samples:
            kwargs['mc_dropout'] = True

        if self.data_dims is None:
            batch_dim = 1
        else:
            batch_dim = x.dim() - self.data_dims
        batch_shape = x.shape[:batch_dim]
        x_flat = x.flatten(end_dim=batch_dim - 1)

        if self.samples > 1 and self.parallel:
            x_rep = x_flat.unsqueeze(0).expand(self.samples, *x_flat.shape)
            batched = vmap(lambda inp: self.model(inp, **kwargs), in_dims=0, randomness='different')(x_rep)
            if self.average:
                output_logits = torch.utils._pytree.tree_map(lambda t: t.mean(0), batched)
                if self.softmax:
                    probs = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1), batched)
                    conf_input = torch.utils._pytree.tree_map(lambda t: t.mean(0), probs)
                else:
                    conf_input = output_logits
            else:
                output_logits = torch.utils._pytree.tree_map(
                    lambda t: t.permute(1, 0, *range(2, t.ndim)), batched
                )
                if self.softmax:
                    conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1),
                                                              output_logits)
                else:
                    conf_input = output_logits
        elif self.samples > 1:
            outputs = []
            for _ in range(self.samples):
                outputs.append(self.model(x_flat, **kwargs))
            if self.average:
                output_logits = self._aggregate_outputs(outputs, dim=0)
                if self.softmax:
                    probs = [torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1), o) for o in
                             outputs]
                    conf_input = self._aggregate_outputs(probs, dim=0)
                else:
                    conf_input = output_logits
            else:
                stacked = torch.stack(outputs, dim=0)
                output_logits = stacked.permute(1, 0, *range(2, stacked.ndim))
                if self.softmax:
                    conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1),
                                                              output_logits)
                else:
                    conf_input = output_logits
        else:
            output_logits = self.model(x_flat, **kwargs)
            if self.softmax:
                conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1),
                                                          output_logits)
            else:
                conf_input = output_logits
            if not self.average:
                output_logits = torch.utils._pytree.tree_map(lambda t: t.unsqueeze(1), output_logits)
                conf_input = torch.utils._pytree.tree_map(lambda t: t.unsqueeze(1), conf_input)

        def restore_batch(tensor):
            if tensor.dim() >= 2:
                flat_first = tensor.size(0)
                new_shape = batch_shape + tuple(tensor.shape[1:])
                return tensor.view(*new_shape)
            else:
                return tensor.view(*batch_shape)

        returned_output = torch.utils._pytree.tree_map(restore_batch, output_logits)
        conf_input = torch.utils._pytree.tree_map(restore_batch, conf_input)

        confidence = self.confidence(conf_input, y)
        if self.index is None:
            return confidence, returned_output
        return confidence, returned_output[self.index]


class LastLayerMonteCarloDropoutConfidence(ModelBasedConfidence):
    """
    This class applies Monte Carlo Dropout on the last layer of a sequential model during inference to compute confidence scores.
    This is done by splitting the model into a backbone and a head (the last layer), running the backbone once and the head multiple times.
    This is more computationally efficient if the last layer is small compared to the rest of the model.
    It only sets the last layer to training mode to enable dropout.
    """

    def __init__(self, model, confidence, samples=4, data_dims=None, index=None, average=True, softmax=False):
        """
        Initializes the LastLayerMonteCarloDropoutConfidence class.

        Args:
            model: A torch.nn.Sequential model.
            confidence: A ConfidenceModule that computes the confidence scores.
            samples: Number of forward passes to perform on the head. Default is 4.
            data_dims: Number of non-batch dimensions of the input data.
                Default is None which means assume a single leading batch dimension (i.e. batch_dim = 1).
            index: Index of the output to use if the model returns multiple outputs. Default is None.
            average: Whether to return averaged outputs or full MC sample tensor. Default is True.
            softmax: Whether to apply softmax to logits before averaging MC samples. Default is False.
        """
        super(LastLayerMonteCarloDropoutConfidence, self).__init__(model, confidence, index)
        if not isinstance(model, torch.nn.Sequential):
            raise TypeError("Model must be torch.nn.Sequential for LastLayerMonteCarloDropoutConfidence.")

        mc_dropout_layer_index = -1
        for i, layer in enumerate(model):
            if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout2d)):
                mc_dropout_layer_index = i

        if mc_dropout_layer_index == -1:
            raise ValueError("No Dropout layer found in model.")

        self.backbone = model[:mc_dropout_layer_index]
        self.head = model[mc_dropout_layer_index:]

        self.samples = samples
        self.average = average
        self.softmax = softmax
        self.data_dims = data_dims

        self.eval()

    def eval(self):
        super(LastLayerMonteCarloDropoutConfidence, self).eval()
        self.model.eval()

        if self.samples > 1:
            self.head.train()
        return self

    def train(self, mode=True):
        super(LastLayerMonteCarloDropoutConfidence, self).train(mode)
        return self

    def forward(self, x, y=None):
        """
        Computes confidence scores with MC Dropout on last layer only.

        Args:
            x: Input tensor
            y: Optional labels

        Returns:
            confidence: Confidence scores
            output: Averaged or stacked logits
        """
        if self.data_dims is None:
            batch_dim = 1
        else:
            batch_dim = x.dim() - self.data_dims
        batch_shape = x.shape[:batch_dim]
        x_flat = x.flatten(end_dim=batch_dim - 1)

        features = self.backbone(x_flat)

        if self.samples > 1:
            outputs = []
            for _ in range(self.samples):
                outputs.append(self.head(features))
            if self.average:
                output_logits = MonteCarloDropoutConfidence._aggregate_outputs(outputs, dim=0)
                if self.softmax:
                    probs = [torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1), o) for o in
                             outputs]
                    conf_input = MonteCarloDropoutConfidence._aggregate_outputs(probs, dim=0)
                else:
                    conf_input = output_logits
            else:
                stacked = torch.stack(outputs, dim=0)
                output_logits = stacked.permute(1, 0, *range(2, stacked.ndim))
                if self.softmax:
                    conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1),
                                                              output_logits)
                else:
                    conf_input = output_logits
        else:
            output_logits = self.head(features)
            if self.softmax:
                conf_input = torch.utils._pytree.tree_map(lambda t: torch.nn.functional.softmax(t, dim=-1),
                                                          output_logits)
            else:
                conf_input = output_logits
            if not self.average:
                output_logits = torch.utils._pytree.tree_map(lambda t: t.unsqueeze(1), output_logits)
                conf_input = torch.utils._pytree.tree_map(lambda t: t.unsqueeze(1), conf_input)

        def restore_batch(tensor):
            if tensor.dim() >= 1:
                new_shape = batch_shape + tuple(tensor.shape[1:])
                return tensor.view(*new_shape).contiguous()
            else:
                return tensor.view(*batch_shape).contiguous()

        returned_output = torch.utils._pytree.tree_map(restore_batch, output_logits)
        conf_input = torch.utils._pytree.tree_map(restore_batch, conf_input)

        confidence = self.confidence(conf_input, y)
        if self.index is None:
            return confidence, returned_output
        return confidence, returned_output[self.index]
