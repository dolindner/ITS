from typing import List, Optional, Sequence
from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from confidence.base_confidence import ConfidenceModule


class FeatureScaler(nn.Module):
    """
    Calculates feature scaling params during fit and applies them later.
    Modes:
      - 'none': no-op
      - 'standardize': (x - mean) / (std + eps)
      - 'minmax': affine-map to [low, high] using per-feature min/max
    """

    def __init__(
            self,
            mode: str = "none",
            low: float = 0.02,
            high: float = 0.98,
            eps: float = 1e-6,
            clip: bool = False,
    ):
        super().__init__()
        assert mode in {"none", "standardize", "minmax"}
        assert 0.0 <= low < high <= 1.0
        self.mode = mode
        self.low = float(low)
        self.high = float(high)
        self.eps = float(eps)
        self.clip = bool(clip)
        # Learned stats
        self._fitted = False
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: torch.Tensor) -> "FeatureScaler":
        # X: [N, D]
        if self.mode == "none":
            self._fitted = True
            return self
        if self.mode == "standardize":
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, unbiased=False, keepdim=True)
            self._mean = mean.detach()
            self._std = std.detach()
            self._fitted = True
            return self
        if self.mode == "minmax":
            x_min = torch.amin(X, dim=0, keepdim=True)
            x_max = torch.amax(X, dim=0, keepdim=True)
            self._min = x_min.detach()
            self._max = x_max.detach()
            self._fitted = True
            return self
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.mode == "none":
            return X
        if self.mode == "standardize":
            mean = self._mean.to(X.device, X.dtype)
            std = self._std.to(X.device, X.dtype)
            return (X - mean) / (std + self.eps)
        if self.mode == "minmax":
            x_min = self._min.to(X.device, X.dtype)
            x_max = self._max.to(X.device, X.dtype)
            denom = (x_max - x_min).clamp_min(self.eps)
            t = (X - x_min) / denom  # affine, fully differentiable
            y = self.low + (self.high - self.low) * t
            return y.clamp(self.low, self.high) if self.clip else y
        return X

    def state_dict(self):
        return {
            "mode": self.mode,
            "low": self.low,
            "high": self.high,
            "eps": self.eps,
            "clip": self.clip,
            "fitted": self._fitted,
            "mean": None if self._mean is None else self._mean.detach().cpu(),
            "std": None if self._std is None else self._std.detach().cpu(),
            "min": None if self._min is None else self._min.detach().cpu(),
            "max": None if self._max is None else self._max.detach().cpu(),
        }

    def load_state_dict(self, state: dict):
        """Restore scaler internals from dict"""
        self.mode = state.get("mode", self.mode)
        self.low = float(state.get("low", self.low))
        self.high = float(state.get("high", self.high))
        self.eps = float(state.get("eps", self.eps))
        self.clip = bool(state.get("clip", self.clip))
        self._fitted = bool(state.get("fitted", self._fitted))

        def _maybe_tensor(v):
            return None if v is None else v.clone()

        self._mean = _maybe_tensor(state.get("mean", None))
        self._std = _maybe_tensor(state.get("std", None))
        self._min = _maybe_tensor(state.get("min", None))
        self._max = _maybe_tensor(state.get("max", None))
        return self


class RegressionConfidence(ConfidenceModule):

    def __init__(
            self,
            sub_confs: Sequence[nn.Module],
            aggregator: Optional[nn.Module] = None,
            input_selectors: Optional[Sequence[int]] = None,
            freeze_subs: bool = True,
            scaler: Union[str, "FeatureScaler"] = "none",
            loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor]] = None,
            pos_weight: float = 1.0,
            aggregator_reg_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
            aggregator_reg_weight: float = 0.0,
            lr: float = 1e-3,
            weight_decay: float = 0e-4,
            epochs: int = 50,
            optimizer_factory: Optional[Callable[[Sequence[nn.Parameter]], torch.optim.Optimizer]] = None,
            verbose: Optional[int] = None,
            pred_y=True
    ):
        super().__init__()
        self.sub_confs = nn.ModuleList(sub_confs)
        self.k = len(sub_confs)
        self.input_selectors: list[int] = (
            list(input_selectors) if input_selectors is not None else list(range(self.k))
        )

        # Aggregator may depend on k; build lazily if None.
        self.aggregator = aggregator  # type: Optional[nn.Module]

        # Optim / training config
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimizer_factory = optimizer_factory
        self.verbose = verbose

        # Loss config
        self.loss_fn = loss_fn
        self.pos_weight = float(pos_weight)
        self.aggregator_reg_fn = aggregator_reg_fn
        self.aggregator_reg_weight = float(aggregator_reg_weight)

        # Device anchor
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.pred_y = pred_y

        # Scaler
        if isinstance(scaler, str):
            # Use FeatureScaler with its own defaults
            self.scaler = FeatureScaler(mode=scaler)
        else:
            self.scaler = scaler

        # Optionally freeze submodules
        if freeze_subs:
            for m in self.sub_confs:
                for p in m.parameters():
                    p.requires_grad = False

    @staticmethod
    def _select(x, idx: int):
        if isinstance(x, (tuple, list)):
            return x[idx]
        if idx == 0:
            return x
        raise TypeError(f"Expected tuple/list input for selector {idx}.")

    def _ensure_aggregator(self, device: torch.device, dtype: torch.dtype):
        # Build a default logistic aggregator if none was provided yet.
        if self.aggregator is None:
            self.aggregator = nn.Linear(self.k, 1)
        self.aggregator.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _collect(self, loader, model_wrapper=None):
        xs, ys = [], []
        for x, y in loader:
            if model_wrapper:
                x = x.to(self._anchor.device)
                y = y.to(self._anchor.device) if y is not None else None
                features = model_wrapper(x)
            else:
                features = x

            feats_each = []
            for conf, sel in zip(self.sub_confs, self.input_selectors):
                x_i = self._select(features, sel)
                if not model_wrapper:  # If not pre-wrapped, move to device
                    if isinstance(x_i, torch.Tensor):
                        x_i = x_i.to(self._anchor.device)
                    elif isinstance(x_i, (list, tuple)):
                        x_i = [xi.to(self._anchor.device) for xi in x_i]
                    else:
                        raise TypeError(f"Unsupported input type for selector {sel}: {type(x_i)}")
                # this is wrong features -1 has shape batch,logits so we have to get max logit
                y_c = y if not self.pred_y else features[-1].max(
                    dim=1).values  # use final output as y if pred_y is True
                s = conf(x_i, y_c).view(-1)
                feats_each.append(s)
            feats = torch.stack(feats_each, dim=1).float()  # [B, k]
            xs.append(feats)
            # For fit we need labels only to build Y; assume in_loader is ID (1.0), out_loader is OOD (0.0).
            ys.append(y)  # we won't actually use raw y here, just maintain shape if needed
        if not xs:
            return None, None
        return torch.cat(xs, dim=0), None  # Y will be constructed in fit

    def _default_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        )

    def _compute_total_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> torch.Tensor:
        # Main loss
        if self.loss_fn is None:
            main_loss = self._default_bce_loss(logits, targets)
        else:
            main_loss = self.loss_fn(logits, targets, self.aggregator)

        # Optional aggregator regularizer
        if self.aggregator_reg_fn is not None and self.aggregator_reg_weight != 0.0:
            reg = self.aggregator_reg_fn(self.aggregator)
            main_loss = main_loss + self.aggregator_reg_weight * reg

        return main_loss

    def fit(self, in_loader, out_loader, model_wrapper=None):
        # Collect features
        X_in, _ = self._collect(in_loader, model_wrapper=model_wrapper)  # ID
        X_out, _ = self._collect(out_loader, model_wrapper=model_wrapper)  # OOD
        if X_in is None or X_out is None:
            return self

        # Build labels
        Y_in = torch.ones(X_in.size(0), dtype=torch.float32, device=X_in.device)
        Y_out = torch.zeros(X_out.size(0), dtype=torch.float32, device=X_out.device)

        # Combine and scale
        X = torch.cat([X_in, X_out], dim=0)  # [N, k]
        Y = torch.cat([Y_in, Y_out], dim=0)  # [N]

        # Fit scaler on training features, then transform
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        # Ensure aggregator exists and is on correct device/dtype
        self._ensure_aggregator(device=X.device, dtype=X.dtype)

        # Build optimizer over trainable params (typically aggregator params only)
        params = [p for p in self.aggregator.parameters() if p.requires_grad]
        if self.optimizer_factory is not None:
            opt = self.optimizer_factory(params)
        else:
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # Training loop
        with torch.enable_grad():
            for e in range(self.epochs):
                opt.zero_grad(set_to_none=True)
                logits = self.aggregator(X).view(-1)
                loss = self._compute_total_loss(logits, Y)
                loss.backward()
                opt.step()
                if hasattr(self.aggregator, 'project_weights'):
                    self.aggregator.project_weights()

                accuracy = ((torch.sigmoid(logits) >= 0.5).float() == Y).float().mean().item()

                if self.verbose is not None and (e % self.verbose == 0):
                    print(f"Epoch {e}: Loss = {loss.item():.6f}, Accuracy = {accuracy:.4f}")

        return self

    def evaluate(self, in_loader, out_loader, model_wrapper=None):
        with torch.no_grad():
            # Collect features
            X_in, _ = self._collect(in_loader, model_wrapper=model_wrapper)  # ID
            X_out, _ = self._collect(out_loader, model_wrapper=model_wrapper)  # OOD
            if X_in is None or X_out is None:
                return None, None

            # Build labels
            Y_in = torch.ones(X_in.size(0), dtype=torch.float32, device=X_in.device)
            Y_out = torch.zeros(X_out.size(0), dtype=torch.float32, device=X_out.device)

            # Combine and scale
            X = torch.cat([X_in, X_out], dim=0)  # [N, k]
            Y = torch.cat([Y_in, Y_out], dim=0)  # [N]

            # Transform features
            X = self.scaler.transform(X)

            # Ensure aggregator exists and is on correct device/dtype
            self._ensure_aggregator(device=X.device, dtype=X.dtype)

            # Compute logits
            logits = self.aggregator(X).view(-1)

            # now compute the accuracy
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct = (preds == Y).float().sum()
            accuracy = correct / Y.size(0)
            return accuracy.item()

    def forward(self, x, y=None):
        """
        Compute aggregator input features from sub-confidence modules and return aggregated confidence.

        Args:
            x: Input to the module. For usual use this can be raw model inputs or precomputed features
               depending on whether a model_wrapper was used during fit.
            y: Optional labels. If pred_y is True, per-item predicted values may be derived from x.


        """
        feats_each = []
        for conf, sel in zip(self.sub_confs, self.input_selectors):
            x_i = self._select(x, sel)
            y_c = y if not self.pred_y else x[-1].max(dim=1).values  # use final output as y if pred_y is True
            s = conf(x_i, y_c).view(-1)
            feats_each.append(s)
        feats = torch.stack(feats_each, dim=1).float()  # [B, k]
        feats = self.scaler.transform(feats)

        # Ensure aggregator exists for inference too (in case fit wasn't called)
        self._ensure_aggregator(device=feats.device, dtype=feats.dtype)

        out = self.aggregator(feats).view(-1)
        return out

    def state_dict(self, *args, **kwargs):
        """
        Only save aggregator + scaler, ignore sub_confs.
        """
        state = {}
        # Aggregator params
        if self.aggregator is not None:
            for k, v in self.aggregator.state_dict().items():
                state[f"aggregator.{k}"] = v
        # Scaler params
        scaler_state = self.scaler.state_dict()
        for k, v in scaler_state.items():
            state[f"scaler.{k}"] = v
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load aggregator + scaler while ignoring sub_confs.
        """
        aggregator_state = {}
        scaler_state = {}
        main_state = {}

        for k, v in state_dict.items():
            if k.startswith("aggregator."):
                aggregator_state[k[len("aggregator."):]] = v
            elif k.startswith("scaler."):
                scaler_state[k[len("scaler."):]] = v
            else:
                # old-style state might have sub_confs; ignore them
                main_state[k] = v

        # Try loading aggregator if present
        if self.aggregator is not None and aggregator_state:
            self.aggregator.load_state_dict(aggregator_state, strict=False)
        elif aggregator_state:
            raise ValueError("State dict contains aggregator parameters but aggregator is None.")

        # Load scaler
        self.scaler.load_state_dict(scaler_state)


# methods to combine different confidence measures.

class LinearAggregator(nn.Module):
    def __init__(self, k: int, bias: bool = True, non_negative: bool = False):
        super().__init__()
        self.k = int(k)
        self.linear = nn.Linear(self.k, 1, bias=bias)
        self.non_negative = non_negative

    def project_weights(self):
        """Project weights to non-negative space after optimizer step."""
        if self.non_negative:
            with torch.no_grad():
                self.linear.weight.data.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.project_weights()
        return self.linear(x).view(-1)


class MLPAggregator(nn.Module):
    """
    MLP aggregator that maps the k-dimensional feature vector to a single logit.
    hidden_sizes: sequence of hidden layer widths (may be empty for single linear layer).
    activation: activation between layers (default GELU).
    """

    def __init__(
            self,
            k: int,
            hidden_sizes: Sequence[int] = (32,),
            activation: Optional[nn.Module] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.k = int(k)
        if activation is None:
            activation = nn.GELU()
        layers: List[nn.Module] = []
        in_dim = self.k
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h), bias=bias))
            layers.append(activation)
            in_dim = int(h)
        layers.append(nn.Linear(in_dim, 1, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, k]
        out = self.net(x)  # [B, 1]
        return out.view(-1)  # [B]


def make_aggregator(k: int, kind: str = "linear", **kwargs) -> nn.Module:
    """
    Factory to create an aggregator.
    kind: 'linear' | 'mlp' | 'flexible' | 'flexible_monotonic'
    """
    kind = kind.lower()

    if kind == "linear":
        return LinearAggregator(k=k, bias=kwargs.get("bias", True), non_negative=True)

    if kind == "mlp":
        return MLPAggregator(
            k=k,
            hidden_sizes=kwargs.get("hidden_sizes", (32,)),
            activation=kwargs.get("activation", None),
            bias=kwargs.get("bias", True)
        )

    if kind == "flexible":
        return FlexibleAggregator(
            k=k,
            feature_extractors=kwargs.get("feature_extractors", None),
            bias=kwargs.get("bias", True)
        )

    if kind == "flexible_monotonic":
        # Build monotonic feature extractors
        feature_extractors = []
        hidden_sizes = kwargs.get("hidden_sizes", (16,))
        activation = kwargs.get("activation", nn.ReLU())

        for _ in range(k):
            layers = []
            in_dim = 1
            for h in hidden_sizes:
                linear = nn.Linear(in_dim, h)
                # Ensure weights are non-negative by constraining them via ReLU
                nn.init.uniform_(linear.weight, 0, 0.1)
                linear.weight.data = linear.weight.data.abs()
                layers.append(NonNegativeLinear(in_dim, h))
                layers.append(activation)
                in_dim = h
            layers.append(NonNegativeLinear(in_dim, 1))
            feature_extractors.append(nn.Sequential(*layers))

        return FlexibleAggregator(k=k, feature_extractors=feature_extractors, bias=kwargs.get("bias", True),
                                  non_negative=True)

    raise ValueError(f"Unknown aggregator kind: {kind}")


class NonNegativeLinear(nn.Linear):
    """
    Linear layer with non-negative weights to enforce monotonicity.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Use softplus to ensure weights stay positive
        weight = F.softplus(self.weight)
        return F.linear(input, weight, self.bias)


class FlexibleAggregator(nn.Module):
    def __init__(self, k: int, feature_extractors: Optional[Sequence[nn.Module]] = None,
                 bias: bool = True, non_negative: bool = False):
        super().__init__()
        self.k = int(k)
        self.non_negative = non_negative
        if feature_extractors is None:
            self.feature_extractors = nn.ModuleList([nn.Identity() for _ in range(self.k)])
        else:
            if len(feature_extractors) != self.k:
                raise ValueError("Length of feature_extractors must equal k")
            self.feature_extractors = nn.ModuleList(feature_extractors)
        self.combiner = nn.Linear(self.k, 1, bias=bias)

    def project_weights(self):
        """Project combiner weights to non-negative space."""
        if self.non_negative:
            with torch.no_grad():
                self.combiner.weight.data.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = []
        for i in range(self.k):
            col = x[:, i:i + 1]
            out_i = self.feature_extractors[i](col)
            if out_i.dim() == 1:
                out_i = out_i.view(-1, 1)
            elif out_i.dim() == 2 and out_i.shape[1] == 1:
                pass
            else:
                out_i = out_i.reshape(out_i.shape[0], -1)
                if out_i.shape[1] != 1:
                    out_i = out_i.mean(dim=1, keepdim=True)
            transformed.append(out_i)
        transformed = torch.cat(transformed, dim=1)
        out = self.combiner(transformed)
        return out.view(-1)
