from typing import Dict, Any, List, Optional

import optuna
import torch

from confidence.control.regression import RegressionConfidence, make_aggregator
from confidence.control.regression_wrapper import RegressionWrapper
from confidence.unsupervised.classic.prototype import ClassPrototypeConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
    OOD_MODEL_PARAM_EXTRACTORS,
)
from model.get_model import get_max_layer_index, get_network_layer
from src.utils.transformation_problem import TransformationProblem


# variant of prototype that supports multiple layers


def _extract_prototype_model_params(conf_module: RegressionWrapper) -> List[Dict[str, torch.Tensor]]:
    """Extracts the state dict of the aggregator from a Mahalanobis-style problem."""
    if not hasattr(conf_module, 'regression_confidence'):
        return []

    raw_state = conf_module.regression_confidence.state_dict()

    safe_state = {}
    for k, v in raw_state.items():
        if isinstance(v, torch.Tensor):
            safe_state[k] = v.cpu().detach()
        else:
            # Leave None or non-tensor values unchanged
            safe_state[k] = v

    return [safe_state]


# -------------------------
# Default params / sampler
# -------------------------

def _default_prototype_multi_params(**kwargs) -> Dict[str, Any]:
    """Default params for multi-layer prototype."""
    # Get max_layer to create proper mask
    dataset_info = kwargs.get("dataset_info")
    architecture = kwargs.get("architecture")
    max_layer = 0
    if dataset_info and architecture:
        try:
            max_layer = get_max_layer_index(dataset_info, architecture) or 0
        except Exception:
            pass

    # Create params with only first layer enabled
    params = {
        "reducer_name": None,
        "metric": "cosine",
        "mixed_alpha": 0.5,
        "mixed_squared": False,
        "mixed_normalize_euclid": True,
        "aggregator_kind": "linear",
        "scaler": "none",
        "freeze_subs": True,
        "lr": 1e-2,
        "epochs": 250,
        "use_correct_only": False,
        "layer_indices": [0],  # Default to first layer only
    }

    # Add mask format for Optuna
    for i in range(max_layer + 1):
        params[f"use_layer_{i}"] = 1 if i == 0 else 0

    return params


def _sample_prototype_multi_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None,
                                   **kwargs) -> Dict[str, Any]:
    # Get the actual maximum layer index
    max_layer = get_max_layer_index(dataset_info, architecture)
    if max_layer is None:
        max_layer = 0

    # Sample binary mask for which layers to use
    mask = [trial.suggest_categorical(f"use_layer_{i}", [0, 1])
            for i in range(max_layer + 1)]

    layer_indices = [i for i, flag in enumerate(mask) if flag]
    if not layer_indices:
        # Ensure at least one layer
        layer_indices = [trial.suggest_int("fallback_layer", 0, max_layer)]
        for i in range(max_layer + 1):
            mask[i] = 1 if i in layer_indices else 0

    reducer_name = None
    if train_cache:
        available = train_cache.reducer_name
        names = [None] + (available if available else [])
        reducer_name = trial.suggest_categorical("reducer_name", names)
        if reducer_name is not None:
            for li in layer_indices:
                layer, layer_io = get_network_layer(dataset_info, architecture, li)
                layer_available = train_cache.get_available_reducers(layer, layer_io)
                if reducer_name not in layer_available:
                    reducer_name = None
                    break
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "mixed"])
    params = {
        "layer_indices": layer_indices,
        "reducer_name": reducer_name,
        "metric": metric,
        "aggregator_kind": trial.suggest_categorical("aggregator_kind", ["linear", "flexible_monotonic"]),
        "scaler": trial.suggest_categorical("scaler", ["none", "standardize"]),
        "freeze_subs": True,
        "lr": float(trial.suggest_float("lr", 1e-4, 1e-1, log=True)),
        "epochs": int(trial.suggest_int("epochs", 50, 500)),
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    if metric == "mixed":
        params["mixed_alpha"] = trial.suggest_float("mixed_alpha", 0.0, 1.0)
        params["mixed_squared"] = trial.suggest_categorical("mixed_squared", [True, False])
        params["mixed_normalize_euclid"] = trial.suggest_categorical("mixed_normalize_euclid", [True, False])
    if metric == "mahalanobis":
        params["shared_covariance"] = True

    # attach mask flags so returned dict matches default factory format
    for i in range(max_layer + 1):
        params[f"use_layer_{i}"] = int(mask[i])

    return params


def _resolve_layer_indices_explicit_or_all(
        layer_indices: Optional[List[int]],
        train_cache,
        dataset_info,
        architecture,
) -> List[int]:
    """Resolve layer indices: explicit list or all available layers."""
    if layer_indices is not None:
        return list(layer_indices)
    try:
        max_layer = get_max_layer_index(dataset_info, architecture)
        if max_layer is None:
            return [0]
        return list(range(0, int(max_layer) + 1))
    except Exception:
        return [0]


# -------------------------
# Factory
# -------------------------

@torch.no_grad()
def _create_prototype_multi_problem(
        params: Dict[str, Any],
        train_cache,
        transform_seq,
        dataset_info,
        architecture,
        val_id_loader=None,
        val_ood_loader=None,
        device: str = "cpu",
        **kwargs
) -> TransformationProblem:
    """Multi-layer prototype factory."""
    layer_indices: Optional[List[int]] = params.get("layer_indices", None)
    reducer_name = params.get("reducer_name", None)
    use_correct_only = params.get("use_correct_only", False)

    layer_indices = _resolve_layer_indices_explicit_or_all(layer_indices, train_cache, dataset_info, architecture)

    if reducer_name is not None:
        for li in layer_indices:
            layer, layer_io = get_network_layer(dataset_info, architecture, li)
            available = train_cache.get_available_reducers(layer, layer_io)
            if reducer_name not in available:
                reducer_name = None
                break

    sub_confs = []
    layer_names = []
    layer_ios = []

    for li in layer_indices:
        layer, layer_io = get_network_layer(dataset_info, architecture, li)
        print(f"Prototype Multi: Processing layer {li} ({layer}) with IO {layer_io} and reducer {reducer_name}")
        if use_correct_only:
            embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
                layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
            )
        else:
            embeddings_t, _, classes_t = train_cache(
                layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True,
                reducer_select=reducer_name
            )

        # Add validation
        print(f"  Embeddings shape: {embeddings_t.shape}")
        print(f"  Classes shape: {classes_t.shape}")
        print(f"  Unique classes: {torch.unique(classes_t)}")
        print(f"  Class range: [{classes_t.min()}, {classes_t.max()}]")

        # Ensure classes are 0-indexed and contiguous
        unique_classes = torch.unique(classes_t)
        if unique_classes.min() != 0 or unique_classes.max() != len(unique_classes) - 1:
            print(f"  WARNING: Classes are not 0-indexed and contiguous, remapping...")
            class_mapping = {old.item(): new for new, old in enumerate(unique_classes)}
            classes_t = torch.tensor([class_mapping[c.item()] for c in classes_t],
                                     device=classes_t.device, dtype=classes_t.dtype)

        detector = ClassPrototypeConfidence(
            metric=params.get("metric", "euclidean"),
            shared_covariance=params.get("shared_covariance", True),
            mixed_alpha=params.get("mixed_alpha", 0.5),
            mixed_squared=params.get("mixed_squared", False),
            mixed_normalize_euclid=params.get("mixed_normalize_euclid", True),
        )
        detector.fit(embeddings_t, classes_t)

        sub_confs.append(detector)
        layer_names.append(layer)
        layer_ios.append(layer_io)

    model_wrapper = train_cache.make_wrapper(
        layer_names, capture_modes=layer_ios, concat=False, flatten=True, reducer_select=reducer_name,
        return_final=False
    )

    aggregator = make_aggregator(k=len(sub_confs), kind=params.get("aggregator_kind", "linear"))
    reg_conf = RegressionConfidence(
        sub_confs=sub_confs,
        aggregator=aggregator,
        input_selectors=list(range(len(sub_confs))),
        freeze_subs=params.get("freeze_subs", True),
        scaler=params.get("scaler", "none"),
        lr=params.get("lr", 1e-2),
        epochs=int(params.get("epochs", 50)), pred_y=False,
    )

    final_conf = RegressionWrapper(model_wrapper, reg_conf)
    final_conf.to(device)

    model_params = kwargs.get("model_params")
    if model_params and len(model_params) > 0:
        final_conf.regression_confidence.load_state_dict(model_params[0])
        print("Loaded prototype multi-layer aggregator model parameters.")
    elif val_id_loader is not None and val_ood_loader is not None:
        final_conf.fit(val_id_loader, val_ood_loader)
        print("Fitted prototype multi-layer aggregator on validation data.")

    return TransformationProblem(final_conf, transform_seq, consolidate_method="consolidate_simple")


# -------------------------
# Register
# -------------------------

OOD_DEFAULT_PARAM_FACTORIES["prototype_multi"] = _default_prototype_multi_params
OOD_PARAM_SAMPLERS["prototype_multi"] = _sample_prototype_multi_params
OOD_PROBLEM_FACTORIES["prototype_multi"] = _create_prototype_multi_problem
OOD_MODEL_PARAM_EXTRACTORS["prototype_multi"] = _extract_prototype_model_params
