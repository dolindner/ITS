from typing import Dict, Any, List, Optional

import optuna
import torch

from confidence.control.regression import RegressionConfidence, make_aggregator
from confidence.control.regression_wrapper import RegressionWrapper
from confidence.unsupervised.classic.mahalanobis import PrototypeMahalanobisConfidence
from confidence.unsupervised.classic.mahalanobis_relative import RelativeMahalanobisConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
    OOD_MODEL_PARAM_EXTRACTORS,
)
from model.get_model import get_max_layer_index, get_network_layer
from src.utils.transformation_problem import TransformationProblem


# -------------------------
# Model Parameter Extractor, from the regresssion module so that it can be restored when applied to the test set.
# -------------------------

def _extract_maha_model_params(conf_module) -> List[Dict[str, torch.Tensor]]:
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

def _default_maha_params(**kwargs) -> Dict[str, Any]:
    """
    Default params for classical Mahalanobis OOD detector factory.
    """
    # Get max_layer to create proper mask
    dataset_info = kwargs.get("dataset_info")
    architecture = kwargs.get("architecture")
    max_layer = 0
    if dataset_info and architecture:
        try:
            max_layer = get_max_layer_index(dataset_info, architecture) or 0
        except Exception:
            pass

    params = {
        "reducer_name": None,
        "eps": 0,
        "shared_covariance": True,
        "aggregator_kind": "linear",
        "scaler": "none",
        "freeze_subs": True,
        "lr": 1e-2,
        "epochs": 250,
        "use_correct_only": False,
        "layer_indices": [0],  # Default to first layer only
    }

    # Add mask format for Optuna sampling
    for i in range(max_layer + 1):
        params[f"use_layer_{i}"] = 1 if i == 0 else 0

    return params


def _default_rmd_params(**kwargs) -> Dict[str, Any]:
    p = _default_maha_params(**kwargs)
    p.update({
        "shared_covariance": True,
        "mahalanobis_eps": 0,
        "aggregator_kind": "linear",
        "scaler": "none",
        "use_correct_only": False,
    })
    return p


def _sample_maha_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[
    str, Any]:
    # Get the actual maximum layer index
    max_layer = get_max_layer_index(dataset_info, architecture)
    if max_layer is None:
        max_layer = 0

    # Sample binary mask for which layers to use
    mask = [trial.suggest_categorical(f"use_layer_{i}", [0, 1])
            for i in range(max_layer + 1)]
    # build layer indices from the sampled values.
    layer_indices = [i for i, flag in enumerate(mask) if flag]
    if not layer_indices:
        # Ensure at least one layer
        layer_indices = [trial.suggest_int("fallback_layer", 0, max_layer)]
        # reflect fallback in mask
        for i in range(max_layer + 1):
            mask[i] = 1 if i in layer_indices else 0

    reducer_name = None
    if train_cache:
        # Get all possible reducer names from cache
        available = train_cache.reducer_name
        names = [None] + (available if available else [])
        reducer_name = trial.suggest_categorical("reducer_name", names)
        # Check if reducer_name is supported by all selected layers; if not, default to None
        if reducer_name is not None:
            for li in layer_indices:
                layer, layer_io = get_network_layer(dataset_info, architecture, li)
                layer_available = train_cache.get_available_reducers(layer, layer_io)
                if reducer_name not in layer_available:
                    reducer_name = None
                    break

    # build returned params and include mask flags to mirror defaults
    params = {
        "layer_indices": layer_indices,
        "reducer_name": reducer_name,  # Now single value
        "eps": 0,
        "shared_covariance": True,
        "aggregator_kind": trial.suggest_categorical("aggregator_kind", ["linear", ]),
        # Original used linear combination.
        "scaler": trial.suggest_categorical("scaler", ["none", ]),
        "freeze_subs": True,
        "lr": float(trial.suggest_float("lr", 1e-4, 1e-1, log=True)),
        "epochs": int(trial.suggest_int("epochs", 50, 500)),
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

    # attach mask keys so returned dict matches default factory format
    for i in range(max_layer + 1):
        params[f"use_layer_{i}"] = int(mask[i])

    return params


def _sample_rmd_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[
    str, Any]:
    p = _sample_maha_params(trial, train_cache, dataset_info, architecture, **kwargs)
    p["mahalanobis_eps"] = p.get("eps", 0)
    p["shared_covariance"] = p.get("shared_covariance", True)
    return p


# -------------------------
# Factory helpers
# -------------------------

def _resolve_layer_indices_explicit_or_all(
        layer_indices: Optional[List[int]],
        train_cache,
        dataset_info,
        architecture,
) -> List[int]:
    """
    Simplified resolver: if layer_indices provided, return it.
    If None => use all layers [0..max_layer] where max_layer is obtained via get_max_layer_index.
    Fallback to [0] on any error.
    """
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
# Factory to build problem (Mahalanobis)
# -------------------------

@torch.no_grad()
def _create_maha_problem(
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
    """
    Mahalanobis factory (plain Mahalanobis). Uses its own defaults and expects
    RMD to be created via the dedicated _create_rmd_problem.
    """
    layer_indices: Optional[List[int]] = params.get("layer_indices", None)
    reducer_name = params.get("reducer_name", None)  # Now single value
    eps = params.get("eps", 0)
    use_correct_only = params.get("use_correct_only", False)

    # resolve all-layers semantics the simple way (use dataset_info & architecture)
    layer_indices = _resolve_layer_indices_explicit_or_all(layer_indices, train_cache, dataset_info, architecture)

    # Check if reducer_name is supported by all selected layers; if not, default to None
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
        print(f"Mahalanobis: Processing layer {li} ({layer}) with IO {layer_io} and reducer {reducer_name}")
        if use_correct_only:
            embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
                layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
            )
        else:
            embeddings_t, _, classes_t = train_cache(
                layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True,
                reducer_select=reducer_name
            )

        cov_mode = params.get("cov_mode", "full")
        low_rank_r = params.get("low_rank_r", 64)

        detector = PrototypeMahalanobisConfidence(
            eps=params.get("eps", 0),
            shared_covariance=params.get("shared_covariance", True),
            use_raw_scatter=params.get("use_raw_scatter", False),
            cov_mode=cov_mode,
            low_rank_r=low_rank_r,
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
        epochs=int(params.get("epochs", 50)),
    )

    final_conf = RegressionWrapper(model_wrapper, reg_conf)
    final_conf.to(device)

    model_params = kwargs.get("model_params")
    if model_params and len(model_params) > 0:
        final_conf.regression_confidence.load_state_dict(model_params[0])
        print("Loaded Mahalanobis aggregator model parameters.")
    elif val_id_loader is not None and val_ood_loader is not None:
        final_conf.fit(val_id_loader, val_ood_loader)
        print("Fitted Mahalanobis aggregator on validation data.")

    return TransformationProblem(final_conf, transform_seq, consolidate_method="consolidate_simple")


# -------------------------
# Factory to build problem (Relative Mahalanobis / RMD)
# -------------------------

@torch.no_grad()
def _create_rmd_problem(
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
    """
    Dedicated factory for Relative Mahalanobis Distance (RMD).
    """
    layer_indices: Optional[List[int]] = params.get("layer_indices", None)
    reducer_name = params.get("reducer_name", None)  # Now single value
    eps = params.get("mahalanobis_eps", 0)
    shared_cov = params.get("shared_covariance", True)
    use_correct_only = params.get("use_correct_only", False)

    # use same simple resolver
    layer_indices = _resolve_layer_indices_explicit_or_all(layer_indices, train_cache, dataset_info, architecture)

    # Check if reducer_name is supported by all selected layers; if not, default to None
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
        if use_correct_only:
            embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
                layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
            )
        else:
            embeddings_t, _, classes_t = train_cache(
                layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True,
                reducer_select=reducer_name
            )

        # Get cov_mode/low_rank_r from params if available, default to full/64.
        cov_mode = params.get("cov_mode", "full")
        low_rank_r = params.get("low_rank_r", 64)
        global_cov_mode = params.get("global_cov_mode", None)

        detector = RelativeMahalanobisConfidence(
            mahalanobis_eps=eps,
            shared_covariance=shared_cov,
            use_raw_scatter=params.get("use_raw_scatter", False),
            cov_mode=cov_mode,
            low_rank_r=low_rank_r,
            global_cov_mode=global_cov_mode,
        )
        detector.fit(embeddings_t, classes_t)

        # Use detector directly (no ClassifyingConfidence wrapper needed)
        sub_confs.append(detector)
        layer_names.append(layer)
        layer_ios.append(layer_io)

    # Universal wrapper. Returns ([feat1, feat2, ...], final_output)
    model_wrapper = train_cache.make_wrapper(
        layer_names, capture_modes=layer_ios, concat=False, flatten=True, reducer_select=reducer_name,
        return_final=False
    )

    aggregator = make_aggregator(k=len(sub_confs), kind=params.get("aggregator_kind", "linear"))
    # Use regression wrapper
    reg_conf = RegressionConfidence(
        sub_confs=sub_confs,
        aggregator=aggregator,
        input_selectors=list(range(len(sub_confs))),  # Use integer selectors for indexing into features_list
        freeze_subs=params.get("freeze_subs", True),
        scaler=params.get("scaler", "none"),
        lr=params.get("lr", 1e-2),
        epochs=int(params.get("epochs", 50)),
    )

    final_conf = RegressionWrapper(model_wrapper, reg_conf)
    final_conf.to(device)

    model_params = kwargs.get("model_params")
    if model_params and len(model_params) > 0:
        # RMD problems also only have one model (the aggregator)
        final_conf.regression_confidence.load_state_dict(model_params[0])
        print("Loaded RMD aggregator model parameters.")
    elif val_id_loader is not None and val_ood_loader is not None:
        final_conf.fit(val_id_loader, val_ood_loader)
        print("Fitted RMD aggregator on validation data.")

    return TransformationProblem(final_conf, transform_seq, consolidate_method="consolidate_simple")


OOD_DEFAULT_PARAM_FACTORIES["mahalanobis"] = _default_maha_params
OOD_PARAM_SAMPLERS["mahalanobis"] = _sample_maha_params
OOD_PROBLEM_FACTORIES["mahalanobis"] = _create_maha_problem
OOD_MODEL_PARAM_EXTRACTORS["mahalanobis"] = _extract_maha_model_params  # method to save fitted regression module.

OOD_DEFAULT_PARAM_FACTORIES["rmd"] = _default_rmd_params
OOD_PARAM_SAMPLERS["rmd"] = _sample_rmd_params
OOD_PROBLEM_FACTORIES["rmd"] = _create_rmd_problem
OOD_MODEL_PARAM_EXTRACTORS["rmd"] = _extract_maha_model_params
