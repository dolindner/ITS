from typing import Dict, Any

import optuna
import torch

from confidence.control.split import PredictedSplitConfidence, SplitConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.model.single_pass import SinglePassConfidence
from confidence.unsupervised.classic.prototype import GlobalPrototypeConfidence, ClassPrototypeConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from model.get_model import get_max_layer_index, get_network_layer
from src.utils.transformation_problem import TransformationProblem


# NOTE we do not sample mahalanbis here as this would match mahalanbis distance which is its own existing detector.
# To differentiate we use mixed, cosine and euclidean only.

def default_global_prototype_params() -> Dict[str, Any]:
    return {
        "metric": "cosine",
        "mixed_alpha": 0.5,
        "mixed_squared": False,
        "mixed_normalize_euclid": True,
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }


def sample_global_prototype_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None,
                                   **kwargs) -> Dict[str, Any]:
    max_layer = get_max_layer_index(dataset_info, architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)
    reducer_names = train_cache.reducer_name if train_cache else None
    reducer_names = [None] + reducer_names if reducer_names else [None]
    reducer_name = trial.suggest_categorical("reducer_name", reducer_names)
    if train_cache:
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "mixed"])
    params = {
        "metric": metric,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    if metric == "mixed":
        params["mixed_alpha"] = trial.suggest_float("mixed_alpha", 0.0, 1.0)
        params["mixed_squared"] = trial.suggest_categorical("mixed_squared", [True, False])
        params["mixed_normalize_euclid"] = trial.suggest_categorical("mixed_normalize_euclid", [True, False])
    return params


def create_global_prototype_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                    **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    if params.get("use_correct_only", False):
        embeddings_t, _, _ = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, _, _ = train_cache(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    detector = GlobalPrototypeConfidence(
        metric=params.get("metric", "euclidean"),
        mixed_alpha=params.get("mixed_alpha", 0.5),
        mixed_squared=params.get("mixed_squared", False),
        mixed_normalize_euclid=params.get("mixed_normalize_euclid", True),
    )
    device = kwargs.get("device", torch.device("cpu"))
    detector.to(device)
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
    detector.fit(embeddings_t.to(device))  # global fit needs only embeddings
    detector.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True,
                                                 reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Class Prototype ---

def default_class_prototype_params() -> Dict[str, Any]:
    return {
        "metric": "cosine",
        "mixed_alpha": 0.5,
        "mixed_squared": False,
        "mixed_normalize_euclid": True,
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }


def sample_class_prototype_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None,
                                  **kwargs) -> Dict[str, Any]:
    max_layer = get_max_layer_index(dataset_info, architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)
    reducer_names = train_cache.reducer_name if train_cache else None
    reducer_names = [None] + reducer_names if reducer_names else [None]
    reducer_name = trial.suggest_categorical("reducer_name", reducer_names)
    if train_cache:
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "mixed"])
    params = {
        "metric": metric,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    if metric == "mixed":
        params["mixed_alpha"] = trial.suggest_float("mixed_alpha", 0.0, 1.0)
        params["mixed_squared"] = trial.suggest_categorical("mixed_squared", [True, False])
        params["mixed_normalize_euclid"] = trial.suggest_categorical("mixed_normalize_euclid", [True, False])
    if metric == "mahalanobis":
        params["shared_covariance"] = True
    return params


def create_class_prototype_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                   **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    if params.get("use_correct_only", False):
        embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, _, classes_t = train_cache(
            layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
        )

    detector = ClassPrototypeConfidence(
        metric=params.get("metric", "euclidean"),
        shared_covariance=params.get("shared_covariance", False),
        mixed_alpha=params.get("mixed_alpha", 0.5),
        mixed_squared=params.get("mixed_squared", False),
        mixed_normalize_euclid=params.get("mixed_normalize_euclid", True),
    )
    device = kwargs.get("device", torch.device("cpu"))
    detector.to(device)
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
    detector.fit(embeddings_t.to(device), classes_t.to(device))
    detector.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True,
                                                 reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- All Class Prototype ---

def default_all_class_prototype_params() -> Dict[str, Any]:
    return {
        "metric": "cosine",
        "mixed_alpha": 0.5,
        "mixed_squared": False,
        "mixed_normalize_euclid": True,
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }


def sample_all_class_prototype_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None,
                                      **kwargs) -> Dict[str, Any]:
    max_layer = get_max_layer_index(dataset_info, architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)
    reducer_names = train_cache.reducer_name if train_cache else None
    reducer_names = [None] + reducer_names if reducer_names else [None]
    reducer_name = trial.suggest_categorical("reducer_name", reducer_names)
    if train_cache:
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "mixed"])
    params = {
        "metric": metric,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    if metric == "mixed":
        params["mixed_alpha"] = trial.suggest_float("mixed_alpha", 0.0, 1.0)
        params["mixed_squared"] = trial.suggest_categorical("mixed_squared", [True, False])
        params["mixed_normalize_euclid"] = trial.suggest_categorical("mixed_normalize_euclid", [True, False])
    if metric == "mahalanobis":
        params["shared_covariance"] = True
    return params


def create_all_class_prototype_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                       **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # Double-check reducer_name validity
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    if params.get("use_correct_only", False):
        embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
            layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
        )
    else:
        embeddings_t, _, classes_t = train_cache(
            layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
        )

    detector = ClassPrototypeConfidence(
        metric=params.get("metric", "euclidean"),
        shared_covariance=params.get("shared_covariance", False),
        mixed_alpha=params.get("mixed_alpha", 0.5),
        mixed_squared=params.get("mixed_squared", False),
        mixed_normalize_euclid=params.get("mixed_normalize_euclid", True),
    )
    device = kwargs.get("device", torch.device("cpu"))
    detector.to(device)
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
    detector.fit(embeddings_t.to(device), classes_t.to(device))
    detector.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True,
                                                 reducer_select=reducer_name)
    conf_split = SplitConfidence(detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration ---
OOD_DEFAULT_PARAM_FACTORIES["global_prototype"] = default_global_prototype_params  # single prototype
OOD_PARAM_SAMPLERS["global_prototype"] = sample_global_prototype_params
OOD_PROBLEM_FACTORIES["global_prototype"] = create_global_prototype_problem

OOD_DEFAULT_PARAM_FACTORIES[
    "per_class_prototype"] = default_class_prototype_params  # distance to predicted class prototype
OOD_PARAM_SAMPLERS["per_class_prototype"] = sample_class_prototype_params
OOD_PROBLEM_FACTORIES["per_class_prototype"] = create_class_prototype_problem

OOD_DEFAULT_PARAM_FACTORIES[
    "class_prototype"] = default_all_class_prototype_params  # difference is that this does not use predicted class.
OOD_PARAM_SAMPLERS["class_prototype"] = sample_all_class_prototype_params
OOD_PROBLEM_FACTORIES["class_prototype"] = create_all_class_prototype_problem
