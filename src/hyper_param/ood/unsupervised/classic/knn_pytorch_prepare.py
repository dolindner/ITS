from typing import Dict, Any

import optuna
import torch

from confidence.control.split import PredictedSplitConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.input_transform import InputTransform
from confidence.model.single_pass import SinglePassConfidence
from confidence.unsupervised.classic.nn_pytorch import KNNConfidence, PerClassKNNConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from model.get_model import get_max_layer_index, get_network_layer
from src.utils.transformation_problem import TransformationProblem


# If transform sequence needs to be constructed from dataset_info:


# --- KNNConfidence ---
def default_knn_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    params = {
        "k": 3,
        "metric": "cosine",
        "dtype": "float16",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
    }

    if train_cache is not None:
        layer_index = params["layer_index"]
        reducer_name = params["reducer_name"]
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            params["reducer_name"] = None
    return params


def sample_knn_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    # Get available reducers from the cache if provided
    dataset_info = kwargs.get("dataset_info", None)
    # Get max layer index
    max_layer = get_max_layer_index(dataset_info, architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)
    # get possible reudcers from train cache.
    reducer_names = train_cache.reducer_name if train_cache else None
    # add none
    reducer_names = [None] + reducer_names if reducer_names else [None]
    reducer_name = trial.suggest_categorical("reducer_name", reducer_names)
    if train_cache:
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
        # quick dirty check, remove later.
        if "resnet" in architecture.lower() and layer_index > 0:
            if len(available_reducers) == 0:
                raise ValueError(
                    f"No available reducers for layer {layer_index} in architecture {architecture} with available reducers {available_reducers}")

    metric = trial.suggest_categorical("metric", ["euclidean", "cosine"])
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": metric,
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
    }
    return params


def create_knn_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                       **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with KNNConfidence."""
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    if train_cache is not None:
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

    # 2. Build and fit detector
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    knn_detector = KNNConfidence(
        k=params["k"],
        metric=params["metric"],
        dtype=dtype_map.get(params.get("dtype", "float32"), torch.float32),
        mixed_alpha=params.get("mixed_alpha", 0.0),
        mixed_squared=params.get("mixed_squared", False),
        mixed_normalize_euclid=params.get("mixed_normalize_euclid", True),
    )
    # get device from kwargs
    device = kwargs.get("device", torch.device("cpu"))
    knn_detector.to(device)
    knn_detector.fit(embeddings_t, classes_t)
    knn_detector.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True,
                                                 reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(knn_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- PerClassKNNConfidence ---

def default_per_class_knn_params() -> Dict[str, Any]:
    return {
        "k": 3,
        "metric": "cosine",
        "computation_mode": "masked",
        "dtype": "float16",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
    }


def sample_per_class_knn_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine"])
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": metric,
        "computation_mode": "masked",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0
    }
    return params


def create_per_class_knn_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                 **kwargs) -> TransformationProblem:
    # 1. Get embeddings
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    if train_cache is not None:
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

    # 2. Build and fit detector
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    pcknn_detector = PerClassKNNConfidence(
        k=params["k"],
        metric=params["metric"],
        computation_mode=params["computation_mode"],
        dtype=dtype_map.get(params.get("dtype", "float32"), torch.float32),
        mixed_alpha=params.get("mixed_alpha", 0.0),
        mixed_squared=params.get("mixed_squared", False),
        mixed_normalize_euclid=params.get("mixed_normalize_euclid", True),
        shared_covariance=params.get("shared_covariance", False),
    )
    pcknn_detector.fit(embeddings_t, classes_t)

    # 3. Create the full confidence module structure
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True,
                                                 reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(pcknn_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- KNNConfidence Trap (euclidean, cosine, mixed) ---

def default_knn_trap_params() -> Dict[str, Any]:
    return default_knn_params()


def sample_knn_trap_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "mixed", "mahalanobis"])
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": metric,
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
    }
    if metric == "mixed":
        params["mixed_alpha"] = trial.suggest_float("mixed_alpha", 0.0, 1.0)
        params["mixed_squared"] = trial.suggest_categorical("mixed_squared", [True, False])
        params["mixed_normalize_euclid"] = trial.suggest_categorical("mixed_normalize_euclid", [True, False])
    if metric == "mahalanobis":
        pass
    return params


def create_knn_trap_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                            **kwargs) -> TransformationProblem:
    return create_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- PerClassKNNConfidence Trap (euclidean, cosine, mixed, mahalanobis) ---

def default_per_class_knn_trap_params() -> Dict[str, Any]:
    return default_per_class_knn_params()


def sample_per_class_knn_trap_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[
    str, Any]:
    metric = trial.suggest_categorical("metric", ["euclidean", "cosine", "mixed"])
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": metric,
        "computation_mode": "masked",
        "dtype": "float16",
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
        pass
    return params


def create_per_class_knn_trap_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                      **kwargs) -> TransformationProblem:
    return create_per_class_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- KNN Mixed ---

def default_knn_mixed_params() -> Dict[str, Any]:
    params = default_knn_params()
    params["metric"] = "mixed"
    params["mixed_alpha"] = 0.0
    return params


def sample_knn_mixed_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": "mixed",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "mixed_alpha": trial.suggest_float("mixed_alpha", 0.0, 1.0),
        "mixed_squared": trial.suggest_categorical("mixed_squared", [True, False]),
        "mixed_normalize_euclid": trial.suggest_categorical("mixed_normalize_euclid", [True, False]),
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params


def create_knn_mixed_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                             **kwargs) -> TransformationProblem:
    return create_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- Per-Class KNN Mixed ---

def default_per_class_knn_mixed_params() -> Dict[str, Any]:
    params = default_per_class_knn_params()
    params["metric"] = "mixed"
    return params


def sample_per_class_knn_mixed_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[
    str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": "mixed",
        "computation_mode": "masked",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "mixed_alpha": trial.suggest_float("mixed_alpha", 0.0, 1.0),
        "mixed_squared": trial.suggest_categorical("mixed_squared", [True, False]),
        "mixed_normalize_euclid": trial.suggest_categorical("mixed_normalize_euclid", [True, False]),
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params


def create_per_class_knn_mixed_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                       **kwargs) -> TransformationProblem:
    return create_per_class_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- KNN Mahalanobis ---

def default_knn_mahalanobis_params() -> Dict[str, Any]:
    params = default_knn_params()
    params["metric"] = "mahalanobis"
    return params


def sample_knn_mahalanobis_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": "mahalanobis",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params


def create_knn_mahalanobis_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                   **kwargs) -> TransformationProblem:
    return create_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- Per-Class KNN Mahalanobis ---

def default_per_class_knn_mahalanobis_params() -> Dict[str, Any]:
    params = default_per_class_knn_params()
    params["metric"] = "mahalanobis"
    return params


def sample_per_class_knn_mahalanobis_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[
    str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": "mahalanobis",
        "computation_mode": "masked",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "shared_covariance": True,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params


def create_per_class_knn_mahalanobis_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info,
                                             architecture, **kwargs) -> TransformationProblem:
    return create_per_class_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- KNN Mixed FAISS ---

def default_knn_mixed_faiss_params() -> Dict[str, Any]:
    params = default_knn_params()
    params["metric"] = "mixed_faiss"
    # ensure default alpha is explicit and covered by samplers / logging
    params["mixed_alpha"] = 0.0  # Changed from 0.0 to match the function's default in nn_pytorch.py
    return params


def sample_knn_mixed_faiss_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": "mixed_faiss",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "mixed_alpha": trial.suggest_float("mixed_alpha", 0.0, 1.0),
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params


def create_knn_mixed_faiss_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                                   **kwargs) -> TransformationProblem:
    return create_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- Per-Class KNN Mixed FAISS ---

def default_per_class_knn_mixed_faiss_params() -> Dict[str, Any]:
    params = default_per_class_knn_params()
    params["metric"] = "mixed_faiss"
    # explicit default alpha for per-class mixed_faiss
    params["mixed_alpha"] = 0.0  # Changed from 0.0 to match the function's default in nn_pytorch.py
    return params


def sample_per_class_knn_mixed_faiss_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[
    str, Any]:
    dataset_info = kwargs.get("dataset_info", None)
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
    params = {
        "k": trial.suggest_int("k", 1, 50),
        "metric": "mixed_faiss",
        "computation_mode": "masked",
        "dtype": "float16",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "mixed_alpha": trial.suggest_float("mixed_alpha", 0.0, 1.0),
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }
    return params


def create_per_class_knn_mixed_faiss_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info,
                                             architecture, **kwargs) -> TransformationProblem:
    return create_per_class_knn_problem(params, train_cache, transform_seq, dataset_info, architecture, **kwargs)


# --- KNN with InputTransform ---

def default_knn_itf_params() -> Dict[str, Any]:
    params = default_knn_params()
    params["itf_standardize"] = True
    params["itf_whiten"] = False
    return params


def sample_knn_itf_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    params = sample_knn_params(trial, train_cache, architecture, **kwargs)
    params["itf_standardize"] = trial.suggest_categorical("itf_standardize", [True, False])
    params["itf_whiten"] = trial.suggest_categorical("itf_whiten", [True, False])
    return params


def create_knn_itf_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                           **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with InputTransform + KNNConfidence."""
    # 1. Get embeddings
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # --- Reducer validity check ---
    if train_cache is not None:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    embeddings_t, _, classes_t = train_cache(
        layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True, reducer_select=reducer_name
    )
    device = kwargs.get("device", torch.device("cpu"))

    # 2. Build and fit InputTransform
    itf = InputTransform(
        standardize=params.get("itf_standardize", False),
        whiten=params.get("itf_whiten", False),
        robust_cov=False  # as requested
    )
    itf.to(device)
    itf.fit(embeddings_t)

    # 3. Build and fit KNN detector on transformed data
    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    knn_detector = KNNConfidence(
        k=params["k"],
        metric=params["metric"],
        dtype=dtype_map.get(params.get("dtype", "float32"), torch.float32),
        input_transform=itf,
    )
    knn_detector.to(device)
    knn_detector.fit(embeddings_t, classes_t)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True,
                                                 reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(knn_detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["knn"] = default_knn_params
OOD_PARAM_SAMPLERS["knn"] = sample_knn_params
OOD_PROBLEM_FACTORIES["knn"] = create_knn_problem

OOD_DEFAULT_PARAM_FACTORIES["per_class_knn"] = default_per_class_knn_params
OOD_PARAM_SAMPLERS["per_class_knn"] = sample_per_class_knn_params
OOD_PROBLEM_FACTORIES["per_class_knn"] = create_per_class_knn_problem

# Trap versions
OOD_DEFAULT_PARAM_FACTORIES["knn_trap"] = default_knn_trap_params
OOD_PARAM_SAMPLERS["knn_trap"] = sample_knn_trap_params
OOD_PROBLEM_FACTORIES["knn_trap"] = create_knn_trap_problem

OOD_DEFAULT_PARAM_FACTORIES["per_class_knn_trap"] = default_per_class_knn_trap_params
OOD_PARAM_SAMPLERS["per_class_knn_trap"] = sample_per_class_knn_trap_params
OOD_PROBLEM_FACTORIES["per_class_knn_trap"] = create_per_class_knn_trap_problem

# Mixed versions
OOD_DEFAULT_PARAM_FACTORIES["knn_mixed"] = default_knn_mixed_params
OOD_PARAM_SAMPLERS["knn_mixed"] = sample_knn_mixed_params
OOD_PROBLEM_FACTORIES["knn_mixed"] = create_knn_mixed_problem

OOD_DEFAULT_PARAM_FACTORIES["per_class_knn_mixed"] = default_per_class_knn_mixed_params
OOD_PARAM_SAMPLERS["per_class_knn_mixed"] = sample_per_class_knn_mixed_params
OOD_PROBLEM_FACTORIES["per_class_knn_mixed"] = create_per_class_knn_mixed_problem

# Mahalanobis versions
OOD_DEFAULT_PARAM_FACTORIES["knn_mahalanobis"] = default_knn_mahalanobis_params
OOD_PARAM_SAMPLERS["knn_mahalanobis"] = sample_knn_mahalanobis_params
OOD_PROBLEM_FACTORIES["knn_mahalanobis"] = create_knn_mahalanobis_problem

OOD_DEFAULT_PARAM_FACTORIES["per_class_knn_mahalanobis"] = default_per_class_knn_mahalanobis_params
OOD_PARAM_SAMPLERS["per_class_knn_mahalanobis"] = sample_per_class_knn_mahalanobis_params
OOD_PROBLEM_FACTORIES["per_class_knn_mahalanobis"] = create_per_class_knn_mahalanobis_problem

# Mixed FAISS versions
OOD_DEFAULT_PARAM_FACTORIES["knn_mixed_faiss"] = default_knn_mixed_faiss_params
OOD_PARAM_SAMPLERS["knn_mixed_faiss"] = sample_knn_mixed_faiss_params
OOD_PROBLEM_FACTORIES["knn_mixed_faiss"] = create_knn_mixed_faiss_problem

OOD_DEFAULT_PARAM_FACTORIES["per_class_knn_mixed_faiss"] = default_per_class_knn_mixed_faiss_params
OOD_PARAM_SAMPLERS["per_class_knn_mixed_faiss"] = sample_per_class_knn_mixed_faiss_params
OOD_PROBLEM_FACTORIES["per_class_knn_mixed_faiss"] = create_per_class_knn_mixed_faiss_problem

# KNN with InputTransform
OOD_DEFAULT_PARAM_FACTORIES["knn_itf"] = default_knn_itf_params
OOD_PARAM_SAMPLERS["knn_itf"] = sample_knn_itf_params
OOD_PROBLEM_FACTORIES["knn_itf"] = create_knn_itf_problem
