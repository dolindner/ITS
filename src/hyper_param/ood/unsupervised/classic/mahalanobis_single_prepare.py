from typing import Dict, Any
import optuna
import torch

from confidence.unsupervised.classic.mahalanobis import PrototypeMahalanobisConfidence
from confidence.unsupervised.classic.mahalanobis_relative import RelativeMahalanobisConfidence
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import SplitConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from src.utils.transformation_problem import TransformationProblem
from model.get_model import get_max_layer_index, get_network_layer

# --- Single Layer Mahalanobis ---

def default_single_mahalanobis_params() -> Dict[str, Any]:
    return {
        "shared_covariance": True,
        "use_raw_scatter": False,
        "eps": 1e-6,
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }

def sample_single_mahalanobis_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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
    return {
        "shared_covariance": True,
        "use_raw_scatter": False,
        "eps": 1e-6,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_single_mahalanobis_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
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

    detector = PrototypeMahalanobisConfidence(
        eps=params.get("eps", 1e-6),
        shared_covariance=params.get("shared_covariance", True),
        use_raw_scatter=params.get("use_raw_scatter", False),
        cov_mode=params.get("cov_mode", "full"),
        low_rank_r=params.get("low_rank_r", 64),
    )
    device = kwargs.get("device", torch.device("cpu"))
    detector.to(device)
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
    detector.fit(embeddings_t.to(device), classes_t.to(device))
    detector.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_split = SplitConfidence(detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Single Layer RMD ---

def default_single_rmd_params() -> Dict[str, Any]:
    return {
        "shared_covariance": True,
        "use_raw_scatter": False,
        "mahalanobis_eps": 1e-6,
        "dtype": "float32",
        "layer_index": 0,
        "reducer_name": None,
        "split_b": 0.0,
        "use_correct_only": False,
    }

def sample_single_rmd_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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
    return {
        "shared_covariance": True,
        "use_raw_scatter": False,
        "mahalanobis_eps": 1e-6,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
    }

def create_single_rmd_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
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

    detector = RelativeMahalanobisConfidence(
        mahalanobis_eps=params.get("mahalanobis_eps", 1e-6),
        shared_covariance=params.get("shared_covariance", True),
        use_raw_scatter=params.get("use_raw_scatter", False),
        cov_mode=params.get("cov_mode", "full"),
        low_rank_r=params.get("low_rank_r", 64),
        global_cov_mode=params.get("global_cov_mode", None),
    )
    device = kwargs.get("device", torch.device("cpu"))
    detector.to(device)
    if params.get("dtype") == "float16":
        embeddings_t = embeddings_t.half()
    detector.fit(embeddings_t.to(device), classes_t.to(device))
    detector.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_split = SplitConfidence(detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration (updated) ---
OOD_DEFAULT_PARAM_FACTORIES["single_mahalanobis"] = default_single_mahalanobis_params
OOD_PARAM_SAMPLERS["single_mahalanobis"] = sample_single_mahalanobis_params
OOD_PROBLEM_FACTORIES["single_mahalanobis"] = create_single_mahalanobis_problem


OOD_DEFAULT_PARAM_FACTORIES["single_rmd"] = default_single_rmd_params
OOD_PARAM_SAMPLERS["single_rmd"] = sample_single_rmd_params
OOD_PROBLEM_FACTORIES["single_rmd"] = create_single_rmd_problem

