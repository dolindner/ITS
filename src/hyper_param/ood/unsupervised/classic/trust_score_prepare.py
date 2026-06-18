from typing import Dict, Any
import optuna
import torch

from confidence.unsupervised.classic.trust_score import TrustScoreTorchConfidence
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from src.utils.transformation_problem import TransformationProblem
from model.get_model import get_max_layer_index, get_network_layer


# --- Trust Score ---

def default_trust_score_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    params = {
        "k_neighbors": 5,
        "k_distance": None,
        "eps": 1e-10,
        "dtype": "float32", #not used currently, not implemented properly for trust score.
        "layer_index": 0,
        "reducer_name": None,
        "alpha": 0.0,
        "split_b": 0.0,
        "use_correct_only": False,
        "distance": "euclidean",
    }
    if train_cache is not None:
        layer_index = params["layer_index"]
        reducer_name = params["reducer_name"]
        layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            params["reducer_name"] = None
    return params

def sample_trust_score_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
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
        "k_neighbors": trial.suggest_int("k_neighbors", 1, 50),
        "k_distance": trial.suggest_categorical("k_distance", [None, 1, 3, 5, 10]),
        "eps": 1e-10,
        "dtype": "float32",
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "alpha": trial.suggest_float("alpha", 0.0, 0.5),
        "split_b": 0.0,
        "use_correct_only": trial.suggest_categorical("use_correct_only", [True, False]),
        "distance": trial.suggest_categorical("distance", ["euclidean", "cosine"]),
    }

def create_trust_score_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    # 1. get embeddings
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    # --- Reducer validity check ---
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
    # 2. build and fit detector
    detector = TrustScoreTorchConfidence(
        k_neighbors=params.get("k_neighbors", 5),
        k_distance=params.get("k_distance", None),
        eps=params.get("eps", 1e-10),
        input_transform=None,
        alpha=params.get("alpha", 0.0),
        distance=params.get("distance", "euclidean")
    )
    device = kwargs.get("device", torch.device("cpu"))
    detector.to(device)
    detector.fit(embeddings_t.to(device), classes_t.to(device))
    detector.to(device)
    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select=reducer_name)
    conf_split = PredictedSplitConfidence(detector, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Registration ---
OOD_DEFAULT_PARAM_FACTORIES["trust_score"] = default_trust_score_params
OOD_PARAM_SAMPLERS["trust_score"] = sample_trust_score_params
OOD_PROBLEM_FACTORIES["trust_score"] = create_trust_score_problem