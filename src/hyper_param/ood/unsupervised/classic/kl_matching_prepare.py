from typing import Dict, Any, Optional
import optuna
import torch

from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from model.get_model import get_network_layer
from src.utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.control.classify import ClassifyingConfidence
from confidence.unsupervised.classic.kl_matching import KLMatchingConfidence

# --- KLMatching ---

def default_kl_matching_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    """
    Default parameters for KLMatching prepare:
      - layer_index: index of network layer to extract logits/features from (default: -1 => final layer)
      - reducer_name: optional reducer to apply to embeddings
      - map_function: optional mapping applied to raw scores (None => use default)
    """
    return {
        "layer_index": 0,
        "reducer_name": None,
        "map_function": None,
    }

def sample_kl_matching_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    return {
        "layer_index": 0,
        "reducer_name": None,
        "map_function": None,
    }

def create_kl_matching_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Build a TransformationProblem using KLMatchingConfidence.
    Extracts embeddings/logits from train_cache for fitting.
    """
    layer_index = params.get("layer_index", -1)
    reducer_name = params.get("reducer_name", None)

    # get layer + io descriptor
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)


    _, logits_t, classes_t = train_cache.get_correct_embeddings(
        layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
    )

    # build and fit detector
    kl_detector = KLMatchingConfidence(model=train_cache.model, map_function=params.get("map_function", None))
    kl_detector.fit(logits_t, y=classes_t)

    conf_mod = SinglePassConfidence(train_cache.model, kl_detector)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration ---
OOD_DEFAULT_PARAM_FACTORIES["kl_matching"] = default_kl_matching_params
OOD_PARAM_SAMPLERS["kl_matching"] = sample_kl_matching_params
OOD_PROBLEM_FACTORIES["kl_matching"] = create_kl_matching_problem