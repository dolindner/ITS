from typing import Dict, Any

import optuna

from confidence.control.classify import ClassifyingConfidence
from confidence.model.single_pass import SinglePassConfidence
from confidence.unsupervised.classic.openmax import OpenMaxConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from model.get_model import get_network_layer
from src.utils.transformation_problem import TransformationProblem


# --- OpenMax ---

def default_openmax_params(train_cache=None, dataset_info=None, architecture=None, **kwargs) -> Dict[str, Any]:
    return {
        "tail_size": 25,
        "alpha": 10,
        "euclid_weight": 0.5,
        "layer_index": 0,
        "reducer_name": None,
    }


def sample_openmax_params(trial: optuna.Trial, train_cache=None, dataset_info=None, architecture=None, **kwargs) -> \
Dict[str, Any]:
    tail_size = trial.suggest_int("tail_size", 5, 50)
    alpha = trial.suggest_int("alpha", 1, 20)
    euclid_weight = trial.suggest_float("euclid_weight", 0.0, 1.0)
    return {
        "tail_size": tail_size,
        "alpha": alpha,
        "euclid_weight": euclid_weight,
        "layer_index": 0,
        "reducer_name": None,
    }


def create_openmax_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture,
                           **kwargs) -> TransformationProblem:
    layer_index = params.get("layer_index", -1)
    reducer_name = params.get("reducer_name", None)

    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)

    # second output from get_correct_embeddings is the logits
    _, logits_t, classes_t = train_cache.get_correct_embeddings(
        layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
    )

    openmax_detector = OpenMaxConfidence(
        tail_size=params.get("tail_size", 25),
        alpha=params.get("alpha", 10),
        euclid_weight=params.get("euclid_weight", 0.5),
        input_is_logits=True
    )
    openmax_detector.fit(logits_t, y=classes_t)

    conf_split = ClassifyingConfidence(openmax_detector)
    conf_mod = SinglePassConfidence(train_cache.model, conf_split)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration ---
OOD_DEFAULT_PARAM_FACTORIES["openmax"] = default_openmax_params
OOD_PARAM_SAMPLERS["openmax"] = sample_openmax_params
OOD_PROBLEM_FACTORIES["openmax"] = create_openmax_problem
