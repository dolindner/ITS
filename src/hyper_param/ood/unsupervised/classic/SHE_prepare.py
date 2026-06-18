from typing import Dict, Any
import optuna
import torch

from confidence.unsupervised.classic.SHE import SHETorchConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from src.utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.control.split import PredictedSplitConfidence
from model.get_model import get_max_layer_index, get_network_layer


def default_she_params() -> Dict[str, Any]:
    return {
        "layer_index": 0,
        "reducer_name": None,
        "penalize_missmatches": True,
        "split_b": 0.0,
    }

def sample_she_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    max_layer = get_max_layer_index(kwargs.get("dataset_info", None), architecture)
    layer_index = trial.suggest_int("layer_index", 0, max_layer)
    reducer_names = train_cache.reducer_name if train_cache else None
    reducer_names = [None] + reducer_names if reducer_names else [None]
    reducer_name = trial.suggest_categorical("reducer_name", reducer_names)
    #sampel reducer from the ones defined in the cache.
    if train_cache:
        layer, layer_io = get_network_layer(kwargs.get("dataset_info", None), architecture, layer_index)
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    return {
        "layer_index": layer_index,
        "reducer_name": reducer_name,
        "penalize_missmatches": trial.suggest_categorical("penalize_missmatches", [True, False]),
        "split_b": trial.suggest_float("split_b", 0.0, 0.0),
    }

def create_she_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """Factory for creating a TransformationProblem with SHETorchConfidence."""
    layer_index = params.get("layer_index", 0)
    reducer_name = params.get("reducer_name", None)
    layer, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    if train_cache:
        available_reducers = train_cache.get_available_reducers(layer, layer_io)
        if reducer_name not in available_reducers:
            reducer_name = None
    embeddings_t, _, classes_t = train_cache.get_correct_embeddings(
        layer, capture_modes=layer_io, flatten=True, reducer_select=reducer_name
    )


    she = SHETorchConfidence()
    she.penalize_missmatches = params.get("penalize_missmatches", True)

    device = kwargs.get("device", torch.device("cpu"))
    she.to(device)
    she.fit(embeddings_t.to(device), classes_t.to(device))
    she.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True, reducer_select= reducer_name)
    conf_split = PredictedSplitConfidence(she, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Register
OOD_DEFAULT_PARAM_FACTORIES["she"] = default_she_params
OOD_PARAM_SAMPLERS["she"] = sample_she_params
OOD_PROBLEM_FACTORIES["she"] = create_she_problem