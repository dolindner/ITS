from typing import Dict, Any

import optuna

from confidence.direct.logit_based import EnergyConfidence
from confidence.direct.prob_based import EntropyConfidence
from confidence.model.ash import ReActConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
)
from model.basic_networks import find_last_linear_layer
from model.get_model import get_network_layer
from src.utils.transformation_problem import TransformationProblem


def default_react_params() -> Dict[str, Any]:
    """Default parameters for ReAct. The percentile of 0.9 is from the paper."""
    return {
        "percentile": 0.9,
        "layer_index": 0,  # Fixed to penultimate layer
        "index_logits": None,
        "confidence_type": "energy",
        "use_feature_confidence": False,
    }


def sample_react_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    """Sample hyperparameters for ReAct."""
    return {
        "percentile": trial.suggest_float("percentile", 0.5, 0.9999999),
        "layer_index": 0,  # Fixed to penultimate layer
        "index_logits": None,
        "confidence_type": trial.suggest_categorical("confidence_type", ["energy", "entropy"]),
        "use_feature_confidence": False,
    }


def create_react_problem(
        params: Dict[str, Any],
        model,
        train_cache,
        transform_seq,
        dataset_info,
        architecture,
        **kwargs
) -> TransformationProblem:
    """
    Build and return a TransformationProblem that wraps backbone+head with ReActConfidence.
    This version fits the threshold based on a percentile of training data activations.
    """
    percentile = params.get("percentile", 0.9)
    layer_index = 0  # ReAct only supports the last layer's input
    index_logits = params.get("index_logits", None)
    confidence_type = params.get("confidence_type", "energy")
    use_feature_confidence = params.get("use_feature_confidence", False)

    # 1. Get layer object and create a dual-output model that returns (final_output, layer_features)
    layer_obj, layer_io = get_network_layer(dataset_info, architecture, layer_index)
    dual_output_model = train_cache.make_wrapper(layer_obj, capture_modes=layer_io, concat=False, flatten=True)

    # Extract the head (last linear layer) from the original model
    head = find_last_linear_layer(model)

    # Instantiate confidence based on type
    if confidence_type == "energy":
        confidence = EnergyConfidence()
    elif confidence_type == "entropy":
        confidence = EntropyConfidence(input_logits=True)
    else:
        raise ValueError(f"Invalid confidence_type: {confidence_type}")

    # 2. Instantiate ReActConfidence module
    # The dual_output_model acts as the backbone. index_feat=0 selects the layer features from its output tuple.
    conf_mod = ReActConfidence(
        backbone=dual_output_model,
        head=head,
        percentile=percentile,
        index_feat=0,  # Changed to 0
        index_logits=1,  # Changed to 1
        confidence=confidence,
        use_feature_confidence=use_feature_confidence,
    )

    # 3. Get training embeddings for the chosen layer to fit the threshold
    # Use the train_cache to efficiently get features
    embeddings_t = train_cache(
        layer_obj, capture_modes=layer_io, flatten=True, return_y=False, return_final=False
    )

    # 4. Fit the ReAct module to calculate the threshold
    conf_mod.fit(embeddings_t)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Register into base_prepare registries
OOD_DEFAULT_PARAM_FACTORIES["react"] = default_react_params
OOD_PARAM_SAMPLERS["react"] = sample_react_params
OOD_PROBLEM_FACTORIES["react"] = create_react_problem
