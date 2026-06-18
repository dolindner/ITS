from typing import Dict, Any
import optuna
import torch

from confidence.direct.logit_based import EnergyConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
)
from model.basic_networks import find_last_linear_layer
from model.get_model import get_network_layer
from src.utils.transformation_problem import TransformationProblem

from confidence.model.ash import ASHConfidence
from confidence.direct.prob_based import EntropyConfidence

#Vriant of ash only for the last layer that doesnt rely on splitting a model.

def default_ash_last_params() -> Dict[str, Any]:
    """Default parameters for ASH applied to the last layer."""
    return {
        "percentile": 0.65,
        "variant": "ash-b",
        "layer_index": 0,  # penultimate / selected layer index for the dual-output backbone wrapper
        "index_logits": None,
        "confidence_type": "energy",
        "use_feature_confidence": False,
    }

def sample_ash_last_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    """Sample hyperparameters for ASH-last."""
    return {
        "variant": trial.suggest_categorical("variant", ["ash-s", "ash-b", "ash-p"]),
        "percentile": trial.suggest_float("percentile", 0.05, 0.9999),
        "layer_index": 0,
        "index_logits": None,
        "confidence_type": trial.suggest_categorical("confidence_type", ["energy", "entropy"]),
        "use_feature_confidence": False,
    }

def create_ash_last_problem(
    params: Dict[str, Any],
    model,
    train_cache,
    transform_seq,
    dataset_info,
    architecture,
    **kwargs
) -> TransformationProblem:
    """
    Build and return a TransformationProblem that wraps backbone+head with ASHConfidence.
    This version fits/infer the image_shape (if needed) based on training embeddings.
    """
    variant = params.get("variant", "ash-s")
    percentile = params.get("percentile", 0.65)
    layer_index = 0  # last layer input selection for dual-output wrapper
    index_logits = params.get("index_logits", None)
    confidence_type = params.get("confidence_type", "energy")
    use_feature_confidence = params.get("use_feature_confidence", False)
    provided_image_shape = params.get("image_shape", None)

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



    # 2. Instantiate ASHConfidence module
    conf_mod = ASHConfidence(
        backbone=dual_output_model,
        head=head,
        variant=variant,
        percentile=percentile,
        index_feat=0,   # the wrapper returns features at index 0
        index_logits=1, # logits from wrapper at index 1 when applicable
        confidence=confidence,
        use_feature_confidence=use_feature_confidence,
    )




    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Register into base_prepare registries
OOD_DEFAULT_PARAM_FACTORIES["ash_last"] = default_ash_last_params
OOD_PARAM_SAMPLERS["ash_last"] = sample_ash_last_params
OOD_PROBLEM_FACTORIES["ash_last"] = create_ash_last_problem
