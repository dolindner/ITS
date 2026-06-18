from typing import Dict, Any

import optuna

from confidence.direct.logit_based import EnergyConfidence
from confidence.direct.prob_based import EntropyConfidence
from confidence.model.ash import ASHConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
)
from model.basic_networks import get_max_split_pos_for_flexible_resnet, split_flexible_resnet_for_ash
from src.utils.transformation_problem import TransformationProblem


def default_ash_params() -> Dict[str, Any]:
    return {
        "variant": "ash-b",
        "percentile": 0.65,
        "index_feat": None,
        "index_logits": None,
        "confidence_type": "energy",
        "use_feature_confidence": False,
        "split_pos": 0,  # split position for FlexibleResNet (Image data only)
    }


def sample_ash_params(trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
    max_split = get_max_split_pos_for_flexible_resnet(kwargs.get("train_cache", None).model)
    return {
        "variant": trial.suggest_categorical("variant", ["ash-s", "ash-b", "ash-p"]),
        "percentile": trial.suggest_float("percentile", 0.05, 0.9999),
        "index_feat": None,
        "index_logits": None,
        "confidence_type": trial.suggest_categorical("confidence_type", ["energy", "entropy"]),
        "use_feature_confidence": False,
        "split_pos": trial.suggest_int("split_pos", 0, max_split),  # New: sample split position
    }


def create_ash_problem(
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
    """
    variant = params.get("variant", "ash-s")
    percentile = params.get("percentile", 0.65)
    index_logits = params.get("index_logits", None)
    confidence_type = params.get("confidence_type", "energy")
    use_feature_confidence = params.get("use_feature_confidence", False)
    split_pos = params.get("split_pos", 0)  # New: get split position

    # Split model into backbone and head for ASH
    backbone, head = split_flexible_resnet_for_ash(model, split_pos)

    # Instantiate confidence based on type
    if confidence_type == "energy":
        confidence = EnergyConfidence()
    elif confidence_type == "entropy":
        confidence = EntropyConfidence(input_logits=True)
    else:
        raise ValueError(f"Invalid confidence_type: {confidence_type}")

    # 2. Instantiate ASHConfidence module
    conf_mod = ASHConfidence(
        backbone=backbone,
        head=head,
        variant=variant,
        percentile=percentile,
        index_feat=None,
        index_logits=None,
        confidence=confidence,
        use_feature_confidence=use_feature_confidence,
    )

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Register into base_prepare registries
OOD_DEFAULT_PARAM_FACTORIES["ash"] = default_ash_params
OOD_PARAM_SAMPLERS["ash"] = sample_ash_params
OOD_PROBLEM_FACTORIES["ash"] = create_ash_problem
