from typing import Dict, Any

import optuna
import torch
from tqdm import tqdm

from confidence.direct.logit_based import EnergyConfidence
from confidence.direct.prob_based import EntropyConfidence
from confidence.model.ash import ReActConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
)
from model.basic_networks import get_max_split_pos_for_flexible_resnet, split_flexible_resnet_for_ash
from src.utils.transformation_problem import TransformationProblem


def default_react_all_params() -> Dict[str, Any]:
    """Default parameters for ReAct All. The percentile of 0.9 is from the paper."""
    return {
        "percentile": 0.9,
        "confidence_type": "energy",
        "use_feature_confidence": False,
        "split_pos": 0,
        "threshold": None,  # if set, percentile is ignored
    }


def sample_react_all_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    """Sample hyperparameters for ReAct All."""
    max_split = get_max_split_pos_for_flexible_resnet(train_cache.model)
    params = {
        "confidence_type": trial.suggest_categorical("confidence_type", ["energy", "entropy"]),
        "use_feature_confidence": False,
        "split_pos": trial.suggest_int("split_pos", 0, max_split),
        "percentile": trial.suggest_float("percentile", 0.8, 0.999),
        "threshold": None,
    }

    return params


def create_react_all_problem(
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
    This version splits the model like ASH and handles thresholding flexibly.
    """
    percentile = params.get("percentile")
    confidence_type = params.get("confidence_type", "energy")
    use_feature_confidence = params.get("use_feature_confidence", False)
    split_pos = params.get("split_pos", 0)
    threshold = params.get("threshold")

    backbone, head = split_flexible_resnet_for_ash(model, split_pos)

    if confidence_type == "energy":
        confidence = EnergyConfidence()
    elif confidence_type == "entropy":
        confidence = EntropyConfidence(input_logits=True)
    else:
        raise ValueError(f"Invalid confidence_type: {confidence_type}")

    conf_mod = ReActConfidence(
        backbone=backbone,
        head=head,
        percentile=percentile,
        threshold=threshold,
        index_feat=None,
        index_logits=None,
        confidence=confidence,
        use_feature_confidence=use_feature_confidence,
    )

    # Fit the ReAct module to calculate the threshold if not provided
    if threshold is None:
        # Manually iterate over a subset of the dataloader to get features from the backbone
        device = next(backbone.parameters()).device
        backbone.eval()
        features_list = []
        num_batches_to_sample = 10
        with torch.no_grad():
            for i, batch in enumerate(tqdm(train_cache.dataloader, desc="Sampling for ReAct threshold", leave=False)):
                if i >= num_batches_to_sample:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:  # dict
                    x = batch.get("x")

                x = x.to(device)
                features = backbone(x)
                features_list.append(features.cpu())

        embeddings_t = torch.cat(features_list, dim=0)
        conf_mod.fit(embeddings_t)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# Register into base_prepare registries
OOD_DEFAULT_PARAM_FACTORIES["react_all"] = default_react_all_params
OOD_PARAM_SAMPLERS["react_all"] = sample_react_all_params
OOD_PROBLEM_FACTORIES["react_all"] = create_react_all_problem
