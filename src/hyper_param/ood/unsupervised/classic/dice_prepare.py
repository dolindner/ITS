from typing import Dict, Any
import optuna
import torch

from confidence.unsupervised.classic.dice import DICEConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from model.get_model import get_network_layer
from src.utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence
from confidence.direct.prob_based import EntropyConfidence
from confidence.control.split import PredictedSplitConfidence


def default_dice_params() -> Dict[str, Any]:
    return {
        "percentile": 0.1,
        # note: DICE always uses the last layer; layer_index and reducers are not supported here
        "split_b": 0.0,
        "confidence_type": "energy",
    }

def sample_dice_params(trial: optuna.Trial, train_cache=None, architecture=None, **kwargs) -> Dict[str, Any]:
    # DICE does not support reducers and always uses the last layer, so only sample percentile and split.
    return {
        "percentile": trial.suggest_float("percentile", 0.1,0.99),
        "split_b": trial.suggest_float("split_b", 0.0, 0.0),
        "confidence_type": trial.suggest_categorical("confidence_type", ["energy", "entropy"]),
    }

def create_dice_problem(params: Dict[str, Any], train_cache, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Factory for creating a TransformationProblem with DICEConfidence.

    Notes:
      - DICE always uses the penultimate layer (input to the last linear layer). Any layer_index in params is ignored.
      - Reducer selection is not supported for DICE; embeddings are taken raw (flattened).
    """
    # Determine penultimate layer (index 0) for DICE
    layer, layer_io = get_network_layer(dataset_info, architecture, 0)

    # Extract embeddings from the penultimate layer (no reducer)
    embeddings_t, _, classes_t = train_cache(
        layer, capture_modes=layer_io, flatten=True, return_y=True, return_final=True
    )

    # Build and fit DICE detector
    device = kwargs.get("device", torch.device("cpu"))
    # Instantiate confidence based on type
    if params["confidence_type"] == "energy":
        confidence = EnergyConfidence()
    elif params["confidence_type"] == "entropy":
        confidence = EntropyConfidence(input_logits=True)
    else:
        raise ValueError(f"Invalid confidence_type: {params['confidence_type']}")
    dice_conf = DICEConfidence(model=kwargs.get("model"), percentile=params["percentile"], confidence=confidence)
    dice_conf.to(device)
    # fit expects (X, y)
    dice_conf.fit(embeddings_t, classes_t)
    dice_conf.to(device)

    dual_output_model = train_cache.make_wrapper(layer, capture_modes=layer_io, concat=False, flatten=True)
    conf_split = PredictedSplitConfidence(dice_conf, EnergyConfidence(), mult=False, b=params.get("split_b", 0.0))
    conf_mod = SinglePassConfidence(dual_output_model, conf_split, index=1)

    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- Registration ---
OOD_DEFAULT_PARAM_FACTORIES["dice"] = default_dice_params
OOD_PARAM_SAMPLERS["dice"] = sample_dice_params
OOD_PROBLEM_FACTORIES["dice"] = create_dice_problem