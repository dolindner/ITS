#optimizes predicted or real target only suitable for comparision
from typing import Dict, Any
import optuna

from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from src.utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.targeted import CrossEntropyConfidence

def default_cross_entropy_params() -> Dict[str, Any]:
    return {}

def sample_cross_entropy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}

def create_cross_entropy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    conf_mod = SinglePassConfidence(model, CrossEntropyConfidence())
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["cross_entropy"] = default_cross_entropy_params
OOD_PARAM_SAMPLERS["cross_entropy"] = sample_cross_entropy_params
OOD_PROBLEM_FACTORIES["cross_entropy"] = create_cross_entropy_problem
