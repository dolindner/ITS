from typing import Dict, Any
import optuna
import torch

from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from src.utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.logit_based import EnergyConfidence

# --- EnergyConfidence ---

def default_energy_params() -> Dict[str, Any]:
    return {"t":1.0}

def sample_energy_params(trial: optuna.Trial,**kwargs) -> Dict[str, Any]:
    params = {"t": trial.suggest_float("t", 0.5, 2.0)}
    return params

def create_energy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    Factory for creating a TransformationProblem with EnergyConfidence.
    This detector does not use embeddings or fitting.
    """
    energy_conf = EnergyConfidence(t=params.get("t", 1.0))
    
    # Wrap in SinglePassConfidence to use the base model directly
    conf_mod = SinglePassConfidence(model, energy_conf)
    
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


OOD_DEFAULT_PARAM_FACTORIES["energy"] = default_energy_params
OOD_PARAM_SAMPLERS["energy"] = sample_energy_params
OOD_PROBLEM_FACTORIES["energy"] = create_energy_problem

