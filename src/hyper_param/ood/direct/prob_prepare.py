from typing import Dict, Any
import optuna

from confidence.direct.logit_based import MaxLogitConfidence
from hyper_param.ood.base_prepare import OOD_DEFAULT_PARAM_FACTORIES, OOD_PARAM_SAMPLERS, OOD_PROBLEM_FACTORIES
from src.utils.transformation_problem import TransformationProblem
from confidence.model.single_pass import SinglePassConfidence
from confidence.direct.prob_based import (
    EntropyConfidence,
    MaximumSoftmaxConfidence,
    DifferentiableMaximumSoftmaxConfidence,
    GeneralizedEntropyConfidence,
)

# --- Entropy ---

def default_entropy_params() -> Dict[str, Any]:
    return {}

def sample_entropy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}

def create_entropy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    ent_conf = EntropyConfidence(input_logits=True)
    conf_mod = SinglePassConfidence(model, ent_conf)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Differentiable Max Softmax ---

def default_diff_max_params() -> Dict[str, Any]:
    return {}

def sample_diff_max_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {}

def create_diff_max_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    d_conf = DifferentiableMaximumSoftmaxConfidence(tau=1.0, input_logits=True)
    conf_mod = SinglePassConfidence(model, d_conf)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- Generalized / Adjusted Entropy ---

def default_adjusted_entropy_params() -> Dict[str, Any]:
    # lmbda default 1.0, m default None (use all classes), optional m_fraction to specify fraction of classes
    return {"lmbda": 1.0, "m": None, "m_fraction": 1.0}

def sample_adjusted_entropy_params(trial: optuna.Trial, train_cache=None, dataset_info=None, **kwargs) -> Dict[str, Any]:
    """
    Sampler accepts kwargs (as other sampler functions do). If caller provided 'm_fraction' in kwargs,
    use that; otherwise sample a fraction in [0.0, 1.0]. Return m as None (will be resolved in create()).
    """
    lmbda = trial.suggest_float("lmbda", 0.1, 3.0, log=False)
    # prefer explicit fraction passed via kwargs (e.g., from caller), otherwise sample
    m_fraction = trial.suggest_float("m_fraction", 0.0, 1.0)
    return {"lmbda": lmbda, "m": None, "m_fraction": m_fraction}

def create_adjusted_entropy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    """
    params expected keys: 'lmbda' (float), 'm' (int or None) and optional 'm_fraction' (float in [0,1]).
    If params['m'] is None and 'm_fraction' provided, compute m = round(m_fraction * num_classes) (min 1).
    """
    lmbda = params.get("lmbda", 1.0)
    m = params.get("m", None)
    m_fraction = params.get("m_fraction", None)

    # if m not specified but m_fraction provided and dataset_info yields num_classes, compute m
    if m is None and m_fraction is not None:

        num_classes = dataset_info.get("num_classes", None)

        if num_classes is not None:
            # ensure at least one class is used
            m = max(1, int(round(float(m_fraction) * int(num_classes))))
        else:
            # dataset_info unknown: leave m as None so the confidence module will use all classes
            m = None

    gen_conf = GeneralizedEntropyConfidence(lmbda=float(lmbda), m=m, input_logits=True)
    conf_mod = SinglePassConfidence(model, gen_conf)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")


# --- MaxLogit ---

def default_maxlogit_params() -> Dict[str, Any]:
    return {}

def sample_maxlogit_params(trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
    return {}

def create_maxlogit_problem(
    params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs
) -> TransformationProblem:
    maxlogit_conf = MaxLogitConfidence(
    )
    conf_mod = SinglePassConfidence(model, maxlogit_conf)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")

# --- MaxProb ---

def default_maxprob_params() -> Dict[str, Any]:
    return {}

def sample_maxprob_params(trial: optuna.Trial, **kwargs) -> Dict[str, Any]:
    return {}


def create_maxprob_problem(
    params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs
) -> TransformationProblem:
    maxprob_conf = MaximumSoftmaxConfidence(
        input_logits=True,
    )
    conf_mod = SinglePassConfidence(model, maxprob_conf)
    return TransformationProblem(conf_mod, transform_seq, consolidate_method="consolidate_simple")



# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["entropy"] = default_entropy_params
OOD_PARAM_SAMPLERS["entropy"] = sample_entropy_params
OOD_PROBLEM_FACTORIES["entropy"] = create_entropy_problem


OOD_DEFAULT_PARAM_FACTORIES["differentiable_max_softmax"] = default_diff_max_params
OOD_PARAM_SAMPLERS["differentiable_max_softmax"] = sample_diff_max_params
OOD_PROBLEM_FACTORIES["differentiable_max_softmax"] = create_diff_max_problem

OOD_DEFAULT_PARAM_FACTORIES["adjusted_entropy"] = default_adjusted_entropy_params
OOD_PARAM_SAMPLERS["adjusted_entropy"] = sample_adjusted_entropy_params
OOD_PROBLEM_FACTORIES["adjusted_entropy"] = create_adjusted_entropy_problem

OOD_DEFAULT_PARAM_FACTORIES["MaxLogit"] = default_maxlogit_params
OOD_PARAM_SAMPLERS["MaxLogit"] = sample_maxlogit_params
OOD_PROBLEM_FACTORIES["MaxLogit"] = create_maxlogit_problem

OOD_DEFAULT_PARAM_FACTORIES["MSP"] = default_maxprob_params
OOD_PARAM_SAMPLERS["MSP"] = sample_maxprob_params
OOD_PROBLEM_FACTORIES["MSP"] = create_maxprob_problem