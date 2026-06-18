from typing import Dict, Any, Optional
import optuna
import torch

from confidence.direct.logit_based import EnergyConfidence, CombinedEnergyMultiSampleConfidence
from confidence.model.laplace_conf import LaplaceModelSamplingConfidence
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
)
from src.utils.transformation_problem import TransformationProblem
from embedding_cache import LayerEmbeddingCache

from confidence.direct.multi_samples import MutualInformationCriterion
from confidence.direct.prob_based import MaximumSoftmaxConfidence, CombinedEntropyMultiSampleConfidence, \
    EntropyConfidence


def prepare_laplace_methods(
    model: torch.nn.Module,
    transform_seq,
    dataset_info,
    architecture: str,
    train_cache: Optional[LayerEmbeddingCache] = None,
    criterion: str = "prob",
    device: torch.device = None,
    mode: str = None,         # This is the Laplace fit method, not link_approx
    precision: float = None,  # Only used if mode is "None"
    **kwargs,
) -> Dict[str, TransformationProblem]:
    """
    Build Laplace problems for different criteria.
    """
    crit_name = criterion.lower()
    samples = 64

    laplace_method = mode if mode is not None else "marglik"
    prior_precision = precision if precision is not None else 1.0

    laplace_kwargs = {
        "hessian_structure": "kron",
        "subset_of_weights": "last_layer",
        "method": laplace_method,
        "kwargs_opt_prior": None,
        "prior_precision": prior_precision,
    }


    if crit_name == "prob":
        pred_type = kwargs.get("pred_type", "glm")
        if pred_type == "glm":
            link_approx = kwargs.get("link_approx", "probit")
            average = True
            softmax = True  # probit/bridge outputs probabilities
            conf_module = MaximumSoftmaxConfidence(input_logits=False)
        elif pred_type == "nn":
            link_approx = "mc"
            average = True
            softmax = True
            conf_module = MaximumSoftmaxConfidence(input_logits=False)
        else:
            raise ValueError(f"Unsupported pred_type '{pred_type}' for prob criterion.")
    elif crit_name == "energy":
        # Use MC sampling to get averaged logits for Energy score
        pred_type = "nn"
        link_approx = "mc"
        average = True
        softmax = False  # Energy score needs logits
        conf_module = EnergyConfidence()
    elif crit_name in ("mutual_information", "mi"):
        pred_type = "nn"
        link_approx = "mc"
        average = False
        softmax = True
        conf_module = MutualInformationCriterion(input_logits=False)


    elif crit_name == "entropy":
        pred_type = "nn"
        link_approx = "mc"
        average = True
        softmax = True
        conf_module = EntropyConfidence(input_logits=False)
    elif crit_name == "entropy_plus_mi":
        pred_type = "nn"
        link_approx = "mc"
        average = False
        softmax = True
        alpha = kwargs.get("alpha", 0.5)
        conf_module = CombinedEntropyMultiSampleConfidence(
            multi_sample_confidence=MutualInformationCriterion(input_logits=False),
            alpha=alpha,
            input_logits=False
        )
    elif crit_name in ("weighted", "laplace_weighted"):
        pred_type = "nn"
        link_approx = "mc"
        average = False
        combine_with = kwargs.get("combine_with", "energy")
        alpha = kwargs.get("alpha", 0.5)
        if combine_with.lower() == "entropy":
            softmax = True
            conf_module = CombinedEntropyMultiSampleConfidence(
                multi_sample_confidence=MutualInformationCriterion(input_logits=False),
                alpha=alpha,
                input_logits=False
            )
        else:
            # default to energy
            softmax = False
            conf_module = CombinedEnergyMultiSampleConfidence(
                multi_sample_confidence=MutualInformationCriterion(input_logits=True),
                alpha=alpha,
            )
    else:
        raise ValueError(f"Unknown Laplace criterion '{criterion}'")

    laplace_model = LaplaceModelSamplingConfidence(
        base_model=model,
        confidence=conf_module,
        samples=samples,
        pred_type=pred_type,
        link_approx=link_approx,
        average=average,
        softmax=softmax,
        **laplace_kwargs,
    )

    if train_cache is not None and hasattr(train_cache, "dataloader"):
        method = laplace_kwargs.get("method", "marglik")
        if method == "gridsearch":
            # For gridsearch, we need to provide a validation set
            val_loader = kwargs.get("val_id_loader", None)
            laplace_model.fit(train_cache.dataloader, validation_loader=val_loader)
        laplace_model.fit(train_cache.dataloader)

    if device is not None:
        try:
            laplace_model.to(device)
        except Exception:
            pass

    problem_name = f"laplace_{crit_name}"
    problems = {problem_name: TransformationProblem(laplace_model, transform_seq, consolidate_method="consolidate_simple")}
    return problems

# --- Laplace Factories ---

def _create_laplace_problem_factory(criterion_name: str):
    def create_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
        problems = prepare_laplace_methods(
            model=model,
            transform_seq=transform_seq,
            dataset_info=dataset_info,
            architecture=architecture,
            **params,
            **kwargs
        )
        return problems[f"laplace_{criterion_name}"]
    return create_problem

def _create_laplace_param_factory(criterion_name: str, sample_prob_params: bool = False):
    def default_params() -> Dict[str, Any]:
        params = {"criterion": criterion_name}
        params["mode"] = "marglik"
        params["precision"] = 1.0
        if sample_prob_params:
            params["pred_type"] = "glm"
            params["link_approx"] = "probit"
        if criterion_name == "entropy_plus_mi":
            params["alpha"] = 0.5
        if criterion_name == "weighted":
            params["combine_with"] = "energy"
            params["alpha"] = 0.5
        return params
    return default_params

def _create_laplace_sampler_factory(criterion_name: str, sample_prob_params: bool = False):
    def sample_params(trial: optuna.Trial,**kwargs) -> Dict[str, Any]:
        params = {"criterion": criterion_name}
        if sample_prob_params:
            params["pred_type"] = trial.suggest_categorical("pred_type", ["glm", "nn"])
            if params["pred_type"] == "glm":
                params["link_approx"] = trial.suggest_categorical("link_approx", ["probit", "bridge", "bridge_norm"])


        params["mode"] = trial.suggest_categorical("mode", ["marglik", "none",])
        if params["mode"] == "none":
            params["precision"] = trial.suggest_float("precision", 1e-2, 1e3, log=True)
        if criterion_name == "entropy_plus_mi":
            params["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
        if criterion_name == "weighted":
            # sample whether to combine MI with energy or entropy and the alpha weight
            params["combine_with"] = trial.suggest_categorical("combine_with", ["energy", "entropy"])
            params["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
        return params
    return sample_params

# --- Registration for each criterion ---

# For prob, we want to sample pred_type and link_approx
OOD_DEFAULT_PARAM_FACTORIES["laplace_prob"] = _create_laplace_param_factory("prob", sample_prob_params=True)
OOD_PARAM_SAMPLERS["laplace_prob"] = _create_laplace_sampler_factory("prob", sample_prob_params=True)
OOD_PROBLEM_FACTORIES["laplace_prob"] = _create_laplace_problem_factory("prob")


for crit in ["energy", "mutual_information", "entropy_plus_mi","entropy", "weighted"]:
    name = f"laplace_{crit}"
    OOD_DEFAULT_PARAM_FACTORIES[name] = _create_laplace_param_factory(crit)
    OOD_PARAM_SAMPLERS[name] = _create_laplace_sampler_factory(crit)
    OOD_PROBLEM_FACTORIES[name] = _create_laplace_problem_factory(crit)

# Alias for MI
OOD_DEFAULT_PARAM_FACTORIES["laplace_mi"] = _create_laplace_param_factory("mutual_information")
OOD_PARAM_SAMPLERS["laplace_mi"] = _create_laplace_sampler_factory("mutual_information")
OOD_PROBLEM_FACTORIES["laplace_mi"] = _create_laplace_problem_factory("mutual_information")



def default_laplace_best_criterion_params() -> Dict[str, Any]:
    return {"criterion": "prob", "pred_type": "glm", "link_approx": "probit"}

def sample_laplace_best_criterion_params(trial: optuna.Trial) -> Dict[str, Any]:
    criterion = trial.suggest_categorical("criterion", ["prob", "energy", "mutual_information"])
    params = {"criterion": criterion}
    if criterion == "prob":
        params["pred_type"] = trial.suggest_categorical("pred_type", ["glm", "nn"])
        if params["pred_type"] == "glm":
            params["link_approx"] = trial.suggest_categorical("link_approx", ["probit", "bridge", "bridge_norm"])
    return params

def create_laplace_best_criterion_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture, **kwargs) -> TransformationProblem:
    problems = prepare_laplace_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        **params,
        **kwargs
    )
    return problems[f"laplace_{params['criterion']}"]
#this can choose from multiple criterions
OOD_DEFAULT_PARAM_FACTORIES["laplace_best_criterion"] = default_laplace_best_criterion_params
OOD_PARAM_SAMPLERS["laplace_best_criterion"] = sample_laplace_best_criterion_params
OOD_PROBLEM_FACTORIES["laplace_best_criterion"] = create_laplace_best_criterion_problem

#
OOD_DEFAULT_PARAM_FACTORIES["laplace_entropy_gridsearch"] = lambda: {"criterion": "entropy", "mode": "gridsearch"}
OOD_PARAM_SAMPLERS["laplace_entropy_gridsearch"] = lambda trial, **kwargs: {"criterion": "entropy", "mode": "gridsearch"}
OOD_PROBLEM_FACTORIES["laplace_entropy_gridsearch"] = _create_laplace_problem_factory("entropy")

