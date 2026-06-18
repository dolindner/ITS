from typing import Dict, Any, Optional

import optuna
import torch

from confidence.direct.logit_based import EnergyConfidence
# new imports for criteria
from confidence.direct.multi_samples import MutualInformationCriterion
from confidence.direct.prob_based import MaximumSoftmaxConfidence, EntropyConfidence
from confidence.model.mc_batch_norm import MonteCarloBatchNormConfidence
from confidence.model.mc_droupout import MonteCarloDropoutConfidence, LastLayerMonteCarloDropoutConfidence
from embedding_cache import LayerEmbeddingCache
from hyper_param.ood.base_prepare import (
    OOD_DEFAULT_PARAM_FACTORIES,
    OOD_PARAM_SAMPLERS,
    OOD_PROBLEM_FACTORIES,
)
from src.utils.transformation_problem import TransformationProblem


def prepare_mc_methods(
        model: torch.nn.Module,
        transform_seq,
        dataset_info,
        architecture: str,
        train_cache: Optional[LayerEmbeddingCache] = None,
        mc_samples: int = 16,
        criterion: str = "prob",
        softmax: bool = True,  # deprecated
        use_dropout: bool = True,
        use_last_layer_dropout: bool = False,
        use_batchnorm: bool = False,
        device: torch.device = None,
        mc_batch_size: int = 32,
) -> Dict[str, TransformationProblem]:
    """
    Build MC problems for dropout and batchnorm variants.

      - "prob" (max-prop / maximum softmax probability): average=True, softmax=True, pass MaximumSoftmaxConfidence(input_logits=False)
      - "entropy": average=True, softmax=True, pass EntropyConfidence(input_logits=False)
      - "energy": average=True, softmax=False, pass EnergyConfidence (expects averaged logits)
      - other multi-sample criteria ( mutual_information): average=False, softmax=True(they expect probabilities as input)
    """
    crit_name = criterion.lower()

    if crit_name == "prob":
        average = True
        use_softmax = True
        # pass probabilities into MaximumSoftmaxConfidence
        conf_module_factory = lambda: MaximumSoftmaxConfidence(input_logits=False)
    elif crit_name == "entropy":
        average = True
        use_softmax = True
        # pass probabilities into EntropyConfidence (expects probabilities)
        conf_module_factory = lambda: EntropyConfidence(input_logits=False)
    elif crit_name == "energy":
        average = True
        use_softmax = False
        # EnergyConfidence expects logits input (averaged logits)
        conf_module_factory = lambda: EnergyConfidence()
    elif crit_name in ("mutual_information", "mi"):
        average = False
        use_softmax = True
        # MutualInformationCriterion accepts probabilities if input_logits=False
        conf_module_factory = lambda: MutualInformationCriterion(input_logits=False)
    else:
        raise ValueError(f"Unknown MC criterion '{criterion}'")

    problems: Dict[str, TransformationProblem] = {}

    # Instantiate confidence module via factory (per-problem instance)
    conf_module = conf_module_factory()

    # Monte Carlo Dropout problem
    if use_dropout:
        mc_dropout = MonteCarloDropoutConfidence(
            model=model,
            confidence=conf_module,
            samples=mc_samples,
            average=average,
            softmax=use_softmax,  # renamed parameter
            index=None,
            parallel=False,
        )
        # move to device if provided
        if device is not None:
            try:
                mc_dropout.to(device)
            except Exception:
                pass
        problems["mc_dropout"] = TransformationProblem(mc_dropout, transform_seq,
                                                       consolidate_method="consolidate_simple")

    if use_last_layer_dropout:
        mc_dropout_last_layer = LastLayerMonteCarloDropoutConfidence(
            model=model,
            confidence=conf_module,
            samples=mc_samples,
            average=average,
            softmax=use_softmax,
            index=None,
        )
        if device is not None:
            try:
                mc_dropout_last_layer.to(device)
            except Exception:
                pass
        problems["mc_dropout_last_layer"] = TransformationProblem(mc_dropout_last_layer, transform_seq,
                                                                  consolidate_method="consolidate_simple")

    # Monte Carlo BatchNorm problem
    if use_batchnorm:
        mc_bn = MonteCarloBatchNormConfidence(
            model=model,
            confidence=conf_module,
            num_estimators=mc_samples,
            convert=True,
            mc_batch_size=mc_batch_size,
            device=device,
            average=average,
            index=None,
            softmax=use_softmax,  # renamed parameter
        )
        if device is not None:
            try:
                mc_bn.to(device)
            except Exception:
                pass

        # Fit the MCBatchNorm model to populate batchnorm stats
        if train_cache is not None and hasattr(train_cache, "dataloader"):
            mc_bn.fit(train_cache.dataloader)
        else:
            raise ValueError("train_cache with a dataloader must be provided to fit MCBatchNorm.")

        problems["mc_batchnorm"] = TransformationProblem(mc_bn, transform_seq, consolidate_method="consolidate_simple")

    return problems


# --- MC Dropout ---

def default_mc_dropout_prob_params() -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": 8, "softmax_before_average": True,
        "use_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_prob_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "softmax_before_average": True,
        "use_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_prob_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                   **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="prob",
        softmax=params.get("softmax_before_average", True),
        use_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout"]


def default_mc_dropout_mi_params() -> Dict[str, Any]:
    return {
        "criterion": "mutual_information",
        "mc_samples": 8, "use_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_mi_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "mutual_information",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_mi_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                 **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="mutual_information",
        use_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout"]


def default_mc_dropout_energy_params() -> Dict[str, Any]:
    return {
        "criterion": "energy",
        "mc_samples": 8, "use_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_energy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "energy",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_energy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                     **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="energy",
        use_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout"]


# --- MC Dropout (entropy) ---

def default_mc_dropout_entropy_params() -> Dict[str, Any]:
    return {
        "criterion": "entropy",
        "mc_samples": 8,
        "softmax_before_average": True,
        "use_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_entropy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "entropy",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "softmax_before_average": True,
        "use_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_entropy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                      **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="entropy",
        softmax=params.get("softmax_before_average", True),
        use_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout"]


# --- Last Layer MC Dropout ---

def default_mc_dropout_last_layer_prob_params() -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": 64, "softmax_before_average": True,
        "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_last_layer_prob_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "softmax_before_average": True,
        "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_last_layer_prob_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                              **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="prob",
        softmax=params.get("softmax_before_average", True),
        use_dropout=False,
        use_last_layer_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout_last_layer"]


def default_mc_dropout_last_layer_mi_params() -> Dict[str, Any]:
    return {
        "criterion": "mutual_information",
        "mc_samples": 64, "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_last_layer_mi_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "mutual_information",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_last_layer_mi_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                            **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="mutual_information",
        use_dropout=False,
        use_last_layer_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout_last_layer"]


def default_mc_dropout_last_layer_energy_params() -> Dict[str, Any]:
    return {
        "criterion": "energy",
        "mc_samples": 64, "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_last_layer_energy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "energy",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_last_layer_energy_problem(params: Dict[str, Any], model, transform_seq, dataset_info,
                                                architecture, **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="energy",
        use_dropout=False,
        use_last_layer_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout_last_layer"]


# --- Last Layer MC Dropout (entropy) ---

def default_mc_dropout_last_layer_entropy_params() -> Dict[str, Any]:
    return {
        "criterion": "entropy",
        "mc_samples": 64,
        "softmax_before_average": True,
        "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_last_layer_entropy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "entropy",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "softmax_before_average": True,
        "use_dropout": False,
        "use_last_layer_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_last_layer_entropy_problem(params: Dict[str, Any], model, transform_seq, dataset_info,
                                                 architecture, **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="entropy",
        softmax=params.get("softmax_before_average", True),
        use_dropout=False,
        use_last_layer_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout_last_layer"]


# --- MC BatchNorm ---

def default_mc_batchnorm_prob_params() -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": 8, "softmax_before_average": True,
        "use_dropout": False,
        "use_batchnorm": True,
    }


def sample_mc_batchnorm_prob_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "softmax_before_average": True,
        "use_dropout": False,
        "use_batchnorm": True,
    }


def create_mc_batchnorm_prob_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                     **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="prob",
        softmax=params.get("softmax_before_average", True),
        use_dropout=False,
        use_batchnorm=True,
        device=kwargs.get("device"),
    )
    return problems["mc_batchnorm"]


def default_mc_batchnorm_mi_params() -> Dict[str, Any]:
    return {
        "criterion": "mutual_information",
        "mc_samples": 8, "use_dropout": False,
        "use_batchnorm": True,
    }


def sample_mc_batchnorm_mi_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "mutual_information",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": False,
        "use_batchnorm": True,
    }


def create_mc_batchnorm_mi_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                   **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="mutual_information",
        use_dropout=False,
        use_batchnorm=True,
        device=kwargs.get("device"),
    )
    return problems["mc_batchnorm"]


def default_mc_batchnorm_energy_params() -> Dict[str, Any]:
    return {
        "criterion": "energy",
        "mc_samples": 8, "use_dropout": False,
        "use_batchnorm": True,
    }


def sample_mc_batchnorm_energy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "energy",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": False,
        "use_batchnorm": True,
    }


def create_mc_batchnorm_energy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                       **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="energy",
        use_dropout=False,
        use_batchnorm=True,
        device=kwargs.get("device"),
    )
    return problems["mc_batchnorm"]


# --- MC BatchNorm (entropy) ---

def default_mc_batchnorm_entropy_params() -> Dict[str, Any]:
    return {
        "criterion": "entropy",
        "mc_samples": 8,
        "softmax_before_average": True,
        "use_dropout": False,
        "use_batchnorm": True,
    }


def sample_mc_batchnorm_entropy_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": "entropy",
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "softmax_before_average": True,
        "use_dropout": False,
        "use_batchnorm": True,
    }


def create_mc_batchnorm_entropy_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                        **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion="entropy",
        softmax=params.get("softmax_before_average", True),
        use_dropout=False,
        use_batchnorm=True,
        device=kwargs.get("device"),
    )
    return problems["mc_batchnorm"]


# --- Additional: Best-criterion factories for MC Dropout / MC BatchNorm ---------

def default_mc_dropout_best_criterion_params() -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": 8, "use_dropout": True,
        "use_batchnorm": False,
    }


def sample_mc_dropout_best_criterion_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": trial.suggest_categorical("criterion", ["prob", "mutual_information", "energy"]),
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": True,
        "use_batchnorm": False,
    }


def create_mc_dropout_best_criterion_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                             **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion=params.get("criterion", "prob"),
        softmax=params.get("softmax", True),
        use_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout"]


def default_mc_batchnorm_best_criterion_params() -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": 8, "use_dropout": False,
        "use_batchnorm": True,
    }


def sample_mc_batchnorm_best_criterion_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": trial.suggest_categorical("criterion", ["prob", "mutual_information", "energy"]),
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_dropout": False,
        "use_batchnorm": True,
    }


def create_mc_batchnorm_best_criterion_problem(params: Dict[str, Any], model, transform_seq, dataset_info, architecture,
                                               **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion=params.get("criterion", "prob"),
        softmax=params.get("softmax", True),
        use_dropout=False,
        use_batchnorm=True,
        device=kwargs.get("device"),
    )
    return problems["mc_batchnorm"]


# --- Last Layer MC Dropout ---

def default_mc_dropout_last_layer_best_criterion_params() -> Dict[str, Any]:
    return {
        "criterion": "prob",
        "mc_samples": 8, "use_last_layer_dropout": True,
    }


def sample_mc_dropout_last_layer_best_criterion_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "criterion": trial.suggest_categorical("criterion", ["prob", "mutual_information", "energy"]),
        "mc_samples": trial.suggest_int("mc_samples", 4, 16),
        "use_last_layer_dropout": True,
    }


def create_mc_dropout_last_layer_best_criterion_problem(params: Dict[str, Any], model, transform_seq, dataset_info,
                                                        architecture, **kwargs) -> TransformationProblem:
    problems = prepare_mc_methods(
        model=model,
        transform_seq=transform_seq,
        dataset_info=dataset_info,
        architecture=architecture,
        train_cache=kwargs.get("train_cache"),
        mc_samples=params.get("mc_samples", 16),
        criterion=params.get("criterion", "prob"),
        softmax=params.get("softmax", True),
        use_dropout=False,
        use_last_layer_dropout=True,
        use_batchnorm=False,
        device=kwargs.get("device"),
    )
    return problems["mc_dropout_last_layer"]


# --- Registration ---

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_prob"] = default_mc_dropout_prob_params
OOD_PARAM_SAMPLERS["mc_dropout_prob"] = sample_mc_dropout_prob_params
OOD_PROBLEM_FACTORIES["mc_dropout_prob"] = create_mc_dropout_prob_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_mi"] = default_mc_dropout_mi_params
OOD_PARAM_SAMPLERS["mc_dropout_mi"] = sample_mc_dropout_mi_params
OOD_PROBLEM_FACTORIES["mc_dropout_mi"] = create_mc_dropout_mi_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_energy"] = default_mc_dropout_energy_params
OOD_PARAM_SAMPLERS["mc_dropout_energy"] = sample_mc_dropout_energy_params
OOD_PROBLEM_FACTORIES["mc_dropout_energy"] = create_mc_dropout_energy_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_entropy"] = default_mc_dropout_entropy_params
OOD_PARAM_SAMPLERS["mc_dropout_entropy"] = sample_mc_dropout_entropy_params
OOD_PROBLEM_FACTORIES["mc_dropout_entropy"] = create_mc_dropout_entropy_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_last_layer_entropy"] = default_mc_dropout_last_layer_entropy_params
OOD_PARAM_SAMPLERS["mc_dropout_last_layer_entropy"] = sample_mc_dropout_last_layer_entropy_params
OOD_PROBLEM_FACTORIES["mc_dropout_last_layer_entropy"] = create_mc_dropout_last_layer_entropy_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_last_layer_prob"] = default_mc_dropout_last_layer_prob_params
OOD_PARAM_SAMPLERS["mc_dropout_last_layer_prob"] = sample_mc_dropout_last_layer_prob_params
OOD_PROBLEM_FACTORIES["mc_dropout_last_layer_prob"] = create_mc_dropout_last_layer_prob_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_last_layer_mi"] = default_mc_dropout_last_layer_mi_params
OOD_PARAM_SAMPLERS["mc_dropout_last_layer_mi"] = sample_mc_dropout_last_layer_mi_params
OOD_PROBLEM_FACTORIES["mc_dropout_last_layer_mi"] = create_mc_dropout_last_layer_mi_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_last_layer_energy"] = default_mc_dropout_last_layer_energy_params
OOD_PARAM_SAMPLERS["mc_dropout_last_layer_energy"] = sample_mc_dropout_last_layer_energy_params
OOD_PROBLEM_FACTORIES["mc_dropout_last_layer_energy"] = create_mc_dropout_last_layer_energy_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_batchnorm_prob"] = default_mc_batchnorm_prob_params
OOD_PARAM_SAMPLERS["mc_batchnorm_prob"] = sample_mc_batchnorm_prob_params
OOD_PROBLEM_FACTORIES["mc_batchnorm_prob"] = create_mc_batchnorm_prob_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_batchnorm_mi"] = default_mc_batchnorm_mi_params
OOD_PARAM_SAMPLERS["mc_batchnorm_mi"] = sample_mc_batchnorm_mi_params
OOD_PROBLEM_FACTORIES["mc_batchnorm_mi"] = create_mc_batchnorm_mi_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_batchnorm_energy"] = default_mc_batchnorm_energy_params
OOD_PARAM_SAMPLERS["mc_batchnorm_energy"] = sample_mc_batchnorm_energy_params
OOD_PROBLEM_FACTORIES["mc_batchnorm_energy"] = create_mc_batchnorm_energy_problem

# --- Register best-criterion detectors ---

OOD_DEFAULT_PARAM_FACTORIES["mc_dropout_best_criterion"] = default_mc_dropout_best_criterion_params
OOD_PARAM_SAMPLERS["mc_dropout_best_criterion"] = sample_mc_dropout_best_criterion_params
OOD_PROBLEM_FACTORIES["mc_dropout_best_criterion"] = create_mc_dropout_best_criterion_problem

OOD_DEFAULT_PARAM_FACTORIES["mc_batchnorm_best_criterion"] = default_mc_batchnorm_best_criterion_params
OOD_PARAM_SAMPLERS["mc_batchnorm_best_criterion"] = sample_mc_batchnorm_best_criterion_params
OOD_PROBLEM_FACTORIES["mc_batchnorm_best_criterion"] = create_mc_batchnorm_best_criterion_problem

OOD_DEFAULT_PARAM_FACTORIES[
    "mc_dropout_last_layer_best_criterion"] = default_mc_dropout_last_layer_best_criterion_params
OOD_PARAM_SAMPLERS["mc_dropout_last_layer_best_criterion"] = sample_mc_dropout_last_layer_best_criterion_params
OOD_PROBLEM_FACTORIES["mc_dropout_last_layer_best_criterion"] = create_mc_dropout_last_layer_best_criterion_problem
