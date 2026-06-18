from __future__ import annotations

import gc
import json
import math
import os
import warnings
from typing import Dict, Any, Callable, Optional, Tuple, List, Union

import optuna  # ensure optuna available locally
import torch
from torch.utils.data import DataLoader

from confidence.base_confidence import ConfidenceModule
from confidence.utils import ModelInputOutputWrapper
from embedding_cache import LayerEmbeddingCache
from src.utils.eval.ood_performance import progressive_confidence_evaluation, ConfidenceEvaluator
from src.utils.transformation_problem import TransformationProblem
from transformation import get_transformation_sequence_images


# --- Registries for OOD Detectors ---

OOD_DEFAULT_PARAM_FACTORIES: Dict[str, Callable[[], Dict[str, Any]]] = {}
OOD_PARAM_SAMPLERS: Dict[str, Callable[[optuna.Trial], Dict[str, Any]]] = {}
OOD_PROBLEM_FACTORIES: Dict[str, Callable[..., TransformationProblem]] = {}
OOD_MODEL_PARAM_EXTRACTORS: Dict[str, Callable[[ConfidenceModule], List[Any]]] = {}


# General class to create Detectors and sample their hyperparameters


def get_default_ood_params(detector_name: str) -> Dict[str, Any]:
    """Get default parameters for a given OOD detector."""
    if detector_name not in OOD_DEFAULT_PARAM_FACTORIES:
        raise KeyError(f"Unknown OOD detector '{detector_name}'")
    return OOD_DEFAULT_PARAM_FACTORIES[detector_name]()


def sample_ood_params(trial: optuna.Trial, detector_name: str, **kwargs) -> Dict[str, Any]:
    """Sample parameters for a given OOD detector using Optuna."""
    if detector_name not in OOD_PARAM_SAMPLERS:
        raise KeyError(f"No parameter sampler for OOD detector '{detector_name}'")
    return OOD_PARAM_SAMPLERS[detector_name](trial, **kwargs)


def create_ood_problem(
        detector_name: str,
        params: Dict[str, Any],
        **factory_kwargs,
) -> TransformationProblem:
    """
    Create a TransformationProblem for a given OOD detector using its factory.
    This function is used both during optimization (with sampled params) and
    after optimization (with the best loaded params).

    If callers omit 'transform_seq' we attempt to resolve it from dataset_info.
    """
    if detector_name not in OOD_PROBLEM_FACTORIES:
        raise KeyError(f"No problem factory for OOD detector '{detector_name}'")

    # Resolve transform_seq if missing or None
    if "transform_seq" not in factory_kwargs or factory_kwargs.get("transform_seq") is None:
        dataset_info = factory_kwargs.get("dataset_info")
        resolved = None
        if dataset_info is not None:
            # try direct attribute first
            try:
                resolved = getattr(dataset_info, "transform_seq", None)
            except Exception:
                resolved = None
            if resolved is None:
                # fallback to name-based construction if available
                seq_name = getattr(dataset_info, "transform_seq_name", None)
                resample = getattr(dataset_info, "resample_method", None)
                if seq_name not in (None, "", "none"):
                    try:
                        resolved = get_transformation_sequence_images(name=seq_name, resample_method=resample,
                                                                      init_method="sobol")
                    except Exception:
                        resolved = None
        if resolved is not None:
            factory_kwargs["transform_seq"] = resolved

    # Build the problem via factory
    problem = OOD_PROBLEM_FACTORIES[detector_name](params=params, **factory_kwargs)

    device_arg = factory_kwargs.get("device", None)
    if device_arg is not None:
        problem.confidence_module.to(device_arg)
    dataset_info = factory_kwargs.get("dataset_info", None)
    problem.max_batch_size = dataset_info.batch_size_search

    return problem


def get_best_ood_params_from_study(study: optuna.Study) -> Dict[str, Any]:
    """Extracts the best hyperparameter set from a completed Optuna study."""
    return study.best_trial.user_attrs["full_params"]


@torch.no_grad()
def run_ood_study(
        study_name: str,
        storage_path: str,
        detector_name: str,
        objective_type: str,
        objective_kwargs: Dict[str, Any],
        n_trials: int = 50,
        enqueue_params: Optional[List[Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]]] = None,
) -> optuna.Study:
    """
    High-level wrapper to create, run, and save an Optuna study for an OOD detector.
    Uses median pruning and supports enqueuing initial trials.

    Args:
        study_name: Name of the study
        storage_path: Where to store files.
        detector_name: Name of the detector
        objective_type: if "search" uses optimization otherwise uses auroic and ood detection task.
        objective_kwargs: Kwargs for make_ood_search_objective or make_ood_auc_objective depending on type.
        n_trials: How many runs to find hyperprameters.
        enqueue_params: An optional list of trials to enqueue. Each item can be either
                        a parameter dictionary, or a tuple of (params_dict, user_attrs_dict).
                        If None, the detector's default parameters will be enqueued.
    """
    metric_objectives = {
        "auc": "auroc", "auroc": "auroc", "aupr_in": "aupr_in", "aupr_out": "aupr_out",
        "fpr95": "fpr95", "tnr95": "tnr95", "detection_error": "detection_error",
        "paired_ood_acc": "paired_ood_acc",
    }
    direction = "maximize"
    if objective_type in ["fpr95", "detection_error"]:
        direction = "minimize"

    if objective_type == "search":
        objective = make_ood_search_objective(detector_name=detector_name, **objective_kwargs)
    elif objective_type in metric_objectives:
        metric = metric_objectives[objective_type]
        objective = make_ood_auc_objective(detector_name=detector_name, **{**objective_kwargs, "metric": metric})
    else:
        raise ValueError(f"Unknown objective_type '{objective_type}'.")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, len(enqueue_params) if enqueue_params else 1))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
    )

    # Enqueue initial trials only if the study is new
    if len(study.trials) == 0:
        params_to_enqueue = []
        if enqueue_params is None:
            try:
                default_params = get_default_ood_params(detector_name)
                if default_params:
                    print(f"Enqueuing trial with default parameters for '{detector_name}'.")
                    params_to_enqueue.append(default_params)
            except Exception as e:
                print(f"Warning: Failed to enqueue default parameters for '{detector_name}': {e}")
        else:
            params_to_enqueue.extend(enqueue_params)

        for trial_info in params_to_enqueue:
            params, user_attrs = None, None
            if isinstance(trial_info, tuple):
                params, user_attrs = trial_info
            else:
                params = trial_info

            if params:
                try:
                    study.enqueue_trial(params, user_attrs=user_attrs)
                    print(f"Enqueued trial with parameters: {params}")
                    if user_attrs:
                        print(f"  ... with user_attrs: {list(user_attrs.keys())}")
                except Exception as e:
                    print(f"Warning: Failed to enqueue trial with params {params}: {e}")

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    print(f"Study {study_name} complete.")
    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params: ")
    for key, value in get_best_ood_params_from_study(study).items():
        print(f"    {key}: {value}")

    gc.collect()
    torch.cuda.empty_cache()

    return study


@torch.no_grad()
def make_ood_auc_objective(
        detector_name: str,
        model: torch.nn.Module,
        train_cache: LayerEmbeddingCache,
        id_loader,  # DataLoader for in-distribution (validation/true) samples
        ood_loader,  # DataLoader for out-of-distribution samples #this is what we evalute this on.
        # Can be the same as val_id_loader for valdiation comparision or can be a testloader
        # (reusing values that one gets from fitting on val loader)
        transform_seq: Any,
        dataset_info: Dict,
        architecture: str,
        device: str = "cuda",
        metric: str = "auroc",
        check_percent: float = 0.1,  # how often to calculate metrics.
        prune_at: Optional[float] = 0.1,  # prunes at the first check interval after this ratio.
        max_batches: Optional[int] = None,
        show_progress: bool = False,
        val_id_loader: Optional[DataLoader] = None,  # Added as some require this for fitting
        val_ood_loader: Optional[DataLoader] = None,  # Added as some require this for fitting
) -> Callable[[optuna.Trial], float]:
    """
    Creates an objective function to optimize OOD detector hyperparameters based on an
    OOD detection metric (default AUROC). Uses progressive_confidence_evaluation to
    support intermediate reports and Optuna pruning.
    """

    def objective(trial: optuna.Trial) -> float:

        sampler_kwargs = {"train_cache": train_cache, "architecture": architecture, "dataset_info": dataset_info}
        factory_kwargs = {
            "model": model, "train_cache": train_cache, "transform_seq": transform_seq,
            "dataset_info": dataset_info, "architecture": architecture, "device": device,
            "val_id_loader": val_id_loader, "val_ood_loader": val_ood_loader,
        }

        # Check if model parameters were passed via user_attrs (from an enqueued trial)
        if "model_params" in trial.user_attrs:
            print(f"Trial {trial.number}: Found pre-loaded model_params in user_attrs.")
            factory_kwargs["model_params"] = trial.user_attrs["model_params"]

        # Sample parameters for the trial
        params = sample_ood_params(trial, detector_name, **sampler_kwargs)

        with warnings.catch_warnings():
            fixed_params = trial.system_attrs.get("fixed_params", None)
        if fixed_params:
            for key, fixed_value in fixed_params.items():
                if key in params:
                    sampled_value = params[key]
                    # Use a tolerance for float comparison
                    if isinstance(fixed_value, float) and isinstance(sampled_value, float):
                        assert abs(fixed_value - sampled_value) < 1e-9, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                    else:
                        assert sampled_value == fixed_value, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                else:
                    print(
                        f"Warning: Fixed parameter '{key}' not found in sampled params for trial {trial.number}. Adding it.")
                    print(
                        f"Warning: Fixed parameter '{key}' not found in sampled params for trial {trial.number}. Adding it.")
                    print(
                        f"Warning: Fixed parameter '{key}' not found in sampled params for trial {trial.number}. Adding it.")

            params.update(fixed_params)

        trial.set_user_attr("full_params", params)

        # Create fully-wired problem via factory
        problem = create_ood_problem(
            detector_name,
            params,
            **factory_kwargs,
        )
        # ensure module on device
        try:
            problem.confidence_module.to(device)
        except Exception:
            pass

        # Progressive evaluation with pruning support
        res = progressive_confidence_evaluation(
            confidence_module=problem.confidence_module,
            id_loader=id_loader,
            ood_loader=ood_loader,
            device=device,
            metric=metric,
            trial=trial,
            check_percent=check_percent,
            prune_at=prune_at,
            max_batches=max_batches,
            show_progress=show_progress,
        )

        # Use registered extractor to get model parameters, if any
        if detector_name in OOD_MODEL_PARAM_EXTRACTORS:
            extractor = OOD_MODEL_PARAM_EXTRACTORS[detector_name]
            model_params = extractor(problem.confidence_module)
            if model_params:
                trial.set_user_attr("model_params", model_params)

        # store some info for inspection
        trial.set_user_attr("full_params", params)
        trial.set_user_attr("ood_eval_info", {
            "id_count": res.get("id_count"),
            "ood_count": res.get("ood_count"),
            "metric": res.get("metric"),
        })

        # Explicit cleanup before returning
        for name, module in problem.confidence_module.named_modules():
            if isinstance(module, ModelInputOutputWrapper):
                module.clear()

        del problem
        return float(res["metric"])

    return objective


from typing import Optional, Callable, Any, Dict


def make_ood_search_objective(
        detector_name: str,
        optimizer: Optional[Any] = None,
        model: torch.nn.Module = None,
        train_cache: LayerEmbeddingCache = None,
        val_loader=None,  # TODO this is badly named. This is the loader we run eval on.
        transform_seq: Any = None,
        dataset_info: Dict = None,
        architecture: str = None,
        device: str = "cuda",
        fixed_params: Optional[Dict[str, Any]] = None,
        report_fraction: float = 0.1,
        repeats: Union[int, float] = 1,
        val_id_loader: Optional[DataLoader] = None,  # Some require in and OOD samples for fitting
        val_ood_loader: Optional[DataLoader] = None,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective that tunes OOD detector params by evaluating their
    performance inside the downstream search procedure.
    """

    def objective(trial: optuna.Trial) -> float:
        sampler_kwargs = {"train_cache": train_cache, "architecture": architecture, "dataset_info": dataset_info}
        factory_kwargs = {
            "model": model, "train_cache": train_cache, "transform_seq": transform_seq,
            "dataset_info": dataset_info, "architecture": architecture, "device": device,
            "val_id_loader": val_id_loader, "val_ood_loader": val_ood_loader,
        }

        # Check if model parameters were passed via user_attrs and laod them
        if "model_params" in trial.user_attrs:
            print(f"Trial {trial.number}: Found pre-loaded model_params in user_attrs.")
            factory_kwargs["model_params"] = trial.user_attrs["model_params"]

        # Sample parameters for the trial
        params = sample_ood_params(trial, detector_name, **sampler_kwargs)

        # If fixed_params are provided, ensure the sampled params match for the fixed keys.
        if fixed_params:
            for key, fixed_value in fixed_params.items():
                if key in params:
                    sampled_value = params[key]
                    # Use a tolerance for float comparison
                    if isinstance(fixed_value, float) and isinstance(sampled_value, float):
                        assert abs(fixed_value - sampled_value) < 1e-9, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
                    else:
                        assert sampled_value == fixed_value, \
                            f"Sampled parameter '{key}' ({sampled_value}) does not match fixed value ({fixed_value})."
            # Update params with fixed values to ensure they are used
            params.update(fixed_params)

        trial.set_user_attr("full_params", params)

        # Build problem with current OOD params
        problem = create_ood_problem(
            detector_name,
            params,
            **factory_kwargs,
        )

        # Use ConfidenceEvaluator to run partial evaluation (supports run_until with early stopping)
        evaluator = ConfidenceEvaluator(
            model=model,
            optimizer=optimizer,
            problem=problem,
            test_loader=val_loader,
            repeats=repeats,
            show_progress=True,
        )

        # Phase 1: run until early checkpoint and report intermediate accuracy

        intermediate_acc = evaluator.run_until(report_fraction)["accuracy_mean"]
        trial.report(intermediate_acc, step=1)
        if trial.should_prune():
            # Clean up before pruning
            del evaluator, problem
            raise optuna.TrialPruned()

        # Phase 2: finish evaluation to completion
        final_res = evaluator.run_until(1.0)
        final_acc = final_res["accuracy_mean"]

        # Use registered extractor to get model parameters, if any
        if detector_name in OOD_MODEL_PARAM_EXTRACTORS:
            extractor = OOD_MODEL_PARAM_EXTRACTORS[detector_name]
            model_params = extractor(problem.confidence_module)
            if model_params:
                trial.set_user_attr("model_params", model_params)

        # store info for inspection
        trial.set_user_attr("full_params", params)
        trial.set_user_attr("search_eval_info", {
            "intermediate_acc": intermediate_acc,
            "final_acc": final_acc,
        })

        # Explicit cleanup before returning
        for name, module in problem.confidence_module.named_modules():
            if isinstance(module, ModelInputOutputWrapper):
                module.clear()
                pass

        del evaluator, problem
        return float(final_acc)

    return objective


def find_best_detector_and_instantiate(
        base_results_dir: str,
        detectors: list,
        model,
        train_cache,
        transform_seq_arg,
        dataset_info,
        architecture: str,
        device,
        val_id_loader,
        val_ood_loader,
        prefer_second: Optional[str] = None,
) -> Tuple[
    Optional[str],
    Optional[Any],
    Optional[Dict[str, float]],
    Optional[str],
    Optional[Any],
    Optional[Dict[str, float]],
]:
    """
    Find best detector by recorded `accuracy_mean` (preferring mean) and instantiate its OOD problem.
    Returns:
      (best_detector, best_problem, best_score_dict, second_choice, second_problem, second_score_dict)
    where score dict is `{"mean": float, "se": float}` (values may be math.nan if missing).
    """

    def _load_score(det_name: str) -> Optional[Dict[str, float]]:
        eval_path = os.path.join(base_results_dir, det_name, "eval_results.json")
        eval_default_path = os.path.join(base_results_dir, det_name, "eval_results_default.json")

        for p in (eval_path, eval_default_path):
            if os.path.exists(p):
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue
                mean = data.get("accuracy_mean")
                se = data.get("accuracy_se", data.get("accuracy_std"))  # prefer se, fall back to std
                # normalize types
                mean_val = float(mean) if isinstance(mean, (int, float)) else math.nan
                se_val = float(se) if isinstance(se, (int, float)) else math.nan
                if not math.isnan(mean_val):
                    return {"mean": mean_val, "se": se_val}
        return None

    # 1) scan detectors for optimized accuracy (use mean)
    best_detector: Optional[str] = None
    best_score_val = -math.inf
    best_score_dict: Optional[Dict[str, float]] = None
    for det in detectors:
        score_dict = _load_score(det)
        if score_dict is None:
            continue
        mean = score_dict["mean"]
        if mean > best_score_val:
            best_score_val = mean
            best_detector = det
            best_score_dict = score_dict

    # If we didn't find any valid scored detector, return Nones
    if best_detector is None:
        return None, None, None, None, None, None

    # 2) load best params for best_detector (or default if no params file)
    det_dir = os.path.join(base_results_dir, best_detector)
    params_path = os.path.join(det_dir, "best_params.json")
    best_params = None
    if os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                best_params = json.load(f)
        except Exception:
            best_params = None

    if best_params is None:
        best_params = get_default_ood_params(best_detector)

    # 3) try load saved model states for this detector (if any)
    best_model_params = []
    prefix = os.path.join(det_dir, "best_model")
    i = 0
    while os.path.exists(f"{prefix}_{i}.pt"):
        try:
            best_model_params.append(torch.load(f"{prefix}_{i}.pt", map_location="cpu"))
        except Exception:
            pass
        i += 1

    final_kwargs = {
        "model": model,
        "train_cache": train_cache,
        "transform_seq": transform_seq_arg,
        "dataset_info": dataset_info,
        "architecture": architecture,
        "device": str(device),
        "val_id_loader": val_id_loader,
        "val_ood_loader": val_ood_loader,
    }
    if best_model_params:
        final_kwargs["model_params"] = best_model_params

    best_problem = create_ood_problem(best_detector, best_params, **final_kwargs)

    # 4) prepare the second problem: choose the better of energy vs entropy by recorded performance
    if prefer_second and prefer_second in detectors:
        second_choice = prefer_second
        second_score_dict = _load_score(second_choice)
    else:
        candidates = [d for d in ("energy", "entropy") if d in detectors]
        second_choice = None
        best_cand_val = -math.inf
        second_score_dict = None
        for c in candidates:
            sdict = _load_score(c)
            if sdict is not None and sdict["mean"] > best_cand_val:
                best_cand_val = sdict["mean"]
                second_choice = c
                second_score_dict = sdict

    # avoid selecting same detector twice
    if second_choice == best_detector:
        other = "entropy" if best_detector == "energy" else "energy"
        if other in detectors:
            other_score = _load_score(other)
            if other_score is not None:
                second_choice = other
                second_score_dict = other_score
            else:
                second_choice = None
                second_score_dict = None
        else:
            second_choice = None
            second_score_dict = None

    if second_choice is None:
        second_problem = None
    else:
        det2_dir = os.path.join(base_results_dir, second_choice)
        params2_path = os.path.join(det2_dir, "best_params.json")
        params2 = None
        if os.path.exists(params2_path):
            try:
                with open(params2_path, "r") as f:
                    params2 = json.load(f)
            except Exception:
                params2 = None
        if params2 is None:
            params2 = get_default_ood_params(second_choice)

        model_params2 = []
        prefix2 = os.path.join(det2_dir, "best_model")
        j = 0
        while os.path.exists(f"{prefix2}_{j}.pt"):
            try:
                model_params2.append(torch.load(f"{prefix2}_{j}.pt", map_location="cpu"))
            except Exception:
                pass
            j += 1

        final_kwargs2 = dict(final_kwargs)
        if model_params2:
            final_kwargs2["model_params"] = model_params2

        second_problem = create_ood_problem(second_choice, params2, **final_kwargs2)

    return best_detector, best_problem, best_score_dict, second_choice, second_problem, second_score_dict

