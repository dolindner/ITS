from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

import yaml

# Default parameter values for search. Must match names in objective generators or the value will be ignored.
# Values will be passed to the models as fixed parameters via enqueue trials. As optuna by default ignores non sampled parameters,
# we manually extract them from the fixed parameters in objective generator.
# Quite hacky and one must name the parameters carefully to avoid conflicts, but makes it so we dont have differentiate between constant, sampled and derived paramers.
# We also store a user attribute full parameters that contains all params so not to only keep the sampled ones.
from .objective_generators import (
    _cost_shgo,
    _cost_parallel_sa,
    _cost_pso,
    _cost_pgd,
)


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def default_shgo_params(budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    n_init = int(0.90 * budget)
    # always allow at leaset one local run
    min_n_init = max(0, budget - grad_weight - 1)
    if n_init > min_n_init > 0:
        n_init = min_n_init

    max_local_runs = max(1, int((budget - n_init) / (3 * grad_weight + 1)))
    local_runs = max(1, int(max_local_runs * 0.5))
    per_run_budget = max(0, ((budget - n_init) - local_runs) // (
                local_runs * grad_weight))  # at least local_runs steps for convergence
    local_steps = per_run_budget
    # Adjust if cost overflow
    if _cost_shgo(n_init, local_runs, local_steps, grad_weight) > budget:
        raise ValueError("Error in budget allocation for SHGO")
    params = {
        "shgo_initial_samples": n_init,
        "shgo_local_runs": local_runs,
        "shgo_local_steps": local_steps,
        "shgo_selection_method": "knn",
        "shgo_local_opt": "adam",
        "shgo_lr": 1e-1,
        "shgo_acceptance_criterion": "step",
        "grad_weight": grad_weight,
    }
    current_cost = _cost_shgo(params["shgo_initial_samples"], params["shgo_local_runs"], params["shgo_local_steps"],
                              grad_weight)
    leftover = budget - current_cost
    if leftover > 0:
        params["shgo_initial_samples"] += leftover
        # sanity check
        final_cost = _cost_shgo(params["shgo_initial_samples"], params["shgo_local_runs"], params["shgo_local_steps"],
                                grad_weight)
        if final_cost != budget:
            raise ValueError(f"SHGO default allocation mismatch after top-up: cost={final_cost}, budget={budget}")
    return params


def default_parallel_sa_params(budget: int) -> Dict[str, Any]:
    parallel_runs = _clip(int(0.05 * budget), 1, max(1, budget // 5))
    max_iter = max(1, budget // max(1, parallel_runs))
    while _cost_parallel_sa(parallel_runs, max_iter) > budget and max_iter > 1:
        max_iter -= 1
        if max_iter < 0:
            raise ValueError("Cannot fit Parallel SA parameters within budget")
    return {
        "psa_parallel_runs": parallel_runs,
        "psa_max_iterations": max_iter,
        "psa_init_temp": 50.0,
        "psa_cooling": 0.95,
        "psa_reinit_interval": 9999999999999,  # Disabled by default
        "psa_reinit_amount": 0.0,  # Disabled by default
        "psa_neighbor_hood_size": 0.1,
    }


def default_pso_params(budget: int, min_swarm: int = 4) -> Dict[str, Any]:
    # Use middle of the sampling range as default
    max_steps_possible = max(1, budget // min_swarm - 1)
    steps = max(1, max_steps_possible // 2)  # Middle of range [1, max_steps_possible]

    # Compute swarm size given steps
    swarm = max(min_swarm, budget // (steps + 1))

    # Ensure budget is not exceeded
    while _cost_pso(swarm, steps) > budget and steps > 1:
        steps -= 1

    return {
        "pso_swarm_size": swarm,
        "pso_steps": steps,
        "pso_w": 0.6,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_clamp_velocity": True,
        "pso_vmax_scale": 0.2,
    }


def default_cd_params(budget: int, dim: Optional[int] = None, samples_min: int = 4) -> Dict[str, Any]:
    return {
    }


def default_wcd_params(budget: int, dim: Optional[int] = None) -> Dict[str, Any]:
    # Dimension-agnostic: only persist cycles (rounds) and first-dim weight.
    rounds = max(1, min(5, int(0.1 * (budget ** 0.5))))
    if rounds == 0:
        rounds = 1
    first_factor = 2
    return {
        "wcd_rounds": rounds,
        "wcd_first_dim_factor": first_factor,
        # base and dim omitted; derive later when actual dim known
    }


def default_wcd_lattice_params(budget: int, dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Dimension-agnostic defaults for WCD with lattice initialization.
    Uses same parameter structure as regular WCD.
    """
    return default_wcd_params(budget, dim)


def default_cd_multi_cyclus_params(budget: int, dim: Optional[int] = None) -> Dict[str, Any]:
    """
    Dimension-agnostic defaults for multi-round coordinate descent.
    Like WCD but with first_dim_factor fixed to 1 (no special weighting).
    """
    rounds = max(1, min(5, int(0.1 * (budget ** 0.5))))
    if rounds == 0:
        rounds = 1
    return {
        "wcd_rounds": rounds,
        "wcd_first_dim_factor": 1,  # Fixed to 1 for uniform CD
    }


def default_pgd_params(budget: int, grad_weight: int = 2) -> Dict[str, Any]:
    # Pick reasonable number of parallel runs
    parallel_runs = _clip(
        int(0.05 * budget),
        2,
        max(2, min(16, budget // 10 if budget >= 20 else 4))
    )

    # Compute theoretical max_iterations that fits within the budget
    # cost = par_runs * (max_iter * grad_weight + 1) <= budget
    max_iterations = int((budget - parallel_runs) / (parallel_runs * grad_weight))
    max_iterations = max(1, max_iterations)

    # Ensure at least a small lower bound for practicality
    if max_iterations < 5 and budget > 20:
        max_iterations = 5

    # Adjust if rounding errors made cost exceed budget
    while _cost_pgd(parallel_runs, max_iterations, grad_weight) > budget and max_iterations > 1:
        max_iterations -= 1

    return {
        "pgd_parallel_runs": parallel_runs,
        "pgd_max_iterations": max_iterations,
        "pgd_learning_rate": 1e-1,
        "pgd_lr_decay_rate": 1.0,
        "pgd_optimizer": "adam",
        "grad_weight": grad_weight,
    }


def default_random_search_params(budget: int, **kwargs) -> Dict[str, Any]:
    """
    Random search is parameterless; all budget is used for initial samples at build time.
    """
    return {}


def default_its_params(budget: int, dim: int | None = None) -> Dict[str, Any]:
    """
    Dimension-agnostic ITS defaults (handled like cd/wcd).
    Defer n_samples computation to builder (needs problem dimension & budget).
    Provide a conservative default for hypotheses (=1); builder may derive if absent.
    """
    return {
        "its_n_hypotheses": 1,  # keep simple; builder derives n_samples
        "its_mc_steps": 1,
        "its_change_of_mind": "score",
        "its_gaussian_filter_channel_wise": False,
        "its_unique_class_condition": False,
    }


def default_its2_params(budget: int, dim: int | None = None) -> Dict[str, Any]:
    """
    ITS2 defaults mirror ITS: dimension-agnostic; hypotheses default to 1; n_samples derived at build time.
    """
    return {
        "its_n_hypotheses": 1,
        "its_mc_steps": 1,
        "its_change_of_mind": "score",
        "its_gaussian_filter_channel_wise": False,
        "its_unique_class_condition": False,
    }


# Central registry for default parameter factory functions
ALGO_DEFAULT_PARAM_FACTORIES: Dict[str, Callable[..., Dict[str, Any]]] = {
    "shgo": default_shgo_params,
    "parallel_sa": default_parallel_sa_params,
    "pso": default_pso_params,
    "cd": default_cd_params,
    "wcd": default_wcd_params,
    "wcd_lattice": default_wcd_lattice_params,
    "cd_multi_cyclus": default_cd_multi_cyclus_params,
    "pgd": default_pgd_params,
    "random_search": default_random_search_params,
    "its": default_its_params,
    "its2": default_its2_params,
}

# Prefix patterns to identify parameters per algorithm (used for filtering trial params)
PARAM_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "shgo": ("shgo_", "grad_weight"),
    "parallel_sa": ("psa_",),
    "pso": ("pso_",),
    "cd": ("cd_",),
    "wcd": ("wcd_",),
    "wcd_lattice": ("wcd_",),
    "cd_multi_cyclus": ("wcd_",),
    "pgd": ("pgd_", "grad_weight"),
    "random_search": ("shgo_",),
    "its": ("its_",),
    "its2": ("its_",),
}


def get_default_params(algo: str, budget: int, **kwargs) -> Dict[str, Any]:
    """
    Obtain default parameter dictionary for an algorithm given a budget.
    kwargs are forwarded to the underlying default_* function.
    """
    if algo not in ALGO_DEFAULT_PARAM_FACTORIES:
        raise KeyError(f"Unknown algorithm '{algo}'")
    return ALGO_DEFAULT_PARAM_FACTORIES[algo](budget, **kwargs)


def filter_algo_params(params: Dict[str, Any], algo: str) -> Dict[str, Any]:
    """
    Filter a raw parameter dict (e.g., trial.params) to only those relevant for the algo.
    """
    if algo not in PARAM_PREFIXES:
        raise KeyError(f"Unknown algorithm '{algo}'")
    prefixes = PARAM_PREFIXES[algo]
    out: Dict[str, Any] = {}
    for k, v in params.items():
        for pref in prefixes:
            # exact match (e.g., 'grad_weight') or prefix match 'prefix_'
            if (pref.endswith("_") and k.startswith(pref)) or k == pref:
                out[k] = v
                break
    return out


def save_params(params: Dict[str, Any], path: str | Path):
    """
    Save parameters to YAML (overwrites).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(params, f)


def load_params(path: str | Path) -> Dict[str, Any]:
    """
    Load parameters from YAML. Returns empty dict if file missing.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Parameter file {path} must contain a mapping.")
    return data


def merge_params(base: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Shallow merge helper: overrides win.
    """
    if not overrides:
        return dict(base)
    merged = dict(base)
    merged.update(overrides)
    return merged


# convenience wrapper for backward compatibility with old code that called default_allocation directly
def default_allocation(algo: str, budget: int, **kwargs) -> Dict[str, Any]:
    """
    Backward-compatible allocation provider.
    """
    return get_default_params(algo, budget, **kwargs)
