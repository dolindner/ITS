import json
import os
import warnings  # Used for the sanity checks
from typing import Any
from typing import Union

import optuna
import torch
from torchmetrics import Accuracy, FBetaScore
from tqdm import tqdm

from confidence.model.base_model import ModelBasedConfidence
from src.utils.transform_sequence import TransformSequence
from src.utils.transformation_problem import TransformationProblem
from src.utils.transforms.rotation import Rotation3DEulerUniform, RotationZ3D, RotationY3D, RotationX3D



class ITSWRAPPER(torch.nn.Module):
    """
    Wrapper class for its that follows my optimize structure that is given an optimization problem.

    """

    def __init__(self, its, take_first_hyp: bool = True, x_return=True):
        super(ITSWRAPPER, self).__init__()
        self.its = its
        self.take_first_hyp = take_first_hyp
        self.x_return = x_return

    @staticmethod
    def _prepare_problem(problem):
        """
        Creates a copy of the problem with Rotation3DEulerUniform replaced by individual axis rotations.
        This is done to ensure correct budget calculation before the optimizer is built.
        """
        # Check if modification is needed
        needs_update = any(isinstance(t, Rotation3DEulerUniform) for t in problem.transform_sequence.transformations)
        if not needs_update:
            return problem

        # Create a new TransformSequence with the modified transformations and domains
        # instead of deep-copying the whole problem object.

        original_sequence = problem.transform_sequence
        transformations = original_sequence.transformations
        domains = original_sequence.domains

        new_transformations = []
        new_domains = []

        for i, t in enumerate(transformations):
            if isinstance(t, Rotation3DEulerUniform):
                new_transformations.extend([RotationZ3D, RotationY3D, RotationX3D])
                domain_part = domains[i]
                new_domains.extend([domain_part, domain_part, domain_part])
            else:
                new_transformations.append(t)
                new_domains.append(domains[i])

        new_sequence = TransformSequence(
            transformations=new_transformations,
            domains=new_domains,
            neighbour_hood_size=original_sequence.neighbour_hood_size,
            application_method=original_sequence.application_method,
            device=original_sequence.dummy_param.device,
            dtype=original_sequence.dummy_param.dtype,
            init_method=original_sequence.init_method,
            reflect=original_sequence.reflect,
            invert=original_sequence.invert
        )

        # Create a new TransformationProblem with the new sequence
        new_problem = TransformationProblem(
            confidence_module=problem.confidence_module,
            transform_sequence=new_sequence,
            consolidate_method=problem.consolidate_method,
            max_batch_size=problem.max_batch_size
        )
        return new_problem

    def optimize(self, problem, data, y=None):
        """
                Evaluate/optimize transformations for a batch using the wrapped ITS instance.

                Args:
                    problem: A TransformationProblem-like object providing `confidence_module`,
                        `transform_sequence` and `max_batch_size` used by ITS.
                    data: Batch tensor of inputs to transform and evaluate.
                    y: Optional labels (not used by ITS by default; kept for compatibility).

                Returns:
                    If `take_first_hyp` is True:
                        - best_matrix[:, 0]: Tensor of best transformation parameters for each sample.
                        - best_error[:, 0]: Tensor of associated errors for the chosen hypothesis.
                        - logits[:, 0]: Logits returned by ITS search for the chosen hypothesis.
                        - data_canonic[:, 0] (only if `x_return` True): The transformed inputs passed to the model.
                    Otherwise (when `take_first_hyp` is False):
                        - best_matrix: Full matrix of candidate transformation parameters.
                        - best_error: Full matrix/array of errors.
                        - logits: Full logits per hypothesis.
                        - data_canonic (optional): Transformed inputs per hypothesis (if `x_return` True).
                """
        # The problem passed here should already be prepared by _prepare_problem
        # Use the ITS to infer the transformation
        self.its.confidence_module = problem.confidence_module
        self.its.transformation = problem.transform_sequence.transformations
        self.its.domain = problem.transform_sequence.domains
        self.its.transformation_schedule = self.its.transformation

        # still check that transforms does not contain  Rotation3DEulerUniform
        for t in self.its.transformation:
            if isinstance(t, Rotation3DEulerUniform):
                raise ValueError(
                    "Problem contains Rotation3DEulerUniform which is incompatible with ITSWRAPPER. Please use _prepare_problem to replace it with individual axis rotations.")

        # Pass resampling method to ITS instance
        resampling_method = problem.transform_sequence.application_method
        self.its.resampling_method = resampling_method

        # Infer matrix dimension from the problem
        matrix_dim = None
        if hasattr(problem, 'matrix_dim') and callable(problem.matrix_dim):
            matrix_dim = problem.matrix_dim()

        best_matrix, best_error, logits, data_canonic = self.its.infer(data, return_best=True, matrix_dim=matrix_dim,
                                                                       max_batch_size_override=problem.max_batch_size)
        # Return the transformed data + logits (parameters concept not used for ITS)
        if self.take_first_hyp:
            if self.x_return:
                return best_matrix[:, 0], best_error[:, 0], logits[:, 0], data_canonic[:, 0]

            return best_matrix[:, 0], best_error[:, 0], logits[:, 0]

        if self.x_return:
            return best_matrix, best_error, logits, data_canonic
        else:
            return best_matrix, best_error, logits


def standard_error(std: float, n: int):
    return std / (n ** 0.5) if n > 0 else None


class ConfidenceEvaluator:
    """
    Evalutes metrics on a specific percentage of the data to allow culling of very bad hyperparamter combinations.

    """

    def __init__(
            self,
            model,
            optimizer,
            problem,
            test_loader,
            repeats: Union[int, float] = 1,
            max_batch_override: int = 128,
            show_progress: bool = True,
            return_per_run: bool = True,
            debug: bool = False,
            num_classes: int | None = None,
    ):
        """
               Initialize a ConfidenceEvaluator for measuring performance over a subset or repeats.

               Args:
                   model: PyTorch model used for prediction (device inferred from model parameters).
                   optimizer: Search/optimizer object (may be an `ITSWRAPPER` or other optimizer).
                   problem: TransformationProblem that describes transformations and confidence module.
                   test_loader: DataLoader providing batches for evaluation.
                   repeats: Number of repeats (int) or fraction (float < 1.0) determining how many batches to process.
                   max_batch_override: Maximum batch size override for the problem (int).
                   show_progress: Whether to display a progress bar (bool).
                   return_per_run: If True, store per-repeat metrics in the returned aggregates (bool).
                   debug: If True, print debug messages (bool).
                   num_classes: Optional explicit number of classes. If None, inferred from logits.

               """

        self.model = model
        self.optimizer = optimizer
        self.problem = problem
        self.test_loader = test_loader
        self.repeats = repeats
        self.show_progress = show_progress
        self.max_batch_override = max_batch_override
        self.return_per_run = return_per_run
        self.debug = debug

        self.device = next(model.parameters()).device

        # state
        self._repeat_idx = 0
        self._iterator = None
        self._metrics = None
        self._num_batches = len(test_loader)
        if isinstance(repeats, float) and repeats < 1.0:
            self._num_batches = int(self._num_batches * repeats)
            self.repeats = 1
        self._batches_done_in_repeat = 0
        self._per_run = []
        self._num_classes = num_classes

    def _log(self, msg: str):
        if self.debug:
            print(f"[DEBUG] {msg}")

    def _init_metrics(self, num_classes: int):
        acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        f2_macro = FBetaScore(task="multiclass", num_classes=num_classes, beta=2.0, average="macro").to(self.device)
        f2_micro = FBetaScore(task="multiclass", num_classes=num_classes, beta=2.0, average="micro").to(self.device)
        f2_weighted = FBetaScore(task="multiclass", num_classes=num_classes, beta=2.0, average="weighted").to(
            self.device)
        return {"acc": acc, "f2_macro": f2_macro, "f2_micro": f2_micro, "f2_weighted": f2_weighted}

    def _start_repeat(self):
        self._log(f"Starting repeat {self._repeat_idx}/{self.repeats}")
        self.model.eval()
        self.problem.confidence_module.eval()

        # Only create iterator if batches will be processed
        if self._num_batches > 0:
            # Do not run optimizer/model here to avoid duplicate work.
            # Iterator will be consumed in _step_one_batch where we lazily init metrics.
            self._iterator = iter(self.test_loader)

        # metrics will be lazily initialized on first batch where logits are available
        self._metrics = None
        self._batches_done_in_repeat = 0

    def _step_one_batch(self):
        try:
            data, target = next(self._iterator)
        except StopIteration:
            self._log(
                f"StopIteration in repeat {self._repeat_idx}, batches_done={self._batches_done_in_repeat}/{self._num_batches}"
            )
            if self._batches_done_in_repeat != self._num_batches:
                raise RuntimeError(
                    f"StopIteration occurred unexpectedly: got {self._batches_done_in_repeat} batches, expected {self._num_batches}. "
                    "Check if test_loader length is consistent."
                )
            return False

        data, target = data.to(self.device), target.to(self.device)

        with torch.no_grad():
            # perform optimization / transformation and forward pass once
            # note best classes may be none for some problems
            # check for its wrapper
            if isinstance(self.optimizer, ITSWRAPPER):
                best_param, best_error, best_classes, x_trans = self.optimizer.optimize(self.problem, data, y=None)
            else:
                best_param, best_error, best_classes = self.optimizer.optimize(self.problem, data, y=None)
                x_trans = self.problem.transform(data, best_param)

            logits = self.model(x_trans)
            if logits.dim() < 2:
                raise ValueError("Model logits must have shape [batch, num_classes].")

            # Lazily initialize metrics if not already done (infer num_classes from logits)
            if self._metrics is None:
                inferred_num_classes = logits.shape[-1]
                if self._num_classes is None:
                    self._num_classes = inferred_num_classes
                    self._log(f"Inferred num_classes={self._num_classes} from logits")
                elif self._num_classes != inferred_num_classes:
                    # warn or raise depending on desired strictness; here we raise for clarity
                    raise ValueError(
                        f"Provided num_classes ({self._num_classes}) does not match logits' num_classes ({inferred_num_classes})."
                    )
                self._metrics = self._init_metrics(self._num_classes)

            preds = logits.argmax(dim=-1)

        # update metrics (metrics are guaranteed to be initialized here)
        self._metrics["acc"].update(preds, target)
        self._metrics["f2_macro"].update(preds, target)
        self._metrics["f2_micro"].update(preds, target)
        self._metrics["f2_weighted"].update(preds, target)

        self._batches_done_in_repeat += 1
        self._log(f"Processed batch {self._batches_done_in_repeat}/{self._num_batches} in repeat {self._repeat_idx}")
        return True

    def run_until(self, target_fraction: float) -> dict:
        """
        Run evaluation up to a fraction of all repeats combined (0.0–1.0).
        Returns dict with mean/std/se metrics and optionally per_run list.
        """
        total_batches = self._num_batches * self.repeats
        target_batches_total = int(total_batches * target_fraction)
        batches_done_total = self._repeat_idx * self._num_batches + self._batches_done_in_repeat

        self._log(
            f"run_until({target_fraction}): target_batches_total={target_batches_total}, already_done={batches_done_total}"
        )

        if self._iterator is None and target_batches_total > batches_done_total:
            self._start_repeat()

        if self.show_progress:
            pbar = tqdm(total=max(0, target_batches_total - batches_done_total), desc="Eval", leave=False)
        else:
            pbar = None

        while batches_done_total < target_batches_total:
            if not self._step_one_batch():  # repeat finished
                if self._batches_done_in_repeat > 0:
                    self._per_run.append(self._compute_current_metrics())
                    self._log(f"Finished repeat {self._repeat_idx}, metrics saved")
                self._repeat_idx += 1
                if self._repeat_idx >= self.repeats:
                    self._log("All repeats completed")
                    break
                self._start_repeat()
                continue

            batches_done_total = self._repeat_idx * self._num_batches + self._batches_done_in_repeat
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        return self._aggregate_results()

    def _compute_current_metrics(self):
        if self._metrics is None:
            raise RuntimeError("No metrics available to compute (no batches processed).")
        return {
            "accuracy": self._metrics["acc"].compute().item(),
            "f2_macro": self._metrics["f2_macro"].compute().item(),
            "f2_micro": self._metrics["f2_micro"].compute().item(),
            "f2_weighted": self._metrics["f2_weighted"].compute().item(),
        }

    def _aggregate_results(self):
        all_runs = list(self._per_run)
        if self._batches_done_in_repeat > 0:
            all_runs.append(self._compute_current_metrics())

        def agg(key):
            vals = [m[key] for m in all_runs]
            if not vals:
                return None, None, None
            if len(vals) == 1:
                return vals[0], 0.0, None
            t = torch.tensor(vals, dtype=torch.float32)
            mean = t.mean().item()
            std = t.std(unbiased=True).item()
            se = standard_error(std, len(vals))
            return mean, std, se

        acc_mean, acc_std, acc_se = agg("accuracy")
        f2m_mean, f2m_std, f2m_se = agg("f2_macro")
        f2mi_mean, f2mi_std, f2mi_se = agg("f2_micro")
        f2w_mean, f2w_std, f2w_se = agg("f2_weighted")

        result = {
            "repeats": len(all_runs),
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "accuracy_se": acc_se,
            "f2_macro_mean": f2m_mean,
            "f2_macro_std": f2m_std,
            "f2_macro_se": f2m_se,
            "f2_micro_mean": f2mi_mean,
            "f2_micro_std": f2mi_std,
            "f2_micro_se": f2mi_se,
            "f2_weighted_mean": f2w_mean,
            "f2_weighted_std": f2w_std,
            "f2_weighted_se": f2w_se,
        }

        if self.return_per_run:
            result["per_run"] = all_runs

        return result


def evaluate_confidence_module(
        confidence_module: ModelBasedConfidence,
        data_loader,
        device: torch.device | None = None,
        show_progress: bool = True,
):
    """
    Evaluate a confidence module on a dataset containing both in-distribution (ID)
    and out-of-distribution (OOD) samples. OOD samples are indicated by negative labels.

    Args:
        confidence_module: A ModelBasedConfidence-like module that returns (confidences, logits)
            for a batch of inputs.
        data_loader: DataLoader yielding (inputs, labels) where OOD examples have label < 0.
        device: Optional torch.device to run evaluation on. If None, deduced from the module.
        show_progress: Display a progress bar and live metrics (bool).

    Returns:
        dict containing:
            - id_samples: int number of ID samples seen
            - ood_samples: int number of OOD samples seen
            - id_accuracy: classification accuracy on ID samples (or None if not computable)
            - ood_accuracy: None (not defined when labels < 0)
            - auroc, aupr_in, aupr_out, fpr95, tnr95, detection_error: detection metrics or None
    """
    confidence_module.eval()
    if device is None:
        try:
            device = next(confidence_module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    id_conf_list, ood_conf_list = [], []
    id_labels_list, pred_labels_list = [], []
    # We still collect labels >=0 for accuracy over ID; OOD accuracy often undefined (labels<0)
    with torch.no_grad():
        iterator = data_loader
        if show_progress:
            iterator = tqdm(iterator, desc="Confidence eval", leave=False)
        for inputs, labels in iterator:
            inputs = inputs.to(device)
            labels = labels.to(device)
            confidences, logits = confidence_module(inputs)

            # If logits provided, compute predictions (even in label mode for possible future extensions)
            preds = None
            if logits is not None:
                preds = torch.argmax(logits, dim=-1)

            id_mask = labels >= 0
            ood_mask = labels < 0

            # Split confidences
            if id_mask.any():
                id_conf_list.append(confidences[id_mask].detach().cpu())
                # For ID accuracy (only defined over labels >=0)
                if preds is not None:
                    id_labels_list.append(labels[id_mask].detach().cpu())
                    pred_labels_list.append(preds[id_mask].detach().cpu())
            if ood_mask.any():
                ood_conf_list.append(confidences[ood_mask].detach().cpu())

            if show_progress:
                try:
                    cur_id_conf = torch.cat(id_conf_list) if id_conf_list else torch.tensor([])
                    cur_ood_conf = torch.cat(ood_conf_list) if ood_conf_list else torch.tensor([])
                    # ID accuracy so far
                    if id_labels_list and pred_labels_list:
                        id_labels_so_far = torch.cat(id_labels_list)
                        id_preds_so_far = torch.cat(pred_labels_list)
                        id_acc_so_far = (id_preds_so_far == id_labels_so_far).float().mean().item()
                    else:
                        id_acc_so_far = float('nan')
                    # AUROC so far (only if both present)
                    if len(cur_id_conf) > 0 and len(cur_ood_conf) > 0:
                        scores_tmp = torch.cat([cur_id_conf, cur_ood_conf]).numpy()
                        targets_tmp = torch.cat([
                            torch.ones_like(cur_id_conf),
                            torch.zeros_like(cur_ood_conf),
                        ]).numpy()
                        auroc_so_far = roc_auc_score(targets_tmp, scores_tmp)
                    else:
                        auroc_so_far = float('nan')
                    iterator.set_postfix({
                        "ID": len(cur_id_conf),
                        "OOD": len(cur_ood_conf),
                        "ID_acc": f"{id_acc_so_far:.3f}",
                        "AUROC": f"{auroc_so_far:.3f}" if auroc_so_far == auroc_so_far else "nan",
                    })
                except Exception:
                    # Fail silently to avoid interrupting evaluation
                    pass
            # ---- end progress bar metric updates ----

    # Concatenate
    id_conf = torch.cat(id_conf_list) if id_conf_list else torch.tensor([])
    ood_conf = torch.cat(ood_conf_list) if ood_conf_list else torch.tensor([])

    # Classification accuracy over ID samples (if we have predictions + labels)
    if id_labels_list and pred_labels_list:
        id_labels = torch.cat(id_labels_list)
        id_preds = torch.cat(pred_labels_list)
        id_accuracy = (id_preds == id_labels).float().mean().item()
    else:
        id_accuracy = None

    # ood_accuracy not well-defined when labels<0 (no true class); keep for compatibility -> None
    ood_accuracy = None

    # Prepare detection metrics
    if len(id_conf) > 0 and len(ood_conf) > 0:
        _, metrics = compute_detection_metric_from_confs(id_conf, ood_conf)
        auroc = metrics.get("auroc")
        aupr_in = metrics.get("aupr_in")
        aupr_out = metrics.get("aupr_out")
        fpr95 = metrics.get("fpr95")
        tnr95 = metrics.get("tnr95")
        detection_error = metrics.get("detection_error")
    else:
        auroc = aupr_in = aupr_out = fpr95 = tnr95 = detection_error = None

    return {
        "id_samples": int(len(id_conf)),
        "ood_samples": int(len(ood_conf)),
        "id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "auroc": auroc,
        "aupr_in": aupr_in,
        "aupr_out": aupr_out,
        "fpr95": fpr95,
        "tnr95": tnr95,
        "detection_error": detection_error,
    }


@torch.no_grad()
def evaluate_confidence_and_search(
        model,
        optimizer,
        problem,
        test_loader,
        max_batch_override: int = 128,
        show_progress: bool = True,
        repeats: int = 1,
        return_per_run: bool = True,
        store_val: bool = False,
        store_correct: bool = True,
):
    """
    Evaluate a search algorithm together with a confidence module over a test set.

    For each batch the optimizer/search is run and the confidence module is used within that process.
    Optionally repeats the evaluation multiple times to estimate mean/std/se across runs.

    Args:
        model: PyTorch model used for final predictions.
        optimizer: Search/optimizer object; may be an `ITSWRAPPER` wrapper.
        problem: TransformationProblem providing transform sequence and confidence module.
        test_loader: DataLoader yielding (data, target).
        max_batch_override: Optional batch size override for `problem.max_batch_size`.
        show_progress: Whether to show progress bars.
        repeats: Number of repeats to run for statistical estimates.
        return_per_run: If True, return a list of run-level metrics.
        store_val: If True and the loader is deterministic, store per-sample errors/matrices/labels.
        store_correct: If True and the loader is deterministic, store per-sample correctness flags.

    Returns:
        dict with aggregated metrics:
            - repeats: number of runs aggregated
            - accuracy_mean / std / se
            - f2_macro_mean / std / se
            - f2_micro_mean / std / se
            - f2_weighted_mean / std / se
            - per_run: optional list of per-run dicts (if return_per_run True)
            - When deterministic and store_val/store_correct are True, per_run entries may include
              per-sample arrays (per_sample_errors, per_sample_matrices, etc.).
    """
    model.eval()
    if hasattr(problem, "confidence_module") and problem.confidence_module is not None:
        problem.confidence_module.eval()

    # Get device from model
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cpu")

    problem_old = problem
    # Safely check for ITSWRAPPER and prepare problem state
    is_its_wrapper = False
    if isinstance(optimizer, ITSWRAPPER):
        problem = ITSWRAPPER._prepare_problem(problem_old)
        is_its_wrapper = True

    per_run = []

    # --- PROGRESS BAR SETUP ---
    use_flat_bar = show_progress and repeats > 1
    num_batches = len(test_loader) if hasattr(test_loader, "__len__") else None

    flat_bar = None
    if use_flat_bar and num_batches is not None:
        flat_bar = tqdm(total=repeats * num_batches, desc="Eval (all repeats)", leave=True)

    # Determine determinism and dataset length
    deterministic = _is_deterministic_loader(test_loader)

    dataset_len = len(test_loader.dataset) if (deterministic and hasattr(test_loader, "dataset")) else None

    try:
        for r in range(repeats):
            # Apply batch size override
            old_batch_size = problem.max_batch_size
            if max_batch_override is not None:
                problem.max_batch_size = max_batch_override

            # --- LAZY METRIC INIT SETUP ---
            metrics_initialized = False
            acc, f2_macro, f2_micro, f2_weighted = [None] * 4

            # Containers for per-sample records
            per_sample_errors = [] if (deterministic and store_val) else None
            per_sample_matrices = [] if (deterministic and store_val) else None
            per_sample_true_labels = [] if (deterministic and store_val) else None
            per_sample_pred_labels = [] if (deterministic and store_val) else None
            per_sample_correct = [] if (deterministic and store_correct) else None

            # Setup Inner Progress Bar
            iterator = test_loader
            if show_progress and not use_flat_bar:
                iterator = tqdm(test_loader, desc=f"Search eval" if repeats == 1 else f"Repeat {r + 1}/{repeats}",
                                leave=False)

            with torch.no_grad():
                for data, target in iterator:
                    data = data.to(device)
                    target = target.to(device)
                    bs = data.shape[0]

                    # --- OPTIMIZATION STEP ---
                    if not is_its_wrapper:
                        res = optimizer.optimize(problem, data, y=None)
                        if isinstance(res, tuple) and len(res) >= 2:
                            best_param = res[0]
                            best_error = res[1]
                        else:
                            best_param = res
                            best_error = None

                        try:
                            best_matrix = problem(best_param)
                        except Exception:
                            best_matrix = None
                        x_trans = problem.transform(data, best_param)
                    else:
                        # ITSWRAPPER logic
                        best_matrix, best_error, _, x_trans = optimizer.optimize(problem, data)

                    # --- PREDICTION ---
                    logits = model(x_trans)
                    if logits.dim() < 2:
                        raise ValueError("Model logits must have shape [batch, num_classes].")
                    preds = logits.argmax(dim=-1)

                    # --- INITIALIZE METRICS ON FIRST BATCH ---
                    if not metrics_initialized:
                        num_classes = logits.shape[-1]
                        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
                        f2_macro = FBetaScore(task='multiclass', num_classes=num_classes, beta=2.0, average='macro').to(
                            device)
                        f2_micro = FBetaScore(task='multiclass', num_classes=num_classes, beta=2.0, average='micro').to(
                            device)
                        f2_weighted = FBetaScore(task='multiclass', num_classes=num_classes, beta=2.0,
                                                 average='weighted').to(device)
                        metrics_initialized = True

                    # --- METRIC UPDATES ---
                    acc.update(preds, target)
                    f2_macro.update(preds, target)
                    f2_micro.update(preds, target)
                    f2_weighted.update(preds, target)

                    # --- STORAGE LOGIC ---
                    if deterministic and store_val:
                        # Errors (STRICT Per-Sample Handling)
                        if best_error is not None:
                            try:
                                err_list = best_error.detach().cpu().tolist()
                                # Only store if length matches batch size (Strict per-sample check)
                                if hasattr(err_list, "__len__") and len(err_list) == bs:
                                    per_sample_errors.extend(err_list)
                                else:
                                    # If length does not match (e.g., scalar or wrong size), store None
                                    warnings.warn(
                                        f"Optimizer returned error list of size {len(err_list)} for batch size {bs}. Storing None for this batch's errors.")
                                    per_sample_errors.extend([None] * bs)
                            except Exception:
                                per_sample_errors.extend([None] * bs)
                        else:
                            per_sample_errors.extend([None] * bs)

                        # Matrices (Strict Per-Sample Storage)
                        if best_matrix is not None:
                            try:
                                mats = best_matrix.detach().cpu().numpy()
                                per_sample_matrices.extend(mats.tolist())
                            except Exception:
                                per_sample_matrices.extend([None] * bs)
                        else:
                            per_sample_matrices.extend([None] * bs)

                        # Labels (True/Pred)
                        per_sample_true_labels.extend(target.detach().cpu().tolist())
                        per_sample_pred_labels.extend(preds.detach().cpu().tolist())

                    # Correctness
                    if deterministic and store_correct:
                        correct = (preds == target).detach().cpu().tolist()
                        per_sample_correct.extend(correct)

                    # --- PROGRESS UPDATES ---
                    if use_flat_bar and flat_bar is not None:
                        flat_bar.update(1)
                        if acc is not None:
                            flat_bar.set_postfix({"Acc": f"{acc.compute().item():.3f}"})
                    elif show_progress and not use_flat_bar:
                        if acc is not None:
                            iterator.set_postfix({"Acc": f"{acc.compute().item():.3f}"})

            # End of batches loop - Compute final run metrics

            # Guard against empty dataloaders
            if acc is None:
                run_metrics = {
                    "accuracy": 0.0, "f2_macro": 0.0, "f2_micro": 0.0, "f2_weighted": 0.0
                }
            else:
                run_metrics = {
                    "accuracy": acc.compute().item(),
                    "f2_macro": f2_macro.compute().item(),
                    "f2_micro": f2_micro.compute().item(),
                    "f2_weighted": f2_weighted.compute().item(),
                }

            # --- STORE COLLECTED SAMPLES (Errors, Matrices, Labels) ---
            if deterministic and store_val:
                # Sanity Check (Warning if lengths differ due to drop_last or error)
                if dataset_len is not None and len(per_sample_errors) != dataset_len:
                    warnings.warn(f"Collected {len(per_sample_errors)} samples but dataset has {dataset_len}. "
                                  "This is expected if test_loader has drop_last=True.")

                run_metrics["per_sample_errors"] = per_sample_errors
                run_metrics["per_sample_matrices"] = per_sample_matrices
                run_metrics["per_sample_true_labels"] = per_sample_true_labels
                run_metrics["per_sample_pred_labels"] = per_sample_pred_labels
            else:
                run_metrics["per_sample_errors"] = None
                run_metrics["per_sample_matrices"] = None
                run_metrics["per_sample_true_labels"] = None
                run_metrics["per_sample_pred_labels"] = None

            # --- STORE CORRECTNESS ---
            if deterministic and store_correct:
                if dataset_len is not None and len(per_sample_correct) != dataset_len:
                    warnings.warn(
                        f"Collected {len(per_sample_correct)} correctness samples but dataset has {dataset_len}. "
                        "This is expected if test_loader has drop_last=True.")

                run_metrics["per_sample_correct"] = per_sample_correct
            else:
                run_metrics["per_sample_correct"] = None

            per_run.append(run_metrics)
            problem.max_batch_size = old_batch_size  # Reset for next repeat

    finally:
        if flat_bar is not None:
            flat_bar.close()

    # --- AGGREGATION ---
    def agg(key):
        vals = [m[key] for m in per_run if m[key] is not None]
        if not vals: return None, None, None

        t = torch.tensor(vals, dtype=torch.float32)
        mean = t.mean().item()
        std = t.std(unbiased=True).item() if len(vals) > 1 else 0.0

        # Standard error calculation
        se = std / (len(vals) ** 0.5) if len(vals) > 1 else None
        return mean, std, se

    acc_mean, acc_std, acc_se = agg("accuracy")
    f2m_mean, f2m_std, f2m_se = agg("f2_macro")
    f2mi_mean, f2mi_std, f2mi_se = agg("f2_micro")
    f2w_mean, f2w_std, f2w_se = agg("f2_weighted")

    result = {
        "repeats": len(per_run),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "accuracy_se": acc_se,
        "f2_macro_mean": f2m_mean,
        "f2_macro_std": f2m_std,
        "f2_macro_se": f2m_se,
        "f2_micro_mean": f2mi_mean,
        "f2_micro_std": f2mi_std,
        "f2_micro_se": f2mi_se,
        "f2_weighted_mean": f2w_mean,
        "f2_weighted_std": f2w_std,
        "f2_weighted_se": f2w_se,
    }
    if return_per_run:
        result["per_run"] = per_run
    return result


def save_eval_results(path: str, result: dict) -> None:
    """
    Save evaluation results to disk as JSON.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # delete existing file if exists
    if os.path.exists(path):
        os.remove(path)

    # Use a simpler temp naming to avoid double .tmp extensions
    import tempfile
    dir_for_temp = dirpath if dirpath else "."
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                     dir=dir_for_temp, delete=False,
                                     suffix=".tmp") as f:
        tmp_path = f.name
        json.dump(result, f, indent=2)

    os.replace(tmp_path, path)


import time


def load_or_run_evaluate_confidence_and_search(
        model: Any,
        optimizer: Any,
        problem: Any,
        test_loader: Any,
        save_path: str | None = None,
        max_batch_override: int | None = 128,
        show_progress: bool = True,
        repeats: int = 1,
        return_per_run: bool = True,
        overwrite: bool = False,
        store_correct=False,
        store_val: bool = False,
        rerun_if_duplicates: bool = True,
) -> dict:
    """
    Run `evaluate_confidence_and_search` with optional caching/merging of results.

    Args:
        model: Model passed to the evaluation.
        optimizer: Optimizer/search passed to the evaluation.
        problem: TransformationProblem used by the evaluation.
        test_loader: DataLoader used for evaluation.
        save_path: Optional path to cache results as JSON. If None or empty, caching is disabled.
        max_batch_override: Optional override for problem.max_batch_size.
        show_progress: Whether to show progress bars.
        repeats: Number of repeats to run (or to ensure in cache).
        return_per_run: Whether to include per-run results in the returned dict.
        overwrite: If True, ignore any existing cache and recompute from scratch.
        store_correct: Pass-through to evaluation for storing per-sample correctness (bool).
        store_val: Pass-through to evaluation for storing per-sample values/matrices (bool).
        rerun_if_duplicates: If True, checks cached results for duplicated runs and reruns if detected.

    Returns:
        dict: Evaluation result (same shape as `evaluate_confidence_and_search`). If `save_path`
        is provided, the result is saved to disk (atomic/temporary files used during writes).
    """

    print()
    cache_path = save_path
    # No cache path -> behave exactly like evaluate_confidence_and_search
    if cache_path is None or str(cache_path).strip() == "":
        seed = int(time.time_ns() % (2 ** 32 - 1))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return evaluate_confidence_and_search(
            model=model,
            optimizer=optimizer,
            problem=problem,
            test_loader=test_loader,
            max_batch_override=max_batch_override,
            show_progress=show_progress,
            repeats=repeats,
            return_per_run=return_per_run,
            store_val=store_val,
            store_correct=store_correct,
        )

    # Load from cache if available and not overwriting
    cached_result = None
    if (not overwrite) and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_result = json.load(f)
        except Exception:
            # Fall back to recomputation if the cache is corrupted
            cached_result = None

    # Check if we have cached results with fewer repeats than requested
    if cached_result is not None:
        cached_repeats = cached_result.get("repeats", 0)

        # If cached repeats match or exceed requested, return cached result
        if cached_repeats >= repeats:
            # Check for duplicates as a error preventive method. T
            if rerun_if_duplicates:
                duplicate_detected = _check_for_duplicate_runs(cached_result)
                if duplicate_detected:
                    print("Duplicate runs detected. Rerunning all repeats...")
                    result = evaluate_confidence_and_search(
                        model=model,
                        optimizer=optimizer,
                        problem=problem,
                        test_loader=test_loader,
                        max_batch_override=max_batch_override,
                        show_progress=show_progress,
                        repeats=repeats,
                        return_per_run=return_per_run,
                        store_val=store_val,
                        store_correct=store_correct,
                    )
                    save_eval_results(cache_path, result)
                    return result

            return cached_result

        additional_repeats = repeats - cached_repeats

        # reseed to prevent potential issues with repeated runs
        # should fix issues if someone seeds them outside for deterministic runs
        # added a check function to check for smae resulst ( about 1 per notebook occured, when a restart happend)
        seed = int(time.time_ns() % (2 ** 32 - 1))
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # SHORT TEMP PATHS
        temp_new_path = f"{cache_path}.nr.tmp"  # short: new runs
        temp_merged_path = f"{cache_path}.m.tmp"  # short: merged

        # Compute only the additional repeats
        new_result = evaluate_confidence_and_search(
            model=model,
            optimizer=optimizer,
            problem=problem,
            test_loader=test_loader,
            max_batch_override=max_batch_override,
            show_progress=show_progress,
            repeats=additional_repeats,
            return_per_run=True,  # Always need per_run for merging
            store_val=store_val,
            store_correct=store_correct,
        )

        # Save new runs temporarily
        save_eval_results(temp_new_path, new_result)

        # Merge cached and new results
        merged_result = _merge_eval_results(cached_result, new_result, return_per_run)

        save_eval_results(temp_merged_path, merged_result)

        # Replace original cache with merged result
        os.replace(temp_merged_path, cache_path)

        # Clean up temporary new runs file
        if os.path.exists(temp_new_path):
            os.remove(temp_new_path)

        return merged_result

    seed = int(time.time_ns() % (2 ** 32 - 1))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # No cache or cache is None -> compute from scratch
    result = evaluate_confidence_and_search(
        model=model,
        optimizer=optimizer,
        problem=problem,
        test_loader=test_loader,
        max_batch_override=max_batch_override,
        show_progress=show_progress,
        repeats=repeats,
        return_per_run=return_per_run,
        store_val=store_val,
        store_correct=store_correct,
    )

    save_eval_results(cache_path, result)
    return result


def exclude_duplicate_runs(
        result: dict,
        tolerance: float = 1e-9,
) -> dict:
    """
    Exclude duplicate runs from evaluation results and optionally rerun them.

    Args:
        result: Result dict containing per_run data
        tolerance: Numerical tolerance for comparing floating point values
        rerun_duplicates: If True, rerun duplicate runs with fresh random seeds
        model, optimizer, problem, test_loader: Required if rerun_duplicates=True
        max_batch_override, show_progress, store_val, store_correct: Params for rerun

    Returns:
        Cleaned result dict with duplicates removed/replaced and recomputed aggregate metrics
    """
    per_run = result.get("per_run", [])
    if not per_run or len(per_run) < 2:
        return result

    metrics_to_check = ["accuracy", "f2_macro", "f2_micro", "f2_weighted"]

    # Build signatures and identify duplicates
    run_signatures = []
    for i, run in enumerate(per_run):
        signature = tuple(run.get(m, float('nan')) for m in metrics_to_check)
        run_signatures.append((i, signature))

    # Find all duplicate indices (keep first occurrence)
    duplicate_indices = set()
    for i in range(len(run_signatures)):
        for j in range(i + 1, len(run_signatures)):
            idx_i, sig_i = run_signatures[i]
            idx_j, sig_j = run_signatures[j]

            if all(abs(a - b) < tolerance for a, b in zip(sig_i, sig_j)):
                duplicate_indices.add(idx_j)

    if not duplicate_indices:
        return result

    print(f"\nFound {len(duplicate_indices)} duplicate run(s) at indices: {sorted(duplicate_indices)}")

    # Simply exclude duplicates
    cleaned_per_run = [run for i, run in enumerate(per_run) if i not in duplicate_indices]
    print(f"Excluded {len(duplicate_indices)} duplicate(s). Remaining runs: {len(cleaned_per_run)}\n")

    # Recompute aggregate metrics (excluding composite metrics)
    def agg(key):
        vals = [m[key] for m in cleaned_per_run if key in m]
        if not vals:
            return None, None, None
        if len(vals) == 1:
            return vals[0], 0.0, None
        t = torch.tensor(vals, dtype=torch.float32)
        mean = t.mean().item()
        std = t.std(unbiased=True).item()
        se = standard_error(std, len(vals))
        return mean, std, se

    acc_mean, acc_std, acc_se = agg("accuracy")
    f2m_mean, f2m_std, f2m_se = agg("f2_macro")
    f2mi_mean, f2mi_std, f2mi_se = agg("f2_micro")
    f2w_mean, f2w_std, f2w_se = agg("f2_weighted")

    cleaned_result = {
        "repeats": len(cleaned_per_run),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "accuracy_se": acc_se,
        "f2_macro_mean": f2m_mean,
        "f2_macro_std": f2m_std,
        "f2_macro_se": f2m_se,
        "f2_micro_mean": f2mi_mean,
        "f2_micro_std": f2mi_std,
        "f2_micro_se": f2mi_se,
        "f2_weighted_mean": f2w_mean,
        "f2_weighted_std": f2w_std,
        "f2_weighted_se": f2w_se,
        "per_run": cleaned_per_run,
    }

    return cleaned_result


def _check_for_duplicate_runs(result: dict, tolerance: float = 1e-6) -> None:
    """
    Check if any two runs in the result have identical mean values.
    Prints a warning if duplicates are detected, indicating potential randomness issues.

    Args:
        result: Result dict containing per_run data
        tolerance: Numerical tolerance for comparing floating point values
    """
    per_run = result.get("per_run", [])
    if not per_run or len(per_run) < 2:
        return False

    # Extract key metrics from each run
    metrics_to_check = ["accuracy", "f2_macro", "f2_micro", "f2_weighted"]

    # Build signatures for each run
    run_signatures = []
    for i, run in enumerate(per_run):
        signature = tuple(run.get(m, float('nan')) for m in metrics_to_check)
        run_signatures.append((i, signature))

    # Check for duplicates
    duplicates_found = []
    for i in range(len(run_signatures)):
        for j in range(i + 1, len(run_signatures)):
            idx_i, sig_i = run_signatures[i]
            idx_j, sig_j = run_signatures[j]

            # Compare all metrics within tolerance
            if all(abs(a - b) < tolerance for a, b in zip(sig_i, sig_j)):
                duplicates_found.append((idx_i, idx_j, sig_i))

    # Print warnings for all duplicates
    if len(duplicates_found) > 0:
        print("\n" + "=" * 80)
        print("WARNING: DUPLICATE RUNS DETECTED - POTENTIAL RANDOMNESS ISSUE")
        print("=" * 80)
        print(f"Found {len(duplicates_found)} pair(s) of runs with identical metrics:")
        for idx_i, idx_j, signature in duplicates_found:
            print(f"\n  Runs {idx_i} and {idx_j} are identical:")
            for metric, value in zip(metrics_to_check, signature):
                print(f"    {metric}: {value:.6f}")
        print("\nThis indicates that the random seed may not be changing between runs.")
        print("Check that np.random.seed(), torch.manual_seed(), etc. are properly called")
        print("with different seeds before each evaluation run.")
        print("=" * 80 + "\n")
    return len(duplicates_found) > 0


def _merge_eval_results(cached_result: dict, new_result: dict, return_per_run: bool) -> dict:
    """
    Merge cached evaluation results with newly computed results.
    This is used so one can reuse results with fewer repeats when one wants to increase the repeat number.
    """
    # Extract per_run data from both results
    cached_per_run = cached_result.get("per_run", [])
    new_per_run = new_result.get("per_run", [])

    if not cached_per_run or not new_per_run:
        raise ValueError("Both cached and new results must have per_run data for merging")

    # Concatenate all runs
    all_runs = cached_per_run + new_per_run
    total_repeats = len(all_runs)

    # Helper to aggregate metrics across runs
    def agg(key):
        vals = [m[key] for m in all_runs if key in m]
        if not vals:
            return None, None, None
        if len(vals) == 1:
            return vals[0], 0.0, None
        t = torch.tensor(vals, dtype=torch.float32)
        mean = t.mean().item()
        std = t.std(unbiased=True).item()
        se = standard_error(std, len(vals))
        return mean, std, se

    # Recompute aggregated metrics
    acc_mean, acc_std, acc_se = agg("accuracy")
    f2m_mean, f2m_std, f2m_se = agg("f2_macro")
    f2mi_mean, f2mi_std, f2mi_se = agg("f2_micro")
    f2w_mean, f2w_std, f2w_se = agg("f2_weighted")

    merged = {
        "repeats": total_repeats,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "accuracy_se": acc_se,
        "f2_macro_mean": f2m_mean,
        "f2_macro_std": f2m_std,
        "f2_macro_se": f2m_se,
        "f2_micro_mean": f2mi_mean,
        "f2_micro_std": f2mi_std,
        "f2_micro_se": f2mi_se,
        "f2_weighted_mean": f2w_mean,
        "f2_weighted_std": f2w_std,
        "f2_weighted_se": f2w_se,
    }

    # Include per_run data if requested
    if return_per_run:
        merged["per_run"] = all_runs

    return merged


from typing import Iterable, Optional, Tuple, Dict

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
)


def compute_detection_metric_from_confs(
        conf_id: torch.Tensor,
        conf_ood: torch.Tensor,
        metric: str = "auroc",
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a scalar detection metric from ID/OOD confidence tensors.

    Args:
      conf_id: 1D tensor of confidences for in-distribution samples.
      conf_ood: 1D tensor of confidences for out-of-distribution samples.
      metric: 'auroc' (default), 'aupr_in', 'aupr_out', or 'fpr95'.

    Returns:
      (value, info_dict) where info_dict contains computed auxiliary values.
    """
    # ensure CPU numpy arrays for sklearn
    conf_id_np = conf_id.detach().cpu().numpy() if conf_id.numel() > 0 else np.array([])
    conf_ood_np = conf_ood.detach().cpu().numpy() if conf_ood.numel() > 0 else np.array([])

    info: Dict[str, float] = {
        "auroc": float("nan"),
        "aupr_in": float("nan"),
        "aupr_out": float("nan"),
        "fpr95": float("nan"),
        "tnr95": float("nan"),
        "detection_error": float("nan"),
    }

    if conf_id_np.size > 0 and conf_ood_np.size > 0:
        scores = np.concatenate([conf_id_np, conf_ood_np])
        targets = np.concatenate([np.ones_like(conf_id_np), np.zeros_like(conf_ood_np)])

        try:
            info["auroc"] = float(roc_auc_score(targets, scores))
        except Exception:
            info["auroc"] = float("nan")
        try:
            info["aupr_in"] = float(average_precision_score(targets, scores))
        except Exception:
            info["aupr_in"] = float("nan")
        try:
            info["aupr_out"] = float(average_precision_score(1 - targets, 1 - scores))
        except Exception:
            info["aupr_out"] = float("nan")
        try:
            fpr, tpr, _ = roc_curve(targets, scores)
            mask95 = tpr >= 0.95
            if mask95.any():
                fpr_at_95_tpr = float(np.min(fpr[mask95]))
                info["fpr95"] = fpr_at_95_tpr
                info["tnr95"] = 1.0 - fpr_at_95_tpr
            else:
                info["fpr95"] = float("nan")
                info["tnr95"] = float("nan")

            fnr = 1.0 - tpr
            det_err = (0.5 * (fnr + fpr)).min()
            info["detection_error"] = float(det_err)
        except Exception:
            info["fpr95"] = float("nan")
            info["tnr95"] = float("nan")
            info["detection_error"] = float("nan")

        del conf_id_np, conf_ood_np, scores, targets

    metric_map = {
        "auroc": info["auroc"],
        "aupr_in": info["aupr_in"],
        "aupr_out": info["aupr_out"],
        "fpr95": info["fpr95"],
        "tnr95": info["tnr95"],
        "detection_error": info["detection_error"],
    }
    if metric not in metric_map:
        raise ValueError(f"Unknown metric '{metric}'")
    return metric_map[metric], info


@torch.no_grad()
def progressive_confidence_evaluation(
        confidence_module,
        id_loader: Iterable,
        ood_loader: Iterable,
        device: torch.device | str = "cuda",
        metric: str = "auroc",
        trial: Optional["optuna.Trial"] = None,
        check_percent: float = 0.1,
        max_batches: Optional[int] = None,
        show_progress: bool = False,
        prune_at_checkpoint: bool = False,
        prune_at: Optional[float] = None,
) -> Dict[str, object]:
    """
    Progressively evaluate detection metric(s) on paired ID/OOD loaders with optional Optuna reporting/pruning.

    Args:
        confidence_module: Module returning (confidences, logits) for a batch.
        id_loader: Iterable/DataLoader for in-distribution samples.
        ood_loader: Iterable/DataLoader for out-of-distribution samples.
        device: Device or string (e.g., "cuda") to use for evaluation.
        metric: Detection metric to compute ('auroc', 'aupr_in', 'aupr_out', 'fpr95', 'paired_ood_acc', etc.).
        trial: Optional `optuna.Trial` to report intermediate results to and allow pruning.
        check_percent: Fraction of total paired minibatches that defines a checkpoint interval (0 < check_percent <= 1).
        max_batches: Optional maximum number of paired minibatches to process.
        show_progress: Whether to show a progress bar.
        prune_at_checkpoint: If True, call `trial.should_prune()` at checkpoints and raise `optuna.TrialPruned()` if needed.
        prune_at: Fraction of progress at which to perform a one-time prune check (if prune_at_checkpoint is False).

    Returns:
        dict containing:
            - metric: final metric value (float or NaN)
            - metric_info: dict with auxiliary metric values (auroc, fpr95, etc.)
            - id_count: total number of ID confidences used
            - ood_count: total number of OOD confidences used

    Notes:
        - Walks both loaders in lockstep and stops at the shorter iterator.
        - When `trial` is provided, intermediate values are reported via `trial.report(value, step)`.
    """
    from tqdm import tqdm as _tqdm
    import math

    if not (0.0 < check_percent <= 1.0):
        raise ValueError("check_percent must be in (0.0, 1.0].")
    if prune_at is None:
        prune_at = check_percent
    if not (0.0 < prune_at <= 1.0):
        raise ValueError("prune_at must be in (0.0, 1.0].")

    device = torch.device(device) if not isinstance(device, torch.device) else device
    confidence_module.eval()
    id_conf_list, ood_conf_list = [], []
    paired_ood_acc_list = []

    id_iter = iter(id_loader)
    ood_iter = iter(ood_loader)

    # Determine checkpoint interval in number of paired minibatches
    len_id = len(id_loader)
    len_ood = len(ood_loader)
    total_pairs = min(len_id, len_ood)
    checkpoint_batches = max(1, int(math.ceil(total_pairs * check_percent)))

    # compute the step at/after which we will perform a single prune check
    prune_step = max(1, int(math.ceil(total_pairs * prune_at)))
    prune_checked = False

    step = 0
    pbar = None
    if show_progress:
        pbar = _tqdm(total=None, desc="OOD eval (progressive)", leave=False)

    try:
        while True:
            if max_batches is not None and step >= max_batches:
                break
            try:
                id_inputs, id_labels = next(id_iter)
            except StopIteration:
                break
            try:
                ood_inputs, ood_labels = next(ood_iter)
            except StopIteration:
                break

            id_inputs = id_inputs.to(device)
            ood_inputs = ood_inputs.to(device)

            with torch.no_grad():
                id_conf, _ = confidence_module(id_inputs)
                ood_conf, _ = confidence_module(ood_inputs)

            if isinstance(id_conf, torch.Tensor):
                id_conf_list.append(id_conf.detach().cpu())
            if isinstance(ood_conf, torch.Tensor):
                ood_conf_list.append(ood_conf.detach().cpu())

            if metric == "paired_ood_acc" and isinstance(id_conf, torch.Tensor) and isinstance(ood_conf, torch.Tensor):
                # ensure same batch size for comparison
                min_bs = min(id_conf.shape[0], ood_conf.shape[0])
                if min_bs > 0:
                    # ensure that y matches
                    if id_labels is not None and ood_labels is not None:
                        assert torch.all(torch.eq(id_labels[:min_bs],
                                                  ood_labels[:min_bs])), "Labels for paired OOD accuracy must match."
                    paired_acc = (id_conf[:min_bs] > ood_conf[:min_bs]).float().mean().item()
                    paired_ood_acc_list.append(paired_acc)

            try:
                del id_inputs, ood_inputs, id_conf, ood_conf, id_labels, ood_labels
            except Exception:
                pass

            step += 1
            if pbar is not None:
                pbar.update(1)

            # Only report at coarse checkpoints determined by check_percent
            if step % checkpoint_batches == 0:
                if metric == "paired_ood_acc":
                    val = torch.tensor(paired_ood_acc_list).mean().item() if paired_ood_acc_list else float("nan")
                    info = {"paired_ood_acc": val}
                elif id_conf_list and ood_conf_list:
                    id_concat = torch.cat(id_conf_list)
                    ood_concat = torch.cat(ood_conf_list)
                    val, info = compute_detection_metric_from_confs(id_concat, ood_concat, metric=metric)
                    # release temporary concatenations
                    del id_concat, ood_concat
                else:
                    val, info = float("nan"), {}
                if trial is not None:
                    trial.report(float(val), step)
                    if prune_at_checkpoint:
                        if trial.should_prune():
                            # Close progress bar and drop references before raising
                            if pbar is not None:
                                pbar.close()
                                pbar = None
                            id_conf_list.clear()
                            ood_conf_list.clear()
                            paired_ood_acc_list.clear()
                            # Ensure local tensors from latest step are not referenced
                            try:
                                del id_inputs, ood_inputs, id_conf, ood_conf, id_labels, ood_labels
                            except Exception:
                                pass
                            id_iter = None
                            ood_iter = None
                            raise optuna.TrialPruned()

                    # Perform pruning decision at most once when progress reaches prune_step.
                    if (not prune_checked) and (step >= prune_step) and not prune_at_checkpoint:
                        prune_checked = True
                        if trial.should_prune():
                            # Close progress bar and drop references before raising
                            if pbar is not None:
                                pbar.close()
                                pbar = None
                            id_conf_list.clear()
                            ood_conf_list.clear()
                            paired_ood_acc_list.clear()
                            # Ensure local tensors from latest step are not referenced
                            try:
                                del id_inputs, ood_inputs, id_conf, ood_conf, id_labels, ood_labels
                            except Exception:
                                pass
                            id_iter = None
                            ood_iter = None
                            raise optuna.TrialPruned()
    finally:
        if pbar is not None:
            pbar.close()
        id_iter = None
        ood_iter = None

    # final metric
    if metric == "paired_ood_acc":
        final_val = torch.tensor(paired_ood_acc_list).mean().item() if paired_ood_acc_list else float("nan")
        final_info = {"paired_ood_acc": final_val}
        id_concat = torch.cat(id_conf_list) if id_conf_list else torch.tensor([])
        ood_concat = torch.cat(ood_conf_list) if ood_conf_list else torch.tensor([])
        result = {
            "metric": final_val,
            "metric_info": final_info,
            "id_count": int(id_concat.numel()),
            "ood_count": int(ood_concat.numel()),
        }
        # release temporaries
        del id_concat, ood_concat
    elif id_conf_list and ood_conf_list:
        id_concat = torch.cat(id_conf_list)
        ood_concat = torch.cat(ood_conf_list)
        final_val, final_info = compute_detection_metric_from_confs(id_concat, ood_concat, metric=metric)
        result = {
            "metric": final_val,
            "metric_info": final_info,
            "id_count": int(id_concat.numel()),
            "ood_count": int(ood_concat.numel()),
        }
        del id_concat, ood_concat
    else:
        result = {
            "metric": float("nan"),
            "metric_info": {},
            "id_count": 0,
            "ood_count": 0,
        }
    return result


@torch.no_grad()
def progressive_confidence_evaluation_every_step(
        confidence_module,
        id_loader: Iterable,
        ood_loader: Iterable,
        device: torch.device | str = "cuda",
        metric: str = "auroc",
        trial: Optional["optuna.Trial"] = None,
        check_every_n_batches: int = 10,  # Changed from check_percent
        max_batches: Optional[int] = None,
        show_progress: bool = False,
        prune_at_checkpoint: bool = True,  # Changed default to True
) -> Dict[str, object]:
    """
    Evaluate ID/OOD confidences progressively with fixed-interval reporting
    for compatibility with SuccessiveHalvingPruner.

    Args:
        check_every_n_batches: Report intermediate values every N batches (default: 10)
        prune_at_checkpoint: If True, check for pruning at each checkpoint
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    confidence_module.eval()
    id_conf_list, ood_conf_list = [], []
    paired_ood_acc_list = []

    id_iter = iter(id_loader)
    ood_iter = iter(ood_loader)

    step = 0
    pbar = None
    if show_progress:
        pbar = tqdm(total=None, desc="OOD eval (progressive)", leave=False)

    try:
        while True:
            if max_batches is not None and step >= max_batches:
                break
            try:
                id_inputs, id_labels = next(id_iter)
            except StopIteration:
                break
            try:
                ood_inputs, ood_labels = next(ood_iter)
            except StopIteration:
                break

            id_inputs = id_inputs.to(device)
            ood_inputs = ood_inputs.to(device)

            with torch.no_grad():
                id_conf, _ = confidence_module(id_inputs)
                ood_conf, _ = confidence_module(ood_inputs)

            if isinstance(id_conf, torch.Tensor):
                id_conf_list.append(id_conf.detach().cpu())
            if isinstance(ood_conf, torch.Tensor):
                ood_conf_list.append(ood_conf.detach().cpu())

            if metric == "paired_ood_acc" and isinstance(id_conf, torch.Tensor) and isinstance(ood_conf, torch.Tensor):
                min_bs = min(id_conf.shape[0], ood_conf.shape[0])
                if min_bs > 0:
                    if id_labels is not None and ood_labels is not None:
                        assert torch.all(torch.eq(id_labels[:min_bs], ood_labels[:min_bs]))
                    paired_acc = (id_conf[:min_bs] > ood_conf[:min_bs]).float().mean().item()
                    paired_ood_acc_list.append(paired_acc)

            try:
                del id_inputs, ood_inputs, id_conf, ood_conf, id_labels, ood_labels
            except Exception:
                pass

            step += 1
            if pbar:
                pbar.update(1)

            # Report at fixed intervals for SuccessiveHalvingPruner
            if step % check_every_n_batches == 0:
                if metric == "paired_ood_acc":
                    val = torch.tensor(paired_ood_acc_list).mean().item() if paired_ood_acc_list else float("nan")
                elif id_conf_list and ood_conf_list:
                    id_concat = torch.cat(id_conf_list)
                    ood_concat = torch.cat(ood_conf_list)
                    val, _ = compute_detection_metric_from_confs(id_concat, ood_concat, metric=metric)
                    del id_concat, ood_concat
                else:
                    val = float("nan")

                if trial is not None:
                    trial.report(float(val), step)
                    if prune_at_checkpoint and trial.should_prune():
                        if pbar:
                            pbar.close()
                        raise optuna.TrialPruned()

    finally:
        if pbar:
            pbar.close()

    # Final metric computation
    if metric == "paired_ood_acc":
        final_val = torch.tensor(paired_ood_acc_list).mean().item() if paired_ood_acc_list else float("nan")
        final_info = {"paired_ood_acc": final_val}
    elif id_conf_list and ood_conf_list:
        id_concat = torch.cat(id_conf_list)
        ood_concat = torch.cat(ood_conf_list)
        final_val, final_info = compute_detection_metric_from_confs(id_concat, ood_concat, metric=metric)
        del id_concat, ood_concat
    else:
        final_val, final_info = float("nan"), {}

    result = {
        "metric": final_val,
        "id_count": sum(c.shape[0] for c in id_conf_list) if id_conf_list else 0,
        "ood_count": sum(c.shape[0] for c in ood_conf_list) if ood_conf_list else 0,
        **final_info
    }
    return result


import numpy as _np
from torch.utils.data import SequentialSampler


def _is_deterministic_loader(loader) -> bool:
    """
    Heuristic: consider loader deterministic if it uses a SequentialSampler or
    its sampler does not have a 'shuffle' attribute or 'shuffle' is False.
    """
    try:
        sampler = getattr(loader, "sampler", None)
        if sampler is None and hasattr(loader, "batch_sampler"):
            sampler = getattr(loader.batch_sampler, "sampler", None)
        if isinstance(sampler, SequentialSampler):
            return True
        # If sampler exposes 'shuffle' attribute, require it to be False
        if sampler is not None and hasattr(sampler, "shuffle"):
            return not bool(getattr(sampler, "shuffle"))
        # Fallback: if DataLoader has attribute 'shuffle' and it's True => not deterministic
        if hasattr(loader, "shuffle"):
            return not bool(getattr(loader, "shuffle"))
    except Exception:
        pass
    return False


def analyze_run_results(results_list):
    """
    Analyze a list of run-results (each as returned by evaluate_confidence_and_search).
    For deterministic runs only (per_sample_errors and per_sample_matrices must be present).

    Returns:
      {
        "num_datapoints": N,
        "num_runs": R,
        "per_run": [
           {
             "mean_relative_error": ...,
             "std_relative_error": ...,
             "mean_matrix_fro": ...,
             "std_matrix_fro": ...,
             "mean_matrix_rel_fro": ...,
             "std_matrix_rel_fro": ...,
           }, ...
        ],
        "per_datapoint_best_errors": [...],  # length N
        "per_datapoint_best_matrix_indices": [...],  # index of run that had best
      }
    """
    if not results_list:
        raise ValueError("Empty results_list")

    # Extract per-run arrays and validate
    runs = []
    for res in results_list:
        if "per_run" in res and isinstance(res["per_run"], list) and len(res["per_run"]) == 1:
            # older callers might wrap run inside per_run list; accept either style:
            run_entry = res["per_run"][0]
        elif "per_run" in res and isinstance(res["per_run"], list) and len(res["per_run"]) > 1:
            # If the passed dict itself contains multiple repeats, flatten them
            # treat each repeat as a separate run
            for r in res["per_run"]:
                runs.append(r)
            continue
        else:
            # direct result dict (single run)
            run_entry = res if "per_sample_errors" in res else None
        if run_entry is None:
            continue
        runs.append(run_entry)

    if not runs:
        raise ValueError("No per-run per-sample data found in results_list")

    # All runs must have per_sample_errors and per_sample_matrices
    for r in runs:
        if r.get("per_sample_errors") is None or r.get("per_sample_matrices") is None:
            raise ValueError(
                "All runs must contain per_sample_errors and per_sample_matrices (deterministic loader required).")

    # Check optional label arrays presence
    labels_available = all(
        (r.get("per_sample_true_labels") is not None and r.get("per_sample_pred_labels") is not None) for r in runs)

    num_runs = len(runs)
    # determine number of datapoints from first run
    N = len(runs[0]["per_sample_errors"])

    # Build numpy arrays: errors shape (R, N), matrices as object lists (R, N)
    errors = _np.zeros((num_runs, N), dtype=_np.float64)
    mats = [[None] * N for _ in range(num_runs)]
    for i, r in enumerate(runs):
        errs = r["per_sample_errors"]
        if len(errs) != N:
            raise ValueError("Inconsistent number of datapoints across runs")
        for j, e in enumerate(errs):
            errors[i, j] = float("nan") if e is None else float(e)
        mats[i] = r["per_sample_matrices"]

    # For each datapoint, find the best (minimum) error across runs and index
    best_error_per_dp = _np.nanmin(errors, axis=0)
    best_idx_per_dp = _np.nanargmin(_np.nan_to_num(errors, nan=_np.inf), axis=0)

    # Collect best matrices per datapoint (from the run that had best error)
    best_mats = [None] * N
    for j in range(N):
        idx = int(best_idx_per_dp[j])
        best_mats[j] = mats[idx][j]  # may be nested list or None

    # Compute relative errors per run and datapoint: (err - best) / (abs(best) + eps)
    eps = 1e-8
    rel_errors = _np.zeros_like(errors)
    for i in range(num_runs):
        for j in range(N):
            be = best_error_per_dp[j]
            ei = errors[i, j]
            if _np.isnan(ei) or _np.isnan(be):
                rel = _np.nan
            else:
                rel = (ei - be) / (abs(be) + eps)
            rel_errors[i, j] = rel

    # Compute matrix Frobenius distances to best matrix per datapoint
    fro_dists = _np.zeros((num_runs, N), dtype=_np.float64)
    fro_rel = _np.zeros((num_runs, N), dtype=_np.float64)
    for i in range(num_runs):
        for j in range(N):
            m_run = mats[i][j]
            m_best = best_mats[j]
            if m_run is None or m_best is None:
                fro_dists[i, j] = _np.nan
                fro_rel[i, j] = _np.nan
                continue
            # convert to numpy arrays
            ma = _np.array(m_run, dtype=_np.float64)
            mb = _np.array(m_best, dtype=_np.float64)
            try:
                # ensure shapes match; if not try broadcasting or skip
                if ma.shape != mb.shape:
                    # attempt to broadcast if mb is 2D and ma is same 2D or replicated
                    # fallback: compute norm of flattened difference
                    diff = (ma.flatten() - mb.flatten())
                else:
                    diff = (ma - mb)
                fro = _np.linalg.norm(diff)
                norm_best = _np.linalg.norm(mb)
                fro_dists[i, j] = fro
                fro_rel[i, j] = fro / (norm_best + eps)
            except Exception:
                fro_dists[i, j] = _np.nan
                fro_rel[i, j] = _np.nan

    # Aggregate per-run statistics
    per_run_stats = []
    for i in range(num_runs):
        valid_rel = rel_errors[i, ~_np.isnan(rel_errors[i])]
        valid_fro = fro_dists[i, ~_np.isnan(fro_dists[i])]
        valid_fro_rel = fro_rel[i, ~_np.isnan(fro_rel[i])]
        stat = {
            "mean_relative_error": float(_np.nanmean(valid_rel)) if valid_rel.size > 0 else None,
            "std_relative_error": float(_np.nanstd(valid_rel)) if valid_rel.size > 0 else None,
            "mean_matrix_fro": float(_np.nanmean(valid_fro)) if valid_fro.size > 0 else None,
            "std_matrix_fro": float(_np.nanstd(valid_fro)) if valid_fro.size > 0 else None,
            "mean_matrix_rel_fro": float(_np.nanmean(valid_fro_rel)) if valid_fro_rel.size > 0 else None,
            "std_matrix_rel_fro": float(_np.nanstd(valid_fro_rel)) if valid_fro_rel.size > 0 else None,
        }
        per_run_stats.append(stat)

    result = {
        "num_datapoints": int(N),
        "num_runs": int(num_runs),
        "per_run_stats": per_run_stats,
        "per_datapoint_best_errors": best_error_per_dp.tolist(),
        "per_datapoint_best_run_index": best_idx_per_dp.tolist(),
        # optionally include summary matrices distances
        "per_run_rel_errors_matrix": _np.where(_np.isfinite(rel_errors), rel_errors, _np.nan).tolist(),
        "per_run_fro_dists_matrix": _np.where(_np.isfinite(fro_dists), fro_dists, _np.nan).tolist(),
        "per_run_fro_rel_matrix": _np.where(_np.isfinite(fro_rel), fro_rel, _np.nan).tolist(),
    }

    # If label info is available, include true labels (from first run) and the predicted label
    # from the run that achieved the best error for each datapoint.
    if labels_available:
        # true labels (assumed identical across runs for a deterministic loader)
        true_labels = runs[0].get("per_sample_true_labels", [None] * N)
        best_pred_labels = [None] * N
        for j in range(N):
            idx = int(best_idx_per_dp[j])
            pred_list = runs[idx].get("per_sample_pred_labels", None)
            best_pred_labels[j] = pred_list[j] if (pred_list is not None and len(pred_list) > j) else None

        # per-run predicted labels matrix (R x N)
        per_run_pred_labels_matrix = []
        for i in range(num_runs):
            pl = runs[i].get("per_sample_pred_labels")
            if pl is None:
                per_run_pred_labels_matrix.append([None] * N)
            else:
                per_run_pred_labels_matrix.append(pl[:N])

        result["per_datapoint_true_labels"] = true_labels
        result["per_datapoint_best_pred_labels"] = best_pred_labels
        result["per_run_pred_labels_matrix"] = per_run_pred_labels_matrix
    else:
        result["per_datapoint_true_labels"] = None
        result["per_datapoint_best_pred_labels"] = None
        result["per_run_pred_labels_matrix"] = None

    return result
