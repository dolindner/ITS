from typing import Literal
import torch
import numpy as np
from src.utils.transformation_problem import TransformationProblem

from search.base_opt import BaseOptimizer


class RSLR(BaseOptimizer):
    """
    Random search plus refinement inspired by (SHGO)-style optimizer.
    1. Draw N global samples via transformation_problem.initial_param
    2. Select candidates using either:
       a. Top-K selection (original method)
       b. k-NN graph-based approximation of Delaunay minimizer selection
    3. Refine each of the candidates by local gradient descent

    Args:
        initial_samples: The number of global samples to draw initially.
        local_runs: The number of local optimization runs to perform.
        local_max_steps: The maximum number of steps for each local optimization run.
        local_opt_class: The PyTorch optimizer class to use for local refinement.
        local_opt_kwargs: Keyword arguments to pass to the local optimizer.
        selection_method: The method used to select candidate points for local refinement.
        acceptance_criterion: Decides what criterion is used when comparing the result of local search with random search.
            - 'step': Takes the best result evaluated across all steps of the local search (including the global initialization).
            - 'always': Always takes the final parameters of the local search, regardless of whether they are the best.
            - 'final': Compares the final local parameters against the result from the random search and takes the best of the two.
        project_param: Whether to project parameters.
        include_zero_always: Whether to always include the identity matrix as a candidate.
    """

    def __init__(
            self,
            initial_samples: int = 100,
            local_runs: int = 4,
            local_max_steps: int = 10,
            local_opt_class=torch.optim.Adam,
            local_opt_kwargs=None,
            selection_method:Literal['topk','knn']='topk',
            acceptance_criterion: Literal['always', 'step', 'final'] = 'step', # decides what criterion is used when comparing the result of local search with random search.
            project_param: bool = True,  # Whether to project parameters.
            include_zero_always: bool = False,  # To always include identity matrix as a candidate.
    ):
        self.initial_samples = initial_samples
        self.local_runs = local_runs
        self.local_max_steps = local_max_steps
        self.local_opt_class = local_opt_class
        self.local_opt_kwargs = local_opt_kwargs or {"lr": 1e-1}
        self.selection_method = selection_method
        self.acceptance_criterion = acceptance_criterion
        self.project_param = project_param
        self.include_zero_always = include_zero_always


    def optimize(self, transformation_problem: TransformationProblem, x,y=None, verbose=False):
        with torch.no_grad():
            batch_size = x.shape[0]
            # 1) global sampling
            if self.include_zero_always:
                # get one fewer so we can prepend zero later
                all_params = transformation_problem.initial_param(
                    batch_size, self.initial_samples - 1
                )

                # build a zero tensor matching the trailing dimensions
                zero_shape = (batch_size, 1) + tuple(all_params.shape[2:])
                zero_param = torch.zeros(
                    zero_shape, device=all_params.device, dtype=all_params.dtype
                )

                # concatenate along the sample dimension
                all_params = torch.cat([zero_param, all_params], dim=1)
            else:
                all_params = transformation_problem.initial_param(
                    batch_size, self.initial_samples
                )


            dim = all_params.shape[-1]
            all_params = all_params.view(-1, dim)
            x_rep = x.repeat_interleave(self.initial_samples, dim=0)
            if y is not None:
                y_rep = y.repeat_interleave(self.initial_samples, dim=0)
            else:
                y_rep = None


            all_err, all_other = transformation_problem.calculate_error(x_rep, all_params, y=y_rep)
            # reshape to (batch, samples, ...)
            all_err = all_err.view(batch_size, self.initial_samples)
            all_other = all_other.view(batch_size, self.initial_samples, -1)
            all_params = all_params.view(batch_size, self.initial_samples, dim)



            # 2) select candidates based on chosen method
            best_params, best_err, best_other = self._select_candidates(
                all_params, all_err, all_other, batch_size, dim
            )

            # flatten for local search
            flat_params = best_params.view(-1, dim).clone().detach().requires_grad_(True)
            x_rep2 = x.repeat_interleave(best_params.shape[1], dim=0)
            y_rep2 = y.repeat_interleave(best_params.shape[1], dim=0) if y is not None else None

            # 3) parallel local gradient descent
            if self.acceptance_criterion == 'step':
                # Keep track of the best parameters found so far during local search
                best_step_params = flat_params.clone()
                best_step_err = best_err.view(-1)
                best_step_other = best_other.view(-1, best_other.shape[-1])

            with torch.enable_grad():
                optimizer = self.local_opt_class([flat_params], **self.local_opt_kwargs)
                for step in range(self.local_max_steps):
                    optimizer.zero_grad()
                    # It's the params *before* the step that correspond to `err`
                    params_before_step = flat_params.clone()
                    err, other = transformation_problem.calculate_error(x_rep2, flat_params, y=y_rep2)
                    loss = err.mean()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        if self.project_param:
                            flat_params.data = transformation_problem.correct_param(flat_params).data
                        else:
                            flat_params.data = transformation_problem.normalize(flat_params).data

                        if self.acceptance_criterion == 'step':
                            improved_mask = err < best_step_err
                            best_step_params = torch.where(improved_mask.unsqueeze(-1), params_before_step, best_step_params)
                            best_step_err = torch.where(improved_mask, err, best_step_err)
                            best_step_other = torch.where(improved_mask.unsqueeze(-1), other, best_step_other)

            if self.local_max_steps>0:
                # final evaluation
                with torch.no_grad():
                    # After the loop, flat_params is at the final position. Evaluate it.
                    final_err, final_other = transformation_problem.calculate_error(x_rep2, flat_params, y=y_rep2)

                    if self.acceptance_criterion == 'step':
                        # Compare the final position with the best-so-far from previous steps
                        improved_mask = final_err < best_step_err
                        final_params_flat = torch.where(improved_mask.unsqueeze(-1), flat_params, best_step_params)
                        final_err = torch.where(improved_mask, final_err, best_step_err)
                        final_other = torch.where(improved_mask.unsqueeze(-1), final_other, best_step_other)
                    else: # 'always' or 'final'
                        final_params_flat = flat_params
                        # final_err and final_other are already set from the evaluation above
            else:
                final_params_flat = flat_params.detach()
                final_err = best_err.view(-1).detach()
                final_other = best_other.view(-1, best_other.shape[-1]).detach()

            # reshape back to (batch, runs, ...)
            n_candidates = best_params.shape[1]
            final_params = final_params_flat.view(batch_size, n_candidates, dim)
            final_err = final_err.view(batch_size, n_candidates)
            final_other = final_other.view(batch_size, n_candidates, -1)

            # Accept only improvements if option enabled
            if self.acceptance_criterion == 'final':
                improved = final_err < best_err
                # Where not improved, revert to original (random sample) values
                final_params = torch.where(improved.unsqueeze(-1), final_params, best_params)
                final_err = torch.where(improved, final_err, best_err)
                final_other = torch.where(improved.unsqueeze(-1), final_other, best_other)

            # consolidate best per sample
            return transformation_problem.consolidate(x, final_params, final_err, final_other)




    def _select_candidates(self, all_params, all_err, all_other, batch_size, dim):
        """Select candidates based on the chosen selection method"""
        if self.selection_method == 'topk':
            return self._select_topk(all_params, all_err, all_other, batch_size, dim)
        elif self.selection_method == 'knn':
            return self._select_knn(all_params, all_err, all_other, batch_size, dim)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _select_knn(self, all_params, all_err, all_other, batch_size, dim):
        device = all_params.device
        b, n, d = all_params.shape
        runs = min(self.local_runs, n)

        k = min(max(2 * dim, 12), n - 1) if n > 1 else 0
        if k == 0:
            idx = torch.zeros((b, 1), dtype=torch.long, device=device)
            best_params = torch.gather(all_params, 1, idx.unsqueeze(-1).expand(-1, -1, d))
            best_err = torch.gather(all_err, 1, idx)
            best_other = torch.gather(all_other, 1, idx.unsqueeze(-1).expand(-1, -1, all_other.shape[-1]))
            return best_params, best_err, best_other

        # 1) Pairwise distances
        dists = torch.cdist(all_params, all_params)
        knn_idx = torch.topk(dists, k=k + 1, largest=False).indices[..., 1:]  # (b, n, k)

        # 2) Compare errors with neighbors
        neigh_errs = torch.gather(all_err.unsqueeze(2).expand(-1, -1, k), 1, knn_idx)
        is_min = (all_err.unsqueeze(2) < neigh_errs).all(dim=2)  # (b, n)

        # 3) Mask non-minimizers with +inf
        masked_errs = all_err.clone()
        masked_errs[~is_min] = float("inf")

        # 4) Select minimizers (some slots may still be inf)
        topk_min = torch.topk(masked_errs, runs, largest=False)
        chosen_idx = topk_min.indices  # (b, runs)

        # 5) Ensure we don’t duplicate: fill missing slots only from non-minimizers
        n_mins = is_min.sum(dim=1)  # number of minimizers per batch
        need_fill = runs - torch.minimum(n_mins, torch.tensor(runs, device=device))

        if need_fill.any():
            # For batches with fewer minimizers, compute global-best **excluding minimizers**
            not_min = ~is_min  # (b, n)
            # set minimizer errors to inf so they won’t be picked again
            global_errs = all_err.clone()
            global_errs[~not_min] = float("inf")

            global_best = torch.topk(global_errs, runs, largest=False).indices  # (b, runs)

            # Fill slots that had inf with non-minimizer bests
            fill_mask = torch.isinf(torch.gather(masked_errs, 1, chosen_idx))  # (b, runs)
            chosen_idx = torch.where(fill_mask, global_best, chosen_idx)

        # 6) Gather results
        best_params = torch.gather(all_params, 1, chosen_idx.unsqueeze(-1).expand(-1, -1, d))
        best_err = torch.gather(all_err, 1, chosen_idx)
        best_other = torch.gather(all_other, 1, chosen_idx.unsqueeze(-1).expand(-1, -1, all_other.shape[-1]))

        return best_params, best_err, best_other


    def _select_topk(self, all_params, all_err, all_other, batch_size, dim):
        """Original topK selection method"""
        topk = torch.topk(all_err, min(self.local_runs, all_err.shape[1]), largest=False)
        idx = topk.indices  # (batch, local_runs)

        # gather best params and data
        best_params = torch.gather(
            all_params, 1, idx.unsqueeze(-1).expand(-1, -1, dim)
        )
        best_other = torch.gather(
            all_other, 1, idx.unsqueeze(-1).expand(-1, -1, all_other.shape[-1])
        )
        best_err = torch.gather(all_err, 1, idx)

        return best_params, best_err, best_other
